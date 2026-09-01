//! Paged KV-cache allocation with content-addressed prefix reuse.
//!
//! Attention state is stored in fixed-size blocks of `block_size` tokens, so a
//! sequence's KV entries need not be contiguous and a long context costs
//! exactly as many blocks as it uses. Blocks whose entire prefix path matches
//! are shared by reference rather than recomputed, which is what makes a
//! repeated system prompt nearly free on its second use.
//!
//! ```
//! use stride_kvcache::{KvCache, KvCacheConfig};
//!
//! let mut cache = KvCache::new(KvCacheConfig { num_blocks: 64, block_size: 16 });
//! let prompt: Vec<u32> = (0..48).collect();
//!
//! // First use: nothing cached, every block computed and published.
//! let m = cache.acquire_prefix("acme", &prompt);
//! assert_eq!(m.num_tokens, 0);
//! let blocks = cache.allocate(3).unwrap();
//! cache.publish("acme", &prompt, &blocks);
//! cache.release(&blocks);
//!
//! // Second use: the prefix is still resident.
//! let m = cache.acquire_prefix("acme", &prompt);
//! assert_eq!(m.num_tokens, 32, "all but the final block is reused");
//! ```

pub mod allocator;
pub mod hash;
pub mod prefix;

pub use allocator::{BlockAllocator, BlockId};
pub use hash::BlockHash;
pub use prefix::PrefixCache;

use stride_core::{Error, Result, TokenId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheConfig {
    /// Total physical blocks carved out of device memory.
    pub num_blocks: usize,
    /// Tokens per block. Larger blocks waste more on short sequences; smaller
    /// blocks cost more indirection per attention step.
    pub block_size: usize,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            num_blocks: 2048,
            block_size: 16,
        }
    }
}

/// Result of looking a prompt up in the prefix cache.
#[derive(Debug, Clone, Default)]
pub struct PrefixMatch {
    /// Shared blocks, already referenced on the caller's behalf.
    pub blocks: Vec<BlockId>,
    /// Prompt tokens covered by those blocks.
    pub num_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CacheStats {
    pub capacity: usize,
    /// Blocks referenced by at least one live sequence.
    pub live: usize,
    /// Unreferenced blocks still holding reusable KV entries.
    pub cached: usize,
    pub free: usize,
    pub block_hit_rate: f64,
}

/// Paged KV cache: block allocation, prefix reuse and eviction under one
/// invariant — every block is live, cached, or free, never two at once.
#[derive(Debug)]
pub struct KvCache {
    block_size: usize,
    alloc: BlockAllocator,
    prefix: PrefixCache,
}

impl KvCache {
    pub fn new(config: KvCacheConfig) -> Self {
        assert!(config.block_size > 0, "block_size must be positive");
        Self {
            block_size: config.block_size,
            alloc: BlockAllocator::new(config.num_blocks),
            prefix: PrefixCache::new(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn capacity(&self) -> usize {
        self.alloc.capacity()
    }

    /// Blocks needed to hold `num_tokens`, rounding up.
    pub fn blocks_for(&self, num_tokens: usize) -> usize {
        num_tokens.div_ceil(self.block_size)
    }

    /// Blocks that could be handed out right now, counting cached blocks that
    /// eviction would reclaim.
    pub fn num_allocatable(&self) -> usize {
        self.alloc.num_free() + self.prefix.num_evictable()
    }

    /// Fraction of the pool holding live KV entries.
    pub fn live_utilization(&self) -> f64 {
        let live = self.alloc.capacity() - self.num_allocatable();
        live as f64 / self.alloc.capacity() as f64
    }

    /// Look `tokens` up under `tenant` and take a reference to every cached
    /// block along the matching prefix.
    ///
    /// The match always stops at least one token short of the prompt. A
    /// forward pass needs at least one uncomputed token to produce logits
    /// from; a fully cached prompt would otherwise have nothing to run and
    /// could not emit a first token.
    pub fn acquire_prefix(&mut self, tenant: &str, tokens: &[TokenId]) -> PrefixMatch {
        let hashes = hash::chain(tenant, tokens, self.block_size);
        // Never consume the block that ends exactly at the prompt's last token.
        let ceiling = if tokens.len().is_multiple_of(self.block_size) {
            hashes.len().saturating_sub(1)
        } else {
            hashes.len()
        };

        let mut blocks = Vec::new();
        for &h in hashes.iter().take(ceiling) {
            let Some(block) = self.prefix.get(h) else {
                break;
            };
            if self.alloc.is_live(block) {
                self.alloc.incref(block);
            } else {
                self.alloc.adopt(block);
            }
            self.prefix.mark_in_use(block);
            blocks.push(block);
        }

        PrefixMatch {
            num_tokens: blocks.len() * self.block_size,
            blocks,
        }
    }

    /// Allocate `n` blocks, evicting cached blocks if the free list runs dry.
    ///
    /// Either every block is allocated or none is — a partial allocation is
    /// rolled back before returning, so a failed admission leaves no leak.
    pub fn allocate(&mut self, n: usize) -> Result<Vec<BlockId>> {
        if n > self.num_allocatable() {
            return Err(Error::OutOfBlocks {
                requested: n,
                available: self.num_allocatable(),
            });
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let block = match self.alloc.take_free() {
                Some(b) => b,
                None => match self.prefix.evict_lru() {
                    Some(b) => {
                        debug_assert!(!self.alloc.is_live(b), "evicted a live block");
                        self.alloc.adopt(b);
                        b
                    }
                    None => {
                        self.release(&out);
                        return Err(Error::OutOfBlocks {
                            requested: n,
                            available: out.len(),
                        });
                    }
                },
            };
            out.push(block);
        }
        Ok(out)
    }

    /// Publish filled blocks under their content addresses.
    ///
    /// `blocks` must cover `tokens` from the start. Only blocks that are
    /// completely full are published; a partial tail block is skipped, since
    /// appending a token would change its identity.
    pub fn publish(&mut self, tenant: &str, tokens: &[TokenId], blocks: &[BlockId]) {
        let hashes = hash::chain(tenant, tokens, self.block_size);
        for (&h, &b) in hashes.iter().zip(blocks.iter()) {
            debug_assert!(self.alloc.is_live(b), "publishing an unreferenced block");
            self.prefix.insert(h, b);
        }
    }

    /// Drop one reference per block. Blocks reaching zero become cached if the
    /// index still knows them, and free otherwise.
    pub fn release(&mut self, blocks: &[BlockId]) {
        for &b in blocks {
            if self.alloc.decref(b) == 0 {
                if self.prefix.contains_block(b) {
                    self.prefix.mark_evictable(b);
                } else {
                    self.alloc.push_free(b);
                }
            }
        }
    }

    pub fn stats(&self) -> CacheStats {
        let free = self.alloc.num_free();
        let cached = self.prefix.num_evictable();
        CacheStats {
            capacity: self.alloc.capacity(),
            live: self.alloc.capacity() - free - cached,
            cached,
            free,
            block_hit_rate: self.prefix.hit_rate(),
        }
    }

    pub fn reset_stats(&mut self) {
        self.prefix.reset_stats();
    }

    /// Panics unless the live/cached/free partition holds. Called by tests and
    /// by debug builds of the scheduler after each step.
    pub fn assert_invariants(&self) {
        let s = self.stats();
        assert_eq!(
            s.live + s.cached + s.free,
            s.capacity,
            "block partition does not cover the pool: {s:?}"
        );
    }
}
