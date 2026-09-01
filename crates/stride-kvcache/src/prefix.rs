//! Content-addressed index from block hash to physical block, with LRU
//! eviction restricted to blocks nobody currently holds.
//!
//! An entry stays in the index after its last sequence finishes. That is the
//! whole point: the next request carrying the same system prompt finds the
//! block still resident and skips recomputing it. Only memory pressure removes
//! it, and only if no live sequence is using it.

use std::collections::{BTreeSet, HashMap};

use crate::allocator::BlockId;
use crate::hash::BlockHash;

#[derive(Debug, Default)]
pub struct PrefixCache {
    by_hash: HashMap<BlockHash, BlockId>,
    by_block: HashMap<BlockId, BlockHash>,
    /// Eviction candidates, ordered by staleness. Holds only unreferenced blocks.
    evictable: BTreeSet<(u64, BlockId)>,
    stamp: HashMap<BlockId, u64>,
    clock: u64,
    hits: u64,
    misses: u64,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Look up a cached block. Records a hit or miss for the reuse metric.
    pub fn get(&mut self, hash: BlockHash) -> Option<BlockId> {
        match self.by_hash.get(&hash).copied() {
            Some(block) => {
                self.hits += 1;
                Some(block)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Publish a filled block under its content address. Returns false if the
    /// address was already claimed.
    ///
    /// First writer wins. A duplicate is a block that was recomputed while an
    /// equivalent one was already resident — most often the final prompt block,
    /// which [`crate::KvCache::acquire_prefix`] withholds on purpose. The
    /// existing entry is equally valid and may already be shared, so it is left
    /// alone; the duplicate stays unindexed and returns to the free list when
    /// its owner releases it.
    pub fn insert(&mut self, hash: BlockHash, block: BlockId) -> bool {
        if self.by_hash.contains_key(&hash) {
            return false;
        }
        self.by_hash.insert(hash, block);
        self.by_block.insert(block, hash);
        let t = self.tick();
        self.stamp.insert(block, t);
        true
    }

    /// Mark a block as an eviction candidate — its reference count reached zero
    /// but the contents are still valid and worth keeping.
    pub fn mark_evictable(&mut self, block: BlockId) {
        if !self.by_block.contains_key(&block) {
            return;
        }
        let t = self.tick();
        if let Some(old) = self.stamp.insert(block, t) {
            self.evictable.remove(&(old, block));
        }
        self.evictable.insert((t, block));
    }

    /// Mark a block as live again — it was just handed to a sequence.
    pub fn mark_in_use(&mut self, block: BlockId) {
        if let Some(&t) = self.stamp.get(&block) {
            self.evictable.remove(&(t, block));
        }
        let t = self.tick();
        self.stamp.insert(block, t);
    }

    /// Drop the least recently used unreferenced entry, returning its block so
    /// the caller can recycle it. `None` means every cached block is live.
    pub fn evict_lru(&mut self) -> Option<BlockId> {
        let &(t, block) = self.evictable.iter().next()?;
        self.evictable.remove(&(t, block));
        self.stamp.remove(&block);
        if let Some(hash) = self.by_block.remove(&block) {
            self.by_hash.remove(&hash);
        }
        Some(block)
    }

    /// True if this block currently backs a cache entry.
    pub fn contains_block(&self, block: BlockId) -> bool {
        self.by_block.contains_key(&block)
    }

    pub fn len(&self) -> usize {
        self.by_hash.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_hash.is_empty()
    }

    pub fn num_evictable(&self) -> usize {
        self.evictable.len()
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Share of block lookups served from cache since start.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(n: u64) -> BlockHash {
        BlockHash(n)
    }

    #[test]
    fn evicts_least_recently_used_first() {
        let mut c = PrefixCache::new();
        for i in 0..3u32 {
            c.insert(h(i as u64), BlockId(i));
        }
        // Release in a deliberately non-sequential order.
        c.mark_evictable(BlockId(1));
        c.mark_evictable(BlockId(0));
        c.mark_evictable(BlockId(2));

        assert_eq!(c.evict_lru(), Some(BlockId(1)));
        assert_eq!(c.evict_lru(), Some(BlockId(0)));
        assert_eq!(c.evict_lru(), Some(BlockId(2)));
        assert_eq!(c.evict_lru(), None);
        assert!(c.is_empty());
    }

    #[test]
    fn live_blocks_are_never_evicted() {
        let mut c = PrefixCache::new();
        c.insert(h(1), BlockId(1));
        c.insert(h(2), BlockId(2));
        c.mark_evictable(BlockId(2));
        c.mark_in_use(BlockId(2));

        assert_eq!(c.num_evictable(), 0);
        assert_eq!(c.evict_lru(), None, "nothing is releasable");
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn reuse_after_release_makes_the_entry_recent_again() {
        let mut c = PrefixCache::new();
        c.insert(h(1), BlockId(1));
        c.insert(h(2), BlockId(2));
        c.mark_evictable(BlockId(1));
        c.mark_evictable(BlockId(2));

        // Block 1 is picked up again, then released; block 2 is now the oldest.
        c.mark_in_use(BlockId(1));
        c.mark_evictable(BlockId(1));

        assert_eq!(c.evict_lru(), Some(BlockId(2)));
    }

    #[test]
    fn duplicate_address_keeps_the_incumbent() {
        let mut c = PrefixCache::new();
        assert!(c.insert(h(1), BlockId(7)));
        assert!(!c.insert(h(1), BlockId(9)), "second writer is refused");
        assert_eq!(c.get(h(1)), Some(BlockId(7)));
        assert!(
            !c.contains_block(BlockId(9)),
            "the duplicate stays unindexed"
        );
    }

    #[test]
    fn hit_rate_tracks_lookups() {
        let mut c = PrefixCache::new();
        c.insert(h(1), BlockId(1));
        assert!(c.get(h(1)).is_some());
        assert!(c.get(h(2)).is_none());
        assert!(c.get(h(3)).is_none());
        assert_eq!(c.hits(), 1);
        assert_eq!(c.misses(), 2);
        assert!((c.hit_rate() - 1.0 / 3.0).abs() < 1e-9);
    }
}
