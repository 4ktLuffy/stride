//! Physical block bookkeeping: reference counts plus a list of truly empty
//! blocks.
//!
//! Every block is in exactly one of three states, and the invariant is
//! maintained by [`crate::KvCache`], not here:
//!
//! - **live**    — refcount > 0, held by at least one sequence
//! - **cached**  — refcount == 0, still indexed by content hash, reclaimable
//! - **free**    — refcount == 0, contents meaningless, on the free list
//!
//! A cached block is deliberately *not* on the free list. If it were, an
//! allocation could hand it out while the prefix index still pointed at it,
//! and a later cache hit would read another sequence's KV entries. Keeping the
//! two pools disjoint makes that class of bug unrepresentable.

/// Index of a physical KV block in device memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u32);

impl BlockId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug)]
pub struct BlockAllocator {
    free: Vec<BlockId>,
    refcount: Vec<u32>,
}

impl BlockAllocator {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "block allocator needs at least one block");
        Self {
            free: (0..capacity).rev().map(|i| BlockId(i as u32)).collect(),
            refcount: vec![0; capacity],
        }
    }

    pub fn capacity(&self) -> usize {
        self.refcount.len()
    }

    /// Blocks on the free list. Does not count reclaimable cached blocks.
    pub fn num_free(&self) -> usize {
        self.free.len()
    }

    /// Blocks holding live or cached KV entries.
    pub fn num_occupied(&self) -> usize {
        self.capacity() - self.num_free()
    }

    pub fn refcount(&self, block: BlockId) -> u32 {
        self.refcount[block.index()]
    }

    pub fn is_live(&self, block: BlockId) -> bool {
        self.refcount(block) > 0
    }

    /// Take an empty block, with a reference count of one.
    pub fn take_free(&mut self) -> Option<BlockId> {
        let block = self.free.pop()?;
        debug_assert_eq!(self.refcount[block.index()], 0);
        self.refcount[block.index()] = 1;
        Some(block)
    }

    /// Take the first reference to a block that is currently unreferenced —
    /// a cache hit reviving a cached block, or a block just evicted.
    pub fn adopt(&mut self, block: BlockId) {
        debug_assert_eq!(
            self.refcount[block.index()], 0,
            "adopt is only for unreferenced blocks"
        );
        self.refcount[block.index()] = 1;
    }

    /// Take an additional reference to a live block.
    pub fn incref(&mut self, block: BlockId) -> u32 {
        let rc = &mut self.refcount[block.index()];
        debug_assert!(*rc > 0, "cannot incref an unreferenced block");
        *rc += 1;
        *rc
    }

    /// Drop one reference. Reaching zero does **not** return the block to the
    /// free list — the caller decides whether it becomes cached or free.
    pub fn decref(&mut self, block: BlockId) -> u32 {
        let rc = &mut self.refcount[block.index()];
        debug_assert!(*rc > 0, "double free of block {block:?}");
        *rc -= 1;
        *rc
    }

    /// Return an unreferenced block to the free list.
    pub fn push_free(&mut self, block: BlockId) {
        debug_assert_eq!(
            self.refcount[block.index()], 0,
            "cannot free a referenced block"
        );
        debug_assert!(
            !self.free.contains(&block),
            "block {block:?} freed twice"
        );
        self.free.push(block);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_until_exhausted_then_recover() {
        let mut a = BlockAllocator::new(4);
        let blocks: Vec<_> = (0..4).map(|_| a.take_free().unwrap()).collect();
        assert_eq!(a.num_free(), 0);
        assert!(a.take_free().is_none(), "must not over-allocate");

        for b in &blocks {
            assert_eq!(a.decref(*b), 0);
            a.push_free(*b);
        }
        assert_eq!(a.num_free(), 4);
        assert_eq!(a.num_occupied(), 0);
    }

    #[test]
    fn shared_block_survives_until_last_holder_leaves() {
        let mut a = BlockAllocator::new(2);
        let b = a.take_free().unwrap();
        a.incref(b);
        a.incref(b);
        assert_eq!(a.refcount(b), 3);

        assert_eq!(a.decref(b), 2);
        assert_eq!(a.decref(b), 1);
        assert!(a.is_live(b), "still held by one owner");
        assert_eq!(a.decref(b), 0);
        assert!(!a.is_live(b));
    }

    #[test]
    fn dropping_to_zero_does_not_implicitly_free() {
        let mut a = BlockAllocator::new(2);
        let b = a.take_free().unwrap();
        a.decref(b);
        assert_eq!(
            a.num_free(),
            1,
            "an unreferenced block stays out of the free list until told"
        );
        a.adopt(b);
        assert_eq!(a.refcount(b), 1, "revived without touching the free list");
    }
}
