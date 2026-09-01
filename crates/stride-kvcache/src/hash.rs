//! Content addressing for KV blocks.
//!
//! A block hash covers the tenant, every token in the block, and the hash of
//! the preceding block. Chaining is what makes a hash identify a whole prefix
//! path rather than 16 tokens in isolation: two sequences share a cached block
//! only if their entire history up to that point agrees.
//!
//! Seeding the chain with the tenant is deliberate. Cross-tenant sharing is not
//! prevented by a check that someone can forget to write — identical token
//! content under two tenants simply produces two different hashes, so the
//! lookup misses.

use stride_core::TokenId;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Content address of one full KV block, chained to its prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockHash(pub u64);

#[inline]
fn fnv1a(mut acc: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        acc ^= b as u64;
        acc = acc.wrapping_mul(FNV_PRIME);
    }
    acc
}

/// Root of a tenant's hash chain. Never collides with a block hash by
/// construction, because every block hash mixes in at least one token.
pub fn tenant_root(tenant: &str) -> BlockHash {
    BlockHash(fnv1a(FNV_OFFSET, tenant.as_bytes()))
}

/// Hash one full block of tokens, chained onto its parent.
pub fn block_hash(parent: BlockHash, tokens: &[TokenId]) -> BlockHash {
    let mut acc = fnv1a(FNV_OFFSET, &parent.0.to_le_bytes());
    for &t in tokens {
        acc = fnv1a(acc, &t.to_le_bytes());
    }
    BlockHash(acc)
}

/// Hash every complete `block_size` chunk of `tokens` for `tenant`.
///
/// A trailing partial block is not hashed: a block is only content-addressable
/// once it is full, because appending a token would change its identity.
pub fn chain(tenant: &str, tokens: &[TokenId], block_size: usize) -> Vec<BlockHash> {
    let mut parent = tenant_root(tenant);
    let mut out = Vec::with_capacity(tokens.len() / block_size.max(1));
    for chunk in tokens.chunks_exact(block_size) {
        parent = block_hash(parent, chunk);
        out.push(parent);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_content_under_different_tenants_does_not_collide() {
        let tokens: Vec<TokenId> = (0..32).collect();
        let a = chain("tenant-a", &tokens, 16);
        let b = chain("tenant-b", &tokens, 16);
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
        assert_ne!(a[0], b[0], "tenant must be part of the address");
        assert_ne!(a[1], b[1]);
    }

    #[test]
    fn chain_is_prefix_stable() {
        let short: Vec<TokenId> = (0..32).collect();
        let long: Vec<TokenId> = (0..64).collect();
        let a = chain("t", &short, 16);
        let b = chain("t", &long, 16);
        assert_eq!(a[..], b[..2], "a shared prefix must hash identically");
    }

    #[test]
    fn diverging_history_breaks_the_chain() {
        let a: Vec<TokenId> = vec![1; 16].into_iter().chain(vec![9; 16]).collect();
        let b: Vec<TokenId> = vec![2; 16].into_iter().chain(vec![9; 16]).collect();
        let (ha, hb) = (chain("t", &a, 16), chain("t", &b, 16));
        assert_ne!(ha[0], hb[0]);
        assert_ne!(
            ha[1], hb[1],
            "identical second block must differ once the prefix differs"
        );
    }

    #[test]
    fn partial_trailing_block_is_not_addressable() {
        let tokens: Vec<TokenId> = (0..20).collect();
        assert_eq!(chain("t", &tokens, 16).len(), 1);
    }
}
