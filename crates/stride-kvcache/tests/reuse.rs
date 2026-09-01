//! End-to-end behaviour of the paged cache: reuse, isolation, eviction and the
//! partition invariant that keeps a cached block from being handed out twice.

use stride_kvcache::{KvCache, KvCacheConfig};

const BS: usize = 16;

fn cache(num_blocks: usize) -> KvCache {
    KvCache::new(KvCacheConfig {
        num_blocks,
        block_size: BS,
    })
}

/// Run one sequence through the cache the way the scheduler would, and return
/// how many prompt tokens the prefix cache covered.
fn serve(c: &mut KvCache, tenant: &str, prompt: &[u32]) -> usize {
    let m = c.acquire_prefix(tenant, prompt);
    let need = c.blocks_for(prompt.len()) - m.blocks.len();
    let fresh = c.allocate(need).expect("capacity for this test");

    let mut all = m.blocks.clone();
    all.extend_from_slice(&fresh);
    c.publish(tenant, prompt, &all);
    c.release(&all);
    c.assert_invariants();
    m.num_tokens
}

#[test]
fn repeated_prompt_is_served_from_cache() {
    let mut c = cache(64);
    let prompt: Vec<u32> = (0..64).collect();

    assert_eq!(serve(&mut c, "acme", &prompt), 0, "cold start");
    // Four full blocks, but the last is withheld so there is something to run.
    assert_eq!(serve(&mut c, "acme", &prompt), 48);
    assert_eq!(serve(&mut c, "acme", &prompt), 48, "stays warm");
}

#[test]
fn shared_system_prompt_is_reused_across_different_suffixes() {
    let mut c = cache(64);
    let system: Vec<u32> = (0..32).collect();

    let a: Vec<u32> = system.iter().copied().chain(100..140).collect();
    let b: Vec<u32> = system.iter().copied().chain(200..250).collect();

    assert_eq!(serve(&mut c, "acme", &a), 0);
    assert_eq!(
        serve(&mut c, "acme", &b),
        32,
        "the shared 32-token system prompt must be reused"
    );
}

#[test]
fn a_diverging_prefix_stops_the_match_at_the_divergence() {
    let mut c = cache(64);
    let a: Vec<u32> = (0..16).chain(0..16).chain(0..16).collect();
    let mut b = a.clone();
    b[20] = 9999; // corrupt the second block

    serve(&mut c, "acme", &a);
    assert_eq!(
        serve(&mut c, "acme", &b),
        16,
        "only the first block may be reused"
    );
}

#[test]
fn tenants_never_share_blocks_even_with_identical_prompts() {
    let mut c = cache(64);
    let prompt: Vec<u32> = (0..64).collect();

    assert_eq!(serve(&mut c, "acme", &prompt), 0);
    assert_eq!(
        serve(&mut c, "globex", &prompt),
        0,
        "byte-identical content under another tenant must miss"
    );
    // And each tenant still gets its own warm path.
    assert_eq!(serve(&mut c, "acme", &prompt), 48);
    assert_eq!(serve(&mut c, "globex", &prompt), 48);
}

#[test]
fn eviction_reclaims_cached_blocks_under_pressure() {
    let mut c = cache(8);
    let first: Vec<u32> = (0..64).collect(); // 4 blocks
    serve(&mut c, "acme", &first);
    assert_eq!(c.stats().cached, 4);

    // A second, unrelated prompt needs more than the free list holds.
    let second: Vec<u32> = (1000..1096).collect(); // 6 blocks
    serve(&mut c, "acme", &second);
    c.assert_invariants();

    // The older prefix has been partly evicted to make room.
    assert!(
        serve(&mut c, "acme", &first) < 48,
        "evicted blocks must not report as hits"
    );
}

#[test]
fn live_blocks_survive_pressure_from_other_sequences() {
    let mut c = cache(8);
    let held: Vec<u32> = (0..32).collect();

    let m = c.acquire_prefix("acme", &held);
    let blocks = c
        .allocate(c.blocks_for(held.len()) - m.blocks.len())
        .unwrap();
    c.publish("acme", &held, &blocks);
    // Deliberately do NOT release: this sequence is still generating.

    let other: Vec<u32> = (500..596).collect();
    let m2 = c.acquire_prefix("acme", &other);
    let want = c.blocks_for(other.len()) - m2.blocks.len();
    let got = c.allocate(want.min(c.num_allocatable()));
    assert!(got.is_ok());
    c.assert_invariants();

    assert!(
        c.stats().live >= blocks.len(),
        "a live sequence's blocks must never be reclaimed"
    );
}

#[test]
fn over_allocation_fails_without_leaking() {
    let mut c = cache(4);
    let before = c.num_allocatable();
    let err = c.allocate(5);
    assert!(err.is_err(), "must refuse to over-allocate");
    assert_eq!(
        c.num_allocatable(),
        before,
        "a failed allocation must leave the pool untouched"
    );
    c.assert_invariants();
}

/// Negative control. If the tenant were dropped from the hash chain, the
/// isolation test above would pass for the wrong reason. This asserts the
/// mechanism itself: two tenants hashing the same tokens must land on
/// different addresses.
#[test]
fn negative_control_isolation_comes_from_the_address_not_a_check() {
    use stride_kvcache::hash;
    let tokens: Vec<u32> = (0..48).collect();
    let a = hash::chain("acme", &tokens, BS);
    let b = hash::chain("globex", &tokens, BS);
    assert_eq!(a.len(), 3);
    assert!(
        a.iter().zip(b.iter()).all(|(x, y)| x != y),
        "no position in the chain may collide across tenants"
    );
}
