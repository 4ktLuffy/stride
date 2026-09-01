//! Behavioural tests for the scheduler.
//!
//! Each drives the runtime loop with a stand-in for model execution: whatever
//! the scheduler puts in `decode`, the harness answers with a token. That is
//! enough to exercise batching, chunking, ranking and preemption, none of
//! which depend on what the tokens actually are.

use stride_core::{Request, SamplingParams, ServiceClass, Tick};
use stride_kvcache::KvCacheConfig;
use stride_sched::{Finished, Scheduler, SchedulerConfig};

const STEP_US: Tick = 1_000;
const FILLER: u32 = 7;

fn scheduler(cfg: SchedulerConfig, num_blocks: usize) -> Scheduler {
    Scheduler::new(
        cfg,
        KvCacheConfig {
            num_blocks,
            block_size: 16,
        },
    )
}

/// Run the loop for `steps` steps, answering every decode slot with a token.
fn drive(s: &mut Scheduler, steps: usize, now: &mut Tick) -> Vec<Finished> {
    let mut done = Vec::new();
    for _ in 0..steps {
        let batch = s.step(*now);
        let outputs: Vec<_> = batch.decode.iter().map(|&id| (id, FILLER)).collect();
        done.extend(s.on_tokens(&outputs, *now));
        *now += STEP_US;
    }
    done
}

fn request(tenant: &str, prompt_len: usize, max_tokens: usize, at: Tick) -> Request {
    Request::new(tenant, (0..prompt_len as u32).collect(), at)
        .with_params(SamplingParams::greedy(max_tokens))
}

#[test]
fn a_request_runs_to_its_token_limit_and_reports_latency() {
    let mut s = scheduler(SchedulerConfig::default(), 256);
    let mut now = 0;
    s.admit(request("acme", 40, 8, now)).unwrap();

    let done = drive(&mut s, 24, &mut now);
    assert_eq!(done.len(), 1, "the request should finish");
    let f = &done[0];
    assert_eq!(f.output.len(), 8, "exactly max_tokens generated");
    assert!(f.ttft_us.is_some(), "time-to-first-token must be recorded");
    assert!(f.mean_itl_us.is_some(), "inter-token latency must be recorded");
    assert_eq!(s.num_running(), 0);
    assert_eq!(s.num_waiting(), 0);
}

#[test]
fn a_long_prompt_is_split_across_steps_instead_of_monopolising_one() {
    let cfg = SchedulerConfig {
        max_batch_tokens: 128,
        ..Default::default()
    };
    let mut s = scheduler(cfg, 512);
    let mut now = 0;
    s.admit(request("acme", 1024, 4, now)).unwrap();

    let mut chunks = 0;
    for _ in 0..40 {
        let batch = s.step(now);
        assert!(
            batch.num_tokens() <= 128,
            "the token budget must hold: {}",
            batch.num_tokens()
        );
        chunks += batch.prefill.len();
        let outputs: Vec<_> = batch.decode.iter().map(|&id| (id, FILLER)).collect();
        s.on_tokens(&outputs, now);
        now += STEP_US;
    }
    assert!(
        chunks >= 8,
        "a 1024-token prompt at 128 tokens/step needs at least 8 chunks, saw {chunks}"
    );
}

#[test]
fn decoding_continues_while_a_long_prompt_is_prefilled() {
    let cfg = SchedulerConfig {
        max_batch_tokens: 256,
        ..Default::default()
    };
    let mut s = scheduler(cfg, 1024);
    let mut now = 0;

    // Get one sequence generating first.
    s.admit(request("acme", 32, 200, now)).unwrap();
    drive(&mut s, 4, &mut now);

    // Now a large prompt arrives mid-stream.
    s.admit(request("acme", 2048, 4, now)).unwrap();

    let mut steps_with_prefill = 0;
    let mut decodes_during_prefill = 0;
    for _ in 0..30 {
        let batch = s.step(now);
        if !batch.prefill.is_empty() {
            steps_with_prefill += 1;
            decodes_during_prefill += batch.decode.len();
        }
        let outputs: Vec<_> = batch.decode.iter().map(|&id| (id, FILLER)).collect();
        s.on_tokens(&outputs, now);
        now += STEP_US;
    }
    assert!(steps_with_prefill > 4, "the big prompt should span many steps");
    assert!(
        decodes_during_prefill >= steps_with_prefill,
        "the running sequence must keep emitting tokens throughout the prefill \
         ({decodes_during_prefill} decodes over {steps_with_prefill} prefill steps)"
    );
}

#[test]
fn an_aged_background_request_outranks_a_fresh_interactive_one() {
    let cfg = SchedulerConfig {
        max_batch_seqs: 1,
        max_batch_tokens: 4096,
        ..Default::default()
    };
    let mut s = scheduler(cfg, 512);

    // Background budget is two minutes; this one has nearly spent it.
    let old = s
        .admit(request("acme", 64, 4, 0).with_class(ServiceClass::Background))
        .unwrap()[0];

    let now = 121_000_000; // 121 s later
    let fresh = s
        .admit(request("acme", 64, 4, now).with_class(ServiceClass::Interactive))
        .unwrap()[0];

    let batch = s.step(now);
    assert_eq!(batch.prefill.len(), 1, "only one slot is available");
    assert_eq!(
        batch.prefill[0].seq, old,
        "the request about to breach its deadline must go first, \
         even though its class is lower"
    );
    assert_ne!(batch.prefill[0].seq, fresh);
}

#[test]
fn a_fresh_interactive_request_outranks_a_fresh_background_one() {
    let cfg = SchedulerConfig {
        max_batch_seqs: 1,
        max_batch_tokens: 4096,
        ..Default::default()
    };
    let mut s = scheduler(cfg, 512);

    // Same arrival time: now class is the only thing separating them.
    let bg = s
        .admit(request("acme", 64, 4, 0).with_class(ServiceClass::Background))
        .unwrap()[0];
    let inter = s
        .admit(request("acme", 64, 4, 0).with_class(ServiceClass::Interactive))
        .unwrap()[0];

    let batch = s.step(0);
    assert_eq!(batch.prefill[0].seq, inter, "tighter deadline wins on a tie");
    assert_ne!(batch.prefill[0].seq, bg);
}

#[test]
fn memory_pressure_preempts_and_the_victim_resumes() {
    // Deliberately tiny: 12 blocks of 16 tokens is 192 tokens of KV in total.
    let cfg = SchedulerConfig {
        max_batch_tokens: 512,
        watermark: 0.0,
        ..Default::default()
    };
    let mut s = scheduler(cfg, 12);
    let mut now = 0;

    for _ in 0..4 {
        s.admit(request("acme", 48, 60, now)).unwrap();
    }

    let mut preemptions = 0;
    for _ in 0..120 {
        let batch = s.step(now);
        preemptions += batch.preempted.len();
        let outputs: Vec<_> = batch.decode.iter().map(|&id| (id, FILLER)).collect();
        s.on_tokens(&outputs, now);
        now += STEP_US;
    }

    assert!(
        preemptions > 0,
        "four 108-token sequences cannot fit in 192 tokens of KV at once"
    );
    assert_eq!(
        s.metrics().preemptions as usize,
        preemptions,
        "every preemption must be counted"
    );
    // The point of preemption is progress, not just eviction.
    assert!(
        s.metrics().finished > 0,
        "preempted sequences must still make progress and finish"
    );
}

#[test]
fn a_shared_system_prompt_is_reused_by_later_requests() {
    let mut s = scheduler(SchedulerConfig::default(), 512);
    let mut now = 0;

    let system: Vec<u32> = (0..256).collect();
    let mut first = Request::new("acme", system.clone(), now);
    first.params = SamplingParams::greedy(4);
    s.admit(first).unwrap();
    drive(&mut s, 16, &mut now);

    let before = s.metrics();
    assert_eq!(before.prefill_tokens_reused, 0, "nothing to reuse on a cold start");

    // A second request carrying the same system prompt plus its own suffix.
    let followup: Vec<u32> = system.iter().copied().chain(900..932).collect();
    let mut second = Request::new("acme", followup, now);
    second.params = SamplingParams::greedy(4);
    s.admit(second).unwrap();
    drive(&mut s, 16, &mut now);

    let after = s.metrics();
    assert!(
        after.prefill_tokens_reused >= 240,
        "most of the 256-token system prompt should be reused, got {}",
        after.prefill_tokens_reused
    );
    assert!(after.prefill_reuse_rate() > 0.0);
}

#[test]
fn tenants_do_not_reuse_each_others_context() {
    let mut s = scheduler(SchedulerConfig::default(), 512);
    let mut now = 0;
    let prompt: Vec<u32> = (0..256).collect();

    for tenant in ["acme", "globex"] {
        let mut r = Request::new(tenant, prompt.clone(), now);
        r.params = SamplingParams::greedy(4);
        s.admit(r).unwrap();
        drive(&mut s, 16, &mut now);
    }
    assert_eq!(
        s.metrics().prefill_tokens_reused,
        0,
        "an identical prompt under a second tenant must not hit the cache"
    );
}

#[test]
fn the_token_budget_is_never_exceeded() {
    let cfg = SchedulerConfig {
        max_batch_tokens: 64,
        max_batch_seqs: 8,
        ..Default::default()
    };
    let mut s = scheduler(cfg, 1024);
    let mut now = 0;
    for _ in 0..12 {
        s.admit(request("acme", 200, 10, now)).unwrap();
    }
    for _ in 0..200 {
        let batch = s.step(now);
        assert!(batch.num_tokens() <= 64, "budget breached: {}", batch.num_tokens());
        assert!(batch.num_seqs() <= 8, "sequence cap breached");
        let outputs: Vec<_> = batch.decode.iter().map(|&id| (id, FILLER)).collect();
        s.on_tokens(&outputs, now);
        now += STEP_US;
    }
}

#[test]
fn cancelling_a_request_releases_its_memory() {
    let mut s = scheduler(SchedulerConfig::default(), 128);
    let mut now = 0;
    let req = request("acme", 64, 500, now);
    let id = req.id;
    s.admit(req).unwrap();
    drive(&mut s, 6, &mut now);

    let live_before = s.cache().stats().live;
    assert!(live_before > 0, "the sequence should hold blocks");

    let done = s.cancel(id);
    assert_eq!(done.len(), 1);
    assert_eq!(s.num_running(), 0);
    assert!(
        s.cache().stats().live < live_before,
        "cancellation must release blocks"
    );
    s.cache().assert_invariants();
}

#[test]
fn an_oversized_prompt_is_refused_at_admission() {
    let cfg = SchedulerConfig {
        max_model_len: 128,
        ..Default::default()
    };
    let mut s = scheduler(cfg, 512);
    assert!(s.admit(request("acme", 64, 4, 0)).is_ok());
    assert!(
        s.admit(request("acme", 4096, 4, 0)).is_err(),
        "a prompt past the context limit must be rejected, not truncated"
    );
    assert!(
        s.admit(request("acme", 0, 4, 0)).is_err(),
        "an empty prompt has nothing to run"
    );
}

#[test]
fn invalid_sampling_parameters_are_refused() {
    let mut s = scheduler(SchedulerConfig::default(), 128);
    let mut bad = request("acme", 16, 4, 0);
    bad.params.temperature = -1.0;
    assert!(s.admit(bad).is_err());

    let mut bad = request("acme", 16, 4, 0);
    bad.params.top_p = 0.0;
    assert!(s.admit(bad).is_err());
}
