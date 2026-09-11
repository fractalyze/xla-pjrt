//! A client that stages its large transfers still gets the right bytes back.
//!
//! `staging_threshold_bytes` only moves *how* a host-to-device transfer
//! reaches the device — through the client's pinned staging pool rather than
//! by DMA out of the caller's pageable memory — so, as with
//! `eager_load_executable_modules`, there is nothing to observe from here but
//! the plumbing: the plugin has to accept the key at the type we send it, and
//! a transfer on the far side of the default threshold has to still arrive
//! intact. A misspelled key or a wrong value tag does not degrade quietly;
//! the plugin rejects the option and client creation fails.
//!
//! The input is deliberately over the plugin's 1 GiB default, so the option
//! is what decides which path it takes: unset, this transfer is the pageable
//! one; set above it, it is staged.
//!
//! Needs a plugin carrying the option (fractalyze/xla#718) and a card with
//! room for a 1.25 GiB buffer, so it is `#[ignore]`d like the rest:
//!
//! ```sh
//! export XLA_PJRT_PLUGIN=.../jax_plugins/xla_cuda12/xla_cuda_plugin.so
//! cargo test -- --ignored --test-threads=1
//! ```

use xla_pjrt::sys::PJRT_Buffer_Type_F32 as F32;
use xla_pjrt::{Session, SessionOptions};

/// 1.25 GiB of f32 — past the plugin's 1 GiB staging default, so the copy
/// this test makes is one the option changes the path of.
const LEN: usize = 335_544_320;

/// The same length in the `i64` the buffer API takes.
const LEN_I64: i64 = LEN as i64;

/// Bytes to check at each end. Staging copies the whole transfer through one
/// pool, so a truncated or misaligned stage shows up at a boundary; reading
/// all 1.25 GiB back to compare it would dominate the test's runtime.
const EDGE: usize = 4096;

const DOUBLE: &str = r#"
module {
  func.func @main(%arg0: tensor<335544320xf32>) -> tensor<335544320xf32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<335544320xf32>
    return %0 : tensor<335544320xf32>
  }
}
"#;

fn as_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and no invalid bit patterns.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
#[ignore = "needs an xla GPU plugin carrying fractalyze/xla#718 (see the module docs)"]
fn a_transfer_past_the_default_threshold_survives_staging() {
    // Every element distinct modulo the edge windows, so a stage that lands a
    // chunk at the wrong offset does not read as correct.
    let input: Vec<f32> = (0..LEN).map(|i| (i % 1_000_003) as f32).collect();
    let want_head: Vec<f32> = input[..EDGE].iter().map(|x| x + x).collect();
    let want_tail: Vec<f32> = input[LEN - EDGE..].iter().map(|x| x + x).collect();

    unsafe {
        let session = Session::with_options(SessionOptions {
            preallocate: Some(false),
            memory_fraction: None,
            eager_load_executable_modules: None,
            // Above the transfer below, so it is staged rather than DMA'd
            // from pageable memory.
            staging_threshold_bytes: Some(2 << 30),
            allocator: None,
        });

        let compiled = session.compile(DOUBLE.as_bytes());
        let buf = session.input_buffer(as_bytes(&input), &[LEN_I64], F32);
        let out = session.run_buffers(&compiled, &[&buf], 1);

        let got = as_f32(&out[0]);
        assert_eq!(got.len(), LEN);
        assert_eq!(got[..EDGE], want_head[..]);
        assert_eq!(got[LEN - EDGE..], want_tail[..]);

        session.free_buffer(buf);
        session.free_executable(compiled);
    }
}
