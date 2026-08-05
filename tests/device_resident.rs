//! An output kept on the device feeds the next execution.
//!
//! This is the whole point of `run_buffers_to_device`, so the test is the
//! use case: run a module, hand its output straight back in without touching
//! the host, and check the result composed.
//!
//! Needs the plugin, so it is `#[ignore]`d:
//!
//! ```sh
//! export XLA_PJRT_PLUGIN=.../jax_plugins/xla_cuda12/xla_cuda_plugin.so
//! cargo test -- --ignored --test-threads=1
//! ```

use xla_pjrt::sys::PJRT_Buffer_Type_F32 as F32;
use xla_pjrt::Session;

/// `f(x) = x + x`. Composing it twice quadruples, which distinguishes "the
/// output was reused" from "the input was run twice".
const DOUBLE: &str = r#"
module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"#;

/// The same, over an output big enough that leaking one per call is fatal
/// rather than merely untidy: 16 MiB x 1024 runs is 16 GiB.
const WIDE: &str = r#"
module {
  func.func @main(%arg0: tensor<4194304xf32>) -> tensor<4194304xf32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<4194304xf32>
    return %0 : tensor<4194304xf32>
  }
}
"#;

const WIDE_LEN: usize = 4 << 20;
const WIDE_RUNS: usize = 4096;

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
#[ignore = "needs the xla GPU plugin (see the module docs)"]
fn device_output_feeds_the_next_execution() {
    unsafe {
        let session = Session::new();
        let exe = session.compile(DOUBLE.as_bytes());

        let input = session.input_buffer(as_bytes(&[1.0, 2.0, 3.0, 4.0]), &[4], F32);

        // First run: keep the result on the device.
        let mut mid = session.run_buffers_to_device(&exe, &[&input], 1);
        assert_eq!(mid.len(), 1);
        let mid = mid.pop().unwrap();

        // Second run consumes it directly — no host copy in between.
        let out = session.run_buffers(&exe, &[&mid], 1);
        assert_eq!(as_f32(&out[0]), vec![4.0, 8.0, 12.0, 16.0]);

        session.free_buffer(mid);
        session.free_buffer(input);
    }
}

#[test]
#[ignore = "needs the xla GPU plugin (see the module docs)"]
fn one_output_reads_back_without_the_others() {
    // The case that motivates it: keep everything resident, fetch the one
    // value the host actually needs.
    unsafe {
        let session = Session::new();
        let exe = session.compile(DOUBLE.as_bytes());
        let input = session.input_buffer(as_bytes(&[1.0, 2.0, 3.0, 4.0]), &[4], F32);

        let mut outs = session.run_buffers_to_device(&exe, &[&input], 1);
        let out = outs.pop().unwrap();
        assert_eq!(as_f32(&session.buffer_to_host(&out)), vec![2.0, 4.0, 6.0, 8.0]);

        // Still usable afterwards: reading is a copy, not a move.
        let again = session.run_buffers(&exe, &[&out], 1);
        assert_eq!(as_f32(&again[0]), vec![4.0, 8.0, 12.0, 16.0]);

        session.free_buffer(out);
        session.free_buffer(input);
    }
}

#[test]
#[ignore = "needs the xla GPU plugin (see the module docs)"]
fn repeated_runs_do_not_exhaust_device_memory() {
    // `run_buffers_timed` used to leak every output: it copied each to host
    // and dropped the pointer. A leak is invisible in one call and fatal in a
    // prover, so this runs enough wide outputs that leaking them exhausts the
    // device — 16 GiB total against a ~24 GiB allocator.
    unsafe {
        let session = Session::new();
        let exe = session.compile(WIDE.as_bytes());
        let ones = vec![1.0f32; WIDE_LEN];
        let input = session.input_buffer(as_bytes(&ones), &[WIDE_LEN as i64], F32);

        for _ in 0..WIDE_RUNS {
            let out = session.run_buffers(&exe, &[&input], 1);
            debug_assert_eq!(out[0].len(), WIDE_LEN * 4);
        }
        session.free_buffer(input);
    }
}
