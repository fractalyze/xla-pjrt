//! What the client's allocator held, read back from the plugin.
//!
//! Unlike the option tests here, this one has something to observe: a session
//! that has run something must report a peak above zero, and the allocator
//! must have held at least as much as was ever live in it. Both are cheap and
//! both are the failure modes that matter -- a struct whose fields land in the
//! wrong order reads plausibly, and an `is_set` flag ignored turns "this
//! allocator does not keep that statistic" into a confident zero.
//!
//! Needs the plugin, so it is `#[ignore]`d like the rest:
//!
//! ```sh
//! export XLA_PJRT_PLUGIN=.../jax_plugins/xla_cuda12/xla_cuda_plugin.so
//! cargo test -- --ignored --test-threads=1
//! ```

use xla_pjrt::sys::PJRT_Buffer_Type_F32 as F32;
use xla_pjrt::{Session, SessionOptions};

/// 64 MiB in, 64 MiB out: large enough that a peak of zero is a reading
/// error rather than a rounding one.
const LEN: usize = 16_777_216;

const DOUBLE: &str = r#"
module {
  func.func @main(%arg0: tensor<16777216xf32>) -> tensor<16777216xf32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<16777216xf32>
    return %0 : tensor<16777216xf32>
  }
}
"#;

fn as_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and no invalid bit patterns.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[test]
#[ignore = "needs the xla GPU plugin (see the module docs)"]
fn a_session_that_has_run_reports_what_its_allocator_held() {
    unsafe {
        let session = Session::with_options(SessionOptions {
            preallocate: Some(false),
            ..SessionOptions::default()
        });

        let before = session.memory_stats().expect("the BFC allocator keeps stats");
        let compiled = session.compile(DOUBLE.as_bytes());
        let input = session.input_buffer(as_bytes(&vec![1.0f32; LEN]), &[LEN as i64], F32);
        let _ = session.run_buffers(&compiled, &[&input], 1);
        let after = session.memory_stats().expect("the BFC allocator keeps stats");

        let peak = after.peak_bytes_in_use.expect("BFC keeps a peak");
        assert!(peak >= 64 << 20, "peak {peak} below the input it ran on");
        assert!(peak >= before.peak_bytes_in_use.unwrap_or(0));
        assert!(after.largest_alloc_size.expect("BFC keeps this") >= 64 << 20);
        // What the allocator holds is never less than what was live in it.
        if let Some(pool) = after.peak_pool_bytes {
            assert!(pool >= peak, "pool {pool} under peak in use {peak}");
        }

        session.free_buffer(input);
        session.free_executable(compiled);
    }
}
