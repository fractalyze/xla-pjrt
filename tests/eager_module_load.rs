//! A client that loads executables eagerly still runs them.
//!
//! `eager_load_executable_modules` only moves *when* an executable's modules
//! reach the CUDA context — from its first execution to
//! `deserialize_and_load` — so there is nothing to observe from here but the
//! plumbing: the plugin has to accept the key at the type we send it, and the
//! executable it hands back has to still execute. Both are the failure modes
//! worth pinning. A misspelled key or a wrong value tag does not degrade
//! quietly; the plugin rejects the option and client creation fails.
//!
//! Needs a plugin carrying the option (fractalyze/xla#664), so it is
//! `#[ignore]`d like the rest:
//!
//! ```sh
//! export XLA_PJRT_PLUGIN=.../jax_plugins/xla_cuda12/xla_cuda_plugin.so
//! cargo test -- --ignored --test-threads=1
//! ```

use xla_pjrt::sys::PJRT_Buffer_Type_F32 as F32;
use xla_pjrt::{Session, SessionOptions};

/// `f(x) = x + x` — enough to tell "it ran" from "it returned the input".
const DOUBLE: &str = r#"
module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
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
#[ignore = "needs an xla GPU plugin carrying fractalyze/xla#664 (see the module docs)"]
fn a_deserialized_executable_runs_under_eager_module_loads() {
    unsafe {
        let session = Session::with_options(SessionOptions {
            preallocate: Some(false),
            memory_fraction: None,
            eager_load_executable_modules: Some(true),
            staging_threshold_bytes: None,
            allocator: None,
        });

        // The path the option changes: the modules load inside this call
        // rather than inside the run below.
        let compiled = session.compile(DOUBLE.as_bytes());
        let bytes = session.serialize(&compiled);
        let exe = session.deserialize_and_load(&bytes);

        let input = session.input_buffer(as_bytes(&[1.0, 2.0, 3.0, 4.0]), &[4], F32);
        let out = session.run_buffers(&exe, &[&input], 1);
        assert_eq!(as_f32(&out[0]), vec![2.0, 4.0, 6.0, 8.0]);

        session.free_buffer(input);
        session.free_executable(exe);
        session.free_executable(compiled);
    }
}
