//! A client built on a non-default allocator still runs what it is given.
//!
//! `allocator` picks which device allocator the plugin builds, and the kinds
//! differ in how an allocation is placed rather than in what a program
//! computes — so, as with `eager_load_executable_modules` and
//! `staging_threshold_bytes`, there is nothing to observe from here but the
//! plumbing. Two failure modes are worth pinning: this is the only option
//! that rides as a *string*, so a value length left at the scalar 1 or a
//! wrong type tag would truncate the spelling into one the plugin does not
//! know; and a plugin that does not know the kind rejects it outright. Both
//! surface as a panic out of client creation rather than as a quiet fall back
//! to the default.
//!
//! The kind under test is the one a caller reaches for: `cuda_async` places
//! out of the device's default memory pool instead of one arena. It is paired
//! with the reservation options the caller uses with it, since together they
//! mean something different than they do under BFC (see `SessionOptions`).
//!
//! Needs a plugin whose GPU client knows the kind, so it is `#[ignore]`d like
//! the rest:
//!
//! ```sh
//! export XLA_PJRT_PLUGIN=.../jax_plugins/xla_cuda12/xla_cuda_plugin.so
//! cargo test -- --ignored --test-threads=1
//! ```

use xla_pjrt::sys::PJRT_Buffer_Type_F32 as F32;
use xla_pjrt::{AllocatorKind, Session, SessionOptions};

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

/// The spellings are the contract with the plugin's own parser, and a typo in
/// one is a client creation that fails on a machine with a card rather than
/// here. No plugin needed, so this one is not `#[ignore]`d.
#[test]
fn every_kind_spells_itself_as_the_plugin_parses_it() {
    assert_eq!(AllocatorKind::Default.as_str(), "default");
    assert_eq!(AllocatorKind::Platform.as_str(), "platform");
    assert_eq!(AllocatorKind::Bfc.as_str(), "bfc");
    assert_eq!(AllocatorKind::CudaAsync.as_str(), "cuda_async");
    assert_eq!(AllocatorKind::Vmm.as_str(), "vmm");
}

#[test]
#[ignore = "needs an xla GPU plugin whose client knows the kind (see the module docs)"]
fn an_executable_runs_on_a_client_that_allocates_asynchronously() {
    unsafe {
        let session = Session::with_options(SessionOptions {
            // What the kind is paired with in practice: a share claimed up
            // front so a co-tenant sizing itself from free memory sees it
            // gone. Under this kind the share is the pool's release
            // threshold rather than a ceiling.
            preallocate: Some(true),
            memory_fraction: Some(0.1),
            eager_load_executable_modules: None,
            staging_threshold_bytes: None,
            allocator: Some(AllocatorKind::CudaAsync),
        });

        let compiled = session.compile(DOUBLE.as_bytes());
        let input = session.input_buffer(as_bytes(&[1.0, 2.0, 3.0, 4.0]), &[4], F32);
        let out = session.run_buffers(&compiled, &[&input], 1);

        assert_eq!(as_f32(&out[0]), vec![2.0, 4.0, 6.0, 8.0]);

        session.free_buffer(input);
        session.free_executable(compiled);
    }
}
