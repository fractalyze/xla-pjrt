//! Buffers allocated up front and filled afterwards feed an execution the
//! same way an `input_buffer` does.
//!
//! `Session::stage` exists so the allocation can happen before the work a
//! transfer should overlap; this test is the correctness half of that — the
//! bytes arrive, in the right buffer, and the execution that consumes them
//! waits for the transfer without the caller doing so.
//!
//! Needs the plugin, so it is `#[ignore]`d:
//!
//! ```sh
//! export XLA_PJRT_PLUGIN=.../frx_plugins/xla_cuda12/xla_cuda_plugin.so
//! cargo test --test staged_upload -- --ignored --test-threads=1
//! ```

use xla_pjrt::sys::PJRT_Buffer_Type_F32 as F32;
use xla_pjrt::{Session, SessionOptions};

/// `f(x, y) = x - y`. Subtraction is not commutative, so it also catches a
/// transfer landing in the wrong buffer index.
const SUB: &str = r#"
module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"#;

fn bytes(xs: [f32; 4]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
#[ignore]
fn staged_buffers_carry_their_bytes_into_an_execution() {
    unsafe {
        // Allocate on demand rather than claiming the card: this test is
        // four floats, and the host it runs on is shared.
        let session = Session::with_options(SessionOptions {
            preallocate: Some(false),
            memory_fraction: None,
        });
        let exe = session.compile(SUB.as_bytes());
        let dims: &[i64] = &[4];
        let staging = session.stage(&[(dims, F32), (dims, F32)]);

        let lhs = bytes([10.0, 20.0, 30.0, 40.0]);
        let rhs = bytes([1.0, 2.0, 3.0, 4.0]);
        // Enqueued, not waited for: the point of the type is that the host
        // data stays alive until `wait`, not that the caller blocks here.
        let a = staging.transfer(0, &lhs);
        let b = staging.transfer(1, &rhs);
        let x = staging.retrieve(0);
        let y = staging.retrieve(1);
        a.wait();
        b.wait();
        // The bridge drops the manager as soon as it has the buffers, so
        // the buffers have to outlive it. If PJRT freed their memory with
        // the manager this is where it would show.
        drop(staging);

        let out = session.run_buffers(&exe, &[&x, &y], 1);
        let got: Vec<f32> = out[0]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, vec![9.0, 18.0, 27.0, 36.0]);

        session.free_buffer(x);
        session.free_buffer(y);
        session.free_executable(exe);
    }
}
