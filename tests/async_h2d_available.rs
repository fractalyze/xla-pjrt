//! Does the loaded plugin implement the async host-to-device transfer
//! manager? A PJRT plugin may leave any entry point null, and the whole
//! point of that manager here is to allocate a buffer before the transfer
//! that fills it, so a null pointer decides a design question.
//!
//! Needs the plugin, so it is `#[ignore]`d:
//!
//! ```sh
//! export XLA_PJRT_PLUGIN=.../frx_plugins/xla_cuda12/xla_cuda_plugin.so
//! cargo test --test async_h2d_available -- --ignored --nocapture
//! ```

#[test]
#[ignore]
fn plugin_implements_the_async_transfer_manager() {
    let present = unsafe { xla_pjrt::async_h2d_entry_points() };
    for (name, ok) in &present {
        println!("{name}: {}", if *ok { "present" } else { "NULL" });
    }
    assert!(present.iter().all(|(_, ok)| *ok), "plugin is missing an entry point");
}
