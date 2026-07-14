# xla-pjrt

Minimal PJRT C API FFI for driving the Fractalyze **xla** GPU plugin (the
`jax-cuda12` build, bundled as `jax_plugins/xla_cuda12/xla_cuda_plugin.so`).
Curve- and framework-agnostic: it deals only in raw bytes + `zk_dtypes`
buffer-type tags, so a caller assembles its own proofs on top.

It drives an AOT-lowered StableHLO module on the GPU: load plugin -> create
client -> compile -> upload host buffers -> execute -> copy outputs back. A
`Session` keeps the plugin + client resident so several executables share one
client, and input buffers (e.g. a proving key) can be uploaded once and reused
across runs.

## Use

Point `XLA_PJRT_PLUGIN` at the plugin `.so`, then depend on the crate:

```bash
export XLA_PJRT_PLUGIN=.../jax_plugins/xla_cuda12/xla_cuda_plugin.so
```

```toml
[dependencies]
xla-pjrt = { git = "https://github.com/fractalyze/xla-pjrt" }
```

Building needs `clang`/`libclang` — the PJRT bindings are generated with
`bindgen` at build time from the vendored header.

## Vendored PJRT header

`third_party/pjrt/xla/pjrt/c/pjrt_c_api.h` is copied verbatim from the Fractalyze
xla fork (`xla/pjrt/c/pjrt_c_api.h`, PJRT C API 0.104). It is self-contained: the
`PJRT_Buffer_Type` enum — including the BN254 / curve tags — is inlined, so there
is no separate data-types include. To bump it, re-copy from xla and rebuild;
`XLA_PJRT_HEADER` overrides the path for a one-off build.
