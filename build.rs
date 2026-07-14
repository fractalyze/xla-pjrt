fn main() {
    // Vendored PJRT C API header (self-contained, see third_party/pjrt). The
    // `PJRT_Buffer_Type` enum (BN254 / … curve tags) is inlined in the header, so there
    // is no separate data-types include. The include root is `third_party/pjrt`, kept
    // for the vendored layout (the header itself only pulls in system C headers).
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let header = std::env::var("XLA_PJRT_HEADER")
        .unwrap_or_else(|_| format!("{manifest}/third_party/pjrt/xla/pjrt/c/pjrt_c_api.h"));
    let inc = std::env::var("XLA_PJRT_INCLUDE")
        .unwrap_or_else(|_| format!("{manifest}/third_party/pjrt"));
    println!("cargo:rerun-if-env-changed=XLA_PJRT_HEADER");
    println!("cargo:rerun-if-changed={header}");
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindgen::Builder::default()
        .header(&header)
        .clang_arg(format!("-I{inc}"))
        // pjrt_c_api.h uses C++ constructs (reinterpret_cast in PJRT_STRUCT_SIZE).
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        .allowlist_type("PJRT_.*")
        .allowlist_function("GetPjrtApi")
        .allowlist_var("PJRT_.*")
        .default_enum_style(bindgen::EnumVariation::Consts)
        // C enum variants already carry their type prefix (PJRT_Buffer_Type_BN254_SF),
        // so don't double it — consumers name the header-faithful tag, not a bindgen artifact.
        .prepend_enum_name(false)
        .derive_default(true)
        .generate()
        .expect("bindgen failed")
        .write_to_file(out.join("pjrt_sys.rs"))
        .expect("write bindings");
}
