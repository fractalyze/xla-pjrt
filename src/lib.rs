//! Minimal wrapper over the PJRT C API for the xla GPU (jax-cuda12) plugin.
//!
//! Drives an AOT-lowered StableHLO module (uint8 boundary) on GPU: load plugin
//! -> create client -> compile -> host buffers -> execute -> copy outputs back.
//! One-shot use (a PoC binary/test), so buffers/executables are intentionally
//! not freed — the process exits and the OS reclaims everything.
#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code)]

pub mod sys {
    include!(concat!(env!("OUT_DIR"), "/pjrt_sys.rs"));
}

use libloading::{Library, Symbol};
use std::mem::{size_of, zeroed};
use std::os::raw::{c_char, c_void};
use std::ptr;

/// Path to the GPU plugin `.so`, from env `XLA_PJRT_PLUGIN`. The plugin ships in
/// the matched jax-cuda12 wheel (`jax_plugins/xla_cuda12/xla_cuda_plugin.so`);
/// see the README for the install + env-var setup.
pub fn plugin_path() -> String {
    std::env::var("XLA_PJRT_PLUGIN")
        .expect("set XLA_PJRT_PLUGIN to the jax-cuda12 xla_cuda_plugin.so")
}

pub struct Pjrt {
    _lib: Library, // keep the .so resident; `api` points into it
    pub api: *const sys::PJRT_Api,
}

// Custom buffer element types (PJRT_Buffer_Type, inlined in pjrt_c_api.h). The
// element type carries the per-element byte size (sf=32, g1=64, g2=128), so
// buffer `dims` are logical element counts, not byte shapes.
pub const BN254_SF: sys::PJRT_Buffer_Type = sys::PJRT_Buffer_Type_PJRT_Buffer_Type_BN254_SF;
pub const BN254_SF_MONT: sys::PJRT_Buffer_Type = sys::PJRT_Buffer_Type_PJRT_Buffer_Type_BN254_SF_MONT;
pub const BN254_G1_AFFINE: sys::PJRT_Buffer_Type = sys::PJRT_Buffer_Type_PJRT_Buffer_Type_BN254_G1_AFFINE;
pub const BN254_G2_AFFINE: sys::PJRT_Buffer_Type = sys::PJRT_Buffer_Type_PJRT_Buffer_Type_BN254_G2_AFFINE;

/// On-the-wire byte sizes of the zk_dtypes element types (32-byte LE limbs).
pub const SF_BYTES: usize = 32;
pub const G1_BYTES: usize = 64;
pub const G2_BYTES: usize = 128;

/// Panic with the plugin's message if `err` is non-null.
unsafe fn check(api: *const sys::PJRT_Api, err: *mut sys::PJRT_Error, ctx: &str) {
    if err.is_null() {
        return;
    }
    let mut m: sys::PJRT_Error_Message_Args = zeroed();
    m.struct_size = size_of::<sys::PJRT_Error_Message_Args>();
    m.error = err;
    (*api).PJRT_Error_Message.unwrap()(&mut m);
    let msg = std::str::from_utf8(std::slice::from_raw_parts(m.message as *const u8, m.message_size))
        .unwrap_or("<non-utf8>")
        .to_string();
    let mut d: sys::PJRT_Error_Destroy_Args = zeroed();
    d.struct_size = size_of::<sys::PJRT_Error_Destroy_Args>();
    d.error = err;
    (*api).PJRT_Error_Destroy.unwrap()(&mut d);
    panic!("PJRT error in {ctx}: {msg}");
}

impl Pjrt {
    /// dlopen the plugin and fetch its `PJRT_Api` table.
    pub unsafe fn load() -> Self {
        let lib = Library::new(plugin_path()).expect("dlopen GPU plugin");
        let get: Symbol<unsafe extern "C" fn() -> *const sys::PJRT_Api> =
            lib.get(b"GetPjrtApi\0").expect("GetPjrtApi symbol");
        let api = get();
        assert!(!api.is_null(), "GetPjrtApi returned null");
        Pjrt { _lib: lib, api }
    }

    /// `(major, minor)` version reported by the plugin.
    pub unsafe fn version(&self) -> (i32, i32) {
        let v = (*self.api).pjrt_api_version;
        (v.major_version, v.minor_version)
    }

    unsafe fn plugin_initialize(&self) {
        let mut a: sys::PJRT_Plugin_Initialize_Args = zeroed();
        a.struct_size = size_of::<sys::PJRT_Plugin_Initialize_Args>();
        check(self.api, (*self.api).PJRT_Plugin_Initialize.unwrap()(&mut a), "Plugin_Initialize");
    }

    unsafe fn create_client(&self) -> Client {
        let mut a: sys::PJRT_Client_Create_Args = zeroed();
        a.struct_size = size_of::<sys::PJRT_Client_Create_Args>();
        check(self.api, (*self.api).PJRT_Client_Create.unwrap()(&mut a), "Client_Create");
        Client { api: self.api, client: a.client }
    }
}

pub struct Client {
    api: *const sys::PJRT_Api,
    client: *mut sys::PJRT_Client,
}

impl Client {
    /// Compile MLIR bytecode in this plugin's context. On a matched xla stack the
    /// core's `stablehlo.ntt`/`stablehlo.msm` are registered, so the full core
    /// compiles (a version-skewed plugin instead fails on unregistered ops).
    unsafe fn compile(&self, code: &[u8]) -> *mut sys::PJRT_LoadedExecutable {
        let fmt = b"mlir";
        let mut prog: sys::PJRT_Program = zeroed();
        prog.struct_size = size_of::<sys::PJRT_Program>();
        prog.code = code.as_ptr() as *mut c_char;
        prog.code_size = code.len();
        prog.format = fmt.as_ptr() as *const c_char;
        prog.format_size = fmt.len();
        // Minimal xla.CompileOptionsProto: executable_build_options{num_replicas=1,
        // num_partitions=1} — else the GPU client builds a 0x0 device assignment
        // and aborts (Check failed: replica_count > 0). Fields per
        // xla/pjrt/compile_options.proto (3=ebo, 4=num_replicas, 5=num_partitions).
        const COMPILE_OPTS: [u8; 6] = [0x1A, 0x04, 0x20, 0x01, 0x28, 0x01];
        let mut a: sys::PJRT_Client_Compile_Args = zeroed();
        a.struct_size = size_of::<sys::PJRT_Client_Compile_Args>();
        a.client = self.client;
        a.program = &prog;
        a.compile_options = COMPILE_OPTS.as_ptr() as *const c_char;
        a.compile_options_size = COMPILE_OPTS.len();
        check(self.api, (*self.api).PJRT_Client_Compile.unwrap()(&mut a), "Client_Compile");
        a.executable
    }

    unsafe fn first_device(&self) -> *mut sys::PJRT_Device {
        let mut a: sys::PJRT_Client_AddressableDevices_Args = zeroed();
        a.struct_size = size_of::<sys::PJRT_Client_AddressableDevices_Args>();
        a.client = self.client;
        check(self.api, (*self.api).PJRT_Client_AddressableDevices.unwrap()(&mut a), "AddressableDevices");
        assert!(a.num_addressable_devices > 0, "no addressable devices");
        *a.addressable_devices
    }

    unsafe fn await_event(&self, ev: *mut sys::PJRT_Event) {
        if ev.is_null() {
            return;
        }
        let mut a: sys::PJRT_Event_Await_Args = zeroed();
        a.struct_size = size_of::<sys::PJRT_Event_Await_Args>();
        a.event = ev;
        check(self.api, (*self.api).PJRT_Event_Await.unwrap()(&mut a), "Event_Await");
        let mut d: sys::PJRT_Event_Destroy_Args = zeroed();
        d.struct_size = size_of::<sys::PJRT_Event_Destroy_Args>();
        d.event = ev;
        (*self.api).PJRT_Event_Destroy.unwrap()(&mut d);
    }

    unsafe fn buf_from_host(
        &self,
        device: *mut sys::PJRT_Device,
        data: &[u8],
        dims: &[i64],
        elem_type: sys::PJRT_Buffer_Type,
    ) -> *mut sys::PJRT_Buffer {
        let mut a: sys::PJRT_Client_BufferFromHostBuffer_Args = zeroed();
        a.struct_size = size_of::<sys::PJRT_Client_BufferFromHostBuffer_Args>();
        a.client = self.client;
        a.data = data.as_ptr() as *const c_void;
        a.type_ = elem_type;
        a.dims = dims.as_ptr();
        a.num_dims = dims.len();
        a.host_buffer_semantics =
            sys::PJRT_HostBufferSemantics_PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
        a.device = device;
        check(self.api, (*self.api).PJRT_Client_BufferFromHostBuffer.unwrap()(&mut a), "BufferFromHostBuffer");
        // Wait until the runtime has finished reading `data` so it is safe to drop.
        self.await_event(a.done_with_host_buffer);
        a.buffer
    }

    unsafe fn execute(
        &self,
        exe: *mut sys::PJRT_LoadedExecutable,
        inputs: &[*mut sys::PJRT_Buffer],
        num_outputs: usize,
    ) -> Vec<*mut sys::PJRT_Buffer> {
        let mut opts: sys::PJRT_ExecuteOptions = zeroed();
        opts.struct_size = size_of::<sys::PJRT_ExecuteOptions>();

        // argument_lists: [num_devices=1][num_args]
        let args_inner: Vec<*mut sys::PJRT_Buffer> = inputs.to_vec();
        let args_dev: [*const *mut sys::PJRT_Buffer; 1] = [args_inner.as_ptr()];

        // output_lists: [num_devices=1][num_outputs], allocated by caller
        let mut out_inner: Vec<*mut sys::PJRT_Buffer> = vec![ptr::null_mut(); num_outputs];
        let out_dev: [*mut *mut sys::PJRT_Buffer; 1] = [out_inner.as_mut_ptr()];

        let mut a: sys::PJRT_LoadedExecutable_Execute_Args = zeroed();
        a.struct_size = size_of::<sys::PJRT_LoadedExecutable_Execute_Args>();
        a.executable = exe;
        a.options = &mut opts;
        a.argument_lists = args_dev.as_ptr();
        a.num_devices = 1;
        a.num_args = inputs.len();
        a.output_lists = out_dev.as_ptr();
        check(self.api, (*self.api).PJRT_LoadedExecutable_Execute.unwrap()(&mut a), "Execute");
        out_inner
    }

    unsafe fn to_host(&self, buf: *mut sys::PJRT_Buffer) -> Vec<u8> {
        // First pass: query required size (dst = null).
        let mut q: sys::PJRT_Buffer_ToHostBuffer_Args = zeroed();
        q.struct_size = size_of::<sys::PJRT_Buffer_ToHostBuffer_Args>();
        q.src = buf;
        check(self.api, (*self.api).PJRT_Buffer_ToHostBuffer.unwrap()(&mut q), "ToHostBuffer(size)");
        let n = q.dst_size;

        let mut out = vec![0u8; n];
        let mut a: sys::PJRT_Buffer_ToHostBuffer_Args = zeroed();
        a.struct_size = size_of::<sys::PJRT_Buffer_ToHostBuffer_Args>();
        a.src = buf;
        a.dst = out.as_mut_ptr() as *mut c_void;
        a.dst_size = n;
        check(self.api, (*self.api).PJRT_Buffer_ToHostBuffer.unwrap()(&mut a), "ToHostBuffer(copy)");
        self.await_event(a.event);
        out
    }
}

/// `(bytes, dims, elem_type)` per input; `dims` are logical element counts
/// (the element type carries byte size).
pub type Inputs<'a> = [(&'a [u8], Vec<i64>, sys::PJRT_Buffer_Type)];

unsafe fn run_loaded(
    c: &Client,
    exe: *mut sys::PJRT_LoadedExecutable,
    inputs: &Inputs,
    num_outputs: usize,
) -> Vec<Vec<u8>> {
    let dev = c.first_device();
    let bufs: Vec<*mut sys::PJRT_Buffer> =
        inputs.iter().map(|(d, dims, t)| c.buf_from_host(dev, d, dims, *t)).collect();
    let outs = c.execute(exe, &bufs, num_outputs);
    outs.iter().map(|&b| c.to_host(b)).collect()
}

/// Compile MLIR bytecode and run it. Returns one byte vec per output.
pub unsafe fn run_bytecode(code: &[u8], inputs: &Inputs, num_outputs: usize) -> Vec<Vec<u8>> {
    let p = Pjrt::load();
    p.plugin_initialize();
    let c = p.create_client();
    let exe = c.compile(code);
    run_loaded(&c, exe, inputs, num_outputs)
}

/// Slice 0: single `lax.msm` — scalars (BN254_SF, `[n]`), G1 points
/// (BN254_G1_AFFINE, `[n]`) -> one G1 point (64 bytes).
pub unsafe fn run_msm(code: &[u8], scalars: &[u8], points: &[u8]) -> Vec<u8> {
    let ns = (scalars.len() / SF_BYTES) as i64;
    let np = (points.len() / G1_BYTES) as i64;
    let mut r = run_bytecode(
        code,
        &[(scalars, vec![ns], BN254_SF), (points, vec![np], BN254_G1_AFFINE)],
        1,
    );
    r.pop().unwrap()
}

/// A persistent plugin + GPU client. Creating a second client in one process
/// aborts (the plugin throws a C++ exception Rust can't catch), so a caller that
/// runs several executables — e.g. the five MSMs of one Groth16 proof — must
/// reuse one `Session` instead of the one-shot free functions above.
pub struct Session {
    _pjrt: Pjrt, // keeps the .so resident; `client.api` points into it
    client: Client,
}

/// A compiled executable bound to a [`Session`]'s client. Compile once and reuse
/// across runs to avoid recompiling the same module on every call.
pub struct Executable(*mut sys::PJRT_LoadedExecutable);

/// A device-resident input buffer. Upload once and reuse across executions to
/// avoid re-transferring a constant input (e.g. a proving key).
pub struct Buffer(*mut sys::PJRT_Buffer);

impl Session {
    /// Load the plugin and create the single client.
    pub unsafe fn new() -> Self {
        let pjrt = Pjrt::load();
        pjrt.plugin_initialize();
        let client = pjrt.create_client();
        Session { _pjrt: pjrt, client }
    }

    /// Compile MLIR bytecode once on the persistent client.
    pub unsafe fn compile(&self, code: &[u8]) -> Executable {
        Executable(self.client.compile(code))
    }

    /// Run a pre-compiled executable.
    pub unsafe fn run(&self, exe: &Executable, inputs: &Inputs, num_outputs: usize) -> Vec<Vec<u8>> {
        run_loaded(&self.client, exe.0, inputs, num_outputs)
    }

    /// Single G1 `lax.msm` against a pre-compiled executable (see [`run_msm`]).
    pub unsafe fn run_msm(&self, exe: &Executable, scalars: &[u8], points: &[u8]) -> Vec<u8> {
        let ns = (scalars.len() / SF_BYTES) as i64;
        let np = (points.len() / G1_BYTES) as i64;
        let mut r = self.run(
            exe,
            &[(scalars, vec![ns], BN254_SF), (points, vec![np], BN254_G1_AFFINE)],
            1,
        );
        r.pop().unwrap()
    }

    /// Single G2 `lax.msm` against a pre-compiled executable (BN254_G2_AFFINE
    /// points -> one G2 point, 128 bytes).
    pub unsafe fn run_msm_g2(&self, exe: &Executable, scalars: &[u8], points: &[u8]) -> Vec<u8> {
        let ns = (scalars.len() / SF_BYTES) as i64;
        let np = (points.len() / G2_BYTES) as i64;
        let mut r = self.run(
            exe,
            &[(scalars, vec![ns], BN254_SF), (points, vec![np], BN254_G2_AFFINE)],
            1,
        );
        r.pop().unwrap()
    }

    /// Upload a host array to a persistent device buffer (reuse across runs).
    pub unsafe fn input_buffer(
        &self,
        data: &[u8],
        dims: &[i64],
        elem_type: sys::PJRT_Buffer_Type,
    ) -> Buffer {
        let dev = self.client.first_device();
        Buffer(self.client.buf_from_host(dev, data, dims, elem_type))
    }

    /// Execute with already-uploaded input buffers (in the executable's
    /// parameter order). Returns one byte vec per output. Lets a caller reuse
    /// resident buffers (e.g. a proving key) across many runs while only
    /// uploading the per-run inputs.
    pub unsafe fn run_buffers(
        &self,
        exe: &Executable,
        inputs: &[&Buffer],
        num_outputs: usize,
    ) -> Vec<Vec<u8>> {
        self.run_buffers_timed(exe, inputs, num_outputs).0
    }

    /// Like [`run_buffers`], but also returns `(dispatch, readback)` durations:
    /// the execute (enqueue) time and the to-host time. The host transfer for
    /// the outputs is small, so the readback duration is dominated by waiting
    /// on the computation. For profiling.
    pub unsafe fn run_buffers_timed(
        &self,
        exe: &Executable,
        inputs: &[&Buffer],
        num_outputs: usize,
    ) -> (Vec<Vec<u8>>, std::time::Duration, std::time::Duration) {
        let bufs: Vec<*mut sys::PJRT_Buffer> = inputs.iter().map(|b| b.0).collect();
        let t = std::time::Instant::now();
        let outs = self.client.execute(exe.0, &bufs, num_outputs);
        let dispatch = t.elapsed();
        let t = std::time::Instant::now();
        let host = outs.iter().map(|&b| self.client.to_host(b)).collect();
        let readback = t.elapsed();
        (host, dispatch, readback)
    }
}
