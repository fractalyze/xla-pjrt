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

    unsafe fn create_client(&self, options: &SessionOptions) -> Client {
        // Options ride as PJRT named values; `preallocate` is the GPU
        // plugin's allocator switch (bool).
        let mut named: Vec<sys::PJRT_NamedValue> = Vec::new();
        let named_value = |name: &'static [u8]| {
            let mut nv: sys::PJRT_NamedValue = zeroed();
            nv.struct_size = size_of::<sys::PJRT_NamedValue>();
            nv.name = name.as_ptr() as *const c_char;
            nv.name_size = name.len();
            nv.value_size = 1;
            nv
        };
        if let Some(preallocate) = options.preallocate {
            let mut nv = named_value(b"preallocate");
            nv.type_ = sys::PJRT_NamedValue_kBool;
            nv.__bindgen_anon_1.bool_value = preallocate;
            named.push(nv);
        }
        if let Some(fraction) = options.memory_fraction {
            let mut nv = named_value(b"memory_fraction");
            nv.type_ = sys::PJRT_NamedValue_kFloat;
            nv.__bindgen_anon_1.float_value = fraction;
            named.push(nv);
        }
        let mut a: sys::PJRT_Client_Create_Args = zeroed();
        a.struct_size = size_of::<sys::PJRT_Client_Create_Args>();
        a.create_options = named.as_ptr();
        a.num_options = named.len();
        check(self.api, (*self.api).PJRT_Client_Create.unwrap()(&mut a), "Client_Create");
        Client { api: self.api, client: a.client }
    }
}

// The plugin is process-global: PJRT initializes once and keeps threads
// alive past any client, so every `Session` shares one loaded plugin.
struct SharedPjrt(Pjrt);
unsafe impl Send for SharedPjrt {}
unsafe impl Sync for SharedPjrt {}
static PJRT: std::sync::OnceLock<SharedPjrt> = std::sync::OnceLock::new();

unsafe fn shared_pjrt() -> &'static Pjrt {
    &PJRT
        .get_or_init(|| {
            let p = Pjrt::load();
            p.plugin_initialize();
            SharedPjrt(p)
        })
        .0
}

/// Client creation options.
///
/// `preallocate: Some(false)` keeps the GPU plugin's allocator from claiming
/// the card up front — what lets several `Session`s (each its own allocator
/// and stream) coexist in one process alongside other CUDA users. `None`
/// leaves the plugin's default (preallocate most of the card, one client).
///
/// `memory_fraction` is the share of the card the client's allocator may
/// take (the plugin's default is 0.75); with `preallocate: Some(true)` it
/// is claimed at creation, which is how a client reserves memory ahead of
/// other CUDA users that size themselves from what is free.
#[derive(Clone, Copy, Debug, Default)]
pub struct SessionOptions {
    pub preallocate: Option<bool>,
    pub memory_fraction: Option<f32>,
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
            sys::PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
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

    unsafe fn destroy_buffer(&self, buf: *mut sys::PJRT_Buffer) {
        let mut a: sys::PJRT_Buffer_Destroy_Args = zeroed();
        a.struct_size = size_of::<sys::PJRT_Buffer_Destroy_Args>();
        a.buffer = buf;
        check(self.api, (*self.api).PJRT_Buffer_Destroy.unwrap()(&mut a), "Buffer_Destroy");
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
    let c = shared_pjrt().create_client(&SessionOptions::default());
    let exe = c.compile(code);
    run_loaded(&c, exe, inputs, num_outputs)
}

/// A persistent GPU client over the process-global plugin. With the plugin's
/// default allocator a second client in one process aborts (it throws a C++
/// exception Rust can't catch), so a caller that runs several executables
/// must reuse one `Session`; with `SessionOptions { preallocate: Some(false) }`
/// several sessions coexist, each with its own allocator and stream.
pub struct Session {
    client: Client,
}

// PJRT clients, executables and buffers are thread-safe handles; the
// pointers they wrap belong to the plugin, which outlives every session.
unsafe impl Send for Session {}
unsafe impl Sync for Session {}
unsafe impl Send for Executable {}
unsafe impl Sync for Executable {}
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

/// A compiled executable bound to a [`Session`]'s client. Compile once and reuse
/// across runs to avoid recompiling the same module on every call.
pub struct Executable(*mut sys::PJRT_LoadedExecutable);

/// Device-resident data: an uploaded input, or an output kept on the device.
///
/// Upload once and reuse across executions to avoid re-transferring a constant
/// input (e.g. a proving key), or carry one stage's output into the next
/// without a round trip (see [`Session::run_buffers_to_device`]).
///
/// A buffer from [`Session::input_buffer`] or `run_buffers_to_device` owns
/// device memory until passed to [`Session::free_buffer`].
pub struct Buffer(*mut sys::PJRT_Buffer);

impl Session {
    /// Load the plugin (once per process) and create a client with the
    /// plugin's default options.
    pub unsafe fn new() -> Self {
        Self::with_options(SessionOptions::default())
    }

    /// Load the plugin (once per process) and create a client with `options`.
    pub unsafe fn with_options(options: SessionOptions) -> Self {
        let client = shared_pjrt().create_client(&options);
        Session { client }
    }

    /// The executable's serialized form — what `deserialize_and_load` turns
    /// back into a loaded executable without recompiling. Plugin-version
    /// specific: a cache keyed on it must be dropped with the plugin.
    pub unsafe fn serialize(&self, exe: &Executable) -> Vec<u8> {
        let api = self.client.api;
        let mut g: sys::PJRT_LoadedExecutable_GetExecutable_Args = zeroed();
        g.struct_size = size_of::<sys::PJRT_LoadedExecutable_GetExecutable_Args>();
        g.loaded_executable = exe.0;
        check(api, (*api).PJRT_LoadedExecutable_GetExecutable.unwrap()(&mut g), "GetExecutable");
        let mut a: sys::PJRT_Executable_Serialize_Args = zeroed();
        a.struct_size = size_of::<sys::PJRT_Executable_Serialize_Args>();
        a.executable = g.executable;
        check(api, (*api).PJRT_Executable_Serialize.unwrap()(&mut a), "Executable_Serialize");
        let bytes =
            std::slice::from_raw_parts(a.serialized_bytes as *const u8, a.serialized_bytes_size).to_vec();
        if let Some(deleter) = a.serialized_executable_deleter {
            deleter(a.serialized_executable);
        }
        let mut d: sys::PJRT_Executable_Destroy_Args = zeroed();
        d.struct_size = size_of::<sys::PJRT_Executable_Destroy_Args>();
        d.executable = g.executable;
        (*api).PJRT_Executable_Destroy.unwrap()(&mut d);
        bytes
    }

    /// Load an executable `serialize` produced on this plugin version.
    pub unsafe fn deserialize_and_load(&self, bytes: &[u8]) -> Executable {
        let api = self.client.api;
        let mut a: sys::PJRT_Executable_DeserializeAndLoad_Args = zeroed();
        a.struct_size = size_of::<sys::PJRT_Executable_DeserializeAndLoad_Args>();
        a.client = self.client.client;
        a.serialized_executable = bytes.as_ptr() as *const c_char;
        a.serialized_executable_size = bytes.len();
        check(api, (*api).PJRT_Executable_DeserializeAndLoad.unwrap()(&mut a), "Executable_DeserializeAndLoad");
        Executable(a.loaded_executable)
    }

    /// Release a compiled executable's device state.
    pub unsafe fn free_executable(&self, exe: Executable) {
        let mut d: sys::PJRT_LoadedExecutable_Destroy_Args = zeroed();
        d.struct_size = size_of::<sys::PJRT_LoadedExecutable_Destroy_Args>();
        d.executable = exe.0;
        (*self.client.api).PJRT_LoadedExecutable_Destroy.unwrap()(&mut d);
    }

    /// Compile MLIR bytecode once on the persistent client.
    pub unsafe fn compile(&self, code: &[u8]) -> Executable {
        Executable(self.client.compile(code))
    }

    /// Run a pre-compiled executable.
    pub unsafe fn run(&self, exe: &Executable, inputs: &Inputs, num_outputs: usize) -> Vec<Vec<u8>> {
        run_loaded(&self.client, exe.0, inputs, num_outputs)
    }

    /// Allocate one device buffer per `(dims, elem_type)` now, for
    /// [`Staging::transfer`] to fill later.
    ///
    /// The allocation is what happens here, and on a GPU client that is the
    /// point: see [`Staging`]. Call it before the work the transfers should
    /// overlap is enqueued.
    pub unsafe fn stage(&self, shapes: &[(&[i64], sys::PJRT_Buffer_Type)]) -> Staging {
        let api = self.client.api;
        let mut m: sys::PJRT_Device_DefaultMemory_Args = zeroed();
        m.struct_size = size_of::<sys::PJRT_Device_DefaultMemory_Args>();
        m.device = self.client.first_device();
        check(api, (*api).PJRT_Device_DefaultMemory.unwrap()(&mut m), "Device_DefaultMemory");
        let specs: Vec<sys::PJRT_ShapeSpec> = shapes
            .iter()
            .map(|(dims, elem_type)| {
                let mut s: sys::PJRT_ShapeSpec = zeroed();
                s.struct_size = size_of::<sys::PJRT_ShapeSpec>();
                s.dims = dims.as_ptr();
                s.num_dims = dims.len();
                s.element_type = *elem_type;
                s
            })
            .collect();
        let mut a: sys::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args = zeroed();
        a.struct_size = size_of::<sys::PJRT_Client_CreateBuffersForAsyncHostToDevice_Args>();
        a.client = self.client.client;
        a.shape_specs = specs.as_ptr() as *mut sys::PJRT_ShapeSpec;
        a.num_shape_specs = specs.len();
        a.memory = m.memory;
        check(
            api,
            (*api).PJRT_Client_CreateBuffersForAsyncHostToDevice.unwrap()(&mut a),
            "Client_CreateBuffersForAsyncHostToDevice",
        );
        Staging { api, manager: a.transfer_manager }
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
        // The outputs have been copied out; without this their device memory
        // lives until the plugin unloads, which a prover exhausts.
        outs.iter().for_each(|&b| self.client.destroy_buffer(b));
        (host, dispatch, readback)
    }

    /// Like [`run_buffers`], but leaves the outputs on the device.
    ///
    /// The other `run_*` methods copy every output to host, which is the right
    /// default for a result the caller is about to read. It is the wrong one
    /// for an intermediate: a pipeline whose stages are separate executables —
    /// because a host-driven protocol interleaves its own work between them —
    /// otherwise pays a round trip per stage boundary for data neither side
    /// looks at.
    ///
    /// Unlike the copying variants, the returned buffers own device memory:
    /// pass each to [`free_buffer`](Self::free_buffer) when done, or the
    /// allocation lives until the plugin unloads.
    pub unsafe fn run_buffers_to_device(
        &self,
        exe: &Executable,
        inputs: &[&Buffer],
        num_outputs: usize,
    ) -> Vec<Buffer> {
        let bufs: Vec<*mut sys::PJRT_Buffer> = inputs.iter().map(|b| b.0).collect();
        self.client
            .execute(exe.0, &bufs, num_outputs)
            .into_iter()
            .map(Buffer)
            .collect()
    }

    /// Copy one buffer to the host.
    ///
    /// The companion to [`run_buffers_to_device`](Self::run_buffers_to_device):
    /// keeping an execution's outputs on the device is only useful if the few
    /// the host *does* need — a commitment among intermediates it never reads —
    /// can be fetched without dragging the rest back.
    pub unsafe fn buffer_to_host(&self, buffer: &Buffer) -> Vec<u8> {
        self.client.to_host(buffer.0)
    }

    /// Release a buffer's device memory.
    ///
    /// Takes ownership so a freed buffer cannot be executed against. `Buffer`
    /// does not free on drop: it holds only the PJRT pointer, and the API
    /// handle needed to release it belongs to this `Session` — a `Drop` impl
    /// would have to reach a pointer into the plugin that may already have
    /// unloaded.
    pub unsafe fn free_buffer(&self, buffer: Buffer) {
        self.client.destroy_buffer(buffer.0);
    }
}

/// Device buffers allocated now and filled later.
///
/// [`Session::input_buffer`] allocates and transfers in one call, and on a
/// GPU client that costs the transfer any chance of running beside the
/// client's own kernels. XLA's GPU allocation model is
/// `kComputeSynchronized`: a buffer the allocator returns at time t may
/// only be written once the compute stream has drained everything enqueued
/// before t, which PJRT enforces by taking a compute-stream sync point when
/// it allocates and making the host-to-device stream wait on it. A buffer
/// allocated while a computation is in flight therefore cannot be written
/// until that computation finishes.
///
/// Allocating the whole set up front and transferring into it afterwards
/// moves that sync point earlier than the work it would otherwise wait for,
/// which is what lets an upload overlap kernels.
pub struct Staging {
    api: *const sys::PJRT_Api,
    manager: *mut sys::PJRT_AsyncHostToDeviceTransferManager,
}

/// A transfer that has been enqueued and not waited for.
///
/// The runtime reads the host data asynchronously, so it must stay alive
/// and unmodified until [`Transfer::wait`] returns. Dropping one without
/// waiting destroys the event and gives up that guarantee.
#[must_use = "the host data may not be dropped until the transfer completes"]
pub struct Transfer {
    api: *const sys::PJRT_Api,
    event: *mut sys::PJRT_Event,
}

impl Transfer {
    /// Block until the runtime has finished reading the host data.
    pub unsafe fn wait(self) {
        let api = self.api;
        let event = self.event;
        std::mem::forget(self);
        Client { api, client: ptr::null_mut() }.await_event(event);
    }
}

impl Drop for Transfer {
    fn drop(&mut self) {
        unsafe {
            let mut d: sys::PJRT_Event_Destroy_Args = zeroed();
            d.struct_size = size_of::<sys::PJRT_Event_Destroy_Args>();
            d.event = self.event;
            (*self.api).PJRT_Event_Destroy.unwrap()(&mut d);
        }
    }
}

impl Staging {
    /// Copy `data` into buffer `index` without waiting for the transfer.
    ///
    /// Each buffer takes exactly one transfer; a second on the same index
    /// is an error from the runtime.
    pub unsafe fn transfer(&self, index: usize, data: &[u8]) -> Transfer {
        let mut a: sys::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args = zeroed();
        a.struct_size = size_of::<sys::PJRT_AsyncHostToDeviceTransferManager_TransferData_Args>();
        a.transfer_manager = self.manager;
        a.buffer_index = index as i32;
        a.data = data.as_ptr() as *const c_void;
        a.offset = 0;
        a.transfer_size = data.len() as i64;
        a.is_last_transfer = true;
        check(
            self.api,
            (*self.api).PJRT_AsyncHostToDeviceTransferManager_TransferData.unwrap()(&mut a),
            "AsyncHostToDeviceTransferManager_TransferData",
        );
        Transfer { api: self.api, event: a.done_with_h2d_transfer }
    }

    /// Take buffer `index` out of the manager. The buffer owns device
    /// memory until passed to [`Session::free_buffer`], and an execution
    /// that consumes it waits for its transfer on its own.
    pub unsafe fn retrieve(&self, index: usize) -> Buffer {
        let mut a: sys::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args = zeroed();
        a.struct_size =
            size_of::<sys::PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args>();
        a.transfer_manager = self.manager;
        a.buffer_index = index as i32;
        check(
            self.api,
            (*self.api).PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer.unwrap()(&mut a),
            "AsyncHostToDeviceTransferManager_RetrieveBuffer",
        );
        Buffer(a.buffer_out)
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        unsafe {
            let mut d: sys::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args = zeroed();
            d.struct_size =
                size_of::<sys::PJRT_AsyncHostToDeviceTransferManager_Destroy_Args>();
            d.transfer_manager = self.manager;
            (*self.api).PJRT_AsyncHostToDeviceTransferManager_Destroy.unwrap()(&mut d);
        }
    }
}

/// Which of the async host-to-device transfer manager's entry points the
/// loaded plugin implements. A PJRT plugin may leave any of them null, and
/// that manager is the only way through this API to allocate a device
/// buffer before the transfer that fills it.
pub unsafe fn async_h2d_entry_points() -> Vec<(&'static str, bool)> {
    let api = shared_pjrt().api;
    vec![
        (
            "PJRT_Client_CreateBuffersForAsyncHostToDevice",
            (*api).PJRT_Client_CreateBuffersForAsyncHostToDevice.is_some(),
        ),
        (
            "PJRT_AsyncHostToDeviceTransferManager_TransferData",
            (*api).PJRT_AsyncHostToDeviceTransferManager_TransferData.is_some(),
        ),
        (
            "PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer",
            (*api).PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer.is_some(),
        ),
        (
            "PJRT_AsyncHostToDeviceTransferManager_Destroy",
            (*api).PJRT_AsyncHostToDeviceTransferManager_Destroy.is_some(),
        ),
        ("PJRT_Device_DefaultMemory", (*api).PJRT_Device_DefaultMemory.is_some()),
    ]
}
