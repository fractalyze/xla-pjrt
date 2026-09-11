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
        if let Some(eager) = options.eager_load_executable_modules {
            let mut nv = named_value(b"eager_load_executable_modules");
            nv.type_ = sys::PJRT_NamedValue_kBool;
            nv.__bindgen_anon_1.bool_value = eager;
            named.push(nv);
        }
        if let Some(kind) = options.allocator {
            // The one string-valued option here: `value_size` counts the
            // characters rather than standing at the scalar 1, and the
            // spelling is `&'static str` so it outlives the call.
            let mut nv = named_value(b"allocator");
            nv.type_ = sys::PJRT_NamedValue_kString;
            nv.__bindgen_anon_1.string_value = kind.as_str().as_ptr() as *const c_char;
            nv.value_size = kind.as_str().len();
            named.push(nv);
        }
        if let Some(threshold) = options.staging_threshold_bytes {
            let mut nv = named_value(b"staging_threshold_bytes");
            nv.type_ = sys::PJRT_NamedValue_kInt64;
            nv.__bindgen_anon_1.int64_value = threshold;
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

/// What a client's device allocator holds and has held.
///
/// `PJRT_Device_MemoryStats`. Only `bytes_in_use` is required of a plugin;
/// each of the rest arrives with a flag beside it, so an allocator that does
/// not keep one leaves it `None` rather than reporting zero.
///
/// The pair worth reading together is `peak_bytes_in_use` against
/// `peak_pool_bytes`: the first is the live bytes at the high-water mark, the
/// second what the allocator held from the driver to place them in. Their
/// difference is what an arena costs above its data, and it is a measurement
/// on a run that finished rather than a bound from one that died.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryStats {
    pub bytes_in_use: i64,
    pub peak_bytes_in_use: Option<i64>,
    pub largest_alloc_size: Option<i64>,
    pub bytes_limit: Option<i64>,
    pub largest_free_block_bytes: Option<i64>,
    pub pool_bytes: Option<i64>,
    pub peak_pool_bytes: Option<i64>,
}

/// Which device allocator the GPU plugin builds for a client.
///
/// The plugin parses the spelling and rejects anything else
/// (`xla/pjrt/c/pjrt_c_api_gpu_internal.cc`), so the kinds are an enum here
/// rather than a string the caller composes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocatorKind {
    /// Whatever the plugin picks for the platform, which on CUDA is [`Bfc`].
    ///
    /// [`Bfc`]: AllocatorKind::Bfc
    Default,
    /// The platform's own allocator: one driver allocation per request, no
    /// pool and no arena, so nothing is claimed up front.
    Platform,
    /// Best-fit with coalescing over one arena. Every allocation is placed in
    /// that arena, so a large one needs a free block of its size.
    Bfc,
    /// `cudaMallocAsync` out of the device's default memory pool. The driver
    /// maps the pool's pages behind a request, so an allocation needs
    /// contiguous virtual addresses rather than a contiguous free block.
    CudaAsync,
    /// CUDA virtual memory management: virtual address space reserved up
    /// front and mapped to physical pages as they are needed.
    Vmm,
}

impl AllocatorKind {
    /// The spelling the plugin parses. It rides as a counted string, so no
    /// NUL terminator is needed.
    pub fn as_str(self) -> &'static str {
        match self {
            AllocatorKind::Default => "default",
            AllocatorKind::Platform => "platform",
            AllocatorKind::Bfc => "bfc",
            AllocatorKind::CudaAsync => "cuda_async",
            AllocatorKind::Vmm => "vmm",
        }
    }
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
///
/// `eager_load_executable_modules: Some(true)` makes the plugin load a
/// deserialized executable's modules into the CUDA context inside
/// [`Session::deserialize_and_load`] instead of on its first execution. A
/// caller that loads many executables up front and runs each once otherwise
/// pays every module load on its critical path. The plugin must carry the
/// option (fractalyze/xla#664); an older one rejects the unknown key and
/// client creation fails, so leave it `None` against those.
///
/// `allocator` picks which device allocator the client builds; `None` leaves
/// the plugin's default, which on CUDA is the BFC allocator. The kind changes
/// what the two options above mean:
///
/// * Under BFC, `memory_fraction` is a ceiling as well as a claim — the
///   client's arena is that share of the card and an allocation that does not
///   fit a free block in it fails, however much of the card is free.
/// * Under [`AllocatorKind::CudaAsync`], the client allocates from the
///   device's *default* CUDA memory pool. `memory_fraction` of the card
///   becomes the pool's release threshold, which `preallocate` claims once up
///   front and the pool then holds rather than returning to the driver; it is
///   not a ceiling, and the pool grows past it while the card has room. With
///   `preallocate: Some(false)` the threshold is zero, so every free goes
///   straight back to the driver and a co-tenant can take it.
///
/// The key itself is an upstream PJRT GPU create option rather than a
/// fractalyze one, but a plugin still rejects a *kind* it cannot name, and
/// [`AllocatorKind::Vmm`] is the newest of them; a client creation that fails
/// on the kind says which spellings that plugin knows.
///
/// `staging_threshold_bytes` is the size at or above which a host-to-device
/// transfer is DMA'd straight out of the caller's pageable memory instead of
/// being copied through the client's pinned staging pool. The plugin's
/// default is 1 GiB, so a caller whose transfers are larger than that gets
/// the pageable rate on exactly its largest copies; setting this above them
/// buys the pinned rate at the cost of growing the pinned pool by about one
/// transfer. The plugin must carry the option (fractalyze/xla#718); an older
/// one rejects the unknown key and client creation fails, so leave it `None`
/// against those.
#[derive(Clone, Copy, Debug, Default)]
pub struct SessionOptions {
    pub preallocate: Option<bool>,
    pub memory_fraction: Option<f32>,
    pub eager_load_executable_modules: Option<bool>,
    pub staging_threshold_bytes: Option<i64>,
    pub allocator: Option<AllocatorKind>,
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

    /// What the client's device allocator holds and has held, or `None` when
    /// the plugin does not keep the statistics for this allocator kind (the
    /// platform allocator has none to report, and the C API's answer there is
    /// an error rather than zeros).
    unsafe fn memory_stats(&self) -> Option<MemoryStats> {
        let mut a: sys::PJRT_Device_MemoryStats_Args = zeroed();
        a.struct_size = size_of::<sys::PJRT_Device_MemoryStats_Args>();
        a.device = self.first_device();
        let err = (*self.api).PJRT_Device_MemoryStats.unwrap()(&mut a);
        if !err.is_null() {
            let mut d: sys::PJRT_Error_Destroy_Args = zeroed();
            d.struct_size = size_of::<sys::PJRT_Error_Destroy_Args>();
            d.error = err;
            (*self.api).PJRT_Error_Destroy.unwrap()(&mut d);
            return None;
        }
        let set = |value: i64, is_set: bool| is_set.then_some(value);
        Some(MemoryStats {
            bytes_in_use: a.bytes_in_use,
            peak_bytes_in_use: set(a.peak_bytes_in_use, a.peak_bytes_in_use_is_set),
            largest_alloc_size: set(a.largest_alloc_size, a.largest_alloc_size_is_set),
            bytes_limit: set(a.bytes_limit, a.bytes_limit_is_set),
            largest_free_block_bytes: set(
                a.largest_free_block_bytes,
                a.largest_free_block_bytes_is_set,
            ),
            pool_bytes: set(a.pool_bytes, a.pool_bytes_is_set),
            peak_pool_bytes: set(a.peak_pool_bytes, a.peak_pool_bytes_is_set),
        })
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
    /// What this session's device allocator holds and has held; `None` when
    /// the plugin keeps no statistics for the kind it built.
    pub unsafe fn memory_stats(&self) -> Option<MemoryStats> {
        self.client.memory_stats()
    }

    pub unsafe fn free_buffer(&self, buffer: Buffer) {
        self.client.destroy_buffer(buffer.0);
    }
}
