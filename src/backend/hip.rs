use libloading::{Library, Symbol};
use std::ffi::{CStr, CString};
use std::sync::Arc;

use crate::backend::traits::{BackendError, BackendType, ComputeBackend, ComputeDevice};
use crate::gpu::pipeline::Uniforms;
use crate::telemetry::PciBusId;

pub const EMBEDDED_FATBIN_GFX: &[u8] = include_bytes!("../kernels/blobs/dyno_kernels_gfx.co");

pub type HipModuleLaunchKernelFn = unsafe extern "C" fn(
    usize,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    usize,
    *mut *mut std::ffi::c_void,
    *mut *mut std::ffi::c_void,
) -> i32;

/// Raw function pointers loaded dynamically from `libamdhip64.so.6`.
#[allow(non_snake_case)]
pub struct HipFns {
    pub hipInit: unsafe extern "C" fn(u32) -> i32,
    pub hipDriverGetVersion: unsafe extern "C" fn(*mut i32) -> i32,
    pub hipGetDeviceCount: unsafe extern "C" fn(*mut i32) -> i32,
    pub hipSetDevice: unsafe extern "C" fn(i32) -> i32,
    pub hipDeviceGetName: unsafe extern "C" fn(*mut i8, i32, i32) -> i32,
    pub hipDeviceGetPCIBusId: unsafe extern "C" fn(*mut i8, i32, i32) -> i32,
    pub hipMalloc: unsafe extern "C" fn(*mut u64, usize) -> i32,
    pub hipFree: unsafe extern "C" fn(u64) -> i32,
    pub hipMemcpyDtoH: unsafe extern "C" fn(*mut std::ffi::c_void, u64, usize) -> i32,
    pub hipModuleLoadData: unsafe extern "C" fn(*mut usize, *const std::ffi::c_void) -> i32,
    pub hipModuleUnload: unsafe extern "C" fn(usize) -> i32,
    pub hipModuleGetFunction: unsafe extern "C" fn(*mut usize, usize, *const i8) -> i32,
    pub hipModuleLaunchKernel: HipModuleLaunchKernelFn,
    pub hipDeviceSynchronize: unsafe extern "C" fn() -> i32,
}

pub struct HipDriver {
    _lib: Library,
    fns: HipFns,
}

impl HipDriver {
    #[allow(non_snake_case)]
    pub fn load() -> Result<Self, BackendError> {
        unsafe {
            let lib = Library::new("libamdhip64.so.6")
                .or_else(|_| Library::new("libamdhip64.so"))
                .map_err(|e| BackendError::LibraryNotFound(format!("libamdhip64.so.6 ({})", e)))?;

            let hipInit: Symbol<unsafe extern "C" fn(u32) -> i32> = lib
                .get(b"hipInit\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("hipInit ({})", e)))?;

            let hipDriverGetVersion: Symbol<unsafe extern "C" fn(*mut i32) -> i32> =
                lib.get(b"hipDriverGetVersion\0").map_err(|e| {
                    BackendError::SymbolNotFound(format!("hipDriverGetVersion ({})", e))
                })?;

            let hipGetDeviceCount: Symbol<unsafe extern "C" fn(*mut i32) -> i32> = lib
                .get(b"hipGetDeviceCount\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("hipGetDeviceCount ({})", e)))?;

            let hipSetDevice: Symbol<unsafe extern "C" fn(i32) -> i32> = lib
                .get(b"hipSetDevice\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("hipSetDevice ({})", e)))?;

            let hipDeviceGetName: Symbol<unsafe extern "C" fn(*mut i8, i32, i32) -> i32> = lib
                .get(b"hipDeviceGetName\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("hipDeviceGetName ({})", e)))?;

            let hipDeviceGetPCIBusId: Symbol<unsafe extern "C" fn(*mut i8, i32, i32) -> i32> =
                lib.get(b"hipDeviceGetPCIBusId\0").map_err(|e| {
                    BackendError::SymbolNotFound(format!("hipDeviceGetPCIBusId ({})", e))
                })?;

            let hipMalloc: Symbol<unsafe extern "C" fn(*mut u64, usize) -> i32> = lib
                .get(b"hipMalloc\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("hipMalloc ({})", e)))?;

            let hipFree: Symbol<unsafe extern "C" fn(u64) -> i32> = lib
                .get(b"hipFree\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("hipFree ({})", e)))?;

            let hipMemcpyDtoH: Symbol<
                unsafe extern "C" fn(*mut std::ffi::c_void, u64, usize) -> i32,
            > = lib
                .get(b"hipMemcpyDtoH\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("hipMemcpyDtoH ({})", e)))?;

            let hipModuleLoadData: Symbol<
                unsafe extern "C" fn(*mut usize, *const std::ffi::c_void) -> i32,
            > = lib
                .get(b"hipModuleLoadData\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("hipModuleLoadData ({})", e)))?;

            let hipModuleUnload: Symbol<unsafe extern "C" fn(usize) -> i32> = lib
                .get(b"hipModuleUnload\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("hipModuleUnload ({})", e)))?;

            let hipModuleGetFunction: Symbol<
                unsafe extern "C" fn(*mut usize, usize, *const i8) -> i32,
            > = lib.get(b"hipModuleGetFunction\0").map_err(|e| {
                BackendError::SymbolNotFound(format!("hipModuleGetFunction ({})", e))
            })?;

            let hipModuleLaunchKernel: Symbol<HipModuleLaunchKernelFn> =
                lib.get(b"hipModuleLaunchKernel\0").map_err(|e| {
                    BackendError::SymbolNotFound(format!("hipModuleLaunchKernel ({})", e))
                })?;

            let hipDeviceSynchronize: Symbol<unsafe extern "C" fn() -> i32> =
                lib.get(b"hipDeviceSynchronize\0").map_err(|e| {
                    BackendError::SymbolNotFound(format!("hipDeviceSynchronize ({})", e))
                })?;

            let res = hipInit(0);
            if res != 0 {
                return Err(BackendError::DeviceInitFailed(format!(
                    "hipInit returned {}",
                    res
                )));
            }

            Ok(Self {
                fns: HipFns {
                    hipInit: *hipInit,
                    hipDriverGetVersion: *hipDriverGetVersion,
                    hipGetDeviceCount: *hipGetDeviceCount,
                    hipSetDevice: *hipSetDevice,
                    hipDeviceGetName: *hipDeviceGetName,
                    hipDeviceGetPCIBusId: *hipDeviceGetPCIBusId,
                    hipMalloc: *hipMalloc,
                    hipFree: *hipFree,
                    hipMemcpyDtoH: *hipMemcpyDtoH,
                    hipModuleLoadData: *hipModuleLoadData,
                    hipModuleUnload: *hipModuleUnload,
                    hipModuleGetFunction: *hipModuleGetFunction,
                    hipModuleLaunchKernel: *hipModuleLaunchKernel,
                    hipDeviceSynchronize: *hipDeviceSynchronize,
                },
                _lib: lib,
            })
        }
    }
}

pub struct HipComputeDevice {
    index: usize,
    name: String,
    pci_bus_id: PciBusId,
    driver: Arc<HipDriver>,
    module: usize,
    function: usize,
    d_pixels: u64,
    width: u32,
    height: u32,
}

impl ComputeDevice for HipComputeDevice {
    fn index(&self) -> usize {
        self.index
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn pci_bus_id(&self) -> &PciBusId {
        &self.pci_bus_id
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Hip
    }

    fn allocate_storage_buffer(&mut self, width: u32, height: u32) -> Result<(), BackendError> {
        unsafe {
            let _ = (self.driver.fns.hipSetDevice)(self.index as i32);
            if self.d_pixels != 0 {
                let _ = (self.driver.fns.hipFree)(self.d_pixels);
                self.d_pixels = 0;
            }

            let num_bytes = (width as usize) * (height as usize) * 4;
            let mut ptr: u64 = 0;
            let res = (self.driver.fns.hipMalloc)(&mut ptr, num_bytes);
            if res != 0 {
                return Err(BackendError::AllocationFailed(format!(
                    "hipMalloc failed with code {}",
                    res
                )));
            }

            self.d_pixels = ptr;
            self.width = width;
            self.height = height;
            Ok(())
        }
    }

    fn dispatch(&mut self, uniforms: &Uniforms) -> Result<(), BackendError> {
        unsafe {
            let _ = (self.driver.fns.hipSetDevice)(self.index as i32);
            let mut u_copy = *uniforms;

            // Multiple of 64 for AMD wave64 CDNA architecture
            let block_x = 16u32;
            let block_y = 16u32;
            let grid_x = self.width.div_ceil(block_x);
            let grid_y = self.height.div_ceil(block_y);

            let mut arg_d_pixels = self.d_pixels;
            let mut kernel_params: [*mut std::ffi::c_void; 2] = [
                &mut arg_d_pixels as *mut _ as *mut std::ffi::c_void,
                &mut u_copy as *mut _ as *mut std::ffi::c_void,
            ];

            let res = (self.driver.fns.hipModuleLaunchKernel)(
                self.function,
                grid_x,
                grid_y,
                1,
                block_x,
                block_y,
                1,
                0,
                0,
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
            );

            if res != 0 {
                return Err(BackendError::DispatchFailed(format!(
                    "hipModuleLaunchKernel failed with code {}",
                    res
                )));
            }

            self.synchronize()
        }
    }

    fn readback_checksum(&mut self) -> Result<u64, BackendError> {
        unsafe {
            let _ = (self.driver.fns.hipSetDevice)(self.index as i32);
            let mut checksum_bytes = [0u8; 8];
            let res = (self.driver.fns.hipMemcpyDtoH)(
                checksum_bytes.as_mut_ptr() as *mut _,
                self.d_pixels,
                8,
            );
            if res != 0 {
                return Err(BackendError::ReadbackFailed(format!(
                    "hipMemcpyDtoH failed with code {}",
                    res
                )));
            }
            Ok(u64::from_le_bytes(checksum_bytes))
        }
    }

    fn synchronize(&self) -> Result<(), BackendError> {
        unsafe {
            let _ = (self.driver.fns.hipSetDevice)(self.index as i32);
            let res = (self.driver.fns.hipDeviceSynchronize)();
            if res != 0 {
                return Err(BackendError::SyncFailed(format!(
                    "hipDeviceSynchronize failed with code {}",
                    res
                )));
            }
            Ok(())
        }
    }
}

impl Drop for HipComputeDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.fns.hipSetDevice)(self.index as i32);
            if self.d_pixels != 0 {
                let _ = (self.driver.fns.hipFree)(self.d_pixels);
            }
            if self.module != 0 {
                let _ = (self.driver.fns.hipModuleUnload)(self.module);
            }
        }
    }
}

pub struct HipBackend {
    driver: Result<Arc<HipDriver>, BackendError>,
}

impl Default for HipBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl HipBackend {
    pub fn new() -> Self {
        Self {
            driver: HipDriver::load().map(Arc::new),
        }
    }
}

impl ComputeBackend for HipBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Hip
    }

    fn is_available(&self) -> bool {
        self.driver.is_ok()
    }

    fn enumerate_device_pci_ids(&self) -> Result<Vec<PciBusId>, BackendError> {
        let driver = match &self.driver {
            Ok(d) => d,
            Err(e) => return Err(e.clone()),
        };

        unsafe {
            let mut count = 0i32;
            let res = (driver.fns.hipGetDeviceCount)(&mut count);
            if res != 0 {
                return Err(BackendError::DeviceInitFailed(format!(
                    "hipGetDeviceCount failed with code {}",
                    res
                )));
            }

            let mut ids = Vec::with_capacity(count as usize);
            for i in 0..count {
                let mut pci_buf = [0i8; 64];
                let _ = (driver.fns.hipDeviceGetPCIBusId)(pci_buf.as_mut_ptr(), 64, i);
                let c_str = CStr::from_ptr(pci_buf.as_ptr());
                let pci_str = c_str.to_string_lossy();
                ids.push(PciBusId::new(pci_str));
            }

            Ok(ids)
        }
    }

    fn create_device(
        &self,
        index: usize,
        width: u32,
        height: u32,
    ) -> Result<Box<dyn ComputeDevice>, BackendError> {
        let driver = match &self.driver {
            Ok(d) => Arc::clone(d),
            Err(e) => return Err(e.clone()),
        };

        unsafe {
            let res_set = (driver.fns.hipSetDevice)(index as i32);
            if res_set != 0 {
                return Err(BackendError::DeviceInitFailed(format!(
                    "hipSetDevice failed for index {}",
                    index
                )));
            }

            let mut name_buf = [0i8; 128];
            let _ = (driver.fns.hipDeviceGetName)(name_buf.as_mut_ptr(), 128, index as i32);
            let name = CStr::from_ptr(name_buf.as_ptr())
                .to_string_lossy()
                .to_string();

            let mut pci_buf = [0i8; 64];
            let _ = (driver.fns.hipDeviceGetPCIBusId)(pci_buf.as_mut_ptr(), 64, index as i32);
            let pci_id = PciBusId::new(CStr::from_ptr(pci_buf.as_ptr()).to_string_lossy());

            // Load embedded AMD CDNA fatbin
            let mut module = 0usize;
            let res_mod = (driver.fns.hipModuleLoadData)(
                &mut module,
                EMBEDDED_FATBIN_GFX.as_ptr() as *const _,
            );
            if res_mod != 0 {
                return Err(BackendError::DeviceInitFailed(format!(
                    "hipModuleLoadData failed to load embedded CDNA fatbin (code {})",
                    res_mod
                )));
            }

            let fn_name = CString::new("dyno_unified_compute").unwrap();
            let mut function = 0usize;
            let res_fn = (driver.fns.hipModuleGetFunction)(&mut function, module, fn_name.as_ptr());
            if res_fn != 0 {
                let _ = (driver.fns.hipModuleUnload)(module);
                return Err(BackendError::DeviceInitFailed(format!(
                    "hipModuleGetFunction failed to find dyno_unified_compute (code {})",
                    res_fn
                )));
            }

            let mut device = HipComputeDevice {
                index,
                name,
                pci_bus_id: pci_id,
                driver,
                module,
                function,
                d_pixels: 0,
                width: 0,
                height: 0,
            };

            device.allocate_storage_buffer(width, height)?;
            Ok(Box::new(device))
        }
    }
}
