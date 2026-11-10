use libloading::{Library, Symbol};
use std::ffi::{CStr, CString};
use std::sync::Arc;

use crate::backend::traits::{BackendError, BackendType, ComputeBackend, ComputeDevice};
use crate::gpu::pipeline::Uniforms;
use crate::telemetry::PciBusId;

pub const EMBEDDED_PTX_SM90: &[u8] = include_bytes!("../kernels/blobs/dyno_kernels_sm90.ptx");

pub type CuLaunchKernelFn = unsafe extern "C" fn(
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

/// Raw function pointers loaded dynamically from `libcuda.so.1`.
#[allow(non_snake_case)]
pub struct CudaFns {
    pub cuInit: unsafe extern "C" fn(u32) -> i32,
    pub cuDriverGetVersion: unsafe extern "C" fn(*mut i32) -> i32,
    pub cuDeviceGetCount: unsafe extern "C" fn(*mut i32) -> i32,
    pub cuDeviceGet: unsafe extern "C" fn(*mut i32, i32) -> i32,
    pub cuDeviceGetName: unsafe extern "C" fn(*mut i8, i32, i32) -> i32,
    pub cuDeviceGetPCIBusId: unsafe extern "C" fn(*mut i8, i32, i32) -> i32,
    pub cuCtxCreate: unsafe extern "C" fn(*mut usize, u32, i32) -> i32,
    pub cuCtxDestroy: unsafe extern "C" fn(usize) -> i32,
    pub cuCtxSetCurrent: unsafe extern "C" fn(usize) -> i32,
    pub cuMemAlloc: unsafe extern "C" fn(*mut u64, usize) -> i32,
    pub cuMemFree: unsafe extern "C" fn(u64) -> i32,
    pub cuMemcpyHtoD: unsafe extern "C" fn(u64, *const std::ffi::c_void, usize) -> i32,
    pub cuMemcpyDtoH: unsafe extern "C" fn(*mut std::ffi::c_void, u64, usize) -> i32,
    pub cuModuleLoadData: unsafe extern "C" fn(*mut usize, *const std::ffi::c_void) -> i32,
    pub cuModuleUnload: unsafe extern "C" fn(usize) -> i32,
    pub cuModuleGetFunction: unsafe extern "C" fn(*mut usize, usize, *const i8) -> i32,
    pub cuLaunchKernel: CuLaunchKernelFn,
    pub cuCtxSynchronize: unsafe extern "C" fn() -> i32,
}

pub struct CudaDriver {
    _lib: Library,
    fns: CudaFns,
}

impl CudaDriver {
    #[allow(non_snake_case)]
    pub fn load() -> Result<Self, BackendError> {
        unsafe {
            let lib = Library::new("libcuda.so.1")
                .map_err(|e| BackendError::LibraryNotFound(format!("libcuda.so.1 ({})", e)))?;

            let cuInit: Symbol<unsafe extern "C" fn(u32) -> i32> = lib
                .get(b"cuInit\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("cuInit ({})", e)))?;

            let cuDriverGetVersion: Symbol<unsafe extern "C" fn(*mut i32) -> i32> = lib
                .get(b"cuDriverGetVersion\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("cuDriverGetVersion ({})", e)))?;

            let cuDeviceGetCount: Symbol<unsafe extern "C" fn(*mut i32) -> i32> = lib
                .get(b"cuDeviceGetCount\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("cuDeviceGetCount ({})", e)))?;

            let cuDeviceGet: Symbol<unsafe extern "C" fn(*mut i32, i32) -> i32> = lib
                .get(b"cuDeviceGet\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("cuDeviceGet ({})", e)))?;

            let cuDeviceGetName: Symbol<unsafe extern "C" fn(*mut i8, i32, i32) -> i32> = lib
                .get(b"cuDeviceGetName\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("cuDeviceGetName ({})", e)))?;

            let cuDeviceGetPCIBusId: Symbol<unsafe extern "C" fn(*mut i8, i32, i32) -> i32> =
                lib.get(b"cuDeviceGetPCIBusId\0").map_err(|e| {
                    BackendError::SymbolNotFound(format!("cuDeviceGetPCIBusId ({})", e))
                })?;

            let cuCtxCreate: Symbol<unsafe extern "C" fn(*mut usize, u32, i32) -> i32> = lib
                .get(b"cuCtxCreate_v2\0")
                .or_else(|_| lib.get(b"cuCtxCreate\0"))
                .map_err(|e| BackendError::SymbolNotFound(format!("cuCtxCreate ({})", e)))?;

            let cuCtxDestroy: Symbol<unsafe extern "C" fn(usize) -> i32> = lib
                .get(b"cuCtxDestroy_v2\0")
                .or_else(|_| lib.get(b"cuCtxDestroy\0"))
                .map_err(|e| BackendError::SymbolNotFound(format!("cuCtxDestroy ({})", e)))?;

            let cuCtxSetCurrent: Symbol<unsafe extern "C" fn(usize) -> i32> = lib
                .get(b"cuCtxSetCurrent\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("cuCtxSetCurrent ({})", e)))?;

            let cuMemAlloc: Symbol<unsafe extern "C" fn(*mut u64, usize) -> i32> = lib
                .get(b"cuMemAlloc_v2\0")
                .or_else(|_| lib.get(b"cuMemAlloc\0"))
                .map_err(|e| BackendError::SymbolNotFound(format!("cuMemAlloc ({})", e)))?;

            let cuMemFree: Symbol<unsafe extern "C" fn(u64) -> i32> = lib
                .get(b"cuMemFree_v2\0")
                .or_else(|_| lib.get(b"cuMemFree\0"))
                .map_err(|e| BackendError::SymbolNotFound(format!("cuMemFree ({})", e)))?;

            let cuMemcpyHtoD: Symbol<
                unsafe extern "C" fn(u64, *const std::ffi::c_void, usize) -> i32,
            > = lib
                .get(b"cuMemcpyHtoD_v2\0")
                .or_else(|_| lib.get(b"cuMemcpyHtoD\0"))
                .map_err(|e| BackendError::SymbolNotFound(format!("cuMemcpyHtoD ({})", e)))?;

            let cuMemcpyDtoH: Symbol<
                unsafe extern "C" fn(*mut std::ffi::c_void, u64, usize) -> i32,
            > = lib
                .get(b"cuMemcpyDtoH_v2\0")
                .or_else(|_| lib.get(b"cuMemcpyDtoH\0"))
                .map_err(|e| BackendError::SymbolNotFound(format!("cuMemcpyDtoH ({})", e)))?;

            let cuModuleLoadData: Symbol<
                unsafe extern "C" fn(*mut usize, *const std::ffi::c_void) -> i32,
            > = lib
                .get(b"cuModuleLoadData\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("cuModuleLoadData ({})", e)))?;

            let cuModuleUnload: Symbol<unsafe extern "C" fn(usize) -> i32> = lib
                .get(b"cuModuleUnload\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("cuModuleUnload ({})", e)))?;

            let cuModuleGetFunction: Symbol<
                unsafe extern "C" fn(*mut usize, usize, *const i8) -> i32,
            > = lib.get(b"cuModuleGetFunction\0").map_err(|e| {
                BackendError::SymbolNotFound(format!("cuModuleGetFunction ({})", e))
            })?;

            let cuLaunchKernel: Symbol<CuLaunchKernelFn> = lib
                .get(b"cuLaunchKernel\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("cuLaunchKernel ({})", e)))?;

            let cuCtxSynchronize: Symbol<unsafe extern "C" fn() -> i32> = lib
                .get(b"cuCtxSynchronize\0")
                .map_err(|e| BackendError::SymbolNotFound(format!("cuCtxSynchronize ({})", e)))?;

            let res = cuInit(0);
            if res != 0 {
                return Err(BackendError::DeviceInitFailed(format!(
                    "cuInit returned {}",
                    res
                )));
            }

            Ok(Self {
                fns: CudaFns {
                    cuInit: *cuInit,
                    cuDriverGetVersion: *cuDriverGetVersion,
                    cuDeviceGetCount: *cuDeviceGetCount,
                    cuDeviceGet: *cuDeviceGet,
                    cuDeviceGetName: *cuDeviceGetName,
                    cuDeviceGetPCIBusId: *cuDeviceGetPCIBusId,
                    cuCtxCreate: *cuCtxCreate,
                    cuCtxDestroy: *cuCtxDestroy,
                    cuCtxSetCurrent: *cuCtxSetCurrent,
                    cuMemAlloc: *cuMemAlloc,
                    cuMemFree: *cuMemFree,
                    cuMemcpyHtoD: *cuMemcpyHtoD,
                    cuMemcpyDtoH: *cuMemcpyDtoH,
                    cuModuleLoadData: *cuModuleLoadData,
                    cuModuleUnload: *cuModuleUnload,
                    cuModuleGetFunction: *cuModuleGetFunction,
                    cuLaunchKernel: *cuLaunchKernel,
                    cuCtxSynchronize: *cuCtxSynchronize,
                },
                _lib: lib,
            })
        }
    }
}

pub struct CudaComputeDevice {
    index: usize,
    name: String,
    pci_bus_id: PciBusId,
    driver: Arc<CudaDriver>,
    ctx: usize,
    module: usize,
    function: usize,
    d_pixels: u64,
    width: u32,
    height: u32,
}

impl ComputeDevice for CudaComputeDevice {
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
        BackendType::Cuda
    }

    fn allocate_storage_buffer(&mut self, width: u32, height: u32) -> Result<(), BackendError> {
        unsafe {
            let _ = (self.driver.fns.cuCtxSetCurrent)(self.ctx);
            if self.d_pixels != 0 {
                let _ = (self.driver.fns.cuMemFree)(self.d_pixels);
                self.d_pixels = 0;
            }

            let num_bytes = (width as usize) * (height as usize) * 4;
            let mut ptr: u64 = 0;
            let res = (self.driver.fns.cuMemAlloc)(&mut ptr, num_bytes);
            if res != 0 {
                return Err(BackendError::AllocationFailed(format!(
                    "cuMemAlloc failed with code {}",
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
            let _ = (self.driver.fns.cuCtxSetCurrent)(self.ctx);
            let mut u_copy = *uniforms;

            let block_x = 16u32;
            let block_y = 16u32;
            let grid_x = self.width.div_ceil(block_x);
            let grid_y = self.height.div_ceil(block_y);

            let mut arg_d_pixels = self.d_pixels;
            let mut kernel_params: [*mut std::ffi::c_void; 2] = [
                &mut arg_d_pixels as *mut _ as *mut std::ffi::c_void,
                &mut u_copy as *mut _ as *mut std::ffi::c_void,
            ];

            let res = (self.driver.fns.cuLaunchKernel)(
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
                    "cuLaunchKernel failed with code {}",
                    res
                )));
            }

            self.synchronize()
        }
    }

    fn readback_checksum(&mut self) -> Result<u64, BackendError> {
        unsafe {
            let _ = (self.driver.fns.cuCtxSetCurrent)(self.ctx);
            let mut checksum_bytes = [0u8; 8];
            let res = (self.driver.fns.cuMemcpyDtoH)(
                checksum_bytes.as_mut_ptr() as *mut _,
                self.d_pixels,
                8,
            );
            if res != 0 {
                return Err(BackendError::ReadbackFailed(format!(
                    "cuMemcpyDtoH failed with code {}",
                    res
                )));
            }
            Ok(u64::from_le_bytes(checksum_bytes))
        }
    }

    fn synchronize(&self) -> Result<(), BackendError> {
        unsafe {
            let _ = (self.driver.fns.cuCtxSetCurrent)(self.ctx);
            let res = (self.driver.fns.cuCtxSynchronize)();
            if res != 0 {
                return Err(BackendError::SyncFailed(format!(
                    "cuCtxSynchronize failed with code {}",
                    res
                )));
            }
            Ok(())
        }
    }
}

impl Drop for CudaComputeDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.fns.cuCtxSetCurrent)(self.ctx);
            if self.d_pixels != 0 {
                let _ = (self.driver.fns.cuMemFree)(self.d_pixels);
            }
            if self.module != 0 {
                let _ = (self.driver.fns.cuModuleUnload)(self.module);
            }
            if self.ctx != 0 {
                let _ = (self.driver.fns.cuCtxDestroy)(self.ctx);
            }
        }
    }
}

pub struct CudaBackend {
    driver: Result<Arc<CudaDriver>, BackendError>,
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaBackend {
    pub fn new() -> Self {
        Self {
            driver: CudaDriver::load().map(Arc::new),
        }
    }
}

impl ComputeBackend for CudaBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Cuda
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
            let res = (driver.fns.cuDeviceGetCount)(&mut count);
            if res != 0 {
                return Err(BackendError::DeviceInitFailed(format!(
                    "cuDeviceGetCount failed with code {}",
                    res
                )));
            }

            let mut ids = Vec::with_capacity(count as usize);
            for i in 0..count {
                let mut dev = 0i32;
                let _ = (driver.fns.cuDeviceGet)(&mut dev, i);
                let mut pci_buf = [0i8; 64];
                let _ = (driver.fns.cuDeviceGetPCIBusId)(pci_buf.as_mut_ptr(), 64, dev);
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
            let mut dev = 0i32;
            let res = (driver.fns.cuDeviceGet)(&mut dev, index as i32);
            if res != 0 {
                return Err(BackendError::DeviceInitFailed(format!(
                    "cuDeviceGet failed for index {}",
                    index
                )));
            }

            let mut name_buf = [0i8; 128];
            let _ = (driver.fns.cuDeviceGetName)(name_buf.as_mut_ptr(), 128, dev);
            let name = CStr::from_ptr(name_buf.as_ptr())
                .to_string_lossy()
                .to_string();

            let mut pci_buf = [0i8; 64];
            let _ = (driver.fns.cuDeviceGetPCIBusId)(pci_buf.as_mut_ptr(), 64, dev);
            let pci_id = PciBusId::new(CStr::from_ptr(pci_buf.as_ptr()).to_string_lossy());

            let mut ctx = 0usize;
            let res_ctx = (driver.fns.cuCtxCreate)(&mut ctx, 0, dev);
            if res_ctx != 0 {
                return Err(BackendError::DeviceInitFailed(format!(
                    "cuCtxCreate failed with code {}",
                    res_ctx
                )));
            }

            // Load embedded Hopper sm_90 PTX
            let mut module = 0usize;
            let res_mod =
                (driver.fns.cuModuleLoadData)(&mut module, EMBEDDED_PTX_SM90.as_ptr() as *const _);
            if res_mod != 0 {
                let _ = (driver.fns.cuCtxDestroy)(ctx);
                return Err(BackendError::DeviceInitFailed(format!(
                    "cuModuleLoadData failed to JIT-compile embedded sm_90 PTX (code {})",
                    res_mod
                )));
            }

            let fn_name = CString::new("dyno_unified_compute").unwrap();
            let mut function = 0usize;
            let res_fn = (driver.fns.cuModuleGetFunction)(&mut function, module, fn_name.as_ptr());
            if res_fn != 0 {
                let _ = (driver.fns.cuModuleUnload)(module);
                let _ = (driver.fns.cuCtxDestroy)(ctx);
                return Err(BackendError::DeviceInitFailed(format!(
                    "cuModuleGetFunction failed to find dyno_unified_compute (code {})",
                    res_fn
                )));
            }

            let mut device = CudaComputeDevice {
                index,
                name,
                pci_bus_id: pci_id,
                driver,
                ctx,
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
