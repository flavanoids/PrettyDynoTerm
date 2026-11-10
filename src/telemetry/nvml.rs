use crate::telemetry::pci::PciBusId;
use crate::telemetry::types::PerGpuTelemetry;
use std::ffi::{CStr, c_char, c_uint};
use std::sync::Arc;

pub const NVML_TEMPERATURE_GPU: c_uint = 0;
pub const NVML_TEMPERATURE_THRESHOLD_SHUTDOWN: c_uint = 0;
pub const NVML_TEMPERATURE_THRESHOLD_SLOWDOWN: c_uint = 1;

pub const NVML_CLOCK_GRAPHICS: c_uint = 0;
pub const NVML_CLOCK_SM: c_uint = 1;
pub const NVML_CLOCK_MEM: c_uint = 2;

// Standard NVML throttle reasons bitmask flags
pub const NVML_CLOCKS_THROTTLE_REASON_GPU_HARDWARE_SLOWDOWN: u64 = 0x0000000000000008;
pub const NVML_CLOCKS_THROTTLE_REASON_SW_THERMAL_SLOWDOWN: u64 = 0x0000000000000020;
pub const NVML_CLOCKS_THROTTLE_REASON_SW_POWER_CAP: u64 = 0x0000000000000004;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct NvmlPciInfo {
    pub bus_id_legacy: [c_char; 16],
    pub domain: c_uint,
    pub bus: c_uint,
    pub device: c_uint,
    pub pci_device_id: c_uint,
    pub pci_sub_system_id: c_uint,
    pub bus_id: [c_char; 32],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct NvmlUtilization {
    pub gpu: c_uint,
    pub memory: c_uint,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct NvmlMemory {
    pub total: u64,
    pub free: u64,
    pub used: u64,
}

pub type NvmlDevice = *mut std::ffi::c_void;
pub type NvmlReturn = c_uint;
pub const NVML_SUCCESS: NvmlReturn = 0;

/// Trait abstracting NVML interactions, enabling unit testing with mocks.
pub trait NvmlClient: Send + Sync {
    fn is_available(&self) -> bool;
    fn query_all_gpus(&self) -> Result<Vec<PerGpuTelemetry>, String>;
}

/// Dynamic loader that dlopens `libnvidia-ml.so.1` at runtime with zero link-time dependencies.
pub struct DlopenNvml {
    _lib: Option<Arc<libloading::Library>>,
    available: bool,
}

impl Default for DlopenNvml {
    fn default() -> Self {
        Self::new()
    }
}

impl DlopenNvml {
    pub fn new() -> Self {
        let candidates = ["libnvidia-ml.so.1", "libnvidia-ml.so"];
        for candidate in candidates {
            if let Ok(lib) = unsafe { libloading::Library::new(candidate) } {
                let arc_lib = Arc::new(lib);
                // Attempt initialization
                let init_ok = unsafe {
                    if let Ok(init_fn) =
                        arc_lib.get::<unsafe extern "C" fn() -> NvmlReturn>(b"nvmlInit_v2\0")
                    {
                        init_fn() == NVML_SUCCESS
                    } else if let Ok(init_fn) =
                        arc_lib.get::<unsafe extern "C" fn() -> NvmlReturn>(b"nvmlInit\0")
                    {
                        init_fn() == NVML_SUCCESS
                    } else {
                        false
                    }
                };

                if init_ok {
                    return Self {
                        _lib: Some(arc_lib),
                        available: true,
                    };
                }
            }
        }

        Self {
            _lib: None,
            available: false,
        }
    }
}

impl NvmlClient for DlopenNvml {
    fn is_available(&self) -> bool {
        self.available
    }

    fn query_all_gpus(&self) -> Result<Vec<PerGpuTelemetry>, String> {
        let lib = match &self._lib {
            Some(l) if self.available => l,
            _ => {
                return Err(
                    "NVML is unavailable on this system (libnvidia-ml.so.1 missing)".to_string(),
                );
            }
        };

        unsafe {
            let get_count: libloading::Symbol<unsafe extern "C" fn(*mut c_uint) -> NvmlReturn> =
                lib.get(b"nvmlDeviceGetCount_v2\0")
                    .or_else(|_| lib.get(b"nvmlDeviceGetCount\0"))
                    .map_err(|e| format!("Missing nvmlDeviceGetCount: {}", e))?;

            let get_handle: libloading::Symbol<
                unsafe extern "C" fn(c_uint, *mut NvmlDevice) -> NvmlReturn,
            > = lib
                .get(b"nvmlDeviceGetHandleByIndex_v2\0")
                .or_else(|_| lib.get(b"nvmlDeviceGetHandleByIndex\0"))
                .map_err(|e| format!("Missing nvmlDeviceGetHandleByIndex: {}", e))?;

            let mut count: c_uint = 0;
            if get_count(&mut count) != NVML_SUCCESS {
                return Err("Failed to query NVML device count".to_string());
            }

            let mut results = Vec::new();

            for i in 0..count {
                let mut handle: NvmlDevice = std::ptr::null_mut();
                if get_handle(i, &mut handle) != NVML_SUCCESS || handle.is_null() {
                    continue;
                }

                let mut gpu = PerGpuTelemetry::default();

                // 1. PCI Bus ID
                if let Ok(get_pci) = lib
                    .get::<unsafe extern "C" fn(NvmlDevice, *mut NvmlPciInfo) -> NvmlReturn>(
                        b"nvmlDeviceGetPciInfo_v3\0",
                    )
                    .or_else(|_| lib.get(b"nvmlDeviceGetPciInfo\0"))
                {
                    let mut pci_info = std::mem::zeroed::<NvmlPciInfo>();
                    if get_pci(handle, &mut pci_info) == NVML_SUCCESS {
                        let bus_str = CStr::from_ptr(pci_info.bus_id.as_ptr()).to_string_lossy();
                        gpu.pci_bus_id = PciBusId::new(bus_str);
                    }
                }

                // 2. Name
                if let Ok(get_name) =
                    lib.get::<unsafe extern "C" fn(NvmlDevice, *mut c_char, c_uint) -> NvmlReturn>(
                        b"nvmlDeviceGetName\0",
                    )
                {
                    let mut buf = [0 as c_char; 96];
                    if get_name(handle, buf.as_mut_ptr(), buf.len() as c_uint) == NVML_SUCCESS {
                        gpu.gpu_name = CStr::from_ptr(buf.as_ptr()).to_string_lossy().to_string();
                    }
                }

                // 3. Temperatures
                if let Ok(get_temp) =
                    lib.get::<unsafe extern "C" fn(NvmlDevice, c_uint, *mut c_uint) -> NvmlReturn>(
                        b"nvmlDeviceGetTemperature\0",
                    )
                {
                    let mut temp: c_uint = 0;
                    if get_temp(handle, NVML_TEMPERATURE_GPU, &mut temp) == NVML_SUCCESS {
                        gpu.edge_temp_c = Some(temp as f32);
                    }
                }

                if let Ok(get_thresh) =
                    lib.get::<unsafe extern "C" fn(NvmlDevice, c_uint, *mut c_uint) -> NvmlReturn>(
                        b"nvmlDeviceGetTemperatureThreshold\0",
                    )
                {
                    let mut slowdown: c_uint = 0;
                    if get_thresh(handle, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, &mut slowdown)
                        == NVML_SUCCESS
                    {
                        gpu.nvml_slowdown_temp_c = Some(slowdown as f32);
                    }

                    let mut shutdown: c_uint = 0;
                    if get_thresh(handle, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN, &mut shutdown)
                        == NVML_SUCCESS
                    {
                        gpu.nvml_shutdown_temp_c = Some(shutdown as f32);
                    }
                }

                // 4. Power (mW -> W)
                if let Ok(get_power) =
                    lib.get::<unsafe extern "C" fn(NvmlDevice, *mut c_uint) -> NvmlReturn>(
                        b"nvmlDeviceGetPowerUsage\0",
                    )
                {
                    let mut mw: c_uint = 0;
                    if get_power(handle, &mut mw) == NVML_SUCCESS {
                        gpu.power_w = Some(mw as f32 / 1000.0);
                    }
                }

                if let Ok(get_cap) = lib
                    .get::<unsafe extern "C" fn(NvmlDevice, *mut c_uint) -> NvmlReturn>(
                        b"nvmlDeviceGetPowerManagementLimit\0",
                    )
                {
                    let mut cap_mw: c_uint = 0;
                    if get_cap(handle, &mut cap_mw) == NVML_SUCCESS {
                        gpu.power_cap_w = Some(cap_mw as f32 / 1000.0);
                    }
                }

                // 5. Clocks (MHz)
                if let Ok(get_clock) =
                    lib.get::<unsafe extern "C" fn(NvmlDevice, c_uint, *mut c_uint) -> NvmlReturn>(
                        b"nvmlDeviceGetClockInfo\0",
                    )
                {
                    let mut sm_mhz: c_uint = 0;
                    if get_clock(handle, NVML_CLOCK_SM, &mut sm_mhz) == NVML_SUCCESS {
                        gpu.clock_mhz = Some(sm_mhz);
                    }

                    let mut mem_mhz: c_uint = 0;
                    if get_clock(handle, NVML_CLOCK_MEM, &mut mem_mhz) == NVML_SUCCESS {
                        gpu.mem_clock_mhz = Some(mem_mhz);
                    }
                }

                // 6. Utilization
                if let Ok(get_util) =
                    lib.get::<unsafe extern "C" fn(NvmlDevice, *mut NvmlUtilization) -> NvmlReturn>(
                        b"nvmlDeviceGetUtilizationRates\0",
                    )
                {
                    let mut util = std::mem::zeroed::<NvmlUtilization>();
                    if get_util(handle, &mut util) == NVML_SUCCESS {
                        gpu.gpu_util_pct = Some(util.gpu as f32);
                        gpu.mem_util_pct = Some(util.memory as f32);
                    }
                }

                // 7. VRAM (Bytes -> MB)
                if let Ok(get_mem) =
                    lib.get::<unsafe extern "C" fn(NvmlDevice, *mut NvmlMemory) -> NvmlReturn>(
                        b"nvmlDeviceGetMemoryInfo\0",
                    )
                {
                    let mut mem = std::mem::zeroed::<NvmlMemory>();
                    if get_mem(handle, &mut mem) == NVML_SUCCESS {
                        gpu.vram_used_mb = Some(mem.used / (1024 * 1024));
                        gpu.vram_total_mb = Some(mem.total / (1024 * 1024));
                    }
                }

                // 8. Throttle Reasons
                if let Ok(get_throttle) = lib
                    .get::<unsafe extern "C" fn(NvmlDevice, *mut u64) -> NvmlReturn>(
                        b"nvmlDeviceGetCurrentClocksThrottleReasons\0",
                    )
                    .or_else(|_| lib.get(b"nvmlDeviceGetCurrentClocksEventReasons\0"))
                {
                    let mut reasons: u64 = 0;
                    if get_throttle(handle, &mut reasons) == NVML_SUCCESS {
                        gpu.nvml_throttle_reasons = Some(reasons);
                    }
                }

                // Fan sensors on passively cooled server cards (H200, H100) return error or unsupported
                // We leave fan_rpm, fan_pwm as None by default.

                results.push(gpu);
            }

            Ok(results)
        }
    }
}

/// Mock implementation of NVML for deterministic testing of NVIDIA datacenter telemetry.
pub struct MockNvml {
    pub gpus: Vec<PerGpuTelemetry>,
    pub available: bool,
}

impl MockNvml {
    pub fn new(gpus: Vec<PerGpuTelemetry>) -> Self {
        Self {
            gpus,
            available: true,
        }
    }

    pub fn unavailable() -> Self {
        Self {
            gpus: Vec::new(),
            available: false,
        }
    }
}

impl NvmlClient for MockNvml {
    fn is_available(&self) -> bool {
        self.available
    }

    fn query_all_gpus(&self) -> Result<Vec<PerGpuTelemetry>, String> {
        if self.available {
            Ok(self.gpus.clone())
        } else {
            Err("NVML is unavailable on this system".to_string())
        }
    }
}
