use crate::gpu::pipeline::Uniforms;
use crate::telemetry::PciBusId;
use std::fmt;

/// Supported compute backends for PrettyDynoTerm execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendType {
    #[default]
    Wgpu,
    Cuda,
    Hip,
}

impl fmt::Display for BackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendType::Wgpu => write!(f, "wgpu"),
            BackendType::Cuda => write!(f, "cuda"),
            BackendType::Hip => write!(f, "hip"),
        }
    }
}

/// Errors originating from backend initialization, dispatch, or memory management.
#[derive(Debug, Clone)]
pub enum BackendError {
    LibraryNotFound(String),
    SymbolNotFound(String),
    DeviceInitFailed(String),
    AllocationFailed(String),
    DispatchFailed(String),
    SyncFailed(String),
    ReadbackFailed(String),
    OutdatedDriver(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryNotFound(s) => write!(f, "GPU Library not found: {}", s),
            Self::SymbolNotFound(s) => write!(f, "Required symbol missing from GPU library: {}", s),
            Self::DeviceInitFailed(s) => write!(f, "GPU device initialization failed: {}", s),
            Self::AllocationFailed(s) => write!(f, "GPU VRAM allocation failed: {}", s),
            Self::DispatchFailed(s) => write!(f, "Kernel dispatch failed: {}", s),
            Self::SyncFailed(s) => write!(f, "GPU device synchronization failed: {}", s),
            Self::ReadbackFailed(s) => write!(f, "PCIe readback failed: {}", s),
            Self::OutdatedDriver(s) => write!(f, "GPU driver version insufficient: {}", s),
        }
    }
}

impl std::error::Error for BackendError {}

/// Abstract execution context for a single physical GPU device.
pub trait ComputeDevice: Send {
    fn index(&self) -> usize;
    fn name(&self) -> &str;
    fn pci_bus_id(&self) -> &PciBusId;
    fn backend_type(&self) -> BackendType;
    fn allocate_storage_buffer(&mut self, width: u32, height: u32) -> Result<(), BackendError>;
    fn dispatch(&mut self, uniforms: &Uniforms) -> Result<(), BackendError>;
    fn readback_checksum(&mut self) -> Result<u64, BackendError>;
    fn synchronize(&self) -> Result<(), BackendError>;
}

/// Abstract hardware backend driver (wgpu, CUDA driver API, or HIP runtime API).
pub trait ComputeBackend: Send + Sync {
    fn backend_type(&self) -> BackendType;
    fn is_available(&self) -> bool;
    fn enumerate_device_pci_ids(&self) -> Result<Vec<PciBusId>, BackendError>;
    fn create_device(
        &self,
        index: usize,
        width: u32,
        height: u32,
    ) -> Result<Box<dyn ComputeDevice>, BackendError>;
}
