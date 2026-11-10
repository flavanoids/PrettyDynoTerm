pub mod cuda;
pub mod hip;
pub mod traits;
pub mod wgpu_backend;

pub use cuda::CudaBackend;
pub use hip::HipBackend;
pub use traits::{BackendError, BackendType, ComputeBackend, ComputeDevice};
pub use wgpu_backend::WgpuBackend;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// Backend choice requested from CLI options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendChoice {
    #[default]
    Auto,
    Wgpu,
    Cuda,
    Hip,
}

impl std::fmt::Display for BackendChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Wgpu => write!(f, "wgpu"),
            Self::Cuda => write!(f, "cuda"),
            Self::Hip => write!(f, "hip"),
        }
    }
}

/// Resolves the compute backend based on explicit choice or auto-detection.
/// Auto-detection order:
/// 1. `cuda` if NVIDIA driver / libcuda.so.1 is present.
/// 2. `hip` if AMD Instinct / libamdhip64.so.6 is present.
/// 3. `wgpu` otherwise (standard consumer GPU / Vulkan path).
pub fn resolve_backend(choice: BackendChoice) -> Result<Box<dyn ComputeBackend>, BackendError> {
    match choice {
        BackendChoice::Wgpu => {
            let b = WgpuBackend::new();
            if !b.is_available() {
                return Err(BackendError::LibraryNotFound(
                    "wgpu / Vulkan adapter unavailable".to_string(),
                ));
            }
            Ok(Box::new(b))
        }
        BackendChoice::Cuda => {
            let b = CudaBackend::new();
            if !b.is_available() {
                return Err(BackendError::LibraryNotFound(
                    "CUDA backend unavailable: libcuda.so.1 could not be loaded".to_string(),
                ));
            }
            Ok(Box::new(b))
        }
        BackendChoice::Hip => {
            let b = HipBackend::new();
            if !b.is_available() {
                return Err(BackendError::LibraryNotFound(
                    "HIP backend unavailable: libamdhip64.so.6 could not be loaded".to_string(),
                ));
            }
            Ok(Box::new(b))
        }
        BackendChoice::Auto => {
            // Check CUDA first
            let cuda = CudaBackend::new();
            if cuda.is_available()
                && cuda
                    .enumerate_device_pci_ids()
                    .map(|v| !v.is_empty())
                    .unwrap_or(false)
            {
                return Ok(Box::new(cuda));
            }

            // Check HIP next (for headless Instinct MI100 / MI50 without Vulkan)
            let hip = HipBackend::new();
            if hip.is_available()
                && hip
                    .enumerate_device_pci_ids()
                    .map(|v| !v.is_empty())
                    .unwrap_or(false)
            {
                return Ok(Box::new(hip));
            }

            // Fallback to WGPU (consumer GPUs, Vulkan ICDs)
            let wgpu = WgpuBackend::new();
            if wgpu.is_available() {
                return Ok(Box::new(wgpu));
            }

            Err(BackendError::DeviceInitFailed(
                "No usable compute backend available (CUDA, HIP, and WGPU all unavailable)"
                    .to_string(),
            ))
        }
    }
}
