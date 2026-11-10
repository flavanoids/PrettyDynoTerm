use std::sync::Arc;
use wgpu::*;

use crate::telemetry::PciBusId;

#[derive(Debug, Clone)]
pub struct GpuAdapterInfo {
    pub name: String,
    pub backend: Backend,
    pub device_type: DeviceType,
    pub driver: String,
    pub pci_bus_id: PciBusId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuInitError {
    NoAdapterFound(String),
    AdapterIndexOutOfBounds { index: usize, found: usize },
    DeviceRequestFailed(String),
}

impl std::fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapterFound(msg) => write!(f, "No compatible GPU adapter found: {}", msg),
            Self::AdapterIndexOutOfBounds { index, found } => {
                write!(
                    f,
                    "GPU adapter index {} out of bounds (found {} adapters)",
                    index, found
                )
            }
            Self::DeviceRequestFailed(msg) => write!(f, "Failed to request GPU device: {}", msg),
        }
    }
}

impl std::error::Error for GpuInitError {}

fn resolve_pci_bus_id(info: &AdapterInfo, adapter_index: usize) -> PciBusId {
    if !info.device_pci_bus_id.is_empty() {
        return PciBusId::new(&info.device_pci_bus_id);
    }
    let cards = crate::telemetry::AmdSysfsTelemetry::default().query_all_cards();
    if adapter_index < cards.len() {
        return cards[adapter_index].pci_bus_id.clone();
    }
    PciBusId::default()
}

#[allow(dead_code)]
pub struct GpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub adapter_info: GpuAdapterInfo,
}

impl GpuContext {
    pub async fn new(preferred_adapter_index: Option<usize>) -> Result<Self, GpuInitError> {
        let instance = Instance::default();

        let adapters: Vec<Adapter> = instance.enumerate_adapters(Backends::PRIMARY).await;
        if adapters.is_empty() {
            return Err(GpuInitError::NoAdapterFound(
                "No Vulkan/Metal/DX12 primary adapter available on system".to_string(),
            ));
        }

        let selected_index = if let Some(idx) = preferred_adapter_index {
            if idx >= adapters.len() {
                return Err(GpuInitError::AdapterIndexOutOfBounds {
                    index: idx,
                    found: adapters.len(),
                });
            }
            idx
        } else {
            // Find first Discrete GPU, else first Integrated, else fallback to 0
            adapters
                .iter()
                .position(|a| a.get_info().device_type == DeviceType::DiscreteGpu)
                .or_else(|| {
                    adapters
                        .iter()
                        .position(|a| a.get_info().device_type == DeviceType::IntegratedGpu)
                })
                .unwrap_or(0)
        };

        let adapter = adapters.into_iter().nth(selected_index).unwrap();
        let info = adapter.get_info();
        let pci_bus_id = resolve_pci_bus_id(&info, selected_index);

        let adapter_info = GpuAdapterInfo {
            name: info.name.clone(),
            backend: info.backend,
            device_type: info.device_type,
            driver: info.driver.clone(),
            pci_bus_id,
        };

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| GpuInitError::DeviceRequestFailed(format!("{:?}", e)))?;

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        })
    }

    pub async fn new_for_adapter_index(index: usize) -> Result<Self, GpuInitError> {
        let instance = Instance::default();

        let adapters: Vec<Adapter> = instance.enumerate_adapters(Backends::PRIMARY).await;
        if adapters.is_empty() {
            return Err(GpuInitError::NoAdapterFound(
                "No Vulkan/Metal/DX12 primary adapter available on system".to_string(),
            ));
        }

        if index >= adapters.len() {
            return Err(GpuInitError::AdapterIndexOutOfBounds {
                index,
                found: adapters.len(),
            });
        }

        let adapter = adapters.into_iter().nth(index).unwrap();
        let info = adapter.get_info();
        let pci_bus_id = resolve_pci_bus_id(&info, index);

        let adapter_info = GpuAdapterInfo {
            name: info.name.clone(),
            backend: info.backend,
            device_type: info.device_type,
            driver: info.driver.clone(),
            pci_bus_id,
        };

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| GpuInitError::DeviceRequestFailed(format!("{:?}", e)))?;

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        })
    }

    pub fn list_adapters() -> Vec<GpuAdapterInfo> {
        let instance = Instance::default();

        pollster::block_on(instance.enumerate_adapters(Backends::PRIMARY))
            .into_iter()
            .enumerate()
            .map(|(idx, a)| {
                let info = a.get_info();
                let pci_bus_id = resolve_pci_bus_id(&info, idx);
                GpuAdapterInfo {
                    name: info.name,
                    backend: info.backend,
                    device_type: info.device_type,
                    driver: info.driver,
                    pci_bus_id,
                }
            })
            .collect()
    }
}
