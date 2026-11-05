use std::sync::Arc;
use wgpu::*;

#[derive(Debug, Clone)]
pub struct GpuAdapterInfo {
    pub name: String,
    pub backend: Backend,
    pub device_type: DeviceType,
    pub driver: String,
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
    pub async fn new(preferred_adapter_index: Option<usize>) -> Result<Self, String> {
        let instance = Instance::default();

        let adapters: Vec<Adapter> = instance.enumerate_adapters(Backends::PRIMARY).await;
        if adapters.is_empty() {
            return Err("No compatible GPU adapter found (Vulkan/Metal/DX12)".to_string());
        }

        let selected_index = if let Some(idx) = preferred_adapter_index {
            if idx >= adapters.len() {
                return Err(format!(
                    "Adapter index {} out of bounds (found {})",
                    idx,
                    adapters.len()
                ));
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

        let adapter_info = GpuAdapterInfo {
            name: info.name.clone(),
            backend: info.backend,
            device_type: info.device_type,
            driver: info.driver.clone(),
        };

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("Failed to request GPU device: {:?}", e))?;

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
            .map(|a| {
                let info = a.get_info();
                GpuAdapterInfo {
                    name: info.name,
                    backend: info.backend,
                    device_type: info.device_type,
                    driver: info.driver,
                }
            })
            .collect()
    }
}
