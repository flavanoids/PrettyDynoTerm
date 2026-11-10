use std::sync::Arc;

use crate::backend::traits::{BackendError, BackendType, ComputeBackend, ComputeDevice};
use crate::gpu::GpuContext;
use crate::gpu::pipeline::{ComputePipelineManager, Uniforms};
use crate::shaders::UNIFIED_SHADER;
use crate::telemetry::PciBusId;

pub struct WgpuComputeDevice {
    index: usize,
    name: String,
    pci_bus_id: PciBusId,
    pipeline: ComputePipelineManager,
}

impl ComputeDevice for WgpuComputeDevice {
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
        BackendType::Wgpu
    }

    fn allocate_storage_buffer(&mut self, width: u32, height: u32) -> Result<(), BackendError> {
        self.pipeline
            .try_resize(width, height)
            .map_err(BackendError::AllocationFailed)
    }

    fn dispatch(&mut self, uniforms: &Uniforms) -> Result<(), BackendError> {
        self.pipeline.render_compute_sync(uniforms);
        Ok(())
    }

    fn readback_checksum(&mut self) -> Result<u64, BackendError> {
        // Uniforms with benchmark_mode for checksum
        let dummy_uniforms = Uniforms {
            time: 0.0,
            frame: 0,
            width: self.pipeline.width,
            height: self.pipeline.height,
            cam_pos: [0.0, 1.8, -4.5],
            scene_id: 0,
            cam_dir: [0.0, -0.2, 1.0],
            benchmark_mode: 1,
        };
        Ok(self
            .pipeline
            .render_compute_sync_with_checksum(&dummy_uniforms))
    }

    fn synchronize(&self) -> Result<(), BackendError> {
        // render_compute_sync in wgpu already synchronizes with on_submitted_work_done
        Ok(())
    }
}

impl WgpuComputeDevice {
    pub fn new(index: usize, width: u32, height: u32) -> Result<Self, BackendError> {
        let ctx = pollster::block_on(GpuContext::new_for_adapter_index(index))
            .map_err(|e| BackendError::DeviceInitFailed(e.to_string()))?;

        let name = ctx.adapter_info.name.clone();
        let pci_bus_id = ctx.adapter_info.pci_bus_id.clone();

        let pipeline = ComputePipelineManager::new(
            Arc::clone(&ctx.device),
            Arc::clone(&ctx.queue),
            UNIFIED_SHADER,
            width,
            height,
        )
        .map_err(BackendError::AllocationFailed)?;

        Ok(Self {
            index,
            name,
            pci_bus_id,
            pipeline,
        })
    }
}

pub struct WgpuBackend;

impl Default for WgpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl WgpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl ComputeBackend for WgpuBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Wgpu
    }

    fn is_available(&self) -> bool {
        !GpuContext::list_adapters().is_empty()
    }

    fn enumerate_device_pci_ids(&self) -> Result<Vec<PciBusId>, BackendError> {
        let adapters = GpuContext::list_adapters();
        if adapters.is_empty() {
            return Err(BackendError::DeviceInitFailed(
                "No Vulkan/Metal/DX12 primary adapters found".to_string(),
            ));
        }
        Ok(adapters.into_iter().map(|a| a.pci_bus_id).collect())
    }

    fn create_device(
        &self,
        index: usize,
        width: u32,
        height: u32,
    ) -> Result<Box<dyn ComputeDevice>, BackendError> {
        let dev = WgpuComputeDevice::new(index, width, height)?;
        Ok(Box::new(dev))
    }
}
