use crate::telemetry::pci::PciBusId;
use serde::{Deserialize, Serialize};

/// High-fidelity telemetry metrics captured for a single physical GPU adapter,
/// keyed by its canonical PCI bus identifier.
///
/// Missing or unsupported sensors are represented as `None`, never as 0.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PerGpuTelemetry {
    pub pci_bus_id: PciBusId,
    pub gpu_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub driver_version: Option<String>,

    // Silicon Temperatures in °C (missing file = None)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_temp_c: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub junction_temp_c: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mem_temp_c: Option<f32>,

    // Hardware Critical Ceilings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_temp_crit_c: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub junction_temp_crit_c: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mem_temp_crit_c: Option<f32>,

    // Power in Watts
    #[serde(skip_serializing_if = "Option::is_none")]
    pub power_w: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub power_cap_w: Option<f32>,

    // Clocks in MHz
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clock_mhz: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mem_clock_mhz: Option<u32>,

    // Utilization (0 - 100%)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_util_pct: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mem_util_pct: Option<f32>,

    // VRAM Capacity in Megabytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_used_mb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_total_mb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_vendor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pcie_link: Option<String>,

    // Active Fan Cooling (Strictly None on passively cooled cards like H200/MI100)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fan_rpm: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fan_max_rpm: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fan_pwm: Option<u32>,

    // NVIDIA-specific / NVML telemetry
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvml_slowdown_temp_c: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvml_shutdown_temp_c: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvml_throttle_reasons: Option<u64>,
}

impl PerGpuTelemetry {
    /// True if board has physical fan tachometer or PWM sensor.
    pub fn has_fans(&self) -> bool {
        self.fan_rpm.is_some() || self.fan_pwm.is_some()
    }
}

/// Unified system and GPU snapshot struct used across HUD, failsafes, and reporting.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HardwareStats {
    pub pci_bus_id: Option<String>,
    pub gpu_name: String,
    pub gpu_clock_mhz: u32,
    pub gpu_mem_clock_mhz: u32,
    pub gpu_voltage_mv: u32,
    pub gpu_util_pct: f32,
    pub mem_util_pct: f32,
    pub gpu_temp_c: f32,
    pub gpu_hotspot_c: f32,
    pub gpu_mem_temp_c: f32,
    pub gpu_temp_crit_c: f32,
    pub gpu_hotspot_crit_c: f32,
    pub gpu_mem_temp_crit_c: f32,

    // Optional fan fields: None when card is passively cooled
    pub gpu_fan_rpm: Option<u32>,
    pub gpu_fan_max_rpm: Option<u32>,
    pub gpu_fan_pwm: Option<u32>,

    pub gpu_power_w: f32,
    pub gpu_power_cap_w: f32,
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    pub vram_vendor: String,
    pub pcie_link: String,
    pub cpu_name: String,
    pub cpu_usage_pct: f32,
    pub cpu_threads: usize,
    pub peak_gpu_clock_mhz: u32,
    pub peak_gpu_util_pct: f32,

    // NVIDIA / NVML
    pub nvml_slowdown_temp_c: Option<f32>,
    pub nvml_shutdown_temp_c: Option<f32>,
    pub nvml_throttle_reasons: Option<u64>,
}

impl HardwareStats {
    pub fn from_per_gpu(
        gpu: &PerGpuTelemetry,
        cpu_name: String,
        cpu_usage_pct: f32,
        cpu_threads: usize,
    ) -> Self {
        Self {
            pci_bus_id: Some(gpu.pci_bus_id.to_string()),
            gpu_name: gpu.gpu_name.clone(),
            gpu_clock_mhz: gpu.clock_mhz.unwrap_or(0),
            gpu_mem_clock_mhz: gpu.mem_clock_mhz.unwrap_or(0),
            gpu_voltage_mv: 0,
            gpu_util_pct: gpu.gpu_util_pct.unwrap_or(0.0),
            mem_util_pct: gpu.mem_util_pct.unwrap_or(0.0),
            gpu_temp_c: gpu.edge_temp_c.unwrap_or(0.0),
            gpu_hotspot_c: gpu.junction_temp_c.unwrap_or(0.0),
            gpu_mem_temp_c: gpu.mem_temp_c.unwrap_or(0.0),
            gpu_temp_crit_c: gpu.edge_temp_crit_c.unwrap_or(0.0),
            gpu_hotspot_crit_c: gpu.junction_temp_crit_c.unwrap_or(0.0),
            gpu_mem_temp_crit_c: gpu.mem_temp_crit_c.unwrap_or(0.0),
            gpu_fan_rpm: gpu.fan_rpm,
            gpu_fan_max_rpm: gpu.fan_max_rpm,
            gpu_fan_pwm: gpu.fan_pwm,
            gpu_power_w: gpu.power_w.unwrap_or(0.0),
            gpu_power_cap_w: gpu.power_cap_w.unwrap_or(0.0),
            vram_used_mb: gpu.vram_used_mb.unwrap_or(0),
            vram_total_mb: gpu.vram_total_mb.unwrap_or(0),
            vram_vendor: gpu.vram_vendor.clone().unwrap_or_default(),
            pcie_link: gpu.pcie_link.clone().unwrap_or_default(),
            cpu_name,
            cpu_usage_pct,
            cpu_threads,
            peak_gpu_clock_mhz: gpu.clock_mhz.unwrap_or(0),
            peak_gpu_util_pct: gpu.gpu_util_pct.unwrap_or(0.0),
            nvml_slowdown_temp_c: gpu.nvml_slowdown_temp_c,
            nvml_shutdown_temp_c: gpu.nvml_shutdown_temp_c,
            nvml_throttle_reasons: gpu.nvml_throttle_reasons,
        }
    }
}
