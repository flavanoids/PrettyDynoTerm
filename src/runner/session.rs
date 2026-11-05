use crate::benchmark::{BenchmarkReport, ThermalTripReport};
use crate::gpu::{self, ComputePipelineManager};
use crate::telemetry::TelemetryCollector;
use crate::terminal::TerminalRenderer;

/// Bundles terminal session dependencies to keep runner functions decoupled,
/// ergonomic, and free of argument explosion.
pub struct TerminalSession<'a> {
    pub pipeline: &'a mut ComputePipelineManager,
    pub renderer: &'a mut TerminalRenderer,
    pub telemetry: &'a TelemetryCollector,
    pub adapter_info: &'a gpu::GpuAdapterInfo,
    pub term_width: usize,
    pub term_height: usize,
    pub res_tag: &'a str,
}

#[derive(Debug, Clone)]
pub enum BenchmarkPassResult {
    Completed(Box<BenchmarkReport>),
    AbortedByUser,
    ThermalTrip(ThermalTripReport),
}

#[derive(Debug, Clone)]
pub struct DemoOptions {
    pub initial_scene: u32,
    pub initial_uncapped: bool,
    pub max_temp_c: f32,
    pub failsafe_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct BenchmarkPassOptions {
    pub frames_per_scene: u32,
    pub iteration_info: Option<(usize, usize)>,
    pub max_temp_c: f32,
    pub failsafe_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct BenchmarkOptions {
    pub frames_per_scene: u32,
    pub output_path: String,
    pub max_temp_c: f32,
    pub failsafe_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct ConsistencyOptions {
    pub iterations: usize,
    pub cooldown_seconds: u64,
    pub target_temp_c: f32,
    pub max_temp_c: f32,
    pub failsafe_enabled: bool,
    pub frames_per_scene: u32,
    pub output_path: String,
}

pub struct CooldownOptions<'a> {
    pub completed_run: usize,
    pub total_runs: usize,
    pub cooldown_seconds: u64,
    pub target_temp_c: f32,
    pub last_report: &'a BenchmarkReport,
}
