use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use std::time::{SystemTime, UNIX_EPOCH};

pub const SCHEMA_VERSION: &str = "1.0.0";

/// Standardized application exit codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ExitCode {
    Ok = 0,
    WorkloadFailure = 1,
    InvalidArgs = 2,
    NoUsableGpu = 3,
    Cancelled = 4,
    InternalError = 5,
}

impl ExitCode {
    pub fn as_i32(self) -> i32 {
        self as i32
    }

    pub fn exit(self) -> ! {
        std::process::exit(self as i32);
    }
}

/// Output formatting mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    #[default]
    Human,
    Json,
    Jsonl,
}

impl OutputFormat {
    pub fn is_jsonl(&self) -> bool {
        matches!(self, OutputFormat::Jsonl)
    }

    pub fn is_json(&self) -> bool {
        matches!(self, OutputFormat::Json)
    }

    pub fn is_human(&self) -> bool {
        matches!(self, OutputFormat::Human)
    }
}

pub fn current_iso_timestamp() -> String {
    let now = SystemTime::now();
    let duration = now.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = duration.as_secs();
    let millis = duration.subsec_millis();

    // Approximate ISO 8601 formatting (UTC)
    let days = secs / 86400;
    let rem_secs = secs % 86400;
    let hours = rem_secs / 3600;
    let minutes = (rem_secs % 3600) / 60;
    let seconds = rem_secs % 60;

    // Convert days since UNIX_EPOCH (1970-01-01) to Year-Month-Day
    let (year, month, day) = days_to_ymd(days);

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        year, month, day, hours, minutes, seconds, millis
    )
}

fn days_to_ymd(days: u64) -> (u32, u32, u32) {
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as u32, m, d)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStartInfo {
    pub index: usize,
    pub name: String,
    pub backend: String,
    pub driver: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pci_bus_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartRecord {
    pub schema_version: String,
    #[serde(rename = "type")]
    pub record_type: String,
    pub timestamp: String,
    pub pid: u32,
    pub mode: String,
    pub preset: String,
    pub frames_per_scene: u32,
    pub total_scenes: u32,
    pub gpus: Vec<GpuStartInfo>,
    pub failsafe_enabled: bool,
}

impl StartRecord {
    pub fn new(
        mode: &str,
        preset: &str,
        frames_per_scene: u32,
        total_scenes: u32,
        gpus: Vec<GpuStartInfo>,
        failsafe_enabled: bool,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            record_type: "start".to_string(),
            timestamp: current_iso_timestamp(),
            pid: std::process::id(),
            mode: mode.to_string(),
            preset: preset.to_string(),
            frames_per_scene,
            total_scenes,
            gpus,
            failsafe_enabled,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressRecord {
    pub schema_version: String,
    #[serde(rename = "type")]
    pub record_type: String,
    pub timestamp: String,
    pub scene_index: u32,
    pub scene_name: String,
    pub frame: u32,
    pub total_scene_frames: u32,
    pub progress_pct: f32,
    pub fps: f32,
    pub gpu_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pci_bus_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_temp_c: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_power_w: Option<f32>,
}

impl ProgressRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scene_index: u32,
        scene_name: &str,
        frame: u32,
        total_scene_frames: u32,
        fps: f32,
        gpu_index: usize,
        gpu_temp_c: Option<f32>,
        gpu_power_w: Option<f32>,
    ) -> Self {
        let total = total_scene_frames.max(1) as f32;
        let progress_pct = ((frame as f32 / total) * 100.0).clamp(0.0, 100.0);
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            record_type: "progress".to_string(),
            timestamp: current_iso_timestamp(),
            scene_index,
            scene_name: scene_name.to_string(),
            frame,
            total_scene_frames,
            progress_pct,
            fps,
            gpu_index,
            pci_bus_id: None,
            gpu_temp_c,
            gpu_power_w,
        }
    }

    pub fn with_pci_bus_id(mut self, pci: impl Into<String>) -> Self {
        let p = pci.into();
        if !p.is_empty() {
            self.pci_bus_id = Some(p);
        }
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneResultRecord {
    pub scene_id: u32,
    pub name: String,
    pub fps: f64,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuResultRecord {
    pub schema_version: String,
    #[serde(rename = "type")]
    pub record_type: String,
    pub timestamp: String,
    pub gpu_index: usize,
    pub gpu_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pci_bus_id: Option<String>,
    pub score: f64,
    pub avg_fps: f64,
    pub one_percent_low_fps: f64,
    pub is_outlier: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outlier_reasons: Vec<String>,
    pub scenes: Vec<SceneResultRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_temp_c: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_power_w: Option<f32>,
}

impl GpuResultRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu_index: usize,
        gpu_name: &str,
        score: f64,
        avg_fps: f64,
        one_percent_low_fps: f64,
        scenes: Vec<SceneResultRecord>,
        peak_temp_c: Option<f32>,
        peak_power_w: Option<f32>,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            record_type: "gpu_result".to_string(),
            timestamp: current_iso_timestamp(),
            gpu_index,
            gpu_name: gpu_name.to_string(),
            pci_bus_id: None,
            score,
            avg_fps,
            one_percent_low_fps,
            is_outlier: false,
            outlier_reasons: Vec::new(),
            scenes,
            peak_temp_c,
            peak_power_w,
        }
    }

    pub fn with_pci_bus_id(mut self, pci: impl Into<String>) -> Self {
        let p = pci.into();
        if !p.is_empty() {
            self.pci_bus_id = Some(p);
        }
        self
    }

    pub fn with_outliers(mut self, is_outlier: bool, reasons: Vec<String>) -> Self {
        self.is_outlier = is_outlier;
        self.outlier_reasons = reasons;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryRecord {
    pub schema_version: String,
    #[serde(rename = "type")]
    pub record_type: String,
    pub timestamp: String,
    pub status: String,
    pub overall_score: f64,
    pub elapsed_seconds: f64,
    pub total_gpus: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outlier_gpus: Vec<usize>,
    pub exit_code: i32,
}

impl SummaryRecord {
    pub fn new(
        status: &str,
        overall_score: f64,
        elapsed_seconds: f64,
        total_gpus: usize,
        outlier_gpus: Vec<usize>,
        exit_code: i32,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            record_type: "summary".to_string(),
            timestamp: current_iso_timestamp(),
            status: status.to_string(),
            overall_score,
            elapsed_seconds,
            total_gpus,
            outlier_gpus,
            exit_code,
        }
    }

    pub fn single(status: &str, overall_score: f64, elapsed_seconds: f64, exit_code: i32) -> Self {
        Self::new(
            status,
            overall_score,
            elapsed_seconds,
            1,
            Vec::new(),
            exit_code,
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRecord {
    pub schema_version: String,
    #[serde(rename = "type")]
    pub record_type: String,
    pub timestamp: String,
    pub error_type: String,
    pub message: String,
    pub exit_code: i32,
}

impl ErrorRecord {
    pub fn new(error_type: &str, message: &str, exit_code: i32) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            record_type: "error".to_string(),
            timestamp: current_iso_timestamp(),
            error_type: error_type.to_string(),
            message: message.to_string(),
            exit_code,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelledRecord {
    pub schema_version: String,
    #[serde(rename = "type")]
    pub record_type: String,
    pub timestamp: String,
    pub reason: String,
    pub elapsed_seconds: f64,
    pub exit_code: i32,
}

impl CancelledRecord {
    pub fn new(reason: &str, elapsed_seconds: f64) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            record_type: "cancelled".to_string(),
            timestamp: current_iso_timestamp(),
            reason: reason.to_string(),
            elapsed_seconds,
            exit_code: ExitCode::Cancelled.as_i32(),
        }
    }
}

/// OutputEmitter manages structured logging to stdout based on selected OutputFormat.
#[derive(Debug, Clone)]
pub struct OutputEmitter {
    format: OutputFormat,
    headless: bool,
}

impl OutputEmitter {
    pub fn new(format: OutputFormat, headless: bool) -> Self {
        Self { format, headless }
    }

    pub fn format(&self) -> OutputFormat {
        self.format
    }

    pub fn is_headless(&self) -> bool {
        self.headless
    }

    pub fn emit_jsonl<T: Serialize>(&self, record: &T) {
        if self.format == OutputFormat::Jsonl
            && let Ok(json) = serde_json::to_string(record)
        {
            let mut out = io::stdout().lock();
            let _ = writeln!(out, "{}", json);
            let _ = out.flush();
        }
    }

    pub fn emit_error(&self, error_type: &str, message: &str, code: ExitCode) {
        match self.format {
            OutputFormat::Jsonl => {
                let rec = ErrorRecord::new(error_type, message, code.as_i32());
                self.emit_jsonl(&rec);
            }
            OutputFormat::Json => {
                let rec = ErrorRecord::new(error_type, message, code.as_i32());
                if let Ok(json) = serde_json::to_string_pretty(&rec) {
                    println!("{}", json);
                }
            }
            OutputFormat::Human => {
                eprintln!(
                    "\x1b[38;2;255;80;80;1mError [{}]: {}\x1b[0m",
                    error_type, message
                );
            }
        }
    }

    pub fn emit_cancelled(&self, reason: &str, elapsed_seconds: f64) {
        match self.format {
            OutputFormat::Jsonl => {
                let rec = CancelledRecord::new(reason, elapsed_seconds);
                self.emit_jsonl(&rec);
            }
            OutputFormat::Json => {
                let rec = CancelledRecord::new(reason, elapsed_seconds);
                if let Ok(json) = serde_json::to_string_pretty(&rec) {
                    println!("{}", json);
                }
            }
            OutputFormat::Human => {
                eprintln!(
                    "\n\x1b[38;2;255;180;0;1m⚠ Execution cancelled: {} (elapsed: {:.2}s)\x1b[0m",
                    reason, elapsed_seconds
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_serialization() {
        let start = StartRecord::new(
            "benchmark",
            "4k",
            100,
            8,
            vec![GpuStartInfo {
                index: 0,
                name: "Test GPU".to_string(),
                backend: "wgpu".to_string(),
                driver: "Vulkan 1.3".to_string(),
                pci_bus_id: Some("0000:03:00.0".to_string()),
            }],
            true,
        );
        let json = serde_json::to_string(&start).expect("Serialize start record");
        assert!(json.contains("\"type\":\"start\""));
        assert!(json.contains("\"schema_version\":\"1.0.0\""));

        let progress = ProgressRecord::new(0, "Waves", 50, 100, 120.5, 0, Some(65.0), Some(150.0));
        let json = serde_json::to_string(&progress).expect("Serialize progress record");
        assert!(json.contains("\"type\":\"progress\""));
        assert!(json.contains("\"progress_pct\":50.0"));

        let cancelled = CancelledRecord::new("SIGINT received", 4.25);
        let json = serde_json::to_string(&cancelled).expect("Serialize cancelled record");
        assert!(json.contains("\"type\":\"cancelled\""));
        assert!(json.contains("\"exit_code\":4"));
    }

    #[test]
    fn test_exit_codes() {
        assert_eq!(ExitCode::Ok.as_i32(), 0);
        assert_eq!(ExitCode::WorkloadFailure.as_i32(), 1);
        assert_eq!(ExitCode::InvalidArgs.as_i32(), 2);
        assert_eq!(ExitCode::NoUsableGpu.as_i32(), 3);
        assert_eq!(ExitCode::Cancelled.as_i32(), 4);
        assert_eq!(ExitCode::InternalError.as_i32(), 5);
    }
}
