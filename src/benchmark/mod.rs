use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneResult {
    pub scene_id: u32,
    pub scene_name: String,
    pub frames_rendered: u32,
    pub elapsed_seconds: f64,
    pub avg_fps: f64,
    pub avg_frametime_ms: f64,
    pub p1_low_frametime_ms: f64,
    pub score: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub timestamp: String,
    pub system_info: SystemReport,
    pub total_elapsed_seconds: f64,
    pub total_frames: u32,
    pub overall_score: u64,
    pub peak_gpu_temp_c: f32,
    #[serde(default)]
    pub peak_gpu_hotspot_c: f32,
    pub peak_gpu_power_w: f32,
    #[serde(default)]
    pub peak_gpu_clock_mhz: u32,
    pub peak_vram_mb: u64,
    pub scene_results: Vec<SceneResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemReport {
    pub gpu_name: String,
    pub gpu_backend: String,
    pub gpu_driver: String,
    pub cpu_name: String,
    pub cpu_threads: usize,
}

pub struct BenchmarkScorer {
    frametimes: Vec<Duration>,
    pub peak_temp: f32,
    pub peak_hotspot: f32,
    pub peak_power: f32,
    pub peak_clock: u32,
    pub peak_vram: u64,
}

impl BenchmarkScorer {
    pub fn new() -> Self {
        Self {
            frametimes: Vec::with_capacity(2048),
            peak_temp: 0.0,
            peak_hotspot: 0.0,
            peak_power: 0.0,
            peak_clock: 0,
            peak_vram: 0,
        }
    }

    pub fn record_frame(
        &mut self,
        dt: Duration,
        temp: f32,
        hotspot: f32,
        power: f32,
        vram: u64,
        clock: u32,
    ) {
        self.frametimes.push(dt);
        if temp > self.peak_temp {
            self.peak_temp = temp;
        }
        if hotspot > self.peak_hotspot {
            self.peak_hotspot = hotspot;
        }
        if power > self.peak_power {
            self.peak_power = power;
        }
        if vram > self.peak_vram {
            self.peak_vram = vram;
        }
        if clock > self.peak_clock {
            self.peak_clock = clock;
        }
    }

    pub fn current_scene_score(&self, scene_id: u32, pixels_per_frame: u64) -> u64 {
        let count = self.frametimes.len() as f64;
        if count == 0.0 {
            return 0;
        }
        let total_time_secs: f64 = self.frametimes.iter().map(|d| d.as_secs_f64()).sum();
        if total_time_secs <= 0.0 {
            return 0;
        }
        let weight = match scene_id {
            0 => 1.0,
            1 => 1.8,
            2 => 1.4,
            _ => 1.0,
        };
        let raw_workload = (count * pixels_per_frame as f64) * weight;
        ((raw_workload / total_time_secs) / 100.0) as u64
    }

    pub fn compute_scene_result(
        &mut self,
        scene_id: u32,
        scene_name: &str,
        pixels_per_frame: u64,
    ) -> SceneResult {
        let count = self.frametimes.len() as f64;
        if count == 0.0 {
            return SceneResult {
                scene_id,
                scene_name: scene_name.to_string(),
                frames_rendered: 0,
                elapsed_seconds: 0.0,
                avg_fps: 0.0,
                avg_frametime_ms: 0.0,
                p1_low_frametime_ms: 0.0,
                score: 0,
            };
        }

        let total_time_secs: f64 = self.frametimes.iter().map(|d| d.as_secs_f64()).sum();
        let avg_frametime_ms = (total_time_secs / count) * 1000.0;
        let avg_fps = count / total_time_secs;

        // Calculate 1% low (99th percentile frametime)
        let mut sorted_times: Vec<f64> = self
            .frametimes
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();
        sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p99_idx = ((count * 0.99) as usize).min(sorted_times.len() - 1);
        let p1_low_frametime_ms = sorted_times[p99_idx];

        // Scoring: Workload operations / time * scale factor
        // Mandelbulb and waves have different arithmetic weights
        let weight = match scene_id {
            0 => 1.0, // Waves: wave octaves + Fresnel
            1 => 1.8, // Mandelbulb: heavy 3D fractal ALU
            2 => 1.4, // Black hole: curved geodesics
            _ => 1.0,
        };

        let raw_workload = (count * pixels_per_frame as f64) * weight;
        let score = ((raw_workload / total_time_secs) / 100.0) as u64;

        // Clear frametimes for next scene
        self.frametimes.clear();

        SceneResult {
            scene_id,
            scene_name: scene_name.to_string(),
            frames_rendered: count as u32,
            elapsed_seconds: total_time_secs,
            avg_fps,
            avg_frametime_ms,
            p1_low_frametime_ms,
            score,
        }
    }

    pub fn save_report(report: &BenchmarkReport, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(report)?;
        fs::write(path, json)
    }
}
