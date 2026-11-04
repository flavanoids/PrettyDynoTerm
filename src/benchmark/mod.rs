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
    #[serde(default)]
    pub peak_gpu_util_pct: f32,
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
    pub peak_util: f32,
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
            peak_util: 0.0,
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
        util: f32,
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
        if util > self.peak_util {
            self.peak_util = util;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_scorer_peaks() {
        let mut scorer = BenchmarkScorer::new();
        scorer.record_frame(Duration::from_millis(16), 55.0, 65.0, 70.0, 100.0, 2048, 2100);
        scorer.record_frame(Duration::from_millis(14), 62.0, 72.0, 99.0, 130.0, 4096, 2450);
        scorer.record_frame(Duration::from_millis(18), 60.0, 70.0, 85.0, 115.0, 3072, 2300);

        assert_eq!(scorer.peak_temp, 62.0);
        assert_eq!(scorer.peak_hotspot, 72.0);
        assert_eq!(scorer.peak_util, 99.0);
        assert_eq!(scorer.peak_power, 130.0);
        assert_eq!(scorer.peak_vram, 4096);
        assert_eq!(scorer.peak_clock, 2450);
    }

    #[test]
    fn test_benchmark_scorer_empty() {
        let mut scorer = BenchmarkScorer::new();
        let res = scorer.compute_scene_result(0, "Empty", 1920 * 1080);
        assert_eq!(res.frames_rendered, 0);
        assert_eq!(res.avg_fps, 0.0);
        assert_eq!(res.score, 0);
        assert_eq!(scorer.current_scene_score(0, 1920 * 1080), 0);
    }

    #[test]
    fn test_benchmark_scorer_calc_and_1percent_low() {
        let mut scorer = BenchmarkScorer::new();
        // Record 100 frames of 10ms, and 1 frame of 50ms spike
        for _ in 0..99 {
            scorer.record_frame(Duration::from_millis(10), 50.0, 60.0, 90.0, 100.0, 2000, 2000);
        }
        scorer.record_frame(Duration::from_millis(50), 50.0, 60.0, 90.0, 100.0, 2000, 2000);

        let res = scorer.compute_scene_result(0, "Test Scene", 1000);
        assert_eq!(res.frames_rendered, 100);
        assert!(res.avg_fps > 0.0);
        assert!(res.avg_frametime_ms >= 10.0);
        // The 99th percentile frametime should capture the slower frame (50ms)
        assert_eq!(res.p1_low_frametime_ms, 50.0);
    }

    #[test]
    fn test_benchmark_scene_weighting() {
        let mut scorer_waves = BenchmarkScorer::new();
        let mut scorer_mandel = BenchmarkScorer::new();
        let mut scorer_bh = BenchmarkScorer::new();

        for _ in 0..10 {
            scorer_waves.record_frame(Duration::from_millis(10), 0.0, 0.0, 0.0, 0.0, 0, 0);
            scorer_mandel.record_frame(Duration::from_millis(10), 0.0, 0.0, 0.0, 0.0, 0, 0);
            scorer_bh.record_frame(Duration::from_millis(10), 0.0, 0.0, 0.0, 0.0, 0, 0);
        }

        let res_waves = scorer_waves.compute_scene_result(0, "Waves", 1000);
        let res_mandel = scorer_mandel.compute_scene_result(1, "Mandelbulb", 1000);
        let res_bh = scorer_bh.compute_scene_result(2, "Black Hole", 1000);

        assert!(res_mandel.score > res_bh.score);
        assert!(res_bh.score > res_waves.score);
    }

    #[test]
    fn test_benchmark_report_json_roundtrip() {
        let report = BenchmarkReport {
            timestamp: "2026-11-04 10:00:00 UTC".to_string(),
            system_info: SystemReport {
                gpu_name: "Test GPU".to_string(),
                gpu_backend: "Vulkan".to_string(),
                gpu_driver: "radv".to_string(),
                cpu_name: "Test CPU".to_string(),
                cpu_threads: 16,
            },
            total_elapsed_seconds: 12.5,
            total_frames: 600,
            overall_score: 95400,
            peak_gpu_temp_c: 68.0,
            peak_gpu_hotspot_c: 78.0,
            peak_gpu_util_pct: 99.5,
            peak_gpu_power_w: 145.0,
            peak_gpu_clock_mhz: 2550,
            peak_vram_mb: 4800,
            scene_results: vec![SceneResult {
                scene_id: 0,
                scene_name: "Waves".to_string(),
                frames_rendered: 100,
                elapsed_seconds: 1.66,
                avg_fps: 60.2,
                avg_frametime_ms: 16.6,
                p1_low_frametime_ms: 18.2,
                score: 15900,
            }],
        };

        let json = serde_json::to_string_pretty(&report).expect("Serialization failed");
        let deserialized: BenchmarkReport =
            serde_json::from_str(&json).expect("Deserialization failed");

        assert_eq!(deserialized.overall_score, 95400);
        assert_eq!(deserialized.system_info.gpu_name, "Test GPU");
        assert_eq!(deserialized.scene_results.len(), 1);
        assert_eq!(deserialized.scene_results[0].scene_name, "Waves");
    }

    #[test]
    fn test_save_report_file() {
        let report = BenchmarkReport {
            timestamp: "2026-11-04".to_string(),
            system_info: SystemReport {
                gpu_name: "Mock GPU".to_string(),
                gpu_backend: "Vulkan".to_string(),
                gpu_driver: "test".to_string(),
                cpu_name: "Mock CPU".to_string(),
                cpu_threads: 4,
            },
            total_elapsed_seconds: 1.0,
            total_frames: 60,
            overall_score: 1000,
            peak_gpu_temp_c: 50.0,
            peak_gpu_hotspot_c: 55.0,
            peak_gpu_util_pct: 80.0,
            peak_gpu_power_w: 75.0,
            peak_gpu_clock_mhz: 1800,
            peak_vram_mb: 1024,
            scene_results: vec![],
        };

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("dyno_test_report.json");
        let path_str = path.to_str().unwrap();

        BenchmarkScorer::save_report(&report, path_str).expect("Failed to write report");
        let content = fs::read_to_string(path_str).expect("Failed to read report");
        assert!(content.contains("Mock GPU"));
        let _ = fs::remove_file(path);
    }
}
