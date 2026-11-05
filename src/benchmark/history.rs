use super::BenchmarkReport;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

pub const DEFAULT_HISTORY_FILE: &str = "dyno_benchmark_history.jsonl";
pub const DEFAULT_BASELINE_FILE: &str = "dyno_baseline.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneComparison {
    pub scene_id: u32,
    pub scene_name: String,
    pub current_score: f64,
    pub baseline_score: f64,
    pub score_delta_pct: f64,
    pub current_fps: f64,
    pub baseline_fps: f64,
    pub fps_delta_pct: f64,
    pub current_frametime_ms: f64,
    pub baseline_frametime_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub current_timestamp: String,
    pub baseline_timestamp: String,
    pub current_score: f64,
    pub baseline_score: f64,
    pub score_delta_pts: f64,
    pub score_delta_pct: f64,
    pub current_clock_mhz: u32,
    pub baseline_clock_mhz: u32,
    pub clock_delta_pct: f64,
    pub current_temp_c: f32,
    pub baseline_temp_c: f32,
    pub temp_delta_c: f32,
    pub current_power_w: f32,
    pub baseline_power_w: f32,
    pub power_delta_w: f32,
    pub scene_comparisons: Vec<SceneComparison>,
    pub throttle_detected: bool,
    pub throttle_notes: Vec<String>,
}

/// Appends a benchmark report to the persistent JSON-lines history file.
pub fn append_to_history(report: &BenchmarkReport, path: &str) -> std::io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    let line = serde_json::to_string(report)?;
    writeln!(file, "{}", line)?;
    Ok(())
}

/// Loads all benchmark reports from the JSON-lines history file.
pub fn load_history(path: &str) -> std::io::Result<Vec<BenchmarkReport>> {
    if !Path::new(path).exists() {
        return Ok(Vec::new());
    }

    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut reports = Vec::new();

    for line_res in reader.lines() {
        let line = line_res?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(report) = serde_json::from_str::<BenchmarkReport>(trimmed) {
            reports.push(report);
        }
    }

    Ok(reports)
}

/// Saves a benchmark report as the reference baseline JSON.
pub fn save_baseline(report: &BenchmarkReport, path: &str) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    fs::write(path, json)?;
    Ok(())
}

/// Loads the reference baseline JSON report if it exists.
pub fn load_baseline(path: &str) -> Option<BenchmarkReport> {
    if !Path::new(path).exists() {
        return None;
    }
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Compares a current benchmark report against a baseline report.
pub fn compare_reports(
    current: &BenchmarkReport,
    baseline: &BenchmarkReport,
) -> BenchmarkComparison {
    let score_delta_pts = current.overall_score - baseline.overall_score;
    let score_delta_pct = if baseline.overall_score > 0.0 {
        (score_delta_pts / baseline.overall_score) * 100.0
    } else {
        0.0
    };

    let clock_delta_pct = if baseline.peak_gpu_clock_mhz > 0 {
        ((current.peak_gpu_clock_mhz as f64 - baseline.peak_gpu_clock_mhz as f64)
            / baseline.peak_gpu_clock_mhz as f64)
            * 100.0
    } else {
        0.0
    };

    let temp_delta_c = current.peak_gpu_temp_c - baseline.peak_gpu_temp_c;
    let power_delta_w = current.peak_gpu_power_w - baseline.peak_gpu_power_w;

    let mut scene_comparisons = Vec::new();
    for cur_scene in &current.scene_results {
        let base_scene_opt = baseline
            .scene_results
            .iter()
            .find(|s| s.scene_id == cur_scene.scene_id);

        let (base_score, base_fps, base_ft) = if let Some(b) = base_scene_opt {
            (b.score, b.avg_fps, b.avg_frametime_ms)
        } else {
            (0.0, 0.0, 0.0)
        };

        let sc_delta_pct = if base_score > 0.0 {
            ((cur_scene.score - base_score) / base_score) * 100.0
        } else {
            0.0
        };

        let fps_delta_pct = if base_fps > 0.0 {
            ((cur_scene.avg_fps - base_fps) / base_fps) * 100.0
        } else {
            0.0
        };

        scene_comparisons.push(SceneComparison {
            scene_id: cur_scene.scene_id,
            scene_name: cur_scene.scene_name.clone(),
            current_score: cur_scene.score,
            baseline_score: base_score,
            score_delta_pct: sc_delta_pct,
            current_fps: cur_scene.avg_fps,
            baseline_fps: base_fps,
            fps_delta_pct,
            current_frametime_ms: cur_scene.avg_frametime_ms,
            baseline_frametime_ms: base_ft,
        });
    }

    // Thermal throttling detection heuristic:
    // If GPU frequency dropped by > 4.5% while peak temperature exceeded 78°C
    let mut throttle_detected = false;
    let mut throttle_notes = Vec::new();

    if clock_delta_pct < -4.5 && current.peak_gpu_temp_c >= 78.0 {
        throttle_detected = true;
        throttle_notes.push(format!(
            "Thermal throttling detected: Core clock dropped by {:.1}% while silicon reached {:.1}°C",
            clock_delta_pct.abs(),
            current.peak_gpu_temp_c
        ));
    }

    if score_delta_pct < -5.0 && current.peak_gpu_temp_c > baseline.peak_gpu_temp_c + 5.0 {
        throttle_detected = true;
        throttle_notes.push(format!(
            "Thermal performance degradation: Overall score dropped by {:.1}% with +{:.1}°C thermal increase",
            score_delta_pct.abs(),
            temp_delta_c
        ));
    }

    BenchmarkComparison {
        current_timestamp: current.timestamp.clone(),
        baseline_timestamp: baseline.timestamp.clone(),
        current_score: current.overall_score,
        baseline_score: baseline.overall_score,
        score_delta_pts,
        score_delta_pct,
        current_clock_mhz: current.peak_gpu_clock_mhz,
        baseline_clock_mhz: baseline.peak_gpu_clock_mhz,
        clock_delta_pct,
        current_temp_c: current.peak_gpu_temp_c,
        baseline_temp_c: baseline.peak_gpu_temp_c,
        temp_delta_c,
        current_power_w: current.peak_gpu_power_w,
        baseline_power_w: baseline.peak_gpu_power_w,
        power_delta_w,
        scene_comparisons,
        throttle_detected,
        throttle_notes,
    }
}

/// Renders a formatted ANSI terminal comparison table.
pub fn format_comparison_table(comp: &BenchmarkComparison) -> String {
    let mut out = String::with_capacity(4096);

    out.push_str("\x1b[38;2;0;255;200;1m┌──────────────────────────────────────────────────────────────────────────────┐\r\n");
    out.push_str(
        "│               DYNO_TERM BENCHMARK PERFORMANCE COMPARISON                     │\r\n",
    );
    out.push_str("├──────────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

    let cur_time = if comp.current_timestamp.len() > 19 {
        &comp.current_timestamp[..19]
    } else {
        &comp.current_timestamp
    };
    let base_time = if comp.baseline_timestamp.len() > 19 {
        &comp.baseline_timestamp[..19]
    } else {
        &comp.baseline_timestamp
    };

    out.push_str(&format!(
        "  \x1b[38;2;120;140;160mCurrent Run:\x1b[0m  \x1b[1m{:<22}\x1b[0m \x1b[38;2;120;140;160mBaseline:\x1b[0m \x1b[1m{}\x1b[0m\r\n",
        cur_time, base_time
    ));
    out.push_str("\x1b[38;2;0;255;200;1m├──────────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");
    out.push_str("  \x1b[38;2;0;220;255;1mSCENE / METRIC              CURRENT RUN        BASELINE REFERENCE     DELTA   \x1b[0m\r\n");
    out.push_str("\x1b[38;2;0;255;200;1m├──────────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

    // Total Score
    let delta_col = format_delta_badge(comp.score_delta_pct, true);
    out.push_str(&format!(
        "  \x1b[1m{:<26}\x1b[0m \x1b[38;2;0;255;200;1m{:<18}\x1b[0m {:<22} {}\r\n",
        "OVERALL DYNO SCORE",
        format!("{:.2} pts", comp.current_score),
        format!("{:.2} pts", comp.baseline_score),
        delta_col
    ));

    out.push_str("\x1b[38;2;60;70;90m  ────────────────────────────────────────────────────────────────────────────\x1b[0m\r\n");

    // Per Scene Results
    for sc in &comp.scene_comparisons {
        let sc_name_short = if sc.scene_name.len() > 24 {
            &sc.scene_name[..24]
        } else {
            &sc.scene_name
        };

        let cur_fps_str = format!("{:.1} FPS", sc.current_fps);
        let base_fps_str = format!("{:.1} FPS", sc.baseline_fps);
        let fps_delta = format_delta_badge(sc.fps_delta_pct, true);

        out.push_str(&format!(
            "  {:<26} {:<18} {:<22} {}\r\n",
            sc_name_short, cur_fps_str, base_fps_str, fps_delta
        ));
    }

    out.push_str("\x1b[38;2;0;255;200;1m├──────────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");
    out.push_str("  \x1b[38;2;0;220;255;1mHARDWARE TELEMETRY DELTAS                                                    \x1b[0m\r\n");
    out.push_str("\x1b[38;2;0;255;200;1m├──────────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

    // Clocks
    let clock_delta = format_delta_badge(comp.clock_delta_pct, true);
    out.push_str(&format!(
        "  {:<26} {:<18} {:<22} {}\r\n",
        "Peak GPU Clock",
        format!("{} MHz", comp.current_clock_mhz),
        format!("{} MHz", comp.baseline_clock_mhz),
        clock_delta
    ));

    // Temp
    let temp_delta = format_temp_delta_badge(comp.temp_delta_c);
    out.push_str(&format!(
        "  {:<26} {:<18} {:<22} {}\r\n",
        "Peak GPU Temperature",
        format!("{:.1}°C", comp.current_temp_c),
        format!("{:.1}°C", comp.baseline_temp_c),
        temp_delta
    ));

    // Power
    let power_delta = format!("{:+5.1} W", comp.power_delta_w);
    out.push_str(&format!(
        "  {:<26} {:<18} {:<22} \x1b[38;2;180;190;210m{}\x1b[0m\r\n",
        "Peak Board Power",
        format!("{:.1} W", comp.current_power_w),
        format!("{:.1} W", comp.baseline_power_w),
        power_delta
    ));

    if comp.throttle_detected {
        out.push_str("\x1b[38;2;0;255;200;1m├──────────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");
        out.push_str("  \x1b[38;2;255;80;80;1m⚠ THERMAL REGRESSION WARNING:\x1b[0m\r\n");
        for note in &comp.throttle_notes {
            out.push_str(&format!("  \x1b[38;2;255;120;120m• {}\x1b[0m\r\n", note));
        }
    }

    out.push_str("\x1b[38;2;0;255;200;1m└──────────────────────────────────────────────────────────────────────────────┘\x1b[0m\r\n");

    out
}

/// Renders a tabular overview of all historical benchmark runs.
pub fn format_history_table(
    history: &[BenchmarkReport],
    baseline_timestamp: Option<&str>,
) -> String {
    let mut out = String::with_capacity(4096);

    out.push_str("\x1b[38;2;0;255;200;1m================================================================================\r\n");
    out.push_str(
        "                       DYNO_TERM BENCHMARK HISTORY                              \r\n",
    );
    out.push_str("================================================================================\x1b[0m\r\n");

    if history.is_empty() {
        out.push_str("  No historical benchmark runs found. Run a benchmark with --benchmark to record one.\r\n");
        out.push_str("\x1b[38;2;0;255;200;1m================================================================================\x1b[0m\r\n");
        return out;
    }

    out.push_str("  \x1b[38;2;0;220;255;1m#  TIMESTAMP            OVERALL SCORE   GPU ADAPTER              TAG        \x1b[0m\r\n");
    out.push_str("\x1b[38;2;60;70;90m  ────────────────────────────────────────────────────────────────────────────\x1b[0m\r\n");

    for (i, r) in history.iter().enumerate() {
        let is_base = baseline_timestamp
            .map(|bt| bt == r.timestamp)
            .unwrap_or(false);
        let tag = if is_base {
            "\x1b[38;2;0;255;128;1m[BASELINE]\x1b[0m"
        } else {
            ""
        };

        let short_time = if r.timestamp.len() > 19 {
            &r.timestamp[..19]
        } else {
            &r.timestamp
        };

        let gpu_short = if r.system_info.gpu_name.len() > 24 {
            &r.system_info.gpu_name[..24]
        } else {
            &r.system_info.gpu_name
        };

        out.push_str(&format!(
            "  {:<2} {:<20} \x1b[1m{:<15}\x1b[0m {:<24} {}\r\n",
            i + 1,
            short_time,
            format!("{:.2} pts", r.overall_score),
            gpu_short,
            tag
        ));
    }

    out.push_str("\x1b[38;2;0;255;200;1m================================================================================\x1b[0m\r\n");
    out
}

fn format_delta_badge(delta_pct: f64, higher_is_better: bool) -> String {
    if delta_pct.abs() < 0.1 {
        format!("\x1b[38;2;160;160;170m{:>6.1}%  ─\x1b[0m", delta_pct)
    } else if (higher_is_better && delta_pct > 0.0) || (!higher_is_better && delta_pct < 0.0) {
        format!("\x1b[38;2;0;255;128;1m{:+6.1}% ▲\x1b[0m", delta_pct)
    } else {
        format!("\x1b[38;2;255;80;80;1m{:+6.1}% ▼\x1b[0m", delta_pct)
    }
}

fn format_temp_delta_badge(delta_c: f32) -> String {
    if delta_c.abs() < 0.5 {
        format!("\x1b[38;2;160;160;170m{:>6.1}°C ─\x1b[0m", delta_c)
    } else if delta_c < 0.0 {
        // Cooler is better
        format!("\x1b[38;2;0;255;128;1m{:+6.1}°C ▼\x1b[0m", delta_c)
    } else {
        // Hotter
        format!("\x1b[38;2;255;140;60;1m{:+6.1}°C ▲\x1b[0m", delta_c)
    }
}
