use crate::telemetry::HardwareStats;
use serde::{Deserialize, Serialize};

/// Statistical metric for a continuous hardware sensor over the benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorMetric {
    pub min: f32,
    pub max: f32,
    pub avg: f32,
    pub p95: f32,
}

impl SensorMetric {
    pub fn from_samples(samples: &[f32]) -> Self {
        if samples.is_empty() {
            return Self {
                min: 0.0,
                max: 0.0,
                avg: 0.0,
                p95: 0.0,
            };
        }
        let min = samples.iter().copied().fold(f32::INFINITY, f32::min);
        let max = samples.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = samples.iter().sum();
        let avg = sum / samples.len() as f32;

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p95_idx = ((sorted.len() as f32 * 0.95) as usize).min(sorted.len() - 1);
        let p95 = sorted[p95_idx];

        Self { min, max, avg, p95 }
    }
}

/// Instantaneous telemetry sample captured at a periodic interval during testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub elapsed_sec: f32,
    pub act_id: u32,
    pub fps: f32,
    pub frametime_ms: f32,
    pub core_clock_mhz: u32,
    pub mem_clock_mhz: u32,
    pub core_voltage_mv: u32,
    pub gpu_util_pct: f32,
    pub mem_util_pct: f32,
    pub edge_temp_c: f32,
    pub hotspot_temp_c: f32,
    pub mem_temp_c: f32,
    pub fan_rpm: u32,
    pub fan_pwm_pct: f32,
    pub board_power_w: f32,
    pub vram_used_mb: u64,
}

/// Advisory assessment threshold table per hardware vendor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VendorThresholds {
    /// Maximum expected die delta (°C) between junction/hotspot and edge
    pub max_die_delta_c: f32,
    /// Critical thermal headroom safety floor (°C)
    pub min_thermal_headroom_c: f32,
    /// Sustained clock drop ratio (p95 clock < max * ratio triggers throttling flag)
    pub clock_drop_ratio: f32,
    /// Power cap saturation ratio (power >= cap * ratio indicates power limit pegged)
    pub power_cap_ratio: f32,
}

impl VendorThresholds {
    pub const AMD: Self = Self {
        max_die_delta_c: 28.0,
        min_thermal_headroom_c: 5.0,
        clock_drop_ratio: 0.78,
        power_cap_ratio: 0.98,
    };

    pub const NVIDIA: Self = Self {
        max_die_delta_c: 25.0,
        min_thermal_headroom_c: 5.0,
        clock_drop_ratio: 0.80,
        power_cap_ratio: 0.98,
    };

    pub fn for_vendor(is_nvidia: bool) -> Self {
        if is_nvidia { Self::NVIDIA } else { Self::AMD }
    }

    pub fn for_vendor_name(gpu_name: &str) -> Self {
        let lower = gpu_name.to_lowercase();
        let is_nvidia = lower.contains("nvidia")
            || lower.contains("h100")
            || lower.contains("h200")
            || lower.contains("geforce")
            || lower.contains("quadro")
            || lower.contains("tesla");
        Self::for_vendor(is_nvidia)
    }
}

pub fn decode_nvml_throttle_reasons(mask: u64) -> Vec<String> {
    let mut reasons = Vec::new();
    if mask & 0x0000000000000001 != 0 {
        reasons.push("gpu_idle".to_string());
    }
    if mask & 0x0000000000000002 != 0 {
        reasons.push("applications_clocks_setting".to_string());
    }
    if mask & 0x0000000000000004 != 0 {
        reasons.push("sw_power_cap".to_string());
    }
    if mask & 0x0000000000000008 != 0 {
        reasons.push("hw_slowdown".to_string());
    }
    if mask & 0x0000000000000010 != 0 {
        reasons.push("sync_boost".to_string());
    }
    if mask & 0x0000000000000020 != 0 {
        reasons.push("sw_thermal_slowdown".to_string());
    }
    if mask & 0x0000000000000040 != 0 {
        reasons.push("hw_thermal_slowdown".to_string());
    }
    if mask & 0x0000000000000080 != 0 {
        reasons.push("hw_power_brake_slowdown".to_string());
    }
    if mask & 0x0000000000000100 != 0 {
        reasons.push("display_clocks_setting".to_string());
    }
    reasons
}

/// Comprehensive automated evaluation of GPU sensors, thermals, cooling, and throttling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTelemetryAssessment {
    pub vram_vendor: String,
    pub pcie_link: String,
    pub power_cap_w: f32,
    pub temp_crit_c: f32,
    pub hotspot_crit_c: f32,

    pub edge_temp_c: SensorMetric,
    pub hotspot_temp_c: SensorMetric,
    pub mem_temp_c: Option<SensorMetric>,
    pub delta_hotspot_edge_c: f32,
    pub thermal_headroom_c: f32,

    pub core_clock_mhz: SensorMetric,
    pub mem_clock_mhz: SensorMetric,
    pub core_voltage_mv: SensorMetric,
    pub board_power_w: SensorMetric,
    pub fan_rpm: SensorMetric,
    pub peak_fan_pwm_pct: f32,

    pub gpu_util_pct: SensorMetric,
    pub mem_util_pct: SensorMetric,

    pub thermal_verdict: String,
    pub cooling_efficiency_rating: String,
    pub throttling_detected: bool,
    pub power_limit_pegged: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub advisory_flags: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nvml_throttle_reasons_decoded: Vec<String>,
}

impl GpuTelemetryAssessment {
    pub fn evaluate(snapshots: &[TelemetrySnapshot], final_stats: &HardwareStats) -> Self {
        let edge_temps: Vec<f32> = snapshots
            .iter()
            .map(|s| s.edge_temp_c)
            .filter(|&t| t > 0.0)
            .collect();
        let hotspot_temps: Vec<f32> = snapshots
            .iter()
            .map(|s| s.hotspot_temp_c)
            .filter(|&t| t > 0.0)
            .collect();
        let mem_temps: Vec<f32> = snapshots
            .iter()
            .map(|s| s.mem_temp_c)
            .filter(|&t| t > 0.0)
            .collect();
        let core_clocks: Vec<f32> = snapshots
            .iter()
            .map(|s| s.core_clock_mhz as f32)
            .filter(|&c| c > 0.0)
            .collect();
        let mem_clocks: Vec<f32> = snapshots
            .iter()
            .map(|s| s.mem_clock_mhz as f32)
            .filter(|&c| c > 0.0)
            .collect();
        let voltages: Vec<f32> = snapshots
            .iter()
            .map(|s| s.core_voltage_mv as f32)
            .filter(|&v| v > 0.0)
            .collect();
        let powers: Vec<f32> = snapshots
            .iter()
            .map(|s| s.board_power_w)
            .filter(|&p| p > 0.0)
            .collect();
        let fan_rpms: Vec<f32> = snapshots.iter().map(|s| s.fan_rpm as f32).collect();
        let gpu_utils: Vec<f32> = snapshots.iter().map(|s| s.gpu_util_pct).collect();
        let mem_utils: Vec<f32> = snapshots.iter().map(|s| s.mem_util_pct).collect();

        let edge_metric = if edge_temps.is_empty() {
            SensorMetric::from_samples(&[final_stats.gpu_temp_c])
        } else {
            SensorMetric::from_samples(&edge_temps)
        };

        let hotspot_metric = if hotspot_temps.is_empty() {
            SensorMetric::from_samples(&[final_stats.gpu_hotspot_c])
        } else {
            SensorMetric::from_samples(&hotspot_temps)
        };

        let mem_temp_metric = if mem_temps.is_empty() {
            if final_stats.gpu_mem_temp_c > 0.0 {
                Some(SensorMetric::from_samples(&[final_stats.gpu_mem_temp_c]))
            } else {
                None
            }
        } else {
            Some(SensorMetric::from_samples(&mem_temps))
        };

        let core_clock_metric = if core_clocks.is_empty() {
            SensorMetric::from_samples(&[final_stats.gpu_clock_mhz as f32])
        } else {
            SensorMetric::from_samples(&core_clocks)
        };

        let mem_clock_metric = if mem_clocks.is_empty() {
            SensorMetric::from_samples(&[final_stats.gpu_mem_clock_mhz as f32])
        } else {
            SensorMetric::from_samples(&mem_clocks)
        };

        let voltage_metric = if voltages.is_empty() {
            SensorMetric::from_samples(&[final_stats.gpu_voltage_mv as f32])
        } else {
            SensorMetric::from_samples(&voltages)
        };

        let power_metric = if powers.is_empty() {
            SensorMetric::from_samples(&[final_stats.gpu_power_w])
        } else {
            SensorMetric::from_samples(&powers)
        };

        let fan_metric = if fan_rpms.is_empty() {
            SensorMetric::from_samples(&[final_stats.gpu_fan_rpm.unwrap_or(0) as f32])
        } else {
            SensorMetric::from_samples(&fan_rpms)
        };

        let gpu_util_metric = if gpu_utils.is_empty() {
            SensorMetric::from_samples(&[final_stats.gpu_util_pct])
        } else {
            SensorMetric::from_samples(&gpu_utils)
        };

        let mem_util_metric = if mem_utils.is_empty() {
            SensorMetric::from_samples(&[final_stats.mem_util_pct])
        } else {
            SensorMetric::from_samples(&mem_utils)
        };

        let peak_fan_pwm = (final_stats.gpu_fan_pwm.unwrap_or(0) as f32 / 255.0) * 100.0;
        let delta_hotspot_edge_c = if hotspot_metric.max > 0.0 && edge_metric.max > 0.0 {
            (hotspot_metric.max - edge_metric.max).max(0.0)
        } else {
            0.0
        };

        let crit_limit = if final_stats.gpu_hotspot_crit_c > 0.0 {
            final_stats.gpu_hotspot_crit_c
        } else {
            110.0
        };
        let thermal_headroom_c = if hotspot_metric.max > 0.0 {
            (crit_limit - hotspot_metric.max).max(0.0)
        } else {
            0.0
        };

        let thresholds = VendorThresholds::for_vendor_name(&final_stats.gpu_name);
        let mut advisory_flags = Vec::new();

        if delta_hotspot_edge_c > thresholds.max_die_delta_c {
            advisory_flags.push(format!(
                "Junction-edge delta {:.1}°C exceeds nominal {:.1}°C",
                delta_hotspot_edge_c, thresholds.max_die_delta_c
            ));
        }

        if hotspot_metric.max > 0.0 && thermal_headroom_c < thresholds.min_thermal_headroom_c {
            advisory_flags.push(format!(
                "Low thermal headroom: {:.1}°C to critical ceiling",
                thermal_headroom_c
            ));
        }

        let sustained_clock_drop = core_clock_metric.max > 500.0
            && core_clock_metric.p95 < core_clock_metric.max * thresholds.clock_drop_ratio
            && gpu_util_metric.avg > 80.0;
        if sustained_clock_drop {
            advisory_flags.push(format!(
                "Sustained clock drop under load: p95 {:.0} MHz vs peak {:.0} MHz",
                core_clock_metric.p95, core_clock_metric.max
            ));
        }

        let power_limit_pegged = final_stats.gpu_power_cap_w > 0.0
            && power_metric.max >= final_stats.gpu_power_cap_w * thresholds.power_cap_ratio;
        if power_limit_pegged {
            advisory_flags.push(format!(
                "Power cap reached: {:.1}W >= {:.0}% of {:.1}W cap",
                power_metric.max,
                thresholds.power_cap_ratio * 100.0,
                final_stats.gpu_power_cap_w
            ));
        }

        let mut nvml_throttle_reasons_decoded = Vec::new();
        if let Some(mask) = final_stats.nvml_throttle_reasons {
            nvml_throttle_reasons_decoded = decode_nvml_throttle_reasons(mask);
            for reason in &nvml_throttle_reasons_decoded {
                advisory_flags.push(format!("NVML Throttle Reason: {}", reason));
            }
        }

        let nvml_thermal_or_slowdown = nvml_throttle_reasons_decoded
            .iter()
            .any(|r| r.contains("slowdown") || r.contains("thermal"));

        // Automated Assessment Evaluation (Advisory only)
        let throttling_detected = (hotspot_metric.max > 0.0
            && ((thermal_headroom_c <= 2.0 && hotspot_metric.max >= 95.0)
                || (hotspot_metric.max >= crit_limit)
                || sustained_clock_drop))
            || nvml_thermal_or_slowdown;

        let cooling_efficiency_rating = if hotspot_metric.max == 0.0 || edge_metric.max == 0.0 {
            "UNKNOWN (Thermal Sensors Inactive)".to_string()
        } else if delta_hotspot_edge_c <= 18.0 {
            "OPTIMAL (Die Δ ≤ 18°C, Vapor Chamber Coupling)".to_string()
        } else if delta_hotspot_edge_c <= thresholds.max_die_delta_c {
            format!(
                "NOMINAL (Die Δ ≤ {:.0}°C, Healthy Thermal Interface)",
                thresholds.max_die_delta_c
            )
        } else if delta_hotspot_edge_c <= thresholds.max_die_delta_c + 7.0 {
            "ELEVATED (Die Δ Warm Thermal Gradient)".to_string()
        } else {
            "CRITICAL (Die Δ Degraded Paste/Coupling)".to_string()
        };

        let thermal_verdict = if hotspot_metric.max == 0.0 {
            "UNKNOWN (Hotspot Sensor Unavailable)".to_string()
        } else if throttling_detected {
            "THERMAL THROTTLING — Restricted by thermal ceiling".to_string()
        } else if thermal_headroom_c <= 8.0 {
            format!(
                "WARM MARGIN — Within {:.1}°C of limit, cooling loaded",
                thermal_headroom_c
            )
        } else {
            format!(
                "HEALTHY — {:.1}°C thermal headroom, sustained boost",
                thermal_headroom_c
            )
        };

        Self {
            vram_vendor: if final_stats.vram_vendor.is_empty() {
                "Standard VRAM".to_string()
            } else {
                final_stats.vram_vendor.clone()
            },
            pcie_link: if final_stats.pcie_link.is_empty() {
                "PCIe Link Active".to_string()
            } else {
                final_stats.pcie_link.clone()
            },
            power_cap_w: final_stats.gpu_power_cap_w,
            temp_crit_c: final_stats.gpu_temp_crit_c,
            hotspot_crit_c: crit_limit,
            edge_temp_c: edge_metric,
            hotspot_temp_c: hotspot_metric,
            mem_temp_c: mem_temp_metric,
            delta_hotspot_edge_c,
            thermal_headroom_c,
            core_clock_mhz: core_clock_metric,
            mem_clock_mhz: mem_clock_metric,
            core_voltage_mv: voltage_metric,
            board_power_w: power_metric,
            fan_rpm: fan_metric,
            peak_fan_pwm_pct: peak_fan_pwm,
            gpu_util_pct: gpu_util_metric,
            mem_util_pct: mem_util_metric,
            thermal_verdict,
            cooling_efficiency_rating,
            throttling_detected,
            power_limit_pegged,
            advisory_flags,
            nvml_throttle_reasons_decoded,
        }
    }
}

/// Formats the GPU telemetry assessment into an ANSI box scorecard table.
pub fn format_gpu_assessment_box(assessment: &GpuTelemetryAssessment) -> String {
    let mut out = String::with_capacity(2048);

    out.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\n");
    out.push_str("\x1b[38;2;0;220;255m│\x1b[0m            \x1b[1mAUTOMATED GPU SENSORS & THERMAL HEALTH ASSESSMENT\x1b[0m             \x1b[38;2;0;220;255m│\x1b[0m\n");
    out.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\n");

    let vram_line = format!(
        "Memory & Bus     : \x1b[38;2;255;255;255;1m{}\x1b[38;2;140;150;170m │ \x1b[38;2;0;255;200;1m{}\x1b[0m",
        assessment.vram_vendor, assessment.pcie_link
    );
    out.push_str(&format_assessment_line(&vram_line, 72));

    let mem_clk_str = if assessment.mem_clock_mhz.max > 0.0 {
        format!("{:.0} MHz", assessment.mem_clock_mhz.max)
    } else {
        "Auto".to_string()
    };
    let clock_line = format!(
        "Core Frequency   : Avg: \x1b[38;2;0;255;200;1m{:>4.0} MHz\x1b[38;2;140;150;170m │ Peak: \x1b[38;2;255;180;0;1m{:>4.0} MHz\x1b[38;2;140;150;170m │ Mem: \x1b[38;2;255;255;255m{}\x1b[0m",
        assessment.core_clock_mhz.avg, assessment.core_clock_mhz.max, mem_clk_str
    );
    out.push_str(&format_assessment_line(&clock_line, 72));

    let power_cap_str = if assessment.power_cap_w > 0.0 {
        format!("{:.0}W Cap", assessment.power_cap_w)
    } else {
        "Auto".to_string()
    };
    let vcore_str = if assessment.core_voltage_mv.max > 0.0 {
        format!("{:.0}mV", assessment.core_voltage_mv.max)
    } else {
        "N/A".to_string()
    };
    let power_line = format!(
        "Power & Voltage  : Avg: \x1b[38;2;0;255;200;1m{:>5.1}W\x1b[38;2;140;150;170m │ Peak: \x1b[38;2;255;180;0;1m{:>5.1}W\x1b[38;2;140;150;170m ({}) │ Vcore: \x1b[38;2;255;255;255m{}\x1b[0m",
        assessment.board_power_w.avg, assessment.board_power_w.max, power_cap_str, vcore_str
    );
    out.push_str(&format_assessment_line(&power_line, 72));

    let fan_rpm_str = if assessment.fan_rpm.max > 0.0 {
        format!(
            "{:.0} RPM ({:.0}% PWM)",
            assessment.fan_rpm.max, assessment.peak_fan_pwm_pct
        )
    } else {
        "0 RPM (Zero-RPM Idle / Passive Mode)".to_string()
    };
    let fan_line = format!(
        "Acoustics & Fans : \x1b[38;2;255;255;255;1m{}\x1b[0m",
        fan_rpm_str
    );
    out.push_str(&format_assessment_line(&fan_line, 72));

    let mem_temp_str = if let Some(ref m) = assessment.mem_temp_c {
        format!(
            "\x1b[38;2;140;150;170m │ VRAM: \x1b[38;2;255;255;255m{:.1}°C\x1b[0m",
            m.max
        )
    } else {
        String::new()
    };
    let temp_line = format!(
        "Die Temperatures : Edge: \x1b[38;2;0;255;200;1m{:.1}°C\x1b[38;2;140;150;170m │ Hotspot: \x1b[38;2;255;180;0;1m{:.1}°C\x1b[38;2;140;150;170m{}",
        assessment.edge_temp_c.max, assessment.hotspot_temp_c.max, mem_temp_str
    );
    out.push_str(&format_assessment_line(&temp_line, 72));

    let delta_line = format!(
        "Thermal Coupling : Delta (Jct-Edge): \x1b[38;2;255;255;255;1m{:.1}°C\x1b[38;2;140;150;170m │ Headroom: \x1b[38;2;0;255;200;1m{:.1}°C\x1b[0m",
        assessment.delta_hotspot_edge_c, assessment.thermal_headroom_c
    );
    out.push_str(&format_assessment_line(&delta_line, 72));

    let cooling_line = format!(
        "Cooling Health   : \x1b[38;2;255;255;255;1m{}\x1b[0m",
        assessment.cooling_efficiency_rating
    );
    out.push_str(&format_assessment_line(&cooling_line, 72));

    let (verdict_sym, verdict_col) = if assessment.throttling_detected {
        ("✗", "\x1b[38;2;255;60;60;1m")
    } else if assessment.thermal_headroom_c <= 8.0 && assessment.hotspot_temp_c.max > 0.0 {
        ("⚠", "\x1b[38;2;255;180;0;1m")
    } else {
        ("✓", "\x1b[38;2;0;255;128;1m")
    };
    let verdict_line = format!(
        "Verdict          : {}{} {}\x1b[0m",
        verdict_col, verdict_sym, assessment.thermal_verdict
    );
    out.push_str(&format_assessment_line(&verdict_line, 72));

    for flag in &assessment.advisory_flags {
        let flag_line = format!("Advisory Note    : \x1b[38;2;255;180;0m⚠ {}\x1b[0m", flag);
        out.push_str(&format_assessment_line(&flag_line, 72));
    }

    out
}

fn format_assessment_line(content: &str, target_width: usize) -> String {
    let vis = crate::terminal::visible_len(content);
    let pad = target_width.saturating_sub(vis);
    format!(
        "\x1b[38;2;0;220;255m│\x1b[0m {}{} \x1b[38;2;0;220;255m│\x1b[0m\n",
        content,
        " ".repeat(pad)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_metric_calculations() {
        let samples = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let metric = SensorMetric::from_samples(&samples);
        assert_eq!(metric.min, 10.0);
        assert_eq!(metric.max, 50.0);
        assert_eq!(metric.avg, 30.0);
        assert_eq!(metric.p95, 50.0);
    }

    #[test]
    fn test_empty_sensor_metric() {
        let metric = SensorMetric::from_samples(&[]);
        assert_eq!(metric.min, 0.0);
        assert_eq!(metric.max, 0.0);
        assert_eq!(metric.avg, 0.0);
        assert_eq!(metric.p95, 0.0);
    }

    #[test]
    fn test_gpu_telemetry_assessment_optimal() {
        let stats = HardwareStats {
            gpu_name: "Radeon RX 6650 XT".to_string(),
            gpu_clock_mhz: 2600,
            gpu_mem_clock_mhz: 2248,
            gpu_voltage_mv: 1137,
            gpu_util_pct: 99.0,
            mem_util_pct: 85.0,
            gpu_temp_c: 62.0,
            gpu_hotspot_c: 76.0,
            gpu_mem_temp_c: 68.0,
            gpu_temp_crit_c: 100.0,
            gpu_hotspot_crit_c: 110.0,
            gpu_fan_rpm: Some(1450),
            gpu_fan_max_rpm: Some(3200),
            gpu_fan_pwm: Some(115),
            gpu_power_w: 143.0,
            gpu_power_cap_w: 157.0,
            vram_used_mb: 4096,
            vram_total_mb: 8192,
            vram_vendor: "Samsung GDDR6".to_string(),
            pcie_link: "16.0 GT/s, x8".to_string(),
            cpu_name: "i5-12600K".to_string(),
            cpu_usage_pct: 25.0,
            cpu_threads: 16,
            peak_gpu_clock_mhz: 2650,
            peak_gpu_util_pct: 100.0,
            ..Default::default()
        };

        let snapshots = vec![
            TelemetrySnapshot {
                elapsed_sec: 0.5,
                act_id: 0,
                fps: 60.0,
                frametime_ms: 16.6,
                core_clock_mhz: 2550,
                mem_clock_mhz: 2248,
                core_voltage_mv: 1125,
                gpu_util_pct: 98.0,
                mem_util_pct: 80.0,
                edge_temp_c: 60.0,
                hotspot_temp_c: 74.0,
                mem_temp_c: 66.0,
                fan_rpm: 1400,
                fan_pwm_pct: 45.0,
                board_power_w: 140.0,
                vram_used_mb: 4000,
            },
            TelemetrySnapshot {
                elapsed_sec: 1.0,
                act_id: 0,
                fps: 60.0,
                frametime_ms: 16.6,
                core_clock_mhz: 2620,
                mem_clock_mhz: 2248,
                core_voltage_mv: 1137,
                gpu_util_pct: 99.5,
                mem_util_pct: 85.0,
                edge_temp_c: 62.0,
                hotspot_temp_c: 76.0,
                mem_temp_c: 68.0,
                fan_rpm: 1450,
                fan_pwm_pct: 45.0,
                board_power_w: 143.0,
                vram_used_mb: 4096,
            },
        ];

        let assessment = GpuTelemetryAssessment::evaluate(&snapshots, &stats);
        assert_eq!(assessment.delta_hotspot_edge_c, 14.0); // 76 - 62 = 14
        assert_eq!(assessment.thermal_headroom_c, 34.0); // 110 - 76 = 34
        assert!(!assessment.throttling_detected);
        assert!(assessment.cooling_efficiency_rating.contains("OPTIMAL"));
        assert!(assessment.thermal_verdict.contains("HEALTHY"));

        let box_rendered = format_gpu_assessment_box(&assessment);
        assert!(box_rendered.contains("AUTOMATED GPU SENSORS & THERMAL HEALTH ASSESSMENT"));
        assert!(box_rendered.contains("Samsung GDDR6"));
        assert!(box_rendered.contains("16.0 GT/s, x8"));
        assert!(box_rendered.contains("143.0W"));
    }

    #[test]
    fn test_gpu_telemetry_throttling_detection() {
        let stats = HardwareStats {
            gpu_name: "Hot GPU".to_string(),
            gpu_clock_mhz: 1800,
            gpu_mem_clock_mhz: 2000,
            gpu_voltage_mv: 1100,
            gpu_util_pct: 99.0,
            mem_util_pct: 90.0,
            gpu_temp_c: 90.0,
            gpu_hotspot_c: 109.5,
            gpu_mem_temp_c: 95.0,
            gpu_temp_crit_c: 100.0,
            gpu_hotspot_crit_c: 110.0,
            gpu_fan_rpm: Some(3200),
            gpu_fan_max_rpm: Some(3200),
            gpu_fan_pwm: Some(255),
            gpu_power_w: 160.0,
            gpu_power_cap_w: 160.0,
            vram_used_mb: 7000,
            vram_total_mb: 8192,
            vram_vendor: "Micron".to_string(),
            pcie_link: "16.0 GT/s, x16".to_string(),
            cpu_name: "Test CPU".to_string(),
            cpu_usage_pct: 30.0,
            cpu_threads: 16,
            peak_gpu_clock_mhz: 2600,
            peak_gpu_util_pct: 100.0,
            ..Default::default()
        };

        let snapshots = vec![
            TelemetrySnapshot {
                elapsed_sec: 1.0,
                act_id: 1,
                fps: 50.0,
                frametime_ms: 20.0,
                core_clock_mhz: 2600,
                mem_clock_mhz: 2000,
                core_voltage_mv: 1100,
                gpu_util_pct: 99.0,
                mem_util_pct: 90.0,
                edge_temp_c: 85.0,
                hotspot_temp_c: 105.0,
                mem_temp_c: 90.0,
                fan_rpm: 3000,
                fan_pwm_pct: 95.0,
                board_power_w: 160.0,
                vram_used_mb: 7000,
            },
            TelemetrySnapshot {
                elapsed_sec: 2.0,
                act_id: 1,
                fps: 35.0,
                frametime_ms: 28.5,
                core_clock_mhz: 1800,
                mem_clock_mhz: 2000,
                core_voltage_mv: 1050,
                gpu_util_pct: 99.0,
                mem_util_pct: 90.0,
                edge_temp_c: 90.0,
                hotspot_temp_c: 109.5,
                mem_temp_c: 95.0,
                fan_rpm: 3200,
                fan_pwm_pct: 100.0,
                board_power_w: 160.0,
                vram_used_mb: 7000,
            },
        ];

        let assessment = GpuTelemetryAssessment::evaluate(&snapshots, &stats);
        assert!(assessment.throttling_detected);
        assert!(assessment.thermal_verdict.contains("THERMAL THROTTLING"));
        assert!(assessment.power_limit_pegged);
    }
}
