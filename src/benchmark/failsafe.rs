use crate::telemetry::HardwareStats;
use wgpu::Device;

/// Real-time hardware thermal failsafe that monitors GPU sensors during
/// active rendering/stress workloads and aborts or throttles before silicon damage occurs.
#[derive(Debug, Clone)]
pub struct ThermalFailsafe {
    pub enabled: bool,
    pub max_hotspot_c: f32,
    pub max_edge_c: f32,
    pub fan_stall_threshold_c: f32,
    pub fan_stall_consecutive_frames: u32,
}

/// Report produced when an emergency thermal tripwire triggers.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalTripReport {
    pub sensor_type: String,
    pub current_temp_c: f32,
    pub ceiling_temp_c: f32,
    pub crit_limit_c: f32,
    pub message: String,
}

impl Default for ThermalFailsafe {
    fn default() -> Self {
        Self::new(105.0, true)
    }
}

impl ThermalFailsafe {
    pub fn new(max_hotspot_c: f32, enabled: bool) -> Self {
        Self {
            enabled,
            max_hotspot_c,
            max_edge_c: 95.0,
            fan_stall_threshold_c: 78.0,
            fan_stall_consecutive_frames: 0,
        }
    }

    /// Evaluates current hardware stats against critical safety ceilings.
    /// Returns Some(ThermalTripReport) if an emergency cutoff is required.
    pub fn check(&mut self, stats: &HardwareStats) -> Option<ThermalTripReport> {
        if !self.enabled {
            return None;
        }

        // 1. Silicon Junction / Hotspot Ceiling Check
        // Trigger if hotspot reaches user ceiling (default 105°C) or within 2.0°C of VBIOS crit limit.
        let hotspot_limit = if stats.gpu_hotspot_crit_c > 0.0 {
            (stats.gpu_hotspot_crit_c - 2.0).min(self.max_hotspot_c)
        } else {
            self.max_hotspot_c
        };

        if stats.gpu_hotspot_c > 0.0 && stats.gpu_hotspot_c >= hotspot_limit {
            return Some(ThermalTripReport {
                sensor_type: "GPU Silicon Junction (Hotspot)".to_string(),
                current_temp_c: stats.gpu_hotspot_c,
                ceiling_temp_c: hotspot_limit,
                crit_limit_c: stats.gpu_hotspot_crit_c,
                message: format!(
                    "Silicon junction hotspot reached {:.1}°C, exceeding safety ceiling of {:.1}°C (Hardware Crit: {:.1}°C)",
                    stats.gpu_hotspot_c, hotspot_limit, stats.gpu_hotspot_crit_c
                ),
            });
        }

        // 2. GPU Edge / Package Temperature Ceiling Check
        // Trigger if edge temp reaches 95°C or within 3.0°C of VBIOS crit limit.
        let edge_limit = if stats.gpu_temp_crit_c > 0.0 {
            (stats.gpu_temp_crit_c - 3.0).min(self.max_edge_c)
        } else {
            self.max_edge_c
        };

        if stats.gpu_temp_c > 0.0 && stats.gpu_temp_c >= edge_limit {
            return Some(ThermalTripReport {
                sensor_type: "GPU Package / Edge".to_string(),
                current_temp_c: stats.gpu_temp_c,
                ceiling_temp_c: edge_limit,
                crit_limit_c: stats.gpu_temp_crit_c,
                message: format!(
                    "GPU edge temperature reached {:.1}°C, exceeding safety ceiling of {:.1}°C (Hardware Crit: {:.1}°C)",
                    stats.gpu_temp_c, edge_limit, stats.gpu_temp_crit_c
                ),
            });
        }

        // 3. Cooling Fan Stall Detection Under Load
        // If GPU temperature exceeds 78°C while fan tachometer reads 0 RPM and PWM is actively driven (>50)
        if stats.gpu_temp_c >= self.fan_stall_threshold_c
            && stats.gpu_fan_rpm == 0
            && stats.gpu_fan_pwm > 50
        {
            self.fan_stall_consecutive_frames += 1;
            // 20 consecutive frames of zero RPM under heavy heat indicates fan failure/deadlock
            if self.fan_stall_consecutive_frames >= 20 {
                return Some(ThermalTripReport {
                    sensor_type: "Cooling Fan Tachometer".to_string(),
                    current_temp_c: stats.gpu_temp_c,
                    ceiling_temp_c: self.fan_stall_threshold_c,
                    crit_limit_c: 0.0,
                    message: format!(
                        "Cooling fan stall detected: Fan reads 0 RPM while GPU silicon is {:.1}°C with PWM duty at {}",
                        stats.gpu_temp_c, stats.gpu_fan_pwm
                    ),
                });
            }
        } else {
            self.fan_stall_consecutive_frames = 0;
        }

        None
    }
}

/// Pre-flight VRAM and hardware buffer limits assessment.
#[derive(Debug, Clone)]
pub struct VramHeadroomCheck {
    pub width: u32,
    pub height: u32,
    pub pixel_count: u64,
    pub single_buffer_bytes: u64,
    pub total_estimated_bytes: u64,
    pub max_buffer_size: u64,
    pub max_storage_binding_size: u64,
    pub detected_vram_total_mb: u64,
    pub detected_vram_free_mb: u64,
    pub is_safe: bool,
    pub warning: Option<String>,
}

impl VramHeadroomCheck {
    /// Validates requested compute resolution against hardware limits and available VRAM headroom.
    pub fn evaluate(
        device: &Device,
        width: u32,
        height: u32,
        stats: Option<&HardwareStats>,
    ) -> Result<Self, String> {
        let pixel_count = (width as u64) * (height as u64);
        let single_buffer_bytes = pixel_count.saturating_mul(4); // RGBA8
        // Estimated pipeline allocation: storage + staging + preview + overhead (~2.2x buffer size)
        let total_estimated_bytes = single_buffer_bytes
            .saturating_mul(2)
            .saturating_add(64 * 1024 * 1024);

        let limits = device.limits();

        // Check 1: WebGPU Device max_buffer_size
        if single_buffer_bytes > limits.max_buffer_size {
            return Err(format!(
                "Resolution {}x{} ({:.1} Mpix) requires single buffer size of {:.1} MB, exceeding GPU hardware limit of {:.1} MB.",
                width,
                height,
                pixel_count as f64 / 1_000_000.0,
                single_buffer_bytes as f64 / 1_000_000.0,
                limits.max_buffer_size as f64 / 1_000_000.0
            ));
        }

        // Check 2: WebGPU Device max_storage_buffer_binding_size
        if single_buffer_bytes > limits.max_storage_buffer_binding_size as u64 {
            return Err(format!(
                "Resolution {}x{} storage binding ({:.1} MB) exceeds GPU limit of {:.1} MB.",
                width,
                height,
                single_buffer_bytes as f64 / 1_000_000.0,
                limits.max_storage_buffer_binding_size as f64 / 1_000_000.0
            ));
        }

        let mut warning = None;
        let mut vram_total_mb = 0;
        let mut vram_free_mb = 0;

        if let Some(s) = stats {
            vram_total_mb = s.vram_total_mb;
            if s.vram_total_mb > 0 {
                vram_free_mb = s.vram_total_mb.saturating_sub(s.vram_used_mb);
                let required_mb = total_estimated_bytes / (1024 * 1024);

                if vram_free_mb > 0 && required_mb > (vram_free_mb * 85 / 100) {
                    warning = Some(format!(
                        "Tight VRAM margin: Workload estimated at ~{} MB, but only {} MB free VRAM detected (out of {} MB total).",
                        required_mb, vram_free_mb, s.vram_total_mb
                    ));
                }
            }
        }

        Ok(Self {
            width,
            height,
            pixel_count,
            single_buffer_bytes,
            total_estimated_bytes,
            max_buffer_size: limits.max_buffer_size,
            max_storage_binding_size: limits.max_storage_buffer_binding_size as u64,
            detected_vram_total_mb: vram_total_mb,
            detected_vram_free_mb: vram_free_mb,
            is_safe: true,
            warning,
        })
    }
}

/// Helper to format a single line within an emergency red border box.
/// Total line width: '│ ' (2) + content (vlen) + pad (72 - vlen) + ' │' (2) = 76 characters.
pub fn format_emergency_line(content: &str) -> String {
    let vlen = crate::terminal::menu::visible_len(content);
    let target = 72;
    let pad = if vlen < target { target - vlen } else { 0 };
    format!(
        "\x1b[38;2;255;60;60;1m│\x1b[0m {}{} \x1b[38;2;255;60;60;1m│\x1b[0m",
        content,
        " ".repeat(pad)
    )
}

/// Formats the full emergency box string (for stderr display and unit testing).
pub fn format_emergency_box(trip: &ThermalTripReport) -> String {
    let mut out = String::new();
    out.push_str("\x1b[38;2;255;60;60;1m┌──────────────────────────────────────────────────────────────────────────┐\x1b[0m\r\n");
    out.push_str(&format_emergency_line("\x1b[38;2;255;255;255;1m       🚨 EMERGENCY HARDWARE THERMAL FAILSAFE TRIPPED 🚨        \x1b[0m"));
    out.push_str("\r\n\x1b[38;2;255;60;60;1m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

    let line1 = format!("SENSOR  : \x1b[38;2;255;255;255;1m{}\x1b[0m", trip.sensor_type);
    out.push_str(&format_emergency_line(&line1));
    out.push_str("\r\n");

    let line2 = format!(
        "READING : \x1b[38;2;255;60;60;1m{:.1}°C\x1b[0m \x1b[38;2;140;150;170m(Ceiling: \x1b[38;2;255;180;0;1m{:.1}°C\x1b[0m, Hardware Crit: \x1b[38;2;255;255;255m{:.1}°C\x1b[0m)\x1b[0m",
        trip.current_temp_c, trip.ceiling_temp_c, trip.crit_limit_c
    );
    out.push_str(&format_emergency_line(&line2));
    out.push_str("\r\n");

    let truncated_msg = if trip.message.len() > 60 {
        &trip.message[..60]
    } else {
        &trip.message
    };
    let line3 = format!("DETAILS : \x1b[38;2;255;180;0m{}\x1b[0m", truncated_msg);
    out.push_str(&format_emergency_line(&line3));
    out.push_str("\r\n");

    let line4 = "ACTION  : \x1b[38;2;0;255;200;1mAborting workload immediately to protect GPU silicon\x1b[0m";
    out.push_str(&format_emergency_line(line4));
    out.push_str("\r\n");

    let line5 = "ADVICE  : \x1b[38;2;140;150;170mInspect cooling fan, clean heatsink, or lower resolution\x1b[0m";
    out.push_str(&format_emergency_line(line5));
    out.push_str("\r\n");

    out.push_str("\x1b[38;2;255;60;60;1m└──────────────────────────────────────────────────────────────────────────┘\x1b[0m\r\n");
    out
}

/// Formats and displays an emergency thermal trip box to stderr.
pub fn print_emergency_thermal_box(trip: &ThermalTripReport) {
    eprintln!("\r\n{}", format_emergency_box(trip));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_failsafe_normal_temps() {
        let mut failsafe = ThermalFailsafe::new(105.0, true);
        let stats = HardwareStats {
            gpu_temp_c: 65.0,
            gpu_hotspot_c: 75.0,
            gpu_temp_crit_c: 100.0,
            gpu_hotspot_crit_c: 110.0,
            gpu_fan_rpm: 1500,
            gpu_fan_pwm: 100,
            ..Default::default()
        };

        assert_eq!(failsafe.check(&stats), None);
    }

    #[test]
    fn test_thermal_failsafe_hotspot_breach() {
        let mut failsafe = ThermalFailsafe::new(105.0, true);
        let stats = HardwareStats {
            gpu_temp_c: 80.0,
            gpu_hotspot_c: 106.5,
            gpu_hotspot_crit_c: 110.0,
            ..Default::default()
        };

        let trip = failsafe.check(&stats).expect("Should trip on hotspot breach");
        assert_eq!(trip.sensor_type, "GPU Silicon Junction (Hotspot)");
        assert_eq!(trip.current_temp_c, 106.5);
    }

    #[test]
    fn test_thermal_failsafe_edge_breach() {
        let mut failsafe = ThermalFailsafe::new(105.0, true);
        let stats = HardwareStats {
            gpu_temp_c: 96.0,
            gpu_hotspot_c: 98.0,
            gpu_temp_crit_c: 100.0,
            gpu_hotspot_crit_c: 110.0,
            ..Default::default()
        };

        let trip = failsafe.check(&stats).expect("Should trip on edge breach");
        assert_eq!(trip.sensor_type, "GPU Package / Edge");
        assert_eq!(trip.current_temp_c, 96.0);
    }

    #[test]
    fn test_thermal_failsafe_crit_margin() {
        let mut failsafe = ThermalFailsafe::new(105.0, true);
        // If VBIOS crit is 98.0°C, 98 - 2.0 = 96.0°C limit
        let stats = HardwareStats {
            gpu_temp_c: 82.0,
            gpu_hotspot_c: 96.5,
            gpu_hotspot_crit_c: 98.0,
            ..Default::default()
        };

        let trip = failsafe.check(&stats).expect("Should trip on proximity to crit limit");
        assert_eq!(trip.ceiling_temp_c, 96.0);
    }

    #[test]
    fn test_thermal_failsafe_fan_stall() {
        let mut failsafe = ThermalFailsafe::new(105.0, true);
        let stats = HardwareStats {
            gpu_temp_c: 80.0,
            gpu_hotspot_c: 85.0,
            gpu_fan_rpm: 0,
            gpu_fan_pwm: 180, // Driver demanding fan
            ..Default::default()
        };

        // Needs 20 consecutive frames
        for _ in 0..19 {
            assert_eq!(failsafe.check(&stats), None);
        }

        let trip = failsafe.check(&stats).expect("Should trip on fan stall");
        assert_eq!(trip.sensor_type, "Cooling Fan Tachometer");
    }

    #[test]
    fn test_thermal_failsafe_disabled() {
        let mut failsafe = ThermalFailsafe::new(105.0, false);
        let stats = HardwareStats {
            gpu_temp_c: 100.0,
            gpu_hotspot_c: 115.0,
            ..Default::default()
        };

        assert_eq!(failsafe.check(&stats), None);
    }

    #[test]
    fn test_emergency_box_line_widths() {
        let trip = ThermalTripReport {
            sensor_type: "GPU Silicon Junction (Hotspot)".to_string(),
            current_temp_c: 106.5,
            ceiling_temp_c: 105.0,
            crit_limit_c: 110.0,
            message: "Silicon junction hotspot reached 106.5°C, exceeding safety ceiling of 105.0°C".to_string(),
        };

        let box_str = format_emergency_box(&trip);
        for (idx, line) in box_str.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let vlen = crate::terminal::menu::visible_len(line);
            assert_eq!(
                vlen, 76,
                "Emergency box line {} has visible length {}, expected 76. Content: {:?}",
                idx, vlen, line
            );
        }
    }
}
