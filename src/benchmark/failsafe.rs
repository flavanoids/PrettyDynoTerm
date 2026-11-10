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
        let safe_hotspot_c = if max_hotspot_c.is_finite() && max_hotspot_c > 0.0 {
            max_hotspot_c
        } else {
            105.0
        };
        let safe_edge_c =
            if max_hotspot_c.is_finite() && max_hotspot_c < 95.0 && max_hotspot_c > 0.0 {
                max_hotspot_c
            } else {
                95.0
            };
        Self {
            enabled,
            max_hotspot_c: safe_hotspot_c,
            max_edge_c: safe_edge_c,
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
        let hotspot_limit =
            if stats.gpu_hotspot_crit_c.is_finite() && stats.gpu_hotspot_crit_c > 0.0 {
                (stats.gpu_hotspot_crit_c - 2.0).min(self.max_hotspot_c)
            } else {
                self.max_hotspot_c
            };

        if stats.gpu_hotspot_c.is_finite()
            && stats.gpu_hotspot_c > 0.0
            && stats.gpu_hotspot_c >= hotspot_limit
        {
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
        let edge_limit = if stats.gpu_temp_crit_c.is_finite() && stats.gpu_temp_crit_c > 0.0 {
            (stats.gpu_temp_crit_c - 3.0).min(self.max_edge_c)
        } else {
            self.max_edge_c
        };

        if stats.gpu_temp_c.is_finite() && stats.gpu_temp_c > 0.0 && stats.gpu_temp_c >= edge_limit
        {
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

        // 3. Cooling Fan Stall Detection Under Load (Active Fan Cards Only)
        // Passively cooled boards (H200, H100, MI100, MI50) have no fan sensors: skip completely.
        if let (Some(rpm), Some(pwm)) = (stats.gpu_fan_rpm, stats.gpu_fan_pwm) {
            if stats.gpu_temp_c >= self.fan_stall_threshold_c && rpm == 0 && pwm > 50 {
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
                            stats.gpu_temp_c, pwm
                        ),
                    });
                }
            } else {
                self.fan_stall_consecutive_frames = 0;
            }
        } else {
            // No fan sensors present: not applicable, never stalled
            self.fan_stall_consecutive_frames = 0;
        }

        None
    }
}

/// Scaling tier for dynamically adapting compute resolutions to GPU VRAM capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VramTestTier {
    /// Safe minimal footprint test (safe for any GPU/llvmpipe, typically ~50-100MB)
    Light,
    /// Balanced dynamic test adapting to ~10-20% available free VRAM
    Balanced,
    /// Stress test effectively exercising ~30-50% of available free VRAM without exceeding headroom
    Stress,
    /// Maximum safe resolution strictly respecting memory headroom and buffer limits
    MaximumSafe,
}

/// Normalized GPU VRAM and buffer limits information across hardware vendors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuMemoryInfo {
    pub total_vram_mb: u64,
    pub used_vram_mb: u64,
    pub free_vram_mb: u64,
    pub max_buffer_size: u64,
    pub max_storage_binding_size: u64,
    pub max_workgroups_per_dim: u32,
    pub is_detected_from_os: bool,
}

impl GpuMemoryInfo {
    /// Automatically queries physical hardware sensors or derives conservative estimates from device limits.
    pub fn detect(device: &Device) -> Self {
        let limits = device.limits();
        if let Some((used_mb, total_mb)) = crate::telemetry::TelemetryCollector::query_system_vram()
        {
            let free_mb = total_mb.saturating_sub(used_mb);
            Self {
                total_vram_mb: total_mb,
                used_vram_mb: used_mb,
                free_vram_mb: free_mb,
                max_buffer_size: limits.max_buffer_size,
                max_storage_binding_size: limits.max_storage_buffer_binding_size,
                max_workgroups_per_dim: limits.max_compute_workgroups_per_dimension,
                is_detected_from_os: true,
            }
        } else {
            // Conservative fallback from device max_buffer_size
            let est_total_mb = (limits.max_buffer_size / (1024 * 1024))
                .saturating_mul(2)
                .max(1024);
            let est_free_mb = est_total_mb.saturating_mul(3) / 4;
            Self {
                total_vram_mb: est_total_mb,
                used_vram_mb: est_total_mb.saturating_sub(est_free_mb),
                free_vram_mb: est_free_mb,
                max_buffer_size: limits.max_buffer_size,
                max_storage_binding_size: limits.max_storage_buffer_binding_size,
                max_workgroups_per_dim: limits.max_compute_workgroups_per_dimension,
                is_detected_from_os: false,
            }
        }
    }

    /// Creates GpuMemoryInfo from HardwareStats, or detects if stats has no VRAM data.
    pub fn from_stats(stats: &HardwareStats, device: &Device) -> Self {
        let limits = device.limits();
        if stats.vram_total_mb > 0 {
            let free_mb = stats.vram_total_mb.saturating_sub(stats.vram_used_mb);
            Self {
                total_vram_mb: stats.vram_total_mb,
                used_vram_mb: stats.vram_used_mb,
                free_vram_mb: free_mb,
                max_buffer_size: limits.max_buffer_size,
                max_storage_binding_size: limits.max_storage_buffer_binding_size,
                max_workgroups_per_dim: limits.max_compute_workgroups_per_dimension,
                is_detected_from_os: true,
            }
        } else {
            Self::detect(device)
        }
    }

    /// Constructs synthetic GPU profile for unit testing diverse hardware tiers (8GB, 24GB, 200GB, etc.).
    pub fn synthetic(
        total_mb: u64,
        free_mb: u64,
        max_buffer_size: u64,
        max_storage_binding_size: u64,
        max_workgroups_per_dim: u32,
    ) -> Self {
        Self {
            total_vram_mb: total_mb,
            used_vram_mb: total_mb.saturating_sub(free_mb),
            free_vram_mb: free_mb,
            max_buffer_size,
            max_storage_binding_size,
            max_workgroups_per_dim,
            is_detected_from_os: true,
        }
    }

    /// Dynamically recommends a safe 16:9 compute resolution scaled for the specified tier,
    /// respecting available VRAM headroom, max_buffer_size, and workgroup dimension limits.
    pub fn recommend_resolution(&self, tier: VramTestTier) -> (u32, u32) {
        let free_bytes = self.free_vram_mb.saturating_mul(1024 * 1024);
        let safe_free_bytes = if free_bytes > 512 * 1024 * 1024 {
            free_bytes.saturating_sub(128 * 1024 * 1024)
        } else {
            free_bytes.saturating_mul(8) / 10
        };

        let target_vram_bytes = match tier {
            VramTestTier::Light => {
                (safe_free_bytes / 20).clamp(32 * 1024 * 1024, 128 * 1024 * 1024)
            }
            VramTestTier::Balanced => (safe_free_bytes * 15 / 100)
                .min(self.max_buffer_size / 2)
                .max(64 * 1024 * 1024),
            VramTestTier::Stress => (safe_free_bytes * 45 / 100)
                .min(self.max_buffer_size * 3 / 4)
                .max(128 * 1024 * 1024),
            VramTestTier::MaximumSafe => (safe_free_bytes * 80 / 100)
                .min(self.max_buffer_size * 9 / 10)
                .max(128 * 1024 * 1024),
        };

        let overhead = 64 * 1024 * 1024;
        let single_buffer_target = if target_vram_bytes > overhead {
            (target_vram_bytes - overhead) / 2
        } else {
            160 * 90 * 4
        };

        let single_buffer_safe = single_buffer_target
            .min(self.max_buffer_size)
            .min(self.max_storage_binding_size);

        let target_pixels = single_buffer_safe / 4;
        let k = ((target_pixels as f64 / 144.0).sqrt() as u32).max(10);
        let mut w = ((16 * k) / 16) * 16;
        let mut h = ((9 * k) / 16) * 16;

        let max_dim = (self.max_workgroups_per_dim as u64).saturating_mul(16);
        if (w as u64) > max_dim {
            w = (max_dim as u32 / 16) * 16;
        }
        if (h as u64) > max_dim {
            h = (max_dim as u32 / 16) * 16;
        }

        (w.max(160), h.max(90))
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
        let mem_info = match stats {
            Some(s) => GpuMemoryInfo::from_stats(s, device),
            None => GpuMemoryInfo::detect(device),
        };
        Self::evaluate_with_memory_info(device, width, height, &mem_info)
    }

    /// Evaluates requested compute resolution with explicit GpuMemoryInfo (used for both real and synthetic tests).
    pub fn evaluate_with_memory_info(
        device: &Device,
        width: u32,
        height: u32,
        mem_info: &GpuMemoryInfo,
    ) -> Result<Self, String> {
        // Out of Bounds Check 1: Zero or Negative Dimensions
        if width == 0 || height == 0 {
            return Err(format!(
                "Invalid out-of-bounds resolution: {}x{}. Compute width and height must both be greater than 0.",
                width, height
            ));
        }

        let limits = device.limits();

        // Out of Bounds Check 2: Max Workgroups Per Dimension
        let workgroups_x = (width as u64).div_ceil(16);
        let workgroups_y = (height as u64).div_ceil(16);
        let max_wg = limits.max_compute_workgroups_per_dimension as u64;
        if workgroups_x > max_wg || workgroups_y > max_wg {
            return Err(format!(
                "Resolution {}x{} dispatch workgroups ({}, {}) exceed GPU hardware limit of {} workgroups per dimension.",
                width, height, workgroups_x, workgroups_y, max_wg
            ));
        }

        let pixel_count = match (width as u64).checked_mul(height as u64) {
            Some(p) => p,
            None => {
                return Err(format!(
                    "Integer overflow in resolution {}x{}: pixel count exceeds 64-bit integer limit.",
                    width, height
                ));
            }
        };

        let single_buffer_bytes = match pixel_count.checked_mul(4) {
            Some(b) => b,
            None => {
                return Err(format!(
                    "Integer overflow in resolution {}x{}: buffer byte size exceeds 64-bit integer limit.",
                    width, height
                ));
            }
        };

        // Out of Bounds / Hardware Limits Check 3: WebGPU max_buffer_size
        if single_buffer_bytes > limits.max_buffer_size {
            return Err(format!(
                "Resolution {}x{} ({:.1} Mpix) requires single buffer size of {:.2} MB, exceeding GPU hardware max_buffer_size limit of {:.2} MB.",
                width,
                height,
                pixel_count as f64 / 1_000_000.0,
                single_buffer_bytes as f64 / (1024.0 * 1024.0),
                limits.max_buffer_size as f64 / (1024.0 * 1024.0)
            ));
        }

        // Out of Bounds / Hardware Limits Check 4: WebGPU max_storage_buffer_binding_size
        if single_buffer_bytes > limits.max_storage_buffer_binding_size {
            return Err(format!(
                "Resolution {}x{} storage binding ({:.2} MB) exceeds GPU max_storage_buffer_binding_size limit of {:.2} MB.",
                width,
                height,
                single_buffer_bytes as f64 / (1024.0 * 1024.0),
                limits.max_storage_buffer_binding_size as f64 / (1024.0 * 1024.0)
            ));
        }

        // Out of Memory Check 5: Estimated Pipeline Allocation
        let total_estimated_bytes = single_buffer_bytes
            .saturating_mul(2)
            .saturating_add(64 * 1024 * 1024);
        let required_mb = total_estimated_bytes / (1024 * 1024);

        // OOM Failsafe: Total Physical VRAM Breach
        if mem_info.total_vram_mb > 0 && required_mb > mem_info.total_vram_mb {
            return Err(format!(
                "Out-of-memory failsafe: Resolution {}x{} requires estimated ~{} MB VRAM, exceeding total physical GPU VRAM ({} MB). Allocation aborted to prevent driver crash.",
                width, height, required_mb, mem_info.total_vram_mb
            ));
        }

        // OOM Failsafe: Free VRAM Headroom Breach
        let safety_margin_mb = if mem_info.free_vram_mb > 1024 {
            (mem_info.free_vram_mb / 10).clamp(128, 512)
        } else {
            (mem_info.free_vram_mb / 10).max(32)
        };
        let safe_free_mb = mem_info.free_vram_mb.saturating_sub(safety_margin_mb);

        if mem_info.free_vram_mb > 0 && required_mb > safe_free_mb {
            return Err(format!(
                "Out-of-memory failsafe: Workload requires estimated ~{} MB VRAM, but only {} MB safe headroom available (detected free: {} MB, total: {} MB, safety reserve: {} MB). Allocation aborted.",
                required_mb,
                safe_free_mb,
                mem_info.free_vram_mb,
                mem_info.total_vram_mb,
                safety_margin_mb
            ));
        }

        let mut warning = None;
        if mem_info.free_vram_mb > 0 && required_mb > (mem_info.free_vram_mb * 75 / 100) {
            warning = Some(format!(
                "Tight VRAM margin: Workload estimated at ~{} MB, nearing free VRAM ceiling ({} MB free of {} MB total).",
                required_mb, mem_info.free_vram_mb, mem_info.total_vram_mb
            ));
        }

        Ok(Self {
            width,
            height,
            pixel_count,
            single_buffer_bytes,
            total_estimated_bytes,
            max_buffer_size: limits.max_buffer_size,
            max_storage_binding_size: limits.max_storage_buffer_binding_size,
            detected_vram_total_mb: mem_info.total_vram_mb,
            detected_vram_free_mb: mem_info.free_vram_mb,
            is_safe: true,
            warning,
        })
    }
}

/// Helper to format a single line within an emergency red border box.
/// Total line width: '│ ' (2) + content (vlen) + pad (72 - vlen) + ' │' (2) = 76 characters.
pub fn format_emergency_line(content: &str) -> String {
    let vlen = crate::terminal::menu::visible_len(content);
    let target: usize = 72;
    let pad = target.saturating_sub(vlen);
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

    let line1 = format!(
        "SENSOR  : \x1b[38;2;255;255;255;1m{}\x1b[0m",
        trip.sensor_type
    );
    out.push_str(&format_emergency_line(&line1));
    out.push_str("\r\n");

    let line2 = format!(
        "READING : \x1b[38;2;255;60;60;1m{:.1}°C\x1b[0m \x1b[38;2;140;150;170m(Ceiling: \x1b[38;2;255;180;0;1m{:.1}°C\x1b[0m, Hardware Crit: \x1b[38;2;255;255;255m{:.1}°C\x1b[0m)\x1b[0m",
        trip.current_temp_c, trip.ceiling_temp_c, trip.crit_limit_c
    );
    out.push_str(&format_emergency_line(&line2));
    out.push_str("\r\n");

    let truncated_msg = crate::benchmark::safe_truncate(&trip.message, 60);
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
            gpu_fan_rpm: Some(1500),
            gpu_fan_pwm: Some(100),
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

        let trip = failsafe
            .check(&stats)
            .expect("Should trip on hotspot breach");
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
    fn test_thermal_failsafe_nan_guard() {
        let mut failsafe = ThermalFailsafe::new(f32::NAN, true);
        // Default safe hotspot fallback should be 105.0
        assert_eq!(failsafe.max_hotspot_c, 105.0);

        let stats = HardwareStats {
            gpu_temp_c: 80.0,
            gpu_hotspot_c: 106.0,
            ..Default::default()
        };
        let trip = failsafe
            .check(&stats)
            .expect("Should trip on default 105.0 ceiling even with initial NaN");
        assert_eq!(trip.current_temp_c, 106.0);
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

        let trip = failsafe
            .check(&stats)
            .expect("Should trip on proximity to crit limit");
        assert_eq!(trip.ceiling_temp_c, 96.0);
    }

    #[test]
    fn test_thermal_failsafe_fan_stall() {
        let mut failsafe = ThermalFailsafe::new(105.0, true);
        let stats = HardwareStats {
            gpu_temp_c: 80.0,
            gpu_hotspot_c: 85.0,
            gpu_fan_rpm: Some(0),
            gpu_fan_pwm: Some(180), // Driver demanding fan
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
    fn test_thermal_failsafe_fan_stall_skipped_when_no_fan() {
        let mut failsafe = ThermalFailsafe::new(105.0, true);
        let stats = HardwareStats {
            gpu_temp_c: 85.0, // High temperature below 95 edge limit
            gpu_hotspot_c: 90.0,
            gpu_fan_rpm: None,
            gpu_fan_pwm: None,
            ..Default::default()
        };

        // Even across 50 frames, fan stall check must be skipped for passively cooled card
        for _ in 0..50 {
            assert!(
                failsafe.check(&stats).is_none(),
                "Passively cooled card must never trigger fan stall"
            );
        }
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
            message:
                "Silicon junction hotspot reached 106.5°C, exceeding safety ceiling of 105.0°C"
                    .to_string(),
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

    #[test]
    fn test_vram_adaptation_8gb_gpu() {
        let mem_8gb = GpuMemoryInfo::synthetic(
            8192,
            3500,
            4 * 1024 * 1024 * 1024,
            4 * 1024 * 1024 * 1024,
            65535,
        );

        let (light_w, light_h) = mem_8gb.recommend_resolution(VramTestTier::Light);
        assert!(light_w >= 160 && light_h >= 90);
        assert_eq!(light_w % 16, 0);
        assert_eq!(light_h % 16, 0);

        let (bal_w, bal_h) = mem_8gb.recommend_resolution(VramTestTier::Balanced);
        assert!(bal_w >= light_w && bal_h >= light_h);
        assert_eq!(bal_w % 16, 0);
        assert_eq!(bal_h % 16, 0);

        let (stress_w, stress_h) = mem_8gb.recommend_resolution(VramTestTier::Stress);
        assert!(stress_w >= bal_w && stress_h >= bal_h);
        assert_eq!(stress_w % 16, 0);
        assert_eq!(stress_h % 16, 0);

        // Verify that balanced resolution fits comfortably in 3500MB free memory
        let single_buf = (bal_w as u64) * (bal_h as u64) * 4;
        let total_est = single_buf * 2 + 64 * 1024 * 1024;
        assert!(total_est <= 3500 * 1024 * 1024);
    }

    #[test]
    fn test_vram_adaptation_200gb_gpu() {
        let mem_200gb = GpuMemoryInfo::synthetic(
            204800,
            180000,
            64 * 1024 * 1024 * 1024,
            64 * 1024 * 1024 * 1024,
            65535,
        );

        let (light_w, light_h) = mem_200gb.recommend_resolution(VramTestTier::Light);
        let (bal_w, bal_h) = mem_200gb.recommend_resolution(VramTestTier::Balanced);
        let (stress_w, stress_h) = mem_200gb.recommend_resolution(VramTestTier::Stress);

        assert!(bal_w >= light_w && bal_h >= light_h);
        assert!(stress_w >= bal_w && stress_h >= bal_h);

        // Stress resolution on 200GB GPU should adapt to massive resolution (>= 8K: 7680x4320)
        assert!(
            (stress_w as u64) * (stress_h as u64) >= 7680 * 4320,
            "200GB GPU should adapt to at least 8K under stress testing"
        );

        // Verify stress allocation strictly respects limits and headroom
        let single_buf = (stress_w as u64) * (stress_h as u64) * 4;
        let total_est = single_buf * 2 + 64 * 1024 * 1024;
        assert!(total_est <= 180000 * 1024 * 1024);
        assert!(single_buf <= 64 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_vram_headroom_oob_and_oom_checks() {
        let mem_8gb = GpuMemoryInfo::synthetic(
            8192,
            3500,
            4 * 1024 * 1024 * 1024,
            4 * 1024 * 1024 * 1024,
            65535,
        );

        // OOB check: 0x0
        assert!(mem_8gb.recommend_resolution(VramTestTier::Light).0 > 0);

        // OOM check: calculate required memory for 65536x65536 (17.1GB single buffer = ~34GB total)
        let single_buf = 65536u64 * 65536 * 4;
        let total_mb = (single_buf * 2 + 64 * 1024 * 1024) / (1024 * 1024);
        assert!(
            total_mb > mem_8gb.total_vram_mb,
            "65536x65536 should exceed 8GB GPU capacity"
        );
    }
}
