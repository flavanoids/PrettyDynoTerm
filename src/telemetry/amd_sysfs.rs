use crate::telemetry::pci::PciBusId;
use crate::telemetry::types::PerGpuTelemetry;
use std::fs;
use std::path::{Path, PathBuf};

/// Reads per-card AMD hardware telemetry directly from Linux sysfs trees.
pub struct AmdSysfsTelemetry {
    sysfs_root: PathBuf,
}

impl Default for AmdSysfsTelemetry {
    fn default() -> Self {
        Self::new(Path::new("/sys"))
    }
}

impl AmdSysfsTelemetry {
    pub fn new(sysfs_root: &Path) -> Self {
        Self {
            sysfs_root: sysfs_root.to_path_buf(),
        }
    }

    /// Scans sysfs DRM directory and discovers all AMD cards with their PCI bus IDs.
    pub fn query_all_cards(&self) -> Vec<PerGpuTelemetry> {
        let drm_dir = self.sysfs_root.join("class/drm");
        let entries = match fs::read_dir(&drm_dir) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };

        let mut results = Vec::new();

        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let name_str = file_name.to_string_lossy();

            // Match only `card[0-9]+` (ignore `card0-DP-1`, `renderD128`, etc.)
            if !name_str.starts_with("card") {
                continue;
            }
            let suffix = &name_str[4..];
            if suffix.is_empty() || !suffix.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }

            let card_path = entry.path();
            let device_path = card_path.join("device");
            if !device_path.exists() {
                continue;
            }

            // Identify PCI bus ID
            let pci_id = match Self::read_pci_bus_id(&device_path) {
                Some(id) if !id.is_empty() => id,
                _ => continue,
            };

            let mut gpu = PerGpuTelemetry {
                pci_bus_id: pci_id,
                gpu_name: format!("AMD GPU ({})", name_str),
                ..Default::default()
            };

            // Read device-level sysfs properties
            Self::read_device_metrics(&device_path, &mut gpu);

            // Read hwmon-level sensor properties
            let hwmon_base = device_path.join("hwmon");
            if let Ok(hwmon_entries) = fs::read_dir(&hwmon_base) {
                for hwmon_entry in hwmon_entries.flatten() {
                    let hw_path = hwmon_entry.path();
                    if hw_path.is_dir() {
                        Self::read_hwmon_metrics(&hw_path, &mut gpu);
                        break; // Typically one hwmon per card
                    }
                }
            }

            results.push(gpu);
        }

        // Sort deterministically by canonical PCI bus ID
        results.sort_by(|a, b| a.pci_bus_id.cmp(&b.pci_bus_id));
        results
    }

    fn read_pci_bus_id(device_path: &Path) -> Option<PciBusId> {
        // 1. Try reading symlink target: `card0/device -> ../../../0000:03:00.0`
        if let Ok(target) = fs::read_link(device_path)
            && let Some(name) = target.file_name()
        {
            let id = PciBusId::new(name.to_string_lossy());
            if !id.is_empty() {
                return Some(id);
            }
        }

        // 2. Fallback: Parse `uevent` file for `PCI_SLOT_NAME=0000:03:00.0`
        let uevent_path = device_path.join("uevent");
        if let Ok(content) = fs::read_to_string(uevent_path) {
            for line in content.lines() {
                if let Some(val) = line.strip_prefix("PCI_SLOT_NAME=") {
                    let id = PciBusId::new(val.trim());
                    if !id.is_empty() {
                        return Some(id);
                    }
                }
            }
        }

        None
    }

    fn read_device_metrics(device_path: &Path, gpu: &mut PerGpuTelemetry) {
        // GPU Busy Percentage (0 - 100)
        if let Ok(s) = fs::read_to_string(device_path.join("gpu_busy_percent"))
            && let Ok(v) = s.trim().parse::<f32>()
        {
            gpu.gpu_util_pct = Some(v.clamp(0.0, 100.0));
        }

        // Memory Controller Busy Percentage (0 - 100)
        if let Ok(s) = fs::read_to_string(device_path.join("mem_busy_percent"))
            && let Ok(v) = s.trim().parse::<f32>()
        {
            gpu.mem_util_pct = Some(v.clamp(0.0, 100.0));
        }

        // VRAM Used (Bytes -> MB)
        if let Ok(s) = fs::read_to_string(device_path.join("mem_info_vram_used"))
            && let Ok(bytes) = s.trim().parse::<u64>()
        {
            gpu.vram_used_mb = Some(bytes / (1024 * 1024));
        }

        // VRAM Total (Bytes -> MB)
        if let Ok(s) = fs::read_to_string(device_path.join("mem_info_vram_total"))
            && let Ok(bytes) = s.trim().parse::<u64>()
        {
            gpu.vram_total_mb = Some(bytes / (1024 * 1024));
        }

        // VRAM Vendor
        if let Ok(s) = fs::read_to_string(device_path.join("mem_info_vram_vendor")) {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                gpu.vram_vendor = Some(trimmed.to_string());
            }
        }

        // Active DPM clocks fallback if hwmon freq is absent
        if gpu.clock_mhz.is_none() {
            let (active, _) = Self::read_pp_dpm_clock(&device_path.join("pp_dpm_sclk"));
            if active > 0 {
                gpu.clock_mhz = Some(active);
            }
        }
        if gpu.mem_clock_mhz.is_none() {
            let (active, _) = Self::read_pp_dpm_clock(&device_path.join("pp_dpm_mclk"));
            if active > 0 {
                gpu.mem_clock_mhz = Some(active);
            }
        }
    }

    fn read_hwmon_metrics(hwmon_path: &Path, gpu: &mut PerGpuTelemetry) {
        // 1. Temperature Sensors: match by `temp*_label`
        let mut has_labels = false;
        if let Ok(entries) = fs::read_dir(hwmon_path) {
            for entry in entries.flatten() {
                let file_name = entry.file_name();
                let name = file_name.to_string_lossy();
                if name.starts_with("temp") && name.ends_with("_label") {
                    has_labels = true;
                    let idx_str = &name[4..name.len() - 6];
                    if let Ok(content) = fs::read_to_string(entry.path()) {
                        let label = content.trim().to_ascii_lowercase();
                        let input_file = hwmon_path.join(format!("temp{}_input", idx_str));
                        let crit_file = hwmon_path.join(format!("temp{}_crit", idx_str));

                        let temp_val = Self::read_temp_file(&input_file);
                        let crit_val = Self::read_temp_file(&crit_file);

                        if label.contains("edge") {
                            gpu.edge_temp_c = temp_val;
                            gpu.edge_temp_crit_c = crit_val;
                        } else if label.contains("junction") || label.contains("hotspot") {
                            gpu.junction_temp_c = temp_val;
                            gpu.junction_temp_crit_c = crit_val;
                        } else if label.contains("mem") || label.contains("hbm") {
                            gpu.mem_temp_c = temp_val;
                            gpu.mem_temp_crit_c = crit_val;
                        }
                    }
                }
            }
        }

        // Fallback: If no labels are present, map index 1 -> edge, index 2 -> junction, index 3 -> mem
        if !has_labels {
            gpu.edge_temp_c = Self::read_temp_file(&hwmon_path.join("temp1_input"));
            gpu.edge_temp_crit_c = Self::read_temp_file(&hwmon_path.join("temp1_crit"));

            gpu.junction_temp_c = Self::read_temp_file(&hwmon_path.join("temp2_input"));
            gpu.junction_temp_crit_c = Self::read_temp_file(&hwmon_path.join("temp2_crit"));

            gpu.mem_temp_c = Self::read_temp_file(&hwmon_path.join("temp3_input"));
            gpu.mem_temp_crit_c = Self::read_temp_file(&hwmon_path.join("temp3_crit"));
        }

        // 2. Power: power1_average (or power1_input fallback) and power1_cap (microwatts -> Watts)
        let power_uw = Self::read_u64_file(&hwmon_path.join("power1_average"))
            .or_else(|| Self::read_u64_file(&hwmon_path.join("power1_input")));
        if let Some(uw) = power_uw {
            gpu.power_w = Some(uw as f32 / 1_000_000.0);
        }

        if let Some(cap_uw) = Self::read_u64_file(&hwmon_path.join("power1_cap")) {
            gpu.power_cap_w = Some(cap_uw as f32 / 1_000_000.0);
        }

        // 3. Frequencies: freq1_input (Hz -> MHz) and freq2_input
        if let Some(freq_hz) = Self::read_u64_file(&hwmon_path.join("freq1_input")) {
            gpu.clock_mhz = Some((freq_hz / 1_000_000) as u32);
        }
        if let Some(freq_hz) = Self::read_u64_file(&hwmon_path.join("freq2_input")) {
            gpu.mem_clock_mhz = Some((freq_hz / 1_000_000) as u32);
        }

        // 4. Fans: strictly None if absent (passively cooled boards have no fan files)
        if let Some(rpm) = Self::read_u64_file(&hwmon_path.join("fan1_input")) {
            gpu.fan_rpm = Some(rpm as u32);
        }
        if let Some(max_rpm) = Self::read_u64_file(&hwmon_path.join("fan1_max")) {
            gpu.fan_max_rpm = Some(max_rpm as u32);
        }
        if let Some(pwm) = Self::read_u64_file(&hwmon_path.join("pwm1")) {
            gpu.fan_pwm = Some((pwm as u32).min(255));
        }
    }

    fn read_temp_file(path: &Path) -> Option<f32> {
        let content = fs::read_to_string(path).ok()?;
        let milli = content.trim().parse::<f32>().ok()?;
        Some(milli / 1000.0)
    }

    fn read_u64_file(path: &Path) -> Option<u64> {
        let content = fs::read_to_string(path).ok()?;
        content.trim().parse::<u64>().ok()
    }

    fn read_pp_dpm_clock(path: &Path) -> (u32, u32) {
        if let Ok(content) = fs::read_to_string(path) {
            let mut active = 0u32;
            let mut highest = 0u32;
            for line in content.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let clk_str = parts[1].replace("Mhz", "").replace("MHz", "");
                    if let Ok(c) = clk_str.parse::<u32>() {
                        highest = highest.max(c);
                        if line.contains('*') && !line.starts_with('S') {
                            active = c;
                        }
                    }
                }
            }
            return (active, highest);
        }
        (0, 0)
    }
}
