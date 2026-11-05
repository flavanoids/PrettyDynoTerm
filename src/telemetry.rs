use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Default)]
pub struct HardwareStats {
    pub gpu_name: String,
    pub gpu_clock_mhz: u32,
    pub gpu_mem_clock_mhz: u32,  // Memory clock in MHz
    pub gpu_voltage_mv: u32,     // GPU core voltage in millivolts
    pub gpu_util_pct: f32,       // GPU core load percentage (0 - 100%)
    pub mem_util_pct: f32,       // Memory controller load percentage (0 - 100%)
    pub gpu_temp_c: f32,         // Edge / Package temp (standard GPU temp matching btop)
    pub gpu_hotspot_c: f32,      // Junction / Hotspot temp (silicon localized peak)
    pub gpu_mem_temp_c: f32,     // VRAM / Memory temp (if available)
    pub gpu_temp_crit_c: f32,    // Critical thermal throttle threshold for edge
    pub gpu_hotspot_crit_c: f32, // Critical thermal throttle threshold for junction
    pub gpu_fan_rpm: u32,        // Current fan speed in RPM
    pub gpu_fan_max_rpm: u32,    // Maximum rated fan speed
    pub gpu_fan_pwm: u32,        // PWM duty cycle (0 - 255)
    pub gpu_power_w: f32,        // Current board power draw (Watts)
    pub gpu_power_cap_w: f32,    // Board power limit / PPT cap (Watts)
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    pub vram_vendor: String, // "Samsung GDDR6", "Micron", etc.
    pub pcie_link: String,   // "16.0 GT/s, x8"
    pub cpu_name: String,
    pub cpu_usage_pct: f32,
    pub cpu_threads: usize,
    pub peak_gpu_clock_mhz: u32,
    pub peak_gpu_util_pct: f32,
}

#[derive(Debug, Clone, Default)]
pub struct HwmonSensors {
    pub clock_mhz: u32,
    pub mem_clock_mhz: u32,
    pub voltage_mv: u32,
    pub edge_temp_c: f32,
    pub hotspot_temp_c: f32,
    pub mem_temp_c: f32,
    pub temp_crit_c: f32,
    pub hotspot_crit_c: f32,
    pub fan_rpm: u32,
    pub fan_max_rpm: u32,
    pub fan_pwm: u32,
    pub power_w: f32,
    pub power_cap_w: f32,
}

pub struct TelemetryCollector {
    stats: Arc<RwLock<HardwareStats>>,
    stop_signal: Arc<AtomicBool>,
    worker_handle: Option<JoinHandle<()>>,
}

impl TelemetryCollector {
    pub fn new(default_gpu_name: String) -> Self {
        let (cpu_total, cpu_idle) = Self::read_proc_stat().unwrap_or((0, 0));
        let cpu_name = Self::read_cpu_name();
        let cpu_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);

        let hwmon_path = Self::find_hwmon_path();
        let vram_path = Self::find_drm_device_path();

        let vram_vendor = Self::read_vram_vendor(&vram_path);
        let pcie_link = Self::read_pcie_link(&vram_path);

        let initial_stats = HardwareStats {
            gpu_name: default_gpu_name,
            gpu_clock_mhz: 0,
            gpu_mem_clock_mhz: 0,
            gpu_voltage_mv: 0,
            gpu_util_pct: 0.0,
            mem_util_pct: 0.0,
            gpu_temp_c: 0.0,
            gpu_hotspot_c: 0.0,
            gpu_mem_temp_c: 0.0,
            gpu_temp_crit_c: 0.0,
            gpu_hotspot_crit_c: 0.0,
            gpu_fan_rpm: 0,
            gpu_fan_max_rpm: 0,
            gpu_fan_pwm: 0,
            gpu_power_w: 0.0,
            gpu_power_cap_w: 0.0,
            vram_used_mb: 0,
            vram_total_mb: 0,
            vram_vendor: vram_vendor.clone(),
            pcie_link: pcie_link.clone(),
            cpu_name,
            cpu_usage_pct: 0.0,
            cpu_threads,
            peak_gpu_clock_mhz: 0,
            peak_gpu_util_pct: 0.0,
        };

        let stats = Arc::new(RwLock::new(initial_stats));
        let stop_signal = Arc::new(AtomicBool::new(false));

        let stats_clone = Arc::clone(&stats);
        let stop_clone = Arc::clone(&stop_signal);
        let hwmon_clone = hwmon_path;
        let vram_clone = vram_path;
        let vram_vendor_clone = vram_vendor;
        let pcie_link_clone = pcie_link;

        let worker_handle = thread::spawn(move || {
            let mut last_cpu_total = cpu_total;
            let mut last_cpu_idle = cpu_idle;
            let mut last_active_clock = 0u32;
            let mut last_active_time = Instant::now();
            let mut peak_clock = 0u32;
            let mut peak_util = 0.0f32;

            while !stop_clone.load(Ordering::Relaxed) {
                // 1. CPU Usage
                let mut cpu_usage_pct = 0.0;
                if let Some((curr_total, curr_idle)) = Self::read_proc_stat() {
                    let total_diff = curr_total.saturating_sub(last_cpu_total);
                    let idle_diff = curr_idle.saturating_sub(last_cpu_idle);
                    last_cpu_total = curr_total;
                    last_cpu_idle = curr_idle;
                    if total_diff > 0 {
                        cpu_usage_pct = ((1.0 - (idle_diff as f32 / total_diff as f32)) * 100.0)
                            .clamp(0.0, 100.0);
                    }
                }

                // 2. GPU hwmon
                let hw = Self::read_gpu_hwmon(&hwmon_clone);
                let mut clock_mhz = hw.clock_mhz;
                let mut mem_clock_mhz = hw.mem_clock_mhz;

                // If core clock is 0 or low, check pp_dpm_sclk for active DPM clock
                if clock_mhz == 0
                    && let Some(ref drm) = vram_clone
                {
                    let (active_dpm, max_dpm) = Self::read_pp_dpm_sclk(drm);
                    if active_dpm > 0 {
                        clock_mhz = active_dpm;
                    } else if last_active_time.elapsed() < Duration::from_millis(1500)
                        && max_dpm > 0
                    {
                        clock_mhz = last_active_clock.max(max_dpm);
                    }
                }

                if mem_clock_mhz == 0
                    && let Some(ref drm) = vram_clone
                {
                    mem_clock_mhz = Self::read_pp_dpm_mclk(drm);
                }

                if clock_mhz > 0 {
                    last_active_clock = clock_mhz;
                    last_active_time = Instant::now();
                    peak_clock = peak_clock.max(clock_mhz);
                } else if last_active_time.elapsed() < Duration::from_millis(1500)
                    && last_active_clock > 0
                {
                    // Smooth momentary GFXOFF gaps between dispatches
                    clock_mhz = last_active_clock;
                }

                // 3. GPU Utilization & Memory Controller Utilization
                let gpu_util = Self::read_gpu_utilization(&vram_clone);
                let mem_util = Self::read_mem_busy_percent(&vram_clone);
                peak_util = peak_util.max(gpu_util);

                // 4. VRAM
                let (vram_used_mb, vram_total_mb) = Self::read_vram(&vram_clone);

                // 5. Update shared stats
                if let Ok(mut w) = stats_clone.write() {
                    w.gpu_clock_mhz = clock_mhz;
                    w.gpu_mem_clock_mhz = mem_clock_mhz;
                    w.gpu_voltage_mv = hw.voltage_mv;
                    w.gpu_util_pct = gpu_util;
                    w.mem_util_pct = mem_util;
                    w.gpu_temp_c = hw.edge_temp_c;
                    w.gpu_hotspot_c = hw.hotspot_temp_c;
                    w.gpu_mem_temp_c = hw.mem_temp_c;
                    w.gpu_temp_crit_c = hw.temp_crit_c;
                    w.gpu_hotspot_crit_c = hw.hotspot_crit_c;
                    w.gpu_fan_rpm = hw.fan_rpm;
                    w.gpu_fan_max_rpm = hw.fan_max_rpm;
                    w.gpu_fan_pwm = hw.fan_pwm;
                    w.gpu_power_w = hw.power_w;
                    w.gpu_power_cap_w = hw.power_cap_w;
                    w.vram_used_mb = vram_used_mb;
                    w.vram_total_mb = vram_total_mb;
                    w.vram_vendor = vram_vendor_clone.clone();
                    w.pcie_link = pcie_link_clone.clone();
                    w.cpu_usage_pct = cpu_usage_pct;
                    w.peak_gpu_clock_mhz = peak_clock;
                    w.peak_gpu_util_pct = peak_util;
                }

                thread::sleep(Duration::from_millis(15));
            }
        });

        Self {
            stats,
            stop_signal,
            worker_handle: Some(worker_handle),
        }
    }

    pub fn sample(&self, default_gpu_name: &str) -> HardwareStats {
        if let Ok(r) = self.stats.read() {
            let mut s = r.clone();
            if s.gpu_name.is_empty() {
                s.gpu_name = default_gpu_name.to_string();
            }
            s
        } else {
            HardwareStats {
                gpu_name: default_gpu_name.to_string(),
                ..Default::default()
            }
        }
    }

    fn read_gpu_hwmon(hwmon_path: &Option<String>) -> HwmonSensors {
        let mut sensors = HwmonSensors::default();

        if let Some(path) = hwmon_path {
            // Read core clock (freq1_input is in Hz for AMD)
            if let Ok(freq_str) = fs::read_to_string(format!("{}/freq1_input", path)) {
                if let Ok(freq_hz) = freq_str.trim().parse::<u64>() {
                    sensors.clock_mhz = (freq_hz / 1_000_000) as u32;
                }
            } else if let Ok(freq_str) = fs::read_to_string(format!("{}/freq2_input", path))
                && let Ok(freq_hz) = freq_str.trim().parse::<u64>()
            {
                sensors.clock_mhz = (freq_hz / 1_000_000) as u32;
            }

            // Read memory clock
            if let Ok(freq_str) = fs::read_to_string(format!("{}/freq2_input", path))
                && let Ok(freq_hz) = freq_str.trim().parse::<u64>()
            {
                sensors.mem_clock_mhz = (freq_hz / 1_000_000) as u32;
            }

            // Read core voltage (in0_input is typically vddgfx in mV)
            if let Ok(in0_str) = fs::read_to_string(format!("{}/in0_input", path))
                && let Ok(v) = in0_str.trim().parse::<u32>()
            {
                sensors.voltage_mv = v;
            }

            // Read fan speed (fan1_input is in RPM)
            if let Ok(fan_str) = fs::read_to_string(format!("{}/fan1_input", path))
                && let Ok(rpm) = fan_str.trim().parse::<u32>()
            {
                sensors.fan_rpm = rpm;
            }
            if let Ok(fan_max_str) = fs::read_to_string(format!("{}/fan1_max", path))
                && let Ok(max_rpm) = fan_max_str.trim().parse::<u32>()
            {
                sensors.fan_max_rpm = max_rpm;
            }
            if let Ok(pwm_str) = fs::read_to_string(format!("{}/pwm1", path))
                && let Ok(pwm) = pwm_str.trim().parse::<u32>()
            {
                sensors.fan_pwm = pwm;
            }

            // Read critical temperature thresholds
            if let Ok(crit_str) = fs::read_to_string(format!("{}/temp1_crit", path))
                && let Ok(mc) = crit_str.trim().parse::<f32>()
            {
                sensors.temp_crit_c = mc / 1000.0;
            }
            if let Ok(crit_str) = fs::read_to_string(format!("{}/temp2_crit", path))
                && let Ok(mc) = crit_str.trim().parse::<f32>()
            {
                sensors.hotspot_crit_c = mc / 1000.0;
            }

            // Read GPU temperature sensors and map by sensor label.
            for i in 1..=8 {
                let input_path = format!("{}/temp{}_input", path, i);
                if let Ok(temp_str) = fs::read_to_string(&input_path)
                    && let Ok(temp_mc) = temp_str.trim().parse::<f32>()
                {
                    let temp_val = temp_mc / 1000.0;
                    let label_path = format!("{}/temp{}_label", path, i);
                    let label = fs::read_to_string(&label_path)
                        .unwrap_or_default()
                        .trim()
                        .to_lowercase();

                    if label.contains("edge")
                        || label.contains("gpu")
                        || label.contains("package")
                        || label.contains("composite")
                        || label.contains("die")
                    {
                        sensors.edge_temp_c = temp_val;
                    } else if label.contains("junction")
                        || label.contains("hotspot")
                        || label.contains("tctl")
                    {
                        sensors.hotspot_temp_c = temp_val;
                    } else if label.contains("mem") || label.contains("vram") {
                        sensors.mem_temp_c = temp_val;
                    } else if i == 1 && sensors.edge_temp_c == 0.0 {
                        sensors.edge_temp_c = temp_val;
                    } else if i == 2 && sensors.hotspot_temp_c == 0.0 {
                        sensors.hotspot_temp_c = temp_val;
                    } else if i == 3 && sensors.mem_temp_c == 0.0 {
                        sensors.mem_temp_c = temp_val;
                    }
                }
            }

            // Fallback: if no edge temp identified but temp1_input exists
            if sensors.edge_temp_c == 0.0
                && let Ok(temp_str) = fs::read_to_string(format!("{}/temp1_input", path))
                && let Ok(temp_mc) = temp_str.trim().parse::<f32>()
            {
                sensors.edge_temp_c = temp_mc / 1000.0;
            }

            // If edge temp still not found, fallback to hotspot
            if sensors.edge_temp_c == 0.0 && sensors.hotspot_temp_c > 0.0 {
                sensors.edge_temp_c = sensors.hotspot_temp_c;
            }

            // Read GPU power (power1_average in microwatts)
            if let Ok(pwr_str) = fs::read_to_string(format!("{}/power1_average", path)) {
                if let Ok(pwr_uw) = pwr_str.trim().parse::<f32>() {
                    sensors.power_w = pwr_uw / 1_000_000.0;
                }
            } else if let Ok(pwr_str) = fs::read_to_string(format!("{}/power1_input", path))
                && let Ok(pwr_uw) = pwr_str.trim().parse::<f32>()
            {
                sensors.power_w = pwr_uw / 1_000_000.0;
            }

            // Read GPU power cap / limit
            if let Ok(cap_str) = fs::read_to_string(format!("{}/power1_cap", path))
                && let Ok(cap_uw) = cap_str.trim().parse::<f32>()
            {
                sensors.power_cap_w = cap_uw / 1_000_000.0;
            }
        }

        sensors
    }

    fn read_pp_dpm_mclk(drm_path: &str) -> u32 {
        let path = format!("{}/pp_dpm_mclk", drm_path);
        if let Ok(content) = fs::read_to_string(&path) {
            for line in content.lines() {
                if line.contains('*') {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let clk_str = parts[1].replace("Mhz", "").replace("MHz", "");
                        if let Ok(c) = clk_str.parse::<u32>() {
                            return c;
                        }
                    }
                }
            }
        }
        0
    }

    fn read_mem_busy_percent(drm_path: &Option<String>) -> f32 {
        if let Some(path) = drm_path
            && let Ok(content) = fs::read_to_string(format!("{}/mem_busy_percent", path))
            && let Ok(pct) = content.trim().parse::<f32>()
        {
            return pct.clamp(0.0, 100.0);
        }
        0.0
    }

    fn read_vram_vendor(drm_path: &Option<String>) -> String {
        if let Some(path) = drm_path
            && let Ok(vendor) = fs::read_to_string(format!("{}/mem_info_vram_vendor", path))
        {
            let trimmed = vendor.trim();
            if !trimmed.is_empty() {
                return match trimmed.to_lowercase().as_str() {
                    "samsung" => "Samsung GDDR6".to_string(),
                    "micron" => "Micron GDDR6".to_string(),
                    "hynix" | "skhynix" => "SK Hynix GDDR6".to_string(),
                    other => other.to_uppercase(),
                };
            }
        }
        String::new()
    }

    fn read_pcie_link(drm_path: &Option<String>) -> String {
        if let Some(path) = drm_path
            && let Ok(content) = fs::read_to_string(format!("{}/pp_dpm_pcie", path))
        {
            for line in content.lines() {
                if line.contains('*') {
                    let cleaned = line.replace('*', "");
                    let parts: Vec<&str> = cleaned.split_whitespace().collect();
                    for (i, &p) in parts.iter().enumerate() {
                        if p.to_lowercase().contains("gt/s") {
                            let speed = p.trim_end_matches(',');
                            if i + 1 < parts.len() {
                                return format!("PCIe {} ({})", speed, parts[i + 1]);
                            }
                            return format!("PCIe {}", speed);
                        }
                    }
                    return cleaned.trim().to_string();
                }
            }
        }
        String::new()
    }

    fn read_pp_dpm_sclk(drm_path: &str) -> (u32, u32) {
        let path = format!("{}/pp_dpm_sclk", drm_path);
        if let Ok(content) = fs::read_to_string(&path) {
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

    fn read_vram(drm_path: &Option<String>) -> (u64, u64) {
        if let Some(path) = drm_path {
            let used = fs::read_to_string(format!("{}/mem_info_vram_used", path))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(0)
                / (1024 * 1024);

            let total = fs::read_to_string(format!("{}/mem_info_vram_total", path))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(0)
                / (1024 * 1024);

            return (used, total);
        }
        (0, 0)
    }

    fn find_hwmon_path() -> Option<String> {
        for card_idx in 0..4 {
            let pattern = format!("/sys/class/drm/card{}/device/hwmon", card_idx);
            if let Ok(entries) = fs::read_dir(&pattern) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        return Some(path.to_string_lossy().to_string());
                    }
                }
            }
        }
        None
    }

    fn read_gpu_utilization(drm_path: &Option<String>) -> f32 {
        if let Some(path) = drm_path {
            // 1. AMD gpu_busy_percent (standard sysfs metric, 0-100)
            if let Ok(content) = fs::read_to_string(format!("{}/gpu_busy_percent", path))
                && let Ok(pct) = content.trim().parse::<f32>()
            {
                return pct.clamp(0.0, 100.0);
            }
        }

        // 2. Scan card0..card4 for gpu_busy_percent
        for card_idx in 0..4 {
            let candidate = format!("/sys/class/drm/card{}/device/gpu_busy_percent", card_idx);
            if let Ok(content) = fs::read_to_string(&candidate)
                && let Ok(pct) = content.trim().parse::<f32>()
            {
                return pct.clamp(0.0, 100.0);
            }
        }

        // 3. Intel i915 / Xe frequency ratio fallback
        for card_idx in 0..4 {
            let act_path = format!("/sys/class/drm/card{}/gt_act_freq_mhz", card_idx);
            let max_path = format!("/sys/class/drm/card{}/gt_max_freq_mhz", card_idx);
            if let (Ok(act_str), Ok(max_str)) =
                (fs::read_to_string(&act_path), fs::read_to_string(&max_path))
                && let (Ok(act), Ok(max)) =
                    (act_str.trim().parse::<f32>(), max_str.trim().parse::<f32>())
                && max > 0.0
            {
                let min_path = format!("/sys/class/drm/card{}/gt_min_freq_mhz", card_idx);
                let min = fs::read_to_string(&min_path)
                    .ok()
                    .and_then(|s| s.trim().parse::<f32>().ok())
                    .unwrap_or(0.0);
                let span = max - min;
                if span > 0.0 {
                    return (((act - min) / span) * 100.0).clamp(0.0, 100.0);
                }
            }
        }

        0.0
    }

    fn find_drm_device_path() -> Option<String> {
        for card_idx in 0..4 {
            let path = format!("/sys/class/drm/card{}/device", card_idx);
            if Path::new(&format!("{}/gpu_busy_percent", path)).exists()
                || Path::new(&format!("{}/mem_info_vram_total", path)).exists()
            {
                return Some(path);
            }
        }
        for card_idx in 0..4 {
            let path = format!("/sys/class/drm/card{}/device", card_idx);
            if Path::new(&path).exists() {
                return Some(path);
            }
        }
        None
    }

    /// Queries the operating system for physical GPU VRAM stats (used_mb, total_mb).
    /// Supports AMD DRM sysfs, Intel lmem sysfs, NVIDIA sysfs / procfs, and PCI BAR resources.
    pub fn query_system_vram() -> Option<(u64, u64)> {
        // 1. Check AMD sysfs mem_info_vram_total / used
        for card_idx in 0..8 {
            let path = format!("/sys/class/drm/card{}/device", card_idx);
            let total_path = format!("{}/mem_info_vram_total", path);
            let used_path = format!("{}/mem_info_vram_used", path);
            if let Ok(total_bytes) = fs::read_to_string(&total_path)
                .and_then(|s| s.trim().parse::<u64>().map_err(std::io::Error::other))
                && total_bytes > 0
            {
                let used_bytes = fs::read_to_string(&used_path)
                    .ok()
                    .and_then(|s| s.trim().parse::<u64>().ok())
                    .unwrap_or(0);
                return Some((used_bytes / (1024 * 1024), total_bytes / (1024 * 1024)));
            }
        }

        // 2. Check Intel lmem_total_bytes / used
        for card_idx in 0..8 {
            let pattern = format!("/sys/class/drm/card{}/lmem_total_bytes", card_idx);
            if let Ok(total_str) = fs::read_to_string(&pattern)
                && let Ok(total_bytes) = total_str.trim().parse::<u64>()
                && total_bytes > 0
            {
                let used_path = format!("/sys/class/drm/card{}/lmem_used_bytes", card_idx);
                let used_bytes = fs::read_to_string(&used_path)
                    .ok()
                    .and_then(|s| s.trim().parse::<u64>().ok())
                    .unwrap_or(0);
                return Some((used_bytes / (1024 * 1024), total_bytes / (1024 * 1024)));
            }
        }

        // 3. Check NVIDIA procfs / sysfs
        if let Ok(entries) = fs::read_dir("/proc/driver/nvidia/gpus") {
            for entry in entries.flatten() {
                let info_path = entry.path().join("information");
                if let Ok(content) = fs::read_to_string(info_path) {
                    for line in content.lines() {
                        if (line.contains("Total Memory:") || line.contains("FB Memory:"))
                            && let Some(val_str) = line.split(':').nth(1)
                        {
                            let num_str = val_str
                                .replace("MiB", "")
                                .replace("MB", "")
                                .replace("KiB", "")
                                .replace("KB", "");
                            if let Ok(mb) = num_str.trim().parse::<u64>()
                                && mb > 0
                            {
                                return Some((0, mb));
                            }
                        }
                    }
                }
            }
        }

        // 4. Check PCI resource0 (BAR0 size for discrete GPUs)
        for card_idx in 0..8 {
            let res_path = format!("/sys/class/drm/card{}/device/resource0", card_idx);
            if let Ok(meta) = fs::metadata(&res_path) {
                let bytes = meta.len();
                if bytes >= 256 * 1024 * 1024 {
                    let mb = bytes / (1024 * 1024);
                    return Some((0, mb));
                }
            }
        }

        None
    }

    pub fn read_proc_stat() -> Option<(u64, u64)> {
        let contents = fs::read_to_string("/proc/stat").ok()?;
        let first_line = contents.lines().next()?;
        let parts: Vec<&str> = first_line.split_whitespace().collect();
        if parts.len() < 5 || parts[0] != "cpu" {
            return None;
        }

        let user: u64 = parts[1].parse().ok()?;
        let nice: u64 = parts[2].parse().ok()?;
        let system: u64 = parts[3].parse().ok()?;
        let idle: u64 = parts[4].parse().ok()?;
        let iowait: u64 = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);
        let irq: u64 = parts.get(6).and_then(|s| s.parse().ok()).unwrap_or(0);
        let softirq: u64 = parts.get(7).and_then(|s| s.parse().ok()).unwrap_or(0);

        let total = user + nice + system + idle + iowait + irq + softirq;
        Some((total, idle))
    }

    pub fn read_cpu_name() -> String {
        if let Ok(contents) = fs::read_to_string("/proc/cpuinfo") {
            for line in contents.lines() {
                if line.starts_with("model name")
                    && let Some((_, val)) = line.split_once(':')
                {
                    return val.trim().to_string();
                }
            }
        }
        "Standard Multi-Core CPU".to_string()
    }
}

impl Drop for TelemetryCollector {
    fn drop(&mut self) {
        self.stop_signal.store(true, Ordering::Relaxed);
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_stats_default() {
        let stats = HardwareStats::default();
        assert_eq!(stats.gpu_clock_mhz, 0);
        assert_eq!(stats.gpu_util_pct, 0.0);
        assert_eq!(stats.gpu_temp_c, 0.0);
        assert_eq!(stats.vram_used_mb, 0);
        assert_eq!(stats.cpu_usage_pct, 0.0);
    }

    #[test]
    fn test_cpu_threads_detected() {
        let threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        assert!(threads >= 1);
    }

    #[test]
    fn test_read_cpu_name_non_empty() {
        let name = TelemetryCollector::read_cpu_name();
        assert!(!name.is_empty());
    }

    #[test]
    fn test_read_proc_stat_format() {
        // On Linux, /proc/stat should return Some((total, idle)) where total >= idle
        if let Some((total, idle)) = TelemetryCollector::read_proc_stat() {
            assert!(total >= idle);
            assert!(total > 0);
        }
    }

    #[test]
    fn test_telemetry_collector_lifecycle() {
        let collector = TelemetryCollector::new("Test Adapter".to_string());
        let stats = collector.sample("Test Adapter");
        assert_eq!(stats.gpu_name, "Test Adapter");
        assert!(stats.cpu_threads >= 1);
        drop(collector);
    }
}
