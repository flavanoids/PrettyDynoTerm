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
    pub gpu_temp_c: f32,       // Edge / Package temp (standard GPU temp matching btop)
    pub gpu_hotspot_c: f32,    // Junction / Hotspot temp (silicon localized peak)
    pub gpu_mem_temp_c: f32,   // VRAM / Memory temp (if available)
    pub gpu_power_w: f32,
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    pub cpu_name: String,
    pub cpu_usage_pct: f32,
    pub cpu_threads: usize,
    pub peak_gpu_clock_mhz: u32,
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

        let initial_stats = HardwareStats {
            gpu_name: default_gpu_name,
            gpu_clock_mhz: 0,
            gpu_temp_c: 0.0,
            gpu_hotspot_c: 0.0,
            gpu_mem_temp_c: 0.0,
            gpu_power_w: 0.0,
            vram_used_mb: 0,
            vram_total_mb: 0,
            cpu_name,
            cpu_usage_pct: 0.0,
            cpu_threads,
            peak_gpu_clock_mhz: 0,
        };

        let stats = Arc::new(RwLock::new(initial_stats));
        let stop_signal = Arc::new(AtomicBool::new(false));

        let stats_clone = Arc::clone(&stats);
        let stop_clone = Arc::clone(&stop_signal);
        let hwmon_clone = hwmon_path;
        let vram_clone = vram_path;

        let worker_handle = thread::spawn(move || {
            let mut last_cpu_total = cpu_total;
            let mut last_cpu_idle = cpu_idle;
            let mut last_active_clock = 0u32;
            let mut last_active_time = Instant::now();
            let mut peak_clock = 0u32;

            while !stop_clone.load(Ordering::Relaxed) {
                // 1. CPU Usage
                let mut cpu_usage_pct = 0.0;
                if let Some((curr_total, curr_idle)) = Self::read_proc_stat() {
                    let total_diff = curr_total.saturating_sub(last_cpu_total);
                    let idle_diff = curr_idle.saturating_sub(last_cpu_idle);
                    last_cpu_total = curr_total;
                    last_cpu_idle = curr_idle;
                    if total_diff > 0 {
                        cpu_usage_pct =
                            ((1.0 - (idle_diff as f32 / total_diff as f32)) * 100.0).clamp(0.0, 100.0);
                    }
                }

                // 2. GPU hwmon
                let (mut clock_mhz, edge_temp, hotspot_temp, mem_temp, power_w) =
                    Self::read_gpu_hwmon(&hwmon_clone);

                // If clock is 0 or low, check pp_dpm_sclk for active DPM clock
                if clock_mhz == 0 {
                    if let Some(ref drm) = vram_clone {
                        let (active_dpm, max_dpm) = Self::read_pp_dpm_sclk(drm);
                        if active_dpm > 0 {
                            clock_mhz = active_dpm;
                        } else if last_active_time.elapsed() < Duration::from_millis(1500) && max_dpm > 0 {
                            clock_mhz = last_active_clock.max(max_dpm);
                        }
                    }
                }

                if clock_mhz > 0 {
                    last_active_clock = clock_mhz;
                    last_active_time = Instant::now();
                    peak_clock = peak_clock.max(clock_mhz);
                } else if last_active_time.elapsed() < Duration::from_millis(1500) && last_active_clock > 0 {
                    // Smooth momentary GFXOFF gaps between dispatches
                    clock_mhz = last_active_clock;
                }

                // 3. VRAM
                let (vram_used_mb, vram_total_mb) = Self::read_vram(&vram_clone);

                // 4. Update shared stats
                if let Ok(mut w) = stats_clone.write() {
                    w.gpu_clock_mhz = clock_mhz;
                    w.gpu_temp_c = edge_temp;
                    w.gpu_hotspot_c = hotspot_temp;
                    w.gpu_mem_temp_c = mem_temp;
                    w.gpu_power_w = power_w;
                    w.vram_used_mb = vram_used_mb;
                    w.vram_total_mb = vram_total_mb;
                    w.cpu_usage_pct = cpu_usage_pct;
                    w.peak_gpu_clock_mhz = peak_clock;
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

    fn read_gpu_hwmon(hwmon_path: &Option<String>) -> (u32, f32, f32, f32, f32) {
        let mut clock_mhz = 0;
        let mut edge_temp_c = 0.0;
        let mut hotspot_temp_c = 0.0;
        let mut mem_temp_c = 0.0;
        let mut power_w = 0.0;

        if let Some(path) = hwmon_path {
            // Read GPU clock (freq1_input is in Hz for AMD)
            if let Ok(freq_str) = fs::read_to_string(format!("{}/freq1_input", path)) {
                if let Ok(freq_hz) = freq_str.trim().parse::<u64>() {
                    clock_mhz = (freq_hz / 1_000_000) as u32;
                }
            } else if let Ok(freq_str) = fs::read_to_string(format!("{}/freq2_input", path)) {
                if let Ok(freq_hz) = freq_str.trim().parse::<u64>() {
                    clock_mhz = (freq_hz / 1_000_000) as u32;
                }
            }

            // Read GPU temperature sensors and map by sensor label.
            // On modern GPUs (AMD amdgpu, Intel, etc.):
            // - "edge" / "gpu" / "package" is the primary GPU die temp displayed by btop/mangohud.
            // - "junction" / "hotspot" is the localized hottest silicon temp (can be 20-35°C higher).
            // - "mem" / "vram" is the video memory junction temp.
            for i in 1..=8 {
                let input_path = format!("{}/temp{}_input", path, i);
                if let Ok(temp_str) = fs::read_to_string(&input_path) {
                    if let Ok(temp_mc) = temp_str.trim().parse::<f32>() {
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
                            edge_temp_c = temp_val;
                        } else if label.contains("junction") || label.contains("hotspot") || label.contains("tctl") {
                            hotspot_temp_c = temp_val;
                        } else if label.contains("mem") || label.contains("vram") {
                            mem_temp_c = temp_val;
                        } else if i == 1 && edge_temp_c == 0.0 {
                            edge_temp_c = temp_val;
                        } else if i == 2 && hotspot_temp_c == 0.0 {
                            hotspot_temp_c = temp_val;
                        } else if i == 3 && mem_temp_c == 0.0 {
                            mem_temp_c = temp_val;
                        }
                    }
                }
            }

            // Fallback: if no edge temp identified but temp1_input exists
            if edge_temp_c == 0.0 {
                if let Ok(temp_str) = fs::read_to_string(format!("{}/temp1_input", path)) {
                    if let Ok(temp_mc) = temp_str.trim().parse::<f32>() {
                        edge_temp_c = temp_mc / 1000.0;
                    }
                }
            }

            // If edge temp still not found, fallback to hotspot
            if edge_temp_c == 0.0 && hotspot_temp_c > 0.0 {
                edge_temp_c = hotspot_temp_c;
            }

            // Read GPU power (power1_average in microwatts)
            if let Ok(pwr_str) = fs::read_to_string(format!("{}/power1_average", path)) {
                if let Ok(pwr_uw) = pwr_str.trim().parse::<f32>() {
                    power_w = pwr_uw / 1_000_000.0;
                }
            } else if let Ok(pwr_str) = fs::read_to_string(format!("{}/power1_input", path)) {
                if let Ok(pwr_uw) = pwr_str.trim().parse::<f32>() {
                    power_w = pwr_uw / 1_000_000.0;
                }
            }
        }

        (clock_mhz, edge_temp_c, hotspot_temp_c, mem_temp_c, power_w)
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

    fn find_drm_device_path() -> Option<String> {
        for card_idx in 0..4 {
            let path = format!("/sys/class/drm/card{}/device", card_idx);
            if Path::new(&format!("{}/mem_info_vram_total", path)).exists() {
                return Some(path);
            }
        }
        None
    }

    fn read_proc_stat() -> Option<(u64, u64)> {
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

    fn read_cpu_name() -> String {
        if let Ok(contents) = fs::read_to_string("/proc/cpuinfo") {
            for line in contents.lines() {
                if line.starts_with("model name") {
                    if let Some((_, val)) = line.split_once(':') {
                        return val.trim().to_string();
                    }
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
