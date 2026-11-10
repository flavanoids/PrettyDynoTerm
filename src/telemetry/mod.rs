pub mod amd_sysfs;
pub mod nvml;
pub mod pci;
pub mod types;

pub use amd_sysfs::AmdSysfsTelemetry;
pub use nvml::{DlopenNvml, MockNvml, NvmlClient};
pub use pci::PciBusId;
pub use types::{HardwareStats, PerGpuTelemetry};

use std::collections::HashMap;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub struct TelemetryCollector {
    gpu_stats: Arc<RwLock<HashMap<PciBusId, HardwareStats>>>,
    default_gpu_stats: Arc<RwLock<HardwareStats>>,
    stop_signal: Arc<AtomicBool>,
    worker_handle: Option<JoinHandle<()>>,
}

impl TelemetryCollector {
    pub fn new(default_gpu_name: String) -> Self {
        Self::with_sysfs_and_nvml(
            default_gpu_name,
            AmdSysfsTelemetry::default(),
            Arc::new(DlopenNvml::new()),
        )
    }

    pub fn with_sysfs_and_nvml(
        default_gpu_name: String,
        sysfs: AmdSysfsTelemetry,
        nvml: Arc<dyn NvmlClient>,
    ) -> Self {
        let (cpu_total, cpu_idle) = Self::read_proc_stat().unwrap_or((0, 0));
        let cpu_name = Self::read_cpu_name();
        let cpu_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);

        let initial_stats = HardwareStats {
            gpu_name: default_gpu_name.clone(),
            cpu_name: cpu_name.clone(),
            cpu_threads,
            ..Default::default()
        };

        let gpu_stats = Arc::new(RwLock::new(HashMap::<PciBusId, HardwareStats>::new()));
        let default_gpu_stats = Arc::new(RwLock::new(initial_stats));
        let stop_signal = Arc::new(AtomicBool::new(false));

        let stats_map_clone = Arc::clone(&gpu_stats);
        let default_stats_clone = Arc::clone(&default_gpu_stats);
        let stop_clone = Arc::clone(&stop_signal);

        let worker_handle = thread::spawn(move || {
            let mut last_cpu_total = cpu_total;
            let mut last_cpu_idle = cpu_idle;

            while !stop_clone.load(Ordering::Relaxed) {
                // 1. Calculate CPU usage
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

                // 2. Discover and query all AMD cards via sysfs
                let amd_cards = sysfs.query_all_cards();

                // 3. Discover and query all NVIDIA cards via NVML (if available)
                let nvidia_cards = if nvml.is_available() {
                    nvml.query_all_gpus().unwrap_or_default()
                } else {
                    Vec::new()
                };

                // 4. Update multi-GPU stats map
                let mut all_cards = amd_cards;
                all_cards.extend(nvidia_cards);

                if let Ok(mut map) = stats_map_clone.write() {
                    for card in &all_cards {
                        let prev_peak_clock = map
                            .get(&card.pci_bus_id)
                            .map(|s| s.peak_gpu_clock_mhz)
                            .unwrap_or(0);
                        let prev_peak_util = map
                            .get(&card.pci_bus_id)
                            .map(|s| s.peak_gpu_util_pct)
                            .unwrap_or(0.0);

                        let mut stats = HardwareStats::from_per_gpu(
                            card,
                            cpu_name.clone(),
                            cpu_usage_pct,
                            cpu_threads,
                        );
                        stats.peak_gpu_clock_mhz = prev_peak_clock.max(stats.gpu_clock_mhz);
                        stats.peak_gpu_util_pct = prev_peak_util.max(stats.gpu_util_pct);

                        map.insert(card.pci_bus_id.clone(), stats);
                    }
                }

                // 5. Update fallback default stats
                if let Ok(mut def) = default_stats_clone.write() {
                    def.cpu_usage_pct = cpu_usage_pct;
                    if let Ok(map) = stats_map_clone.read()
                        && let Some((_, first)) = map.iter().next()
                    {
                        let peak_clk = def.peak_gpu_clock_mhz.max(first.gpu_clock_mhz);
                        let peak_ut = def.peak_gpu_util_pct.max(first.gpu_util_pct);
                        let name = if def.gpu_name.is_empty() || def.gpu_name == "Unknown GPU" {
                            first.gpu_name.clone()
                        } else {
                            def.gpu_name.clone()
                        };
                        *def = first.clone();
                        def.gpu_name = name;
                        def.peak_gpu_clock_mhz = peak_clk;
                        def.peak_gpu_util_pct = peak_ut;
                    }
                }

                thread::sleep(Duration::from_millis(100));
            }
        });

        Self {
            gpu_stats,
            default_gpu_stats,
            stop_signal,
            worker_handle: Some(worker_handle),
        }
    }

    /// Samples telemetry for the default active GPU adapter.
    pub fn sample(&self, default_gpu_name: &str) -> HardwareStats {
        if let Ok(r) = self.default_gpu_stats.read() {
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

    /// Samples telemetry for a specific physical GPU keyed by canonical PCI Bus ID.
    pub fn sample_by_pci(&self, pci: &PciBusId) -> Option<HardwareStats> {
        self.gpu_stats.read().ok()?.get(pci).cloned()
    }

    /// Samples telemetry for a specific GPU by canonical PCI Bus ID, falling back to default stats.
    pub fn sample_or_default(&self, pci: &PciBusId, default_gpu_name: &str) -> HardwareStats {
        if let Some(stats) = self.sample_by_pci(pci) {
            return stats;
        }
        self.sample(default_gpu_name)
    }

    /// Returns a snapshot of all discovered physical GPU cards across the node.
    pub fn all_gpus(&self) -> Vec<HardwareStats> {
        match self.gpu_stats.read() {
            Ok(map) => {
                let mut v: Vec<HardwareStats> = map.values().cloned().collect();
                v.sort_by(|a, b| a.pci_bus_id.cmp(&b.pci_bus_id));
                v
            }
            Err(_) => Vec::new(),
        }
    }

    /// Queries system VRAM capacity from any detected GPU.
    pub fn query_system_vram() -> Option<(u64, u64)> {
        let sysfs = AmdSysfsTelemetry::default();
        let cards = sysfs.query_all_cards();
        for card in cards {
            if let (Some(used), Some(total)) = (card.vram_used_mb, card.vram_total_mb)
                && total > 0
            {
                return Some((used, total));
            }
        }
        None
    }

    pub fn read_proc_stat() -> Option<(u64, u64)> {
        let content = fs::read_to_string("/proc/stat").ok()?;
        let first_line = content.lines().next()?;
        let parts: Vec<&str> = first_line.split_whitespace().collect();
        if parts.len() < 5 || parts[0] != "cpu" {
            return None;
        }

        let mut total: u64 = 0;
        let mut idle: u64 = 0;
        for (i, p) in parts[1..].iter().enumerate() {
            if let Ok(v) = p.parse::<u64>() {
                total += v;
                if i == 3 {
                    idle = v;
                }
            }
        }
        Some((total, idle))
    }

    pub fn read_cpu_name() -> String {
        if let Ok(content) = fs::read_to_string("/proc/cpuinfo") {
            for line in content.lines() {
                if line.starts_with("model name")
                    && let Some(pos) = line.find(':')
                {
                    let name = line[pos + 1..].trim();
                    if !name.is_empty() {
                        return name.to_string();
                    }
                }
            }
        }
        "CPU".to_string()
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
