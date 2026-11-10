use crate::telemetry::PciBusId;

/// Per-GPU run summary statistics used for peer comparison and outlier detection.
#[derive(Debug, Clone)]
pub struct GpuRunStats {
    pub gpu_index: usize,
    pub pci_bus_id: PciBusId,
    pub gpu_name: String,
    pub duration_s: f64,
    pub score: f64,
    pub avg_fps: f64,
    pub peak_temp_c: Option<f32>,
    pub peak_power_w: Option<f32>,
}

/// Outlier classification result for an individual GPU.
#[derive(Debug, Clone, PartialEq)]
pub struct OutlierFlag {
    pub gpu_index: usize,
    pub pci_bus_id: PciBusId,
    pub is_outlier: bool,
    pub reasons: Vec<String>,
    pub duration_s: f64,
    pub peer_median_duration_s: Option<f64>,
    pub peak_temp_c: Option<f32>,
    pub peer_median_temp_c: Option<f32>,
}

/// Calculates the median value of a slice of f64 numbers.
pub fn calculate_median_f64(vals: &[f64]) -> Option<f64> {
    if vals.is_empty() {
        return None;
    }
    let mut s = vals.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = s.len() / 2;
    if s.len().is_multiple_of(2) {
        Some((s[mid - 1] + s[mid]) / 2.0)
    } else {
        Some(s[mid])
    }
}

/// Calculates the median value of a slice of f32 numbers.
pub fn calculate_median_f32(vals: &[f32]) -> Option<f32> {
    if vals.is_empty() {
        return None;
    }
    let mut s = vals.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = s.len() / 2;
    if s.len().is_multiple_of(2) {
        Some((s[mid - 1] + s[mid]) / 2.0)
    } else {
        Some(s[mid])
    }
}

/// Detects outliers among peer GPUs according to Prompt C4 rules:
/// - Mark any GPU whose runtime/score is >5% worse than the median of its peers.
/// - Mark any GPU whose max temp is >8°C above the peer median.
pub fn detect_gpu_outliers(runs: &[GpuRunStats]) -> Vec<OutlierFlag> {
    let mut flags = Vec::with_capacity(runs.len());

    // Outlier detection requires at least 2 GPUs to have peer comparison
    if runs.len() < 2 {
        for run in runs {
            flags.push(OutlierFlag {
                gpu_index: run.gpu_index,
                pci_bus_id: run.pci_bus_id.clone(),
                is_outlier: false,
                reasons: Vec::new(),
                duration_s: run.duration_s,
                peer_median_duration_s: None,
                peak_temp_c: run.peak_temp_c,
                peer_median_temp_c: None,
            });
        }
        return flags;
    }

    for (i, run) in runs.iter().enumerate() {
        let mut reasons = Vec::new();

        // 1. Peer durations (all GPUs excluding this one)
        let peer_durations: Vec<f64> = runs
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != i)
            .map(|(_, r)| r.duration_s)
            .collect();

        let peer_median_dur = calculate_median_f64(&peer_durations);
        if let Some(median_dur) = peer_median_dur {
            // >5% worse runtime means duration is > median * 1.05
            if median_dur > 0.0 && run.duration_s > median_dur * 1.05 {
                let diff_pct = ((run.duration_s / median_dur) - 1.0) * 100.0;
                reasons.push(format!(
                    ">5% slower than peers: {:.2}s vs peer median {:.2}s (+{:.1}%)",
                    run.duration_s, median_dur, diff_pct
                ));
            }
        }

        // 2. Peer temperatures (all GPUs excluding this one)
        let peer_median_temp = if let Some(gpu_temp) = run.peak_temp_c
            && gpu_temp > 0.0
        {
            let peer_temps: Vec<f32> = runs
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx != i)
                .filter_map(|(_, r)| r.peak_temp_c)
                .filter(|t| *t > 0.0)
                .collect();

            let med_temp = calculate_median_f32(&peer_temps);
            if let Some(median_t) = med_temp {
                // >8°C above peer median
                if gpu_temp > median_t + 8.0 {
                    let diff_c = gpu_temp - median_t;
                    reasons.push(format!(
                        ">8°C hotter than peers: {:.1}°C vs peer median {:.1}°C (+{:.1}°C)",
                        gpu_temp, median_t, diff_c
                    ));
                }
            }
            med_temp
        } else {
            None
        };

        let is_outlier = !reasons.is_empty();
        flags.push(OutlierFlag {
            gpu_index: run.gpu_index,
            pci_bus_id: run.pci_bus_id.clone(),
            is_outlier,
            reasons,
            duration_s: run.duration_s,
            peer_median_duration_s: peer_median_dur,
            peak_temp_c: run.peak_temp_c,
            peer_median_temp_c: peer_median_temp,
        });
    }

    flags
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_gpu_never_outlier() {
        let runs = vec![GpuRunStats {
            gpu_index: 0,
            pci_bus_id: PciBusId::new("0000:03:00.0"),
            gpu_name: "GPU 0".to_string(),
            duration_s: 100.0,
            score: 1000.0,
            avg_fps: 60.0,
            peak_temp_c: Some(95.0),
            peak_power_w: Some(300.0),
        }];
        let flags = detect_gpu_outliers(&runs);
        assert_eq!(flags.len(), 1);
        assert!(!flags[0].is_outlier);
        assert!(flags[0].reasons.is_empty());
    }

    #[test]
    fn test_two_gpus_performance_outlier() {
        let runs = vec![
            GpuRunStats {
                gpu_index: 0,
                pci_bus_id: PciBusId::new("0000:01:00.0"),
                gpu_name: "GPU 0".to_string(),
                duration_s: 10.0,
                score: 1000.0,
                avg_fps: 100.0,
                peak_temp_c: Some(60.0),
                peak_power_w: Some(250.0),
            },
            GpuRunStats {
                gpu_index: 1,
                pci_bus_id: PciBusId::new("0000:02:00.0"),
                gpu_name: "GPU 1".to_string(),
                duration_s: 11.0, // 10% slower
                score: 900.0,
                avg_fps: 90.0,
                peak_temp_c: Some(62.0),
                peak_power_w: Some(250.0),
            },
        ];
        let flags = detect_gpu_outliers(&runs);
        assert_eq!(flags.len(), 2);
        assert!(!flags[0].is_outlier);
        assert!(flags[1].is_outlier);
        assert!(flags[1].reasons[0].contains(">5% slower"));
    }

    #[test]
    fn test_eight_gpus_thermal_and_performance_outliers() {
        let mut runs = Vec::new();
        // 7 normal GPUs at 10.0s, 65°C
        for i in 0..7 {
            runs.push(GpuRunStats {
                gpu_index: i,
                pci_bus_id: PciBusId::new(format!("0000:{:02}:00.0", i + 1)),
                gpu_name: format!("GPU {}", i),
                duration_s: 10.0,
                score: 1000.0,
                avg_fps: 100.0,
                peak_temp_c: Some(65.0),
                peak_power_w: Some(300.0),
            });
        }
        // GPU 7 is both hot (76°C vs 65°C, +11°C) and slow (11.5s vs 10.0s, +15%)
        runs.push(GpuRunStats {
            gpu_index: 7,
            pci_bus_id: PciBusId::new("0000:08:00.0"),
            gpu_name: "GPU 7".to_string(),
            duration_s: 11.5,
            score: 850.0,
            avg_fps: 85.0,
            peak_temp_c: Some(76.0),
            peak_power_w: Some(300.0),
        });

        let flags = detect_gpu_outliers(&runs);
        assert_eq!(flags.len(), 8);
        for flag in &flags[0..7] {
            assert!(!flag.is_outlier);
        }
        assert!(flags[7].is_outlier);
        assert_eq!(flags[7].reasons.len(), 2);
        assert!(flags[7].reasons[0].contains(">5% slower"));
        assert!(flags[7].reasons[1].contains(">8°C hotter"));
    }
}
