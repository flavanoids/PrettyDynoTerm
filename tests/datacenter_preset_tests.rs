use dyno_term::benchmark::failsafe::ThermalFailsafe;
use dyno_term::benchmark::telemetry::{
    GpuTelemetryAssessment, VendorThresholds, decode_nvml_throttle_reasons,
};
use dyno_term::benchmark::vram_tier::{DatacenterTier, VramAllocationPlan};
use dyno_term::telemetry::HardwareStats;
use std::process::Command;

#[test]
fn test_vram_tiers_all_datacenter_hardware() {
    // 1. AMD Instinct MI50 (16 GB)
    let plan_mi50 = VramAllocationPlan::compute_from_live(16_384, 16_384, 15_000_000_000);
    assert_eq!(plan_mi50.tier, DatacenterTier::Mi50_16Gb);
    assert!(plan_mi50.is_clamped);
    assert_eq!(plan_mi50.allocated_bytes, 16_384 * 1024 * 1024 * 75 / 100);
    assert!(plan_mi50.allocation_ratio <= 0.75);

    // 2. AMD Instinct MI100 (32 GB)
    let plan_mi100 = VramAllocationPlan::compute_from_live(32_768, 32_768, 20_000_000_000);
    assert_eq!(plan_mi100.tier, DatacenterTier::Mi100_32Gb);
    assert!(!plan_mi100.is_clamped);
    assert_eq!(plan_mi100.allocated_bytes, 20_000_000_000);

    // 3. NVIDIA Hopper H100 SXM (80 GB)
    let plan_h100 = VramAllocationPlan::compute_from_live(81_920, 81_920, 100_000_000_000);
    assert_eq!(plan_h100.tier, DatacenterTier::H100_80Gb);
    assert!(plan_h100.is_clamped);
    assert_eq!(plan_h100.allocated_bytes, 81_920 * 1024 * 1024 * 75 / 100);
    assert!(plan_h100.allocation_ratio <= 0.75);

    // 4. NVIDIA Hopper H100 NVL (94 GB)
    let plan_h100nvl = VramAllocationPlan::compute_from_live(96_256, 96_256, 100_000_000_000);
    assert_eq!(plan_h100nvl.tier, DatacenterTier::H100Nvl_94Gb);
    assert!(plan_h100nvl.is_clamped);
    assert_eq!(
        plan_h100nvl.allocated_bytes,
        96_256 * 1024 * 1024 * 75 / 100
    );
    assert!(plan_h100nvl.allocation_ratio <= 0.75);

    // 5. NVIDIA Hopper H200 NVL (141 GB)
    let plan_h200 = VramAllocationPlan::compute_from_live(144_384, 144_384, 120_000_000_000);
    assert_eq!(plan_h200.tier, DatacenterTier::H200_141Gb);
    assert!(plan_h200.is_clamped);
    assert_eq!(plan_h200.allocated_bytes, 144_384 * 1024 * 1024 * 75 / 100);
    assert!(plan_h200.allocation_ratio <= 0.75);
}

#[test]
fn test_vram_never_exceeds_90pct_of_live_free_memory() {
    // 80 GB card with only 10 GB free memory remaining
    let plan = VramAllocationPlan::compute_from_live(81_920, 10_240, 50_000_000_000);
    assert!(plan.is_clamped);
    let max_90_free = 10_240 * 1024 * 1024 * 90 / 100;
    assert_eq!(plan.allocated_bytes, max_90_free);
    assert!(plan.allocation_ratio_free <= 0.90);
}

#[test]
fn test_thermal_guard_trip_debouncing() {
    let mut failsafe = ThermalFailsafe::new(105.0, true).with_gpu_index(2);
    let high_temp_stats = HardwareStats {
        gpu_temp_c: 80.0,
        gpu_hotspot_c: 106.0, // > 105.0 ceiling
        ..Default::default()
    };
    let normal_temp_stats = HardwareStats {
        gpu_temp_c: 70.0,
        gpu_hotspot_c: 85.0,
        ..Default::default()
    };

    // Case 1: Intermittent spike (Sample 1 breach, Sample 2 normal, Sample 3 breach)
    assert_eq!(failsafe.check(&high_temp_stats), None); // Sample 1
    assert_eq!(failsafe.check(&normal_temp_stats), None); // Reset debounce
    assert_eq!(failsafe.check(&high_temp_stats), None); // Sample 1 again (not tripped!)

    // Reset failsafe
    let mut failsafe2 = ThermalFailsafe::new(105.0, true).with_gpu_index(1);

    // Case 2: 3 consecutive samples over ceiling -> trips
    assert_eq!(failsafe2.check(&high_temp_stats), None); // Sample 1
    assert_eq!(failsafe2.check(&high_temp_stats), None); // Sample 2
    let trip = failsafe2
        .check(&high_temp_stats)
        .expect("Must trip on 3rd consecutive breach");

    assert_eq!(trip.gpu_index, 1);
    assert_eq!(trip.sensor_type, "junction");
    assert_eq!(trip.current_temp_c, 106.0);
    assert_eq!(trip.ceiling_temp_c, 105.0);
}

#[test]
fn test_thermal_guard_datacenter_preset_ceilings() {
    let mut failsafe = ThermalFailsafe::new(105.0, true).with_datacenter_preset(true);

    // In datacenter preset:
    // Junction ceiling is min(max_temp_c, 95.0, crit - 5.0) -> 95.0°C
    let stats = HardwareStats {
        gpu_temp_c: 80.0,
        gpu_hotspot_c: 96.0,
        gpu_hotspot_crit_c: 110.0,
        ..Default::default()
    };

    assert_eq!(failsafe.check(&stats), None);
    assert_eq!(failsafe.check(&stats), None);
    let trip = failsafe
        .check(&stats)
        .expect("Must trip on junction >= 95.0 in datacenter preset");
    assert_eq!(trip.ceiling_temp_c, 95.0);
}

#[test]
fn test_passively_cooled_board_never_trips_fan_stall() {
    // Passively cooled cards (H200, H100, MI100, MI50) have fan_rpm = None
    let mut failsafe = ThermalFailsafe::new(105.0, true);
    let passive_stats = HardwareStats {
        gpu_temp_c: 85.0, // High heat
        gpu_hotspot_c: 90.0,
        gpu_fan_rpm: None,
        gpu_fan_pwm: None,
        ..Default::default()
    };

    // Run 100 consecutive frames: must never trip fan stall
    for _ in 0..100 {
        assert!(
            failsafe.check(&passive_stats).is_none(),
            "Passively cooled boards with absent fan sensors must never trip fan stall"
        );
    }
}

#[test]
fn test_vendor_thresholds_table_and_nvml_decoding() {
    let amd = VendorThresholds::for_vendor_name("AMD Radeon Instinct MI100");
    assert_eq!(amd.max_die_delta_c, 28.0);
    assert_eq!(amd.power_cap_ratio, 0.98);

    let nvidia = VendorThresholds::for_vendor_name("NVIDIA H200 NVL");
    assert_eq!(nvidia.max_die_delta_c, 25.0);
    assert_eq!(nvidia.power_cap_ratio, 0.98);

    // Decode NVML bitmask: 0x04 (sw_power_cap) | 0x40 (hw_thermal_slowdown)
    let reasons = decode_nvml_throttle_reasons(0x04 | 0x40);
    assert!(reasons.contains(&"sw_power_cap".to_string()));
    assert!(reasons.contains(&"hw_thermal_slowdown".to_string()));

    // Assessment is advisory and produces notes
    let stats = HardwareStats {
        gpu_name: "NVIDIA H100".to_string(),
        nvml_throttle_reasons: Some(0x04 | 0x40),
        gpu_power_cap_w: 700.0,
        ..Default::default()
    };
    let assessment = GpuTelemetryAssessment::evaluate(&[], &stats);
    assert!(assessment.throttling_detected);
    assert!(
        assessment
            .advisory_flags
            .iter()
            .any(|f| f.contains("NVML Throttle Reason: hw_thermal_slowdown"))
    );
}

#[test]
fn test_cli_preset_datacenter_records() {
    let output = Command::new(env!("CARGO_BIN_EXE_dyno_term"))
        .args([
            "--preset",
            "datacenter",
            "--benchmark",
            "--headless",
            "--output",
            "jsonl",
            "--frames",
            "1",
            "--gpus",
            "0",
        ])
        .output()
        .expect("Execute dyno_term with datacenter preset");

    assert!(
        output.status.success(),
        "Command failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut found_start_preset = false;
    let mut found_summary_preset = false;

    for line in stdout.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if v.get("type").and_then(|t| t.as_str()) == Some("start") {
                assert_eq!(v.get("preset").and_then(|p| p.as_str()), Some("datacenter"));
                found_start_preset = true;
            }
            if v.get("type").and_then(|t| t.as_str()) == Some("summary") {
                assert_eq!(v.get("preset").and_then(|p| p.as_str()), Some("datacenter"));
                found_summary_preset = true;
            }
        }
    }

    assert!(
        found_start_preset,
        "Start record must contain preset: datacenter"
    );
    assert!(
        found_summary_preset,
        "Summary record must contain preset: datacenter"
    );
}

#[test]
fn test_cli_invalid_preset_exit_code_2() {
    let output = Command::new(env!("CARGO_BIN_EXE_dyno_term"))
        .args([
            "--preset",
            "non_existent_preset",
            "--benchmark",
            "--headless",
            "--output",
            "jsonl",
        ])
        .output()
        .expect("Execute dyno_term with invalid preset");

    assert_eq!(output.status.code(), Some(2));
}
