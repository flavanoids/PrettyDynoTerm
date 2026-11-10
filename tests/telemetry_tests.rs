use dyno_term::benchmark::failsafe::ThermalFailsafe;
use dyno_term::telemetry::TelemetryCollector;
use dyno_term::telemetry::amd_sysfs::AmdSysfsTelemetry;
use dyno_term::telemetry::nvml::{MockNvml, NVML_CLOCKS_THROTTLE_REASON_SW_POWER_CAP, NvmlClient};
use dyno_term::telemetry::pci::PciBusId;
use dyno_term::telemetry::types::{HardwareStats, PerGpuTelemetry};
use std::path::Path;
use std::sync::Arc;

#[test]
fn test_sysfs_rdna2_with_fans() {
    let fixture_path = Path::new("tests/fixtures/sysfs/rdna2_fan");
    let reader = AmdSysfsTelemetry::new(fixture_path);
    let cards = reader.query_all_cards();

    assert_eq!(
        cards.len(),
        1,
        "Expected exactly 1 card in rdna2_fan fixture"
    );
    let card = &cards[0];

    // Canonical PCI Bus ID
    assert_eq!(card.pci_bus_id, PciBusId::new("0000:03:00.0"));

    // Silicon Temps & Crits
    assert_eq!(card.edge_temp_c, Some(55.0));
    assert_eq!(card.edge_temp_crit_c, Some(100.0));
    assert_eq!(card.junction_temp_c, Some(68.0));
    assert_eq!(card.junction_temp_crit_c, Some(110.0));
    assert_eq!(card.mem_temp_c, None); // Not exposed on this consumer card

    // Power & Clocks
    assert_eq!(card.power_w, Some(130.0));
    assert_eq!(card.power_cap_w, Some(180.0));
    assert_eq!(card.clock_mhz, Some(2400));
    assert_eq!(card.mem_clock_mhz, Some(1000));

    // Utilization & VRAM
    assert_eq!(card.gpu_util_pct, Some(45.0));
    assert_eq!(card.mem_util_pct, Some(32.0));
    assert_eq!(card.vram_used_mb, Some(2048));
    assert_eq!(card.vram_total_mb, Some(12288));
    assert_eq!(card.vram_vendor.as_deref(), Some("Samsung"));

    // Active Fans Present
    assert_eq!(card.fan_rpm, Some(1450));
    assert_eq!(card.fan_max_rpm, Some(3200));
    assert_eq!(card.fan_pwm, Some(115));
    assert!(card.has_fans());
}

#[test]
fn test_sysfs_mi100_passive_no_fan() {
    let fixture_path = Path::new("tests/fixtures/sysfs/mi100_passive");
    let reader = AmdSysfsTelemetry::new(fixture_path);
    let cards = reader.query_all_cards();

    assert_eq!(
        cards.len(),
        1,
        "Expected exactly 1 card in mi100_passive fixture"
    );
    let card = &cards[0];

    // Canonical PCI Bus ID
    assert_eq!(card.pci_bus_id, PciBusId::new("0000:41:00.0"));

    // 3 Temperature Labels: edge, junction, mem
    assert_eq!(card.edge_temp_c, Some(62.0));
    assert_eq!(card.edge_temp_crit_c, Some(95.0));
    assert_eq!(card.junction_temp_c, Some(79.0));
    assert_eq!(card.junction_temp_crit_c, Some(105.0));
    assert_eq!(card.mem_temp_c, Some(72.0));
    assert_eq!(card.mem_temp_crit_c, Some(100.0));

    // Power & Clocks
    assert_eq!(card.power_w, Some(285.0));
    assert_eq!(card.power_cap_w, Some(300.0));
    assert_eq!(card.clock_mhz, Some(1502));
    assert_eq!(card.mem_clock_mhz, Some(1200));

    // Utilization & 32GB HBM2 VRAM
    assert_eq!(card.gpu_util_pct, Some(88.0));
    assert_eq!(card.mem_util_pct, Some(65.0));
    assert_eq!(card.vram_used_mb, Some(16384));
    assert_eq!(card.vram_total_mb, Some(32768));
    assert_eq!(card.vram_vendor.as_deref(), Some("Hynix"));

    // Passively Cooled: fan fields must be strictly None (never 0)
    assert_eq!(card.fan_rpm, None);
    assert_eq!(card.fan_max_rpm, None);
    assert_eq!(card.fan_pwm, None);
    assert!(!card.has_fans());

    // Verify Thermal Failsafe behavior on passively cooled card
    let stats = HardwareStats::from_per_gpu(card, "EPYC".to_string(), 10.0, 64);
    let mut failsafe = ThermalFailsafe::new(105.0, true);

    // Repeated checks at high temp must never trigger a false fan-stall trip
    for _ in 0..50 {
        assert!(
            failsafe.check(&stats).is_none(),
            "Passively cooled card must never trigger fan stall failsafe"
        );
    }
}

#[test]
fn test_sysfs_node_8gpu_enumeration() {
    let fixture_path = Path::new("tests/fixtures/sysfs/node_8gpu");
    let reader = AmdSysfsTelemetry::new(fixture_path);
    let cards = reader.query_all_cards();

    assert_eq!(
        cards.len(),
        8,
        "Must discover all 8 GPUs on the node (no 0..4 loop limit!)"
    );

    let expected_pci_slots = [
        "0000:01:00.0",
        "0000:02:00.0",
        "0000:03:00.0",
        "0000:04:00.0",
        "0000:05:00.0",
        "0000:06:00.0",
        "0000:07:00.0",
        "0000:08:00.0",
    ];

    for (i, card) in cards.iter().enumerate() {
        assert_eq!(
            card.pci_bus_id.as_str(),
            expected_pci_slots[i],
            "PCI bus ID mismatch at index {}",
            i
        );

        // Verify metrics are unique per card and not contaminated by card0
        let expected_edge = 45.0 + (i as f32 * 2.0);
        let expected_junction = 55.0 + (i as f32 * 2.5);
        let expected_power = 120.0 + (i as f32 * 10.0);

        assert_eq!(card.edge_temp_c, Some(expected_edge));
        assert_eq!(card.junction_temp_c, Some(expected_junction));
        assert_eq!(card.power_w, Some(expected_power));
    }
}

#[test]
fn test_mock_nvml_datacenter_h200() {
    let h200 = PerGpuTelemetry {
        pci_bus_id: PciBusId::new("0000:1a:00.0"),
        gpu_name: "NVIDIA H200 NVL".to_string(),
        driver_version: Some("575.57.08".to_string()),
        edge_temp_c: Some(58.0),
        junction_temp_c: None, // NVML reports single GPU core temperature
        mem_temp_c: Some(65.0),
        edge_temp_crit_c: None,
        junction_temp_crit_c: None,
        mem_temp_crit_c: None,
        power_w: Some(380.0),
        power_cap_w: Some(700.0),
        clock_mhz: Some(1980),
        mem_clock_mhz: Some(3200),
        gpu_util_pct: Some(99.0),
        mem_util_pct: Some(85.0),
        vram_used_mb: Some(42000),
        vram_total_mb: Some(144384), // 141 GB HBM3e
        vram_vendor: Some("HBM3e".to_string()),
        pcie_link: Some("PCIe 5.0 (32.0 GT/s, x16)".to_string()),
        fan_rpm: None, // Passively cooled server board
        fan_max_rpm: None,
        fan_pwm: None,
        nvml_slowdown_temp_c: Some(92.0),
        nvml_shutdown_temp_c: Some(98.0),
        nvml_throttle_reasons: Some(NVML_CLOCKS_THROTTLE_REASON_SW_POWER_CAP),
    };

    let mock_nvml = MockNvml::new(vec![h200]);
    assert!(mock_nvml.is_available());

    let gpus = mock_nvml.query_all_gpus().expect("Query mock NVML GPUs");
    assert_eq!(gpus.len(), 1);
    let gpu = &gpus[0];

    assert_eq!(gpu.gpu_name, "NVIDIA H200 NVL");
    assert_eq!(gpu.pci_bus_id, PciBusId::new("0000:1a:00.0"));
    assert_eq!(gpu.vram_total_mb, Some(144384));
    assert_eq!(gpu.nvml_slowdown_temp_c, Some(92.0));
    assert_eq!(
        gpu.nvml_throttle_reasons,
        Some(NVML_CLOCKS_THROTTLE_REASON_SW_POWER_CAP)
    );
    assert!(!gpu.has_fans());
}

#[test]
fn test_mock_nvml_unavailable_handling() {
    let mock = MockNvml::unavailable();
    assert!(!mock.is_available());
    assert!(mock.query_all_gpus().is_err());
}

#[test]
fn test_telemetry_collector_pci_bus_id_keying() {
    let sysfs = AmdSysfsTelemetry::new(Path::new("tests/fixtures/sysfs/node_8gpu"));
    let mock_nvml = Arc::new(MockNvml::unavailable());

    let collector =
        TelemetryCollector::with_sysfs_and_nvml("AMD Node".to_string(), sysfs, mock_nvml);

    // Allow worker thread to perform initial sampling sweep
    std::thread::sleep(std::time::Duration::from_millis(250));

    // Query card 0 specifically by PCI ID
    let card0_stats = collector
        .sample_by_pci(&PciBusId::new("0000:01:00.0"))
        .expect("Card 0 must exist in collector map");
    assert_eq!(card0_stats.gpu_temp_c, 45.0);

    // Query card 4 specifically by PCI ID
    let card4_stats = collector
        .sample_by_pci(&PciBusId::new("0000:05:00.0"))
        .expect("Card 4 must exist in collector map");
    assert_eq!(card4_stats.gpu_temp_c, 53.0);

    // Query non-existent card
    assert!(
        collector
            .sample_by_pci(&PciBusId::new("0000:99:00.0"))
            .is_none()
    );

    // Query all cards
    let all = collector.all_gpus();
    assert_eq!(all.len(), 8);
}
