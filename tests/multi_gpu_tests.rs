use std::process::Command;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Duration;

use dyno_term::cli::parse_gpu_selection;
use dyno_term::output::ProgressRecord;
use dyno_term::runner::outliers::{GpuRunStats, detect_gpu_outliers};
use dyno_term::telemetry::PciBusId;

fn get_bin_path() -> std::path::PathBuf {
    let mut path = std::env::current_exe().expect("current_exe");
    path.pop(); // pop test exe name
    if path.ends_with("deps") {
        path.pop(); // pop deps
    }
    path.push("dyno_term");
    path
}

#[test]
fn test_cli_gpu_selection_parsing() {
    // "all" on 8 GPUs
    let all_8 = parse_gpu_selection("all", None, 8).unwrap();
    assert_eq!(all_8, vec![0, 1, 2, 3, 4, 5, 6, 7]);

    // "all" on single GPU dev box
    let all_1 = parse_gpu_selection("all", None, 1).unwrap();
    assert_eq!(all_1, vec![0]);

    // Comma-separated list with spaces
    let subset = parse_gpu_selection(" 0 , 2 , 5 ", None, 8).unwrap();
    assert_eq!(subset, vec![0, 2, 5]);

    // Single --gpu flag takes precedence
    let single_override = parse_gpu_selection("all", Some(4), 8).unwrap();
    assert_eq!(single_override, vec![4]);

    // Out of bounds error
    let err_oob = parse_gpu_selection("0,8", None, 8);
    assert!(err_oob.is_err());
    assert!(err_oob.unwrap_err().contains("out of bounds"));

    // Invalid string syntax
    let err_str = parse_gpu_selection("foo,bar", None, 8);
    assert!(err_str.is_err());
    assert!(err_str.unwrap_err().contains("Invalid GPU index"));
}

#[test]
fn test_mock_peer_outlier_detection_suite() {
    // Scenario 1: Solo GPU is never marked as an outlier
    let solo = vec![GpuRunStats {
        gpu_index: 0,
        pci_bus_id: PciBusId::new("0000:03:00.0"),
        gpu_name: "AMD RX 6650 XT".to_string(),
        duration_s: 15.0,
        score: 12000.0,
        avg_fps: 140.0,
        peak_temp_c: Some(65.0),
        peak_power_w: Some(130.0),
    }];
    let solo_flags = detect_gpu_outliers(&solo);
    assert_eq!(solo_flags.len(), 1);
    assert!(!solo_flags[0].is_outlier);

    // Scenario 2: 2 GPUs — GPU 1 is 10% slower than GPU 0
    let dual = vec![
        GpuRunStats {
            gpu_index: 0,
            pci_bus_id: PciBusId::new("0000:01:00.0"),
            gpu_name: "H200 NVL #0".to_string(),
            duration_s: 10.0,
            score: 50000.0,
            avg_fps: 500.0,
            peak_temp_c: Some(55.0),
            peak_power_w: Some(600.0),
        },
        GpuRunStats {
            gpu_index: 1,
            pci_bus_id: PciBusId::new("0000:02:00.0"),
            gpu_name: "H200 NVL #1".to_string(),
            duration_s: 11.0, // 10% slower (>5%)
            score: 45000.0,
            avg_fps: 450.0,
            peak_temp_c: Some(56.0),
            peak_power_w: Some(600.0),
        },
    ];
    let dual_flags = detect_gpu_outliers(&dual);
    assert_eq!(dual_flags.len(), 2);
    assert!(!dual_flags[0].is_outlier);
    assert!(dual_flags[1].is_outlier);
    assert!(dual_flags[1].reasons[0].contains(">5% slower"));

    // Scenario 3: 8 GPUs — GPU 3 is 9°C hotter than peers (>8°C)
    let mut fleet = Vec::new();
    for i in 0..8 {
        let temp = if i == 3 { 74.0 } else { 62.0 }; // +12°C above peers
        fleet.push(GpuRunStats {
            gpu_index: i,
            pci_bus_id: PciBusId::new(format!("0000:{:02x}:00.0", i + 1)),
            gpu_name: format!("MI100 #{}", i),
            duration_s: 20.0,
            score: 35000.0,
            avg_fps: 350.0,
            peak_temp_c: Some(temp),
            peak_power_w: Some(300.0),
        });
    }
    let fleet_flags = detect_gpu_outliers(&fleet);
    assert_eq!(fleet_flags.len(), 8);
    for (i, flag) in fleet_flags.iter().enumerate() {
        if i == 3 {
            assert!(flag.is_outlier);
            assert!(flag.reasons[0].contains(">8°C hotter"));
        } else {
            assert!(!flag.is_outlier);
        }
    }
}

#[test]
fn test_mock_multi_device_concurrency_and_interleaving() {
    const NUM_MOCK_GPUS: usize = 2;
    let barrier = Arc::new(Barrier::new(NUM_MOCK_GPUS));
    let progress_stream = Arc::new(Mutex::new(Vec::new()));

    let mut handles = Vec::new();

    for gpu_idx in 0..NUM_MOCK_GPUS {
        let b = Arc::clone(&barrier);
        let stream = Arc::clone(&progress_stream);

        handles.push(thread::spawn(move || {
            // Synchronize start
            b.wait();

            for scene_idx in 0..3 {
                // Synchronize each scene
                b.wait();

                let pci = format!("0000:{:02}:00.0", gpu_idx + 1);
                let prog = ProgressRecord::new(
                    scene_idx,
                    "TestScene",
                    10,
                    10,
                    200.0,
                    gpu_idx,
                    Some(60.0),
                    Some(250.0),
                )
                .with_pci_bus_id(&pci);

                {
                    let mut s = stream.lock().unwrap();
                    s.push(prog);
                }

                thread::sleep(Duration::from_millis(5));
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let records = progress_stream.lock().unwrap();
    assert_eq!(records.len(), NUM_MOCK_GPUS * 3);

    // Verify records from both GPUs are present
    let gpu_0_count = records.iter().filter(|r| r.gpu_index == 0).count();
    let gpu_1_count = records.iter().filter(|r| r.gpu_index == 1).count();
    assert_eq!(gpu_0_count, 3);
    assert_eq!(gpu_1_count, 3);

    // Verify PCI IDs are properly set
    for r in records.iter() {
        assert!(r.pci_bus_id.is_some());
    }
}

#[test]
fn test_real_hardware_multi_gpu_cli_run() {
    let bin = get_bin_path();
    let output = Command::new(&bin)
        .args([
            "--headless",
            "--output",
            "jsonl",
            "--benchmark",
            "--gpus",
            "all",
            "--frames",
            "5",
        ])
        .output()
        .expect("Failed to execute process");

    assert_eq!(
        output.status.code(),
        Some(0),
        "Exit code 0 expected for --gpus all"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.trim().lines().collect();
    assert!(!lines.is_empty(), "Output should not be empty");

    // First line must be start record
    let first_json: serde_json::Value = serde_json::from_str(lines[0]).expect("Valid start json");
    assert_eq!(first_json["type"], "start");
    assert!(first_json["gpus"].as_array().is_some());

    // Final line must be summary record with exit_code 0
    let last_json: serde_json::Value =
        serde_json::from_str(lines.last().unwrap()).expect("Valid summary json");
    assert_eq!(last_json["type"], "summary");
    assert_eq!(last_json["status"], "ok");
    assert_eq!(last_json["exit_code"], 0);

    // Verify gpu_result record has been emitted
    let has_gpu_result = lines.iter().any(|line| {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            v["type"] == "gpu_result"
        } else {
            false
        }
    });
    assert!(has_gpu_result, "Output must contain gpu_result record");
}

#[test]
fn test_real_hardware_single_gpu_explicit_selection() {
    let bin = get_bin_path();
    let output = Command::new(&bin)
        .args([
            "--headless",
            "--output",
            "jsonl",
            "--benchmark",
            "--gpus",
            "0",
            "--frames",
            "5",
        ])
        .output()
        .expect("Failed to execute process");

    assert_eq!(
        output.status.code(),
        Some(0),
        "Exit code 0 expected for --gpus 0"
    );
}
