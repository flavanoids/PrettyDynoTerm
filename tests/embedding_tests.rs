use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::process::Command;

use dyno_term::cli::parse_gpu_selection;
use dyno_term::output::VersionMetadata;

#[test]
fn test_version_metadata_struct() {
    let meta = VersionMetadata::current();
    assert!(!meta.version.is_empty());
    assert!(!meta.git_commit.is_empty());
    assert_eq!(meta.embedded_kernels.cuda_ptx_sm90_sha256.len(), 64);
    assert_eq!(meta.embedded_kernels.hip_co_gfx_sha256.len(), 64);
    assert_eq!(meta.embedded_kernels.wgsl_shader_sha256.len(), 64);
    assert!(meta.supported_backends.contains(&"wgpu".to_string()));
    assert!(meta.supported_backends.contains(&"cuda".to_string()));
    assert!(meta.supported_backends.contains(&"hip".to_string()));
}

#[test]
fn test_version_flag_json_output() {
    let bin_path = env!("CARGO_BIN_EXE_dyno_term");
    let output = Command::new(bin_path)
        .args(["--version", "--output", "json"])
        .output()
        .expect("Failed to run dyno_term --version --output json");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("Valid utf-8 stdout");
    let parsed: VersionMetadata =
        serde_json::from_str(&stdout).expect("Valid VersionMetadata JSON");

    assert_eq!(parsed.version, env!("CARGO_PKG_VERSION"));
    assert_eq!(parsed.embedded_kernels.cuda_ptx_sm90_sha256.len(), 64);
    assert_eq!(parsed.embedded_kernels.hip_co_gfx_sha256.len(), 64);
    assert_eq!(parsed.embedded_kernels.wgsl_shader_sha256.len(), 64);
}

#[test]
fn test_parse_gpu_selection_visible_devices_env() {
    // 1. Without env vars: --gpus all selects all adapters
    unsafe {
        std::env::remove_var("CUDA_VISIBLE_DEVICES");
        std::env::remove_var("HIP_VISIBLE_DEVICES");
    }
    let all = parse_gpu_selection("all", None, 4).expect("Parse all");
    assert_eq!(all, vec![0, 1, 2, 3]);

    // 2. CUDA_VISIBLE_DEVICES filters --gpus all
    unsafe {
        std::env::set_var("CUDA_VISIBLE_DEVICES", "0,2");
    }
    let filtered = parse_gpu_selection("all", None, 4).expect("Parse with CUDA_VISIBLE_DEVICES");
    assert_eq!(filtered, vec![0, 2]);

    // 3. HIP_VISIBLE_DEVICES when CUDA not set
    unsafe {
        std::env::remove_var("CUDA_VISIBLE_DEVICES");
        std::env::set_var("HIP_VISIBLE_DEVICES", "1");
    }
    let hip_filtered = parse_gpu_selection("all", None, 4).expect("Parse with HIP_VISIBLE_DEVICES");
    assert_eq!(hip_filtered, vec![1]);

    // 4. Masking all devices returns Err
    unsafe {
        std::env::set_var("CUDA_VISIBLE_DEVICES", "");
    }
    let masked = parse_gpu_selection("all", None, 4);
    assert!(masked.is_err());

    // 5. Explicit --gpus overrides visible devices env
    unsafe {
        std::env::set_var("CUDA_VISIBLE_DEVICES", "0");
    }
    let explicit = parse_gpu_selection("2,3", None, 4).expect("Explicit --gpus");
    assert_eq!(explicit, vec![2, 3]);

    // Clean up environment
    unsafe {
        std::env::remove_var("CUDA_VISIBLE_DEVICES");
        std::env::remove_var("HIP_VISIBLE_DEVICES");
    }
}

#[test]
fn test_read_only_cwd_execution() {
    let bin_path = env!("CARGO_BIN_EXE_dyno_term");
    let test_ro_dir = std::env::temp_dir().join(format!("dyno_ro_cwd_{}", std::process::id()));
    fs::create_dir_all(&test_ro_dir).expect("Create test ro dir");

    // Set permissions to read-only (0o555 = r-xr-xr-x)
    let ro_perms = fs::Permissions::from_mode(0o555);
    fs::set_permissions(&test_ro_dir, ro_perms).expect("Set read-only permissions");

    let output = Command::new(bin_path)
        .current_dir(&test_ro_dir)
        .args([
            "--headless",
            "--benchmark",
            "--gpus",
            "0",
            "--res",
            "720p",
            "--frames",
            "5",
            "--output",
            "jsonl",
        ])
        .output()
        .expect("Execute dyno_term from read-only CWD");

    assert!(
        output.status.success(),
        "Process should succeed with exit code 0"
    );

    // Verify zero files created in test_ro_dir
    let entries: Vec<_> = fs::read_dir(&test_ro_dir)
        .expect("Read dir")
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(
        entries.len(),
        0,
        "Read-only CWD must remain pristine with 0 files created"
    );

    // Clean up
    let rw_perms = fs::Permissions::from_mode(0o777);
    let _ = fs::set_permissions(&test_ro_dir, rw_perms);
    let _ = fs::remove_dir_all(&test_ro_dir);
}

#[test]
fn test_report_dir_isolated_output() {
    let bin_path = env!("CARGO_BIN_EXE_dyno_term");
    let test_dir = std::env::temp_dir().join(format!("dyno_report_dir_{}", std::process::id()));
    let _ = fs::remove_dir_all(&test_dir);

    let output = Command::new(bin_path)
        .args([
            "--headless",
            "--benchmark",
            "--gpus",
            "0",
            "--res",
            "720p",
            "--frames",
            "5",
            "--report-dir",
            test_dir.to_str().unwrap(),
            "--output",
            "jsonl",
        ])
        .output()
        .expect("Execute dyno_term with --report-dir");

    assert!(output.status.success());
    let expected_report = test_dir.join("dyno_benchmark_results.json");
    assert!(
        expected_report.exists(),
        "Report must be created in target --report-dir"
    );

    let content = fs::read_to_string(&expected_report).expect("Read report");
    let val: serde_json::Value = serde_json::from_str(&content).expect("Valid JSON");
    assert!(val.get("overall_score").is_some());

    let _ = fs::remove_dir_all(&test_dir);
}

#[test]
fn test_duration_flag_execution() {
    let bin_path = env!("CARGO_BIN_EXE_dyno_term");
    let output = Command::new(bin_path)
        .args([
            "--headless",
            "--benchmark",
            "--gpus",
            "0",
            "--res",
            "720p",
            "--duration",
            "1",
            "--output",
            "jsonl",
        ])
        .output()
        .expect("Execute dyno_term with --duration");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("Valid utf-8 stdout");

    let mut found_summary = false;
    for line in stdout.lines() {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(line)
            && val.get("type").and_then(|t| t.as_str()) == Some("summary")
        {
            found_summary = true;
            let elapsed = val
                .get("elapsed_seconds")
                .and_then(|e| e.as_f64())
                .unwrap_or(0.0);
            assert!(
                (0.8..=3.0).contains(&elapsed),
                "Duration should be close to 1 second, got: {}",
                elapsed
            );
            assert_eq!(val.get("status").and_then(|s| s.as_str()), Some("ok"));
        }
    }
    assert!(found_summary, "Must emit summary record upon completion");
}
