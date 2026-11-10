use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

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
fn test_exit_code_0_ok() {
    let bin = get_bin_path();
    let output = Command::new(&bin)
        .arg("--list-gpus")
        .output()
        .expect("Failed to execute process");

    assert_eq!(
        output.status.code(),
        Some(0),
        "Exit code 0 expected for --list-gpus"
    );
}

#[test]
fn test_exit_code_1_workload_failure() {
    let bin = get_bin_path();
    // Setting max-temp to 1.0 ensures immediate thermal tripwire activation
    let output = Command::new(&bin)
        .args([
            "--headless",
            "--output",
            "jsonl",
            "--benchmark",
            "--frames",
            "5",
            "--max-temp",
            "1.0",
        ])
        .output()
        .expect("Failed to execute process");

    assert_eq!(
        output.status.code(),
        Some(1),
        "Exit code 1 expected on thermal tripwire breach"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("\"type\":\"error\""),
        "JSONL stream should emit error record on workload failure"
    );
    assert!(
        stdout.contains("\"type\":\"thermal_trip\""),
        "JSONL stream should emit thermal_trip record on thermal tripwire breach"
    );
}

#[test]
fn test_exit_code_2_invalid_args() {
    let bin = get_bin_path();

    // 1. Headless without runnable mode flag
    let output = Command::new(&bin)
        .arg("--headless")
        .output()
        .expect("Failed to execute process");

    assert_eq!(
        output.status.code(),
        Some(2),
        "Exit code 2 expected when headless invoked without mode"
    );

    // 2. Invalid value for output format
    let output2 = Command::new(&bin)
        .args(["--output", "invalid_format"])
        .output()
        .expect("Failed to execute process");

    assert_eq!(
        output2.status.code(),
        Some(2),
        "Exit code 2 expected on invalid CLI option value"
    );
}

#[test]
fn test_exit_code_3_no_usable_gpu() {
    let bin = get_bin_path();
    let output = Command::new(&bin)
        .args(["--headless", "--benchmark", "--gpu", "99999"])
        .output()
        .expect("Failed to execute process");

    assert_eq!(
        output.status.code(),
        Some(3),
        "Exit code 3 expected when GPU index is out of bounds or missing"
    );
}

#[test]
fn test_exit_code_4_cancelled_on_sigint() {
    let bin = get_bin_path();
    let mut child = Command::new(&bin)
        .args([
            "--headless",
            "--output",
            "jsonl",
            "--benchmark",
            "--frames",
            "500",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn process");

    let pid = child.id() as i32;

    // Allow process to start up and enter benchmark
    thread::sleep(Duration::from_millis(300));

    // Send SIGINT signal
    unsafe {
        libc::kill(pid, libc::SIGINT);
    }

    let status = child.wait().expect("Failed to wait on child");
    assert_eq!(
        status.code(),
        Some(4),
        "Exit code 4 expected on SIGINT cancellation"
    );

    let mut stdout = String::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_string(&mut stdout);
    }

    assert!(
        stdout.contains("\"type\":\"cancelled\""),
        "JSONL output must contain cancelled record upon signal interception. Output was: {}",
        stdout
    );
}

#[test]
fn test_exit_code_5_internal_error() {
    let bin = get_bin_path();
    let output = Command::new(&bin)
        .arg("--test-internal-error")
        .output()
        .expect("Failed to execute process");

    assert_eq!(
        output.status.code(),
        Some(5),
        "Exit code 5 expected on internal error flag"
    );
}

#[test]
fn test_headless_jsonl_output_cleanliness_and_schema() {
    let bin = get_bin_path();
    let output = Command::new(&bin)
        .args([
            "--headless",
            "--output",
            "jsonl",
            "--benchmark",
            "--frames",
            "2",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("Failed to execute process");

    assert_eq!(output.status.code(), Some(0));

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Assert zero ANSI escape sequences
    assert!(
        !stdout.contains("\x1b["),
        "Headless output must not contain ANSI escape sequences!"
    );

    let lines: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
    assert!(!lines.is_empty(), "Must emit JSONL lines");

    let mut has_start = false;
    let mut has_progress = false;
    let mut has_gpu_result = false;
    let mut has_summary = false;

    for line in lines {
        let val: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("Invalid JSON line: {} ({})", line, e));

        assert_eq!(
            val.get("schema_version").and_then(|v| v.as_str()),
            Some("1.0.0"),
            "Every line must contain schema_version: 1.0.0"
        );

        match val.get("type").and_then(|v| v.as_str()) {
            Some("start") => has_start = true,
            Some("progress") => has_progress = true,
            Some("gpu_result") => has_gpu_result = true,
            Some("summary") => has_summary = true,
            other => panic!("Unexpected record type: {:?}", other),
        }
    }

    assert!(has_start, "Missing start record");
    assert!(has_progress, "Missing progress record");
    assert!(has_gpu_result, "Missing gpu_result record");
    assert!(has_summary, "Missing summary record");
}
