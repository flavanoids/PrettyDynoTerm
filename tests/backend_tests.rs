use std::process::Command;

use dyno_term::backend::WgpuBackend;
use dyno_term::backend::cuda::EMBEDDED_PTX_SM90;
use dyno_term::backend::hip::EMBEDDED_FATBIN_GFX;
use dyno_term::backend::traits::{BackendError, BackendType, ComputeBackend, ComputeDevice};
use dyno_term::gpu::pipeline::Uniforms;
use dyno_term::telemetry::PciBusId;

fn get_bin_path() -> std::path::PathBuf {
    let mut path = std::env::current_exe().expect("current_exe");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push("dyno_term");
    path
}

#[test]
fn test_backend_cuda_absent_exits_code_3() {
    let bin = get_bin_path();
    let output = Command::new(&bin)
        .args([
            "--headless",
            "--output",
            "jsonl",
            "--benchmark",
            "--backend",
            "cuda",
            "--frames",
            "1",
        ])
        .output()
        .expect("Failed to execute process");

    assert_eq!(
        output.status.code(),
        Some(3),
        "Exit code 3 expected when --backend cuda is requested on a system without libcuda.so.1"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("libcuda.so.1"),
        "Error message should mention missing libcuda.so.1"
    );
    assert!(
        stdout.contains("\"error_type\":\"NoUsableGpu\""),
        "Error type should be NoUsableGpu"
    );
}

#[test]
fn test_backend_hip_absent_exits_code_3() {
    let bin = get_bin_path();
    let output = Command::new(&bin)
        .args([
            "--headless",
            "--output",
            "jsonl",
            "--benchmark",
            "--backend",
            "hip",
            "--frames",
            "1",
        ])
        .output()
        .expect("Failed to execute process");

    // If ROCm is not installed on the dev box, it exits with 3
    assert_eq!(
        output.status.code(),
        Some(3),
        "Exit code 3 expected when --backend hip is requested on a system without libamdhip64.so.6"
    );
}

#[test]
fn test_backend_wgpu_cli_run() {
    let bin = get_bin_path();
    let output = Command::new(&bin)
        .args([
            "--headless",
            "--output",
            "jsonl",
            "--benchmark",
            "--backend",
            "wgpu",
            "--gpus",
            "0",
            "--frames",
            "2",
        ])
        .output()
        .expect("Failed to execute process");

    assert_eq!(
        output.status.code(),
        Some(0),
        "--backend wgpu should execute successfully on dev box"
    );
}

#[test]
fn test_backend_auto_cli_run() {
    let bin = get_bin_path();
    let output = Command::new(&bin)
        .args([
            "--headless",
            "--output",
            "jsonl",
            "--benchmark",
            "--backend",
            "auto",
            "--gpus",
            "0",
            "--frames",
            "2",
        ])
        .output()
        .expect("Failed to execute process");

    assert_eq!(
        output.status.code(),
        Some(0),
        "--backend auto should select wgpu and succeed on dev box"
    );
}

#[test]
fn test_embedded_kernel_blobs_present_and_valid() {
    // 1. Validate Hopper sm_90 PTX blob
    assert!(
        !EMBEDDED_PTX_SM90.is_empty(),
        "Embedded sm_90 PTX blob must not be empty"
    );
    let ptx_str = std::str::from_utf8(EMBEDDED_PTX_SM90).expect("PTX must be valid UTF-8 text");
    assert!(
        ptx_str.contains(".version"),
        "PTX must declare .version directive"
    );
    assert!(
        ptx_str.contains(".target sm_90"),
        "PTX must target sm_90 architecture"
    );
    assert!(
        ptx_str.contains("dyno_unified_compute"),
        "PTX must define dyno_unified_compute entry point"
    );

    // 2. Validate AMD CDNA Fatbin / Code Object blob
    assert!(
        !EMBEDDED_FATBIN_GFX.is_empty(),
        "Embedded CDNA fatbin must not be empty"
    );
    assert_eq!(
        &EMBEDDED_FATBIN_GFX[0..4],
        b"\x7fELF",
        "AMDGPU code object must be a valid ELF binary"
    );
    let found_symbol = EMBEDDED_FATBIN_GFX
        .windows(b"dyno_unified_compute".len())
        .any(|window| window == b"dyno_unified_compute");
    assert!(
        found_symbol,
        "AMDGPU code object must contain dyno_unified_compute symbol"
    );
}

#[test]
fn test_wgpu_backend_direct_dispatch() {
    let backend = WgpuBackend::new();
    assert!(
        backend.is_available(),
        "WgpuBackend should be available on dev box"
    );

    let pci_ids = backend
        .enumerate_device_pci_ids()
        .expect("enumerate_device_pci_ids");
    assert!(!pci_ids.is_empty());

    let mut device = backend
        .create_device(0, 160, 100)
        .expect("create_device failed");
    assert_eq!(device.index(), 0);
    assert_eq!(device.backend_type(), BackendType::Wgpu);

    device
        .allocate_storage_buffer(160, 100)
        .expect("allocate_storage_buffer");

    let uniforms = Uniforms {
        time: 1.0,
        frame: 1,
        width: 160,
        height: 100,
        cam_pos: [0.0, 1.8, -4.5],
        scene_id: 1, // Mandelbulb
        cam_dir: [0.0, -0.2, 1.0],
        benchmark_mode: 1,
    };

    device.dispatch(&uniforms).expect("dispatch");
    device.synchronize().expect("synchronize");
    let checksum = device.readback_checksum().expect("readback_checksum");
    // Checksum readback succeeds and produces a valid 64-bit value
    let _ = checksum;
}

/// Mock test verifying abstract ComputeDevice lifecycle and trait contract.
#[test]
fn test_mock_device_dispatch_contract() {
    struct MockDevice {
        name: String,
        pci: PciBusId,
        backend: BackendType,
        dispatched: bool,
    }

    impl ComputeDevice for MockDevice {
        fn index(&self) -> usize {
            0
        }
        fn name(&self) -> &str {
            &self.name
        }
        fn pci_bus_id(&self) -> &PciBusId {
            &self.pci
        }
        fn backend_type(&self) -> BackendType {
            self.backend
        }
        fn allocate_storage_buffer(&mut self, _w: u32, _h: u32) -> Result<(), BackendError> {
            Ok(())
        }
        fn dispatch(&mut self, _u: &Uniforms) -> Result<(), BackendError> {
            self.dispatched = true;
            Ok(())
        }
        fn readback_checksum(&mut self) -> Result<u64, BackendError> {
            Ok(0x1234_5678_9abc_def0)
        }
        fn synchronize(&self) -> Result<(), BackendError> {
            Ok(())
        }
    }

    let mut mock_cuda = MockDevice {
        name: "Mock NVIDIA H200 NVL".to_string(),
        pci: PciBusId::new("0000:01:00.0"),
        backend: BackendType::Cuda,
        dispatched: false,
    };

    assert_eq!(mock_cuda.backend_type(), BackendType::Cuda);
    mock_cuda.allocate_storage_buffer(3840, 2160).unwrap();
    let uniforms = Uniforms {
        time: 0.5,
        frame: 10,
        width: 3840,
        height: 2160,
        cam_pos: [0.0, 0.0, 0.0],
        scene_id: 2,
        cam_dir: [0.0, 0.0, 1.0],
        benchmark_mode: 1,
    };
    mock_cuda.dispatch(&uniforms).unwrap();
    assert!(mock_cuda.dispatched);
    let checksum = mock_cuda.readback_checksum().unwrap();
    assert_eq!(checksum, 0x1234_5678_9abc_def0);
    mock_cuda.synchronize().unwrap();
}
