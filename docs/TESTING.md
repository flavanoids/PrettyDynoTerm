# Testing & Verification Guide: `dyno_term`

`dyno_term` incorporates a comprehensive testing and diagnostics architecture ensuring mathematical correctness, GPU compute stability, sensor reliability, and determinism across all supported platforms.

---

## 1. Quick Start

### Run the CLI Self-Test (Binary Diagnostics)
Run diagnostics directly from the compiled binary:
```bash
./target/release/dyno_term --test

# Short & alias forms
./target/release/dyno_term -t
./target/release/dyno_term --self-test
```

### Run Developer Unit & Integration Tests
Run all Cargo unit tests, CLI tests, and pipeline integration tests:
```bash
cargo test --all-targets
```

### Run Everything (One-Step Script)
Run both the Cargo automated test suite and the binary self-test with one command:
```bash
./scripts/run_tests.sh
```

---

## 2. CLI Self-Test Architecture (`--test` / `--self-test`)

When `-t, --test` is provided, `dyno_term` runs in non-interactive stdout mode, executing 6 sequential subsystem verification suites and exiting with `0` (success) or `1` (failure):

```
================================================================================
          DYNO_TERM SYSTEM SELF-TEST & HARDWARE VERIFICATION SUITE              
================================================================================
  [1/6] CPU Simulation & N-Body Physics Engine     [ PASS ]
        • Initialized 2048 particles across dual colliding galaxies
        • Multi-threaded Rayon physics integration verified (0 NaNs/Infs)
        • Preview frame buffer rasterizer & color blending verified
  [2/6] WGSL Compute Shaders & Uniform Layouts     [ PASS ]
        • UNIFIED_SHADER WGSL structure & 7 GPU acts verified
        • DOWNSAMPLE_SHADER WGSL structure & workgroups verified
        • Uniforms (48B, 16B aligned) and DownsampleUniforms (16B) layout verified
  [3/6] GPU Hardware & Compute Pipeline            [ PASS ]
        • Found 2 GPU adapter(s)
        • Active GPU: AMD Radeon RX 6650 XT (RADV NAVI23) | Vulkan | DiscreteGpu
        • Pipeline instantiated @ 320x180 (57600 pixels)
        • All 7 GPU shader scenes (Waves, Mandelbulb, Black Hole, Tornado, Cardiac, LBM Vortex, Kinematics) compute & downsampling verified
  [4/6] Telemetry & Hardware Sensors               [ PASS ]
        • Detected 16 system CPU threads
        • CPU Model: 12th Gen Intel(R) Core(TM) i5-12600K
        • /proc/stat active: total ticks 100294573, idle ticks 98607848
        • TelemetryCollector background sampling worker verified
  [5/6] Benchmark Engine & Scorer Math             [ PASS ]
        • BenchmarkScorer peak metric tracking verified
        • Scene result scoring & FPS statistics verified (score: 6.91)
        • BenchmarkReport JSON serialization round-trip verified
  [6/6] Terminal Engine & ANSI Renderer            [ PASS ]
        • Resolution formatting preset tags verified
        • Act math model descriptions (Gerstner, Mandelbulb, Schwarzschild, etc.) verified
        • ANSI escape sequence width stripping verified
        • TerminalRenderer half-block compositor instantiated
================================================================================
  RESULT: ALL 6 TEST SUITES PASSED (0 failures, completed in 102.79ms)
================================================================================
```

### Self-Test Suite Breakdown:

| Suite | Component | Checks Performed |
| :--- | :--- | :--- |
| **1. CPU Simulation** | `src/cpu/mod.rs` | 2,048 particles seeded into two colliding spiral galaxies; Rayon multithreaded force accumulation step; asserts non-zero displacement, finite coordinates ($x, y, z, v_x, v_y, v_z \neq \text{NaN}, \infty$); preview buffer rasterization and color grading. |
| **2. WGSL Shaders** | `src/shaders/mod.rs` | Token validation for `UNIFIED_SHADER` (Acts 0–6) and `DOWNSAMPLE_SHADER`; memory alignment validation (`Uniforms` size 48 bytes, 16-byte aligned; `DownsampleUniforms` size 16 bytes). |
| **3. GPU Pipeline** | `src/gpu/` | Hardware adapter enumeration (`GpuContext::list_adapters()`); pipeline compilation at 320×180; headless compute dispatches across all GPU Acts (Waves, Mandelbulb, Black Hole, Tornado, Cardiac, LBM Vortex, Kinematics); VRAM bilinear downsampling verification. |
| **4. Telemetry** | `src/telemetry/` | CPU available parallelism; Linux `/proc/stat` total and idle tick parsing; CPU brand model reader; background sampling thread start and clean drop. |
| **5. Benchmark Math** | `src/benchmark/mod.rs` | `BenchmarkScorer` peak sensor tracking (temp, hotspot, util, power, clock, VRAM); frametime accumulation; average FPS and 1% low calculations; `BenchmarkReport` JSON serialization round-trip. |
| **6. Terminal Engine** | `src/terminal/` | Resolution tag formatting (`16k`, `8k`, `4k`, `1440p`, `1080p`, `720p`); Act mathematical descriptions; ANSI escape sequence stripping (`visible_len`); half-block TrueColor renderer initialization. |

---

## 3. Automated Developer Test Suite (`cargo test`)

The automated test harness spans comprehensive unit, integration, CLI, and datacenter qualification domains:

```bash
cargo test --all-targets -- --nocapture
```

### Test Coverage Architecture:

| Target | Category | Description |
| :--- | :--- | :--- |
| `src/lib.rs` | Unit Tests | Empty result handling, peak tracking, frametime & 1% low percentiles, scene weighting formulas, JSON report round-trip, report file writing, history and baseline workflow, TFLOPS, bandwidth & energy metric calculation, StatSummary calculations (mean, standard deviation, variance, CV%), consistency rating thresholds, sensor assessment (thermal delta, headroom, fan PWM, power cap), thermal throttling heuristics, thermal failsafe tripwire, fan stall breach, hotspot breach, edge breach, VRAM headroom OOB/OOM check, dynamic VRAM adaptation scaling, particle simulation determinism, shader syntax, and terminal rendering. |
| `tests/backend_tests.rs` | Backend & Kernel Blobs | Dynamic dlopen dispatch for CUDA (`libcuda.so.1`) and HIP (`libamdhip64.so`), fallback code 3 exits, and embedded PTX / CO blob integrity. |
| `tests/cli_tests.rs` | CLI Integration | CLI argument parsing, `-t`/`--test`/`--self-test`, `--benchmark`, consistency flags, `--demo`, `--gpu`, `--list-gpus`, `--history`/`--set-baseline`/`--compare`, custom width/height, resolution presets. |
| `tests/datacenter_preset_tests.rs` | Datacenter Qualifications | Datacenter preset validation, thermal ceilings, debouncing, 90% live-free VRAM ceiling, and multi-tier VRAM scaling. |
| `tests/embedding_tests.rs` | Orchestrator Embedding | Version metadata JSON introspection, visible device environment filtering (`CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`), read-only CWD guarantee, report directory isolation, and duration flags. |
| `tests/exit_code_tests.rs` | System Contract & Signals | Process exit codes `0` through `5`, SIGINT/SIGTERM 2-second watchdog handling, clean JSONL stream output. |
| `tests/multi_gpu_tests.rs` | Fleet & Multi-Device | Multi-GPU thread barrier synchronization, concurrent interleaving, and fleet peer outlier detection. |
| `tests/pipeline_integration.rs` | GPU Compute Pipeline | Headless GPU device creation, compute pipeline initialization, compute pass dispatch, staging buffer readback, dynamic VRAM adaptation, OOB and OOM failsafes, headroom guard rejection. |
| `tests/telemetry_tests.rs` | Hardware Sensors | Sysfs DRM enumeration, hwmon parsing, NVML dlopen decoding, and PCI bus ID normalization. |

---

## 4. CI/CD & Headless Environments

In environments without a physical discrete GPU (such as Docker containers or standard CI/CD runners):
- **CPU Vulkan Fallback**: If `llvmpipe` (Vulkan Software Rasterizer) is installed, GPU tests execute smoothly on the CPU.
- **Graceful Handling**: Integration tests in `tests/pipeline_integration.rs` gracefully detect and skip if no Vulkan/Metal graphics driver is present.
- **Pure Unit Tests**: All CPU physics, shader syntax, telemetry, benchmark math, and CLI parsing tests run with zero GPU dependencies.
