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
        • UNIFIED_SHADER WGSL structure & 5 GPU acts verified
        • DOWNSAMPLE_SHADER WGSL structure & workgroups verified
        • Uniforms (48B, 16B aligned) and DownsampleUniforms (16B) layout verified
  [3/6] GPU Hardware & Compute Pipeline            [ PASS ]
        • Found 2 GPU adapter(s)
        • Active GPU: AMD Radeon RX 6650 XT (RADV NAVI23) | Vulkan | DiscreteGpu
        • Pipeline instantiated @ 320x180 (57600 pixels)
        • All 5 GPU shader scenes (Waves, Mandelbulb, Black Hole, Tornado, Cardiac) compute & downsampling verified
  [4/6] Telemetry & Hardware Sensors               [ PASS ]
        • Detected 16 system CPU threads
        • CPU Model: 12th Gen Intel(R) Core(TM) i5-12600K
        • /proc/stat active: total ticks 100294573, idle ticks 98607848
        • TelemetryCollector background sampling worker verified
  [5/6] Benchmark Engine & Scorer Math             [ PASS ]
        • BenchmarkScorer peak metric tracking verified
        • Scene result scoring & FPS statistics verified (score: 1382400)
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
| **2. WGSL Shaders** | `src/shaders/mod.rs` | Token validation for `UNIFIED_SHADER` (Acts 0–5) and `DOWNSAMPLE_SHADER`; memory alignment validation (`Uniforms` size 48 bytes, 16-byte aligned; `DownsampleUniforms` size 16 bytes). |
| **3. GPU Pipeline** | `src/gpu/` | Hardware adapter enumeration (`GpuContext::list_adapters()`); pipeline compilation at 320×180; headless compute dispatches across all 6 GPU Acts (Waves, Mandelbulb, Black Hole, Tornado, Cardiac, LBM Vortex); VRAM bilinear downsampling verification. |
| **4. Telemetry** | `src/telemetry.rs` | CPU available parallelism; Linux `/proc/stat` total and idle tick parsing; CPU brand model reader; background sampling thread start and clean drop. |
| **5. Benchmark Math** | `src/benchmark/mod.rs` | `BenchmarkScorer` peak sensor tracking (temp, hotspot, util, power, clock, VRAM); frametime accumulation; average FPS and 1% low calculations; `BenchmarkReport` JSON serialization round-trip. |
| **6. Terminal Engine** | `src/terminal/` | Resolution tag formatting (`16k`, `8k`, `4k`, `1440p`, `1080p`, `720p`); Act mathematical descriptions; ANSI escape sequence stripping (`visible_len`); half-block TrueColor renderer initialization. |

---

## 3. Automated Developer Test Suite (`cargo test`)

The automated test suite contains **44 tests** spanning unit, integration, and CLI domains:

```bash
cargo test --all-targets -- --nocapture
```

### Test Coverage Matrix:

| Target | Module | Test Count | Description |
| :--- | :--- | :--- | :--- |
| `src/lib.rs` | `benchmark::tests` | 7 | Empty result handling, peak tracking, frametime & 1% low percentiles, scene weighting formulas, JSON report round-trip, report file writing, history and baseline workflow. |
| `src/lib.rs` | `benchmark::consistency::tests` | 4 | StatSummary calculations (mean, standard deviation, variance, range, CV%), empty stat protection, consistency rating thresholds, JSON round-trip save/load, and ANSI scorecard generation. |
| `src/lib.rs` | `benchmark::telemetry::tests` | 4 | SensorMetric min/max/avg/p95 calculations, empty metric fallbacks, automated GPU sensor assessment (thermal delta, headroom, fan PWM, power cap), and thermal throttling heuristic detection. |
| `src/lib.rs` | `cpu::tests` | 4 | Initialization particle distribution, multithreaded step determinism, absence of NaNs/Infs, preview rasterization, buffer bounds safety. |
| `src/lib.rs` | `shaders::tests` | 3 | Unified shader WGSL structure, downsampler WGSL structure, std140 uniform memory struct alignment and size checks. |
| `src/lib.rs` | `telemetry::tests` | 5 | HardwareStats defaults, CPU thread detection, CPU brand string reading, `/proc/stat` format, TelemetryCollector worker lifecycle. |
| `src/lib.rs` | `terminal::tests` | 7 | Half-block downsampling identity and downscaling, renderer initialization, HUD width boundaries, scene math models, resolution presets, ANSI string visible length. |
| `tests/cli_tests.rs` | CLI Integration | 9 | CLI default argument parsing, `-t`/`--test`/`--self-test` flags, `--benchmark` flags, `-c`/`--consistency`/`--stability`/`--cooldown` flags, `--demo` flags, `--gpu` and `--list-gpus` flags, `--history`/`--set-baseline`/`--compare`, custom width/height, resolution preset parsing. |
| `tests/pipeline_integration.rs` | GPU Integration | 1 | Headless GPU device creation, compute pipeline initialization, compute pass dispatch, and staging buffer readback. |

---

## 4. CI/CD & Headless Environments

In environments without a physical discrete GPU (such as Docker containers or standard CI/CD runners):
- **CPU Vulkan Fallback**: If `llvmpipe` (Vulkan Software Rasterizer) is installed, GPU tests execute smoothly on the CPU.
- **Graceful Handling**: Integration tests in `tests/pipeline_integration.rs` gracefully detect and skip if no Vulkan/Metal graphics driver is present.
- **Pure Unit Tests**: All CPU physics, shader syntax, telemetry, benchmark math, and CLI parsing tests run with zero GPU dependencies.
