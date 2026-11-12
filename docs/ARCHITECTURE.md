# Architecture & Design: `dyno_term`

`dyno_term` is a high-performance, cross-platform terminal benchmark and artistic demoparty engine. It bridges mathematical graphics with rigorous hardware evaluation, running natively on **Linux and macOS** without requiring a windowing system (X11, Wayland, Quartz), making it capable of running both on desktop workstations and over headless SSH sessions.

---

## 1. System Philosophy

Traditional PC benchmarks (such as 3DMark or Cinebench) require heavy GUI runtimes, native windowing surfaces, and multi-gigabyte asset downloads. Conversely, terminal tools are typically either text-based monitors (`btop`, `glances`) or demoscene ASCII toys (`asciiquarium`, `cmatrix`).

`dyno_term` bridges this gap with three design pillars:
1. **Zero-Dependency Native Binary**: Compiled to a standalone Rust binary using WebGPU (`wgpu`) compute pipelines.
2. **Aesthetic Mathematical Complexity**: Graphics are procedurally generated in real time using 3D raymarching, fractal calculus, and relativistic curved spacetime geodesics—no external textures, meshes, or assets required.
3. **Deterministic Standardized Workload**: Fixed mathematical operations per frame, calibrated to produce a universal baseline score regardless of device form factor.

---

## 2. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 OFFSCREEN COMPUTE PIPELINE                  │
│                                                             │
│   [ Canonical Resolution: 16K (15360x8640) / 4K / 1080p ]   │
│                              │                              │
│      WGSL Compute Shader (Acts I-VI) or Rayon (Act VII)     │
│                              │                              │
│                    GPU Storage Buffer                       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                [ Asynchronous Readback & DMA ]
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                PRESENTATION & DOWNSAMPLING                  │
│                                                             │
│   Box-Filter Downsample -> Terminal Cell Aspect Ratio       │
│                              │                              │
│   ANSI TrueColor Half-Block Encoding ('▀' U+2280)          │
│                              │                              │
│   Cybernetic Telemetry Frame (Clock, Temp, Power, VRAM)     │
│                              │                              │
│   Throttled 60 FPS Presentation Buffer (Zero stdout stall)  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Decoupling Compute from Presentation

A central technical obstacle in terminal graphics is the **terminal emulator I/O bottleneck**:
- Even high-performance GPU-accelerated terminal emulators (`alacritty`, `kitty`, `wezterm`, `foot`) choke if saturated with hundreds of thousands of ANSI escape codes per second.
- If a GPU benchmark pushes 1,000 frames per second directly to `stdout`, the CPU will peg at 100% simply decoding terminal escape sequences, and you end up benchmarking terminal string parsing rather than hardware compute.

### The Solution:
1. **Offscreen Compute Target**: The GPU renders into an offscreen storage buffer at canonical benchmark resolutions (16K UHD, 4K UHD, 1080p).
2. **Decoupled Terminal Pacing**: In benchmark mode, the GPU computes frames uncapped at full hardware speed, while the terminal preview is throttled to **60 FPS** (updating every 16.6 ms).
3. **Differential ANSI Encoding**: The terminal renderer caches previous foreground/background RGB values and only emits escape codes when adjacent pixel colors transition, reducing `stdout` bandwidth by up to 70%.

---

## 4. Execution Modes

| Feature | Benchmark Mode (`--benchmark`) | Interactive Demo Mode (`--demo`) | Self-Test Mode (`--test` / `--self-test`) |
| :--- | :--- | :--- | :--- |
| **Pacing** | Uncapped GPU/CPU duty cycle | 60 FPS V-Sync locked (or `--uncapped`) | Instantaneous automated diagnostic pass |
| **Compute Resolution** | Fixed canonical preset (Default: 4K / 16K) | Dynamic or fixed preset | Test resolution (320×180 downscaled to 80×40) |
| **Duration** | Finite frame count (Default: 100 frames) | Continuous until `[Q]` or `[ESC]` | Automated (< 250ms completion) |
| **Controls** | Background run with telemetry progress | `[SPACE]` switch scenes, `[U]` toggle uncap | None (Non-interactive CLI stdout report) |
| **Output** | Final Scorecard Badge & JSON report | Real-time interactive demoparty visual | 6-Tier Subsystem verification matrix with exit code |

---

## 5. Verification & Testing Architecture

To guarantee hardware stability and deterministic benchmark scoring across diverse platforms, `dyno_term` implements a dual-layer testing infrastructure:

1. **Hardware & Subsystem Self-Test (`dyno_term --test`)**:
   - Built directly into the compiled binary so end users, sysadmins, and automated benchmarks can verify hardware compute capabilities before launching intensive benchmarks.
   - Executes across 6 core subsystems:
     - **CPU Physics**: Validates 2,048-particle dual-spiral galaxy initialization, Rayon multi-threaded force integration (0 NaNs/Infs), and preview rasterization.
     - **WGSL Shaders**: Validates `UNIFIED_SHADER` and `DOWNSAMPLE_SHADER` syntax, workgroup boundaries, and std140 uniform buffer struct alignments (48-byte layout, 16-byte alignment).
     - **GPU Compute Pipeline**: Tests adapter selection, pipeline instantiation, compute dispatches across all 6 GPU acts, and VRAM downsampling readback.
     - **Telemetry Sensors**: Tests CPU thread discovery, `/proc/stat` delta calculations, and GPU sysfs hwmon probes.
     - **Benchmark Math & Scorer**: Tests frametime accumulation, 1% low calculations, arithmetic scene weighting, and JSON report round-tripping.
     - **Terminal Engine**: Tests resolution tags, ANSI sequence stripping, and half-block TrueColor rasterizer initialization.
2. **Automated Developer Test Harness (`cargo test`)**:
   - Comprehensive automated unit and integration test suites covering unit logic (`src/lib.rs`), CLI parsing (`tests/cli_tests.rs`), multi-GPU concurrency (`tests/multi_gpu_tests.rs`), telemetry integration (`tests/telemetry_tests.rs`), exit codes (`tests/exit_code_tests.rs`), and GPU context execution (`tests/pipeline_integration.rs`).
   - Run in one step via `./scripts/run_tests.sh`.

---

## 6. Dynamic VRAM Adaptation & Headroom Failsafes

To reliably support GPUs spanning 4GB/8GB entry-level discrete hardware up to 200GB+ datacenter accelerators (e.g. NVIDIA H100/H200, AMD Instinct MI300X), `dyno_term` implements a dynamic memory adaptation and failsafe system:

1. **Hardware Memory Querying (`GpuMemoryInfo`)**:
   - Detects physical device VRAM limits via `wgpu::Limits` and Linux DRM `mem_info_vram_total` / `mem_info_vram_used` sysfs nodes.
   - Categorizes hardware into test tiers (`VramTestTier`): `Light`, `Balanced`, `Stress`, and `MaxSafe`.
2. **Pre-flight Headroom Verification (`VramHeadroomCheck`)**:
   - Evaluates memory demand prior to allocating storage buffers or dispatching compute workgroups.
   - Computes expected memory footprints against active available headroom with a safety margin (defaulting to 256 MB or 15% of free memory).
   - Rejects unsupportable allocations cleanly rather than provoking native device crashes or OS OOM kills.
3. **Out-of-Bounds & Out-of-Memory Safeguards**:
   - GPU `ComputePipelineManager::try_resize` inspects dimension validity (`width > 0 && height > 0`) and buffer byte limits before reallocating.
   - CPU `NBodySimulation` enforces zero-particle early returns and clamps massive particle counts safely.
   - Terminal rasterizer guards against zero-dimension slices and mismatched buffer strides.

---

## 7. Decoupled Runner & Backend Architecture

To eliminate the "god-file" anti-pattern and maintain clean modular separation of concerns, execution and compute backends are decoupled into dedicated modules:

```
src/
├── main.rs                   # CLI entrypoint, argument parsing, & signal/headless dispatcher
├── backend/                  # Dynamic ComputeBackend trait (CUDA, HIP, WGPU)
│   ├── mod.rs                # Backend dispatch & autodetection
│   ├── traits.rs             # ComputeBackend & ComputeDevice abstractions
│   ├── cuda.rs               # Dynamic libcuda.so.1 dlopen & PTX dispatch
│   ├── hip.rs                # Dynamic libamdhip64.so dlopen & CO dispatch
│   └── wgpu_backend.rs       # Cross-platform WebGPU compute pipeline
├── runner/                   # Execution runners & workload supervisors
│   ├── mod.rs                # Runner interface & re-exports
│   ├── session.rs            # TerminalSession context & option structs
│   ├── demo.rs               # Interactive demo loop & event processing
│   ├── benchmark.rs          # Finite deterministic benchmark pass execution & scorecards
│   ├── consistency.rs        # Multi-iteration consistency suite & thermal cooldown screens
│   ├── multi_gpu.rs          # Concurrent multi-device orchestration & thread barriers
│   ├── outliers.rs           # Multi-GPU fleet outlier detection
│   └── capture.rs            # Headless screenshot rendering & PPM output
├── telemetry/                # Hardware sensor layer keyed by PCI bus ID
│   ├── mod.rs                # TelemetryCollector supervisor
│   ├── nvml.rs               # Dynamic libnvidia-ml.so.1 dlopen
│   ├── amd_sysfs.rs          # Linux DRM sysfs & hwmon polling
│   └── pci.rs                # PCI bus ID canonical normalization
```

By packaging session dependencies (`pipeline`, `renderer`, `telemetry`, `adapter_info`, `term_width`, `term_height`, `res_tag`) into `TerminalSession<'a>` and encapsulating workload settings into typed options (`DemoOptions`, `BenchmarkOptions`, `ConsistencyOptions`), runner functions maintain low cyclomatic complexity, clear boundaries, and strict compliance with zero-warning lint standards.

---

## 8. First-Principles Physics & Microarchitectural Memory Staging

In accordance with modern GPU architecture design patterns:
1. **Workgroup Shared Memory (LDS / `var<workgroup>`)**:
   - `DOWNSAMPLE_SHADER` utilizes a $16 \times 16$ workgroup-shared tile cache (`var<workgroup> tile_cache: array<array<u32, 16>, 16>`) synchronized with `workgroupBarrier()`. This stages global memory reads cooperatively across all 256 invocations before writing preview pixels, testing Local Data Share bandwidth.
2. **First-Principles Wave Dispersion & Spacetime Curvature**:
   - Act I strictly follows the hydrodynamic gravity-capillary wave dispersion equation $\omega^2 = (g k + \frac{\gamma}{\rho} k^3) \tanh(k h)$.
   - Act III implements Boyer-Lindquist Kerr horizon metrics ($\Sigma = r^2 + a^2 \cos^2\theta$) and curvature-adaptive step sizing ($\Delta t(r)$) to eliminate numerical event horizon leakage.
   - Act IV models Poisson dry adiabatic potential temperature ($\theta = T_0 (P/P_0)^{0.286}$).
   - Act V models myocardial electrical anisotropy with a $4:1$ conductivity tensor ratio ($\mathbf{D} = \sigma_t \mathbf{I} + (\sigma_l - \sigma_t)\vec{f}\vec{f}^T$).
   - Act VIII executes 2nd-order Symplectic Leapfrog (Velocity Verlet) time integration on the CPU.
3. **Explicit Silicon Metrics**:
   - Benchmark scorecards and telemetry JSON logs report **Estimated TFLOPS**, **Effective VRAM Bandwidth (GB/s)**, and **Energy Efficiency (Joules per Mray)** directly.

---

See [docs/TESTING.md](TESTING.md) for full testing documentation, [docs/SHADERS_AND_SCENES.md](SHADERS_AND_SCENES.md) for complete mathematical formulations, and [docs/SCENE_SPECIFICATION_STANDARD.md](SCENE_SPECIFICATION_STANDARD.md) for the mandatory engineering pipeline for contributing new acts and tests.
