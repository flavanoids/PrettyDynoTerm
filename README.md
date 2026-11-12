# PrettyDynoTerm

> **Cross-Platform Terminal GPU & CPU Benchmark Engine & Datacenter Compute Stress Harness**  
> High-performance 16K/4K offscreen 3D raymarching, fractal calculus, relativistic astrophysics, atmospheric vortex dynamics, and cardiac electrophysiology rendered directly inside your terminal in 24-bit TrueColor or streamed headlessly as JSON/JSONL.

<p align="center">
  <img src="assets/screenshots/menu.png" alt="PrettyDynoTerm Interactive CLI Launcher" width="780" />
</p>

---

## Technical Documentation & Architecture

- **[docs/SHADERS_AND_SCENES.md](docs/SHADERS_AND_SCENES.md)**: Complete mathematical formulations, partial differential equations, WGSL compute dispatch mechanics, input/output specifications, and scientific use cases for every scene.
- **[docs/OUTPUT_SCHEMA.md](docs/OUTPUT_SCHEMA.md)**: Full JSON Lines (JSONL) schema specification, stream event types (`start`, `progress`, `gpu_result`, `summary`, `error`, `cancelled`), and Linux exit codes.
- **[docs/EMBEDDING.md](docs/EMBEDDING.md)**: Official host integration contract covering non-interactive headless guarantees, 2-second clean signal teardown, read-only working directory isolation, version introspection, and device masking.
- **[docs/SCENE_SPECIFICATION_STANDARD.md](docs/SCENE_SPECIFICATION_STANDARD.md)**: Mandatory 7-stage specification pipeline for engineering new benchmark scenes.

---

## Benchmark Acts & Mathematical Models

### Act I: Atmospheric Ocean Waves (GPU)
14 Gerstner wave octaves, trochoidal crests, 48-step adiabatic cumulus cloud deck, Clausius-Clapeyron saturation vapor thermodynamics, and Henyey-Greenstein Mie radiative transfer.
<p align="center">
  <img src="assets/screenshots/act1_ocean_waves.png" alt="Act I: Atmospheric Ocean Waves" width="850" />
</p>

---

### Act II: The Mandelbulb Core (GPU)
Infinite recursive 3D fractal caverns ($z \mapsto z^8 + c$) testing raw GPU arithmetic logic and warp/wavefront branch divergence across 160 raymarch steps and 24 polynomial iterations.
<p align="center">
  <img src="assets/screenshots/act2_mandelbulb.png" alt="Act II: The Mandelbulb Core" width="850" />
</p>

---

### Act III: Relativistic Black Hole (GPU)
80 4th-order Runge-Kutta (RK4) curved spacetime geodesic integration steps around a spinning Kerr black hole with frame-dragging, gravitational redshift, and Novikov-Thorne accretion disk.
<p align="center">
  <img src="assets/screenshots/act3_black_hole.png" alt="Act III: Relativistic Black Hole" width="850" />
</p>

---

### Act IV: Tornadic Mesocyclone (GPU - Weather Model)
Navier-Stokes Burgers-Rott vortex dynamics ($v_\theta = \frac{\Gamma}{2\pi r}(1 - e^{-r^2/r_c^2})$), cyclostrophic pressure deficit balance ($\frac{dp}{dr} = \rho \frac{v_\theta^2}{r}$), adiabatic condensation funnel cloud, and surface debris boundary friction layer.
<p align="center">
  <img src="assets/screenshots/act4_tornado.png" alt="Act IV: Tornadic Mesocyclone" width="850" />
</p>

---

### Act V: Cardiac Electrophysiology (GPU - Medical Model)
Aliev-Panfilov 2-variable reaction-diffusion system ($\partial u / \partial t$, $\partial v / \partial t$) simulating 3D scroll-wave re-entry rotors (ventricular fibrillation) across a thick-walled left ventricle with $120^\circ$ transmural myocardial fiber pitch and optical mapping voltage fluorescence.
<p align="center">
  <img src="assets/screenshots/act5_cardiac.png" alt="Act V: Cardiac Electrophysiology" width="850" />
</p>

---

### Act VI: Lattice Boltzmann Vortex Street (GPU - CFD Fluid Model)
Lattice Boltzmann Method (LBM D2Q9) incompressible Navier-Stokes simulation evaluating 9 discrete velocity distribution streams ($c_0 \dots c_8$), BGK collision relaxation, 4-tap central difference vorticity curl ($\omega = \nabla \times \vec{u}$), and bluff body aerodynamic wake shedding forming an unsteady von Kármán Vortex Street.
<p align="center">
  <img src="assets/screenshots/act6_lbm_vortex.png" alt="Act VI: Lattice Boltzmann Vortex Street" width="850" />
</p>

---

### Act VII: Cybernetic Gait Kinematics (GPU - 3D Character Animation)
Hierarchical forward kinematics bipedal gait simulation evaluating 16-joint articulated transformation trees, smooth minimum polynomial ($smin$) muscle contraction bulging, metallic servo reflection, and real-time planar raymarching floor reflections.
<p align="center">
  <img src="assets/screenshots/act7_kinematics.png" alt="Act VII: Cybernetic Gait Kinematics" width="850" />
</p>

---

### Act VIII: Kinetic N-Body Supernova (CPU Rayon)
32,000 gravitational celestial bodies simulated in parallel across all CPU cores/threads via Rayon, with collision dynamics, tidal tail dispersion, and velocity-based relativistic thermal heat gradient coloring.
<p align="center">
  <img src="assets/screenshots/act8_nbody.png" alt="Act VIII: Kinetic N-Body Supernova" width="850" />
</p>

---

## Core Engine Capabilities

- **High-Performance 16K / 4K UHD Offscreen Compute**:
  - Internal compute engine supporting resolutions up to **16K Extreme UHD (15360×8640 = 132.7 Mpix/frame)**.
  - VRAM-native GPU compute downsampling to terminal character grids using UTF-8 half-blocks (`▀`), providing extreme supersampling anti-aliasing (SSAA).
  - In headless mode, PCIe readback is bypassed in favor of synchronous 8-byte GPU frame checksums, eliminating bus readback bottlenecks at massive resolutions.
- **Concurrent Multi-GPU Scaling & Peer Outlier Detection**:
  - Spawns independent worker threads and separate GPU device/queue contexts per physical adapter (`--gpus all` or `--gpus 0,2`).
  - Thread barriers ensure coordinated launch across all devices without serializing compute dispatch.
  - Automated fleet outlier detection identifies individual GPUs deviating $>5\%$ in throughput (slower) or $>8^\circ\text{C}$ in operating temperature (hotter) relative to peer medians.
- **Pluggable Compute Backends (CUDA, HIP, wgpu)**:
  - Runtime dynamic library loading (`dlopen`) for `libvulkan.so.1`, `libcuda.so.1`, `libnvidia-ml.so.1`, and `libamdhip64.so.6`.
  - **Zero build-time linking** against vendor SDKs; starts cleanly on any node and gracefully degrades if a driver backend is absent.
  - Pre-compiled embedded blobs for NVIDIA Hopper (`sm_90` PTX) and AMD Instinct (`gfx906`, `gfx908`, `gfx90a`, `gfx942` fat binaries) compiled from a unified C++ kernel source.
- **Datacenter Preset & VRAM Headroom Protection**:
  - `--preset datacenter`: Automatically targets 16K resolution and validates memory headroom against physical tiers (16 GB, 32 GB, 80 GB, 94 GB, 141 GB).
  - Enforces strict memory limits: buffer allocations must not exceed $\le 75\%$ total VRAM and $\le 90\%$ live free VRAM.
  - Debounced emergency thermal failsafe (3-sample breach window) with vendor-tailored ceilings for edge, junction/hotspot, and HBM memory.
  - Passive cooling awareness: boards without fan tachometers (e.g. H200 NVL, MI100) automatically skip fan-stall tripwires without false positives.
- **Headless Streaming & Host Supervisor Integration**:
  - Non-interactive headless execution (`--headless`) works seamlessly over SSH without a pseudo-terminal (TTY) or when launched by systemd / daemon supervisors.
  - Streams real-time JSON Lines (`--output jsonl`) emitting progress records ($\ge 1\text{/s}$ per GPU), hardware telemetry snapshots, and overall summary records.
  - Clean signal teardown: intercepts `SIGINT` / `SIGTERM`, frees GPU allocations, emits a final `cancelled` JSONL record, and exits with code `4` within $\le 2.0$ seconds (guaranteed by an independent background watchdog thread).
  - Strict read-only working directory isolation: when `--report-dir <DIR>` is omitted, the binary writes zero files to disk, running safely inside read-only containers (`chmod 555`).
- **Flexible Execution Modes (Duration vs Fixed Frames)**:
  - Deterministic fixed-frame benchmarking (`--frames N`, default 100).
  - Fixed-duration profiling (`--duration SECONDS`): repeatedly cycles scenes ($0 \to 7 \to 0 \dots$) until time expires, reporting frame throughput (FPS) and aggregate work completed.

<p align="center">
  <img src="assets/screenshots/scorecard.png" alt="PrettyDynoTerm Official Benchmark Scorecard" width="780" />
</p>

---

## Installation & Building

### 1. Standard Host Build
```bash
git clone git@github.com:flavanoids/PrettyDynoTerm.git
cd PrettyDynoTerm
cargo build --release
```
The compiled binary will be located at `./target/release/dyno_term`.

### 2. Datacenter & Enterprise Linux Container Build ($\le \text{GLIBC } 2.28$)
To build a portable binary that runs out-of-the-box on Debian 11 (glibc 2.31), Rocky Linux 8 (glibc 2.28), Ubuntu 20.04/22.04, and RHEL without host build dependencies:
```bash
./scripts/build_rocky8_binary.sh
```
This builds inside a Rocky Linux 8 container using Docker or Podman, producing `./dyno_term`.

### 3. Static MUSL Build & Zero-Dependency Distribution Packaging
To compile a fully static binary with zero glibc dependencies and package it into a self-contained tarball for fleet deployment (no Docker/Podman or Alpine required):
```bash
./scripts/package_release.sh
```
This produces:
- Standalone static binary: `./dyno_term` (and `./dist/dyno_term-v<VERSION>-linux-x86_64/dyno_term`, ~7.6 MB stripped)
- Fleet deployment tarball: `./dist/dyno_term-v<VERSION>-linux-x86_64.tar.gz` (~3.1 MB)
- Checksum: `./dist/dyno_term-v<VERSION>-linux-x86_64.tar.gz.sha256`

Or build manually via cargo:
```bash
cargo build --target x86_64-unknown-linux-musl --release
```

### 4. GLIBC Dependency Verification
Verify that your binary strictly satisfies datacenter glibc symbol limits ($\le 2.31$):
```bash
./scripts/check_glibc_symbols.sh dyno_term
```

---

## Usage

### 1. Interactive CLI Launcher Menu (Default)
Running the binary without subcommand flags opens the interactive terminal launcher:
```bash
./dyno_term
```
- **Menu Features**:
  - **Launch Interactive Demo**: 60 FPS smooth preview with live telemetry HUD.
  - **Launch 100% Uncapped Stress Test**: Uncapped frame pipeline to peg GPU to 100% power and boost clock.
  - **Run Full Benchmark Suite**: Deterministic workload scoring with instant scorecard report.
  - **Resolution Preset Switcher**: 16K, 8K, 4K UHD, 1440p, 1080p, 720p, Native Viewport, or Custom pixel dimensions.
  - **Switch GPU Adapter**: Switch WebGPU adapter between discrete GPUs, integrated GPUs, and CPU rasterizers.
  - **Live Hardware Telemetry**: 150ms real-time monitoring of GPU temp, board power draw, core clock, VRAM, and CPU load.

### 2. Headless Streaming Mode (`--headless --output jsonl`)
Ideal for automated test pipelines, daemon runners, and remote execution:
```bash
./dyno_term --headless --benchmark --gpus all --preset datacenter --output jsonl
```
Every event is emitted as a single line containing valid JSON and a `schema_version` field:
- `start`: Workload configuration and detected GPU hardware adapters.
- `progress`: Emitted $\ge 1\text{/s}$ per GPU with current scene, frame index, FPS, junction temperature, and board power.
- `gpu_result`: Per-GPU score, average FPS, 1% low frametime, scene breakdown, and peer outlier flags.
- `summary`: Overall node benchmark score, total elapsed time, active GPU count, and exit code.
- `cancelled`: Emitted upon `SIGINT` / `SIGTERM` before exiting with code `4`.

### 3. Multi-GPU Cluster Execution & Device Masking
Run concurrent workloads across all installed accelerators or target specific cards:
```bash
# Run on all detected discrete GPUs
./dyno_term --benchmark --gpus all

# Run on specific GPU ordinals (e.g. GPU 0 and GPU 2)
./dyno_term --benchmark --gpus 0,2

# Transparent environment masking (honours vendor isolation variables)
CUDA_VISIBLE_DEVICES=0,1 ./dyno_term --benchmark --gpus all
HIP_VISIBLE_DEVICES=2,3 ./dyno_term --benchmark --gpus all
```

### 4. Continuous Fixed Duration Stress Mode (`--duration SECONDS`)
Stress test accelerators for a fixed wall-clock time rather than a fixed frame count:
```bash
# Stress all GPUs continuously for 60 seconds repeating scenes
./dyno_term --headless --benchmark --gpus all --duration 60 --output jsonl

# 10-minute burn-in at 16K datacenter preset
./dyno_term --headless --benchmark --preset datacenter --duration 600 --output jsonl
```

### 5. Datacenter Node Smoke Test Script
Validate node runtime libraries, driver ICD registrations, introspection metadata, and run a 5-second sanity benchmark across all GPUs in one step:
```bash
./scripts/node_smoke.sh
```

### 6. Binary Introspection & Version Metadata (`--version --output json`)
Query binary version, git commit hash, supported backends, and SHA-256 cryptographic hashes of embedded compute kernels:
```bash
./dyno_term --version --output json
```
Example Output:
```json
{
  "version": "1.0.0",
  "git_commit": "133e902b4d1...",
  "embedded_kernels": {
    "cuda_ptx_sm90_sha256": "752449575a4c064f92f036a476beb93dbf353bc333c26d273a7b9f690941a57b",
    "hip_co_gfx_sha256": "ae1eae69ccd8159bb569df23a0d8b1d90384d9daf982c9912a417e55c5710b9a",
    "wgsl_shader_sha256": "051eee8f1bce54d8f81fae6b03140424daf14c619ec6b959ee8f844d734915e9"
  },
  "supported_backends": ["wgpu", "cuda", "hip"]
}
```

### 7. Direct Interactive Demo Mode
```bash
./dyno_term --demo
```
- `[SPACE]` : Switch between scenes
- `[U]` : Toggle between 60 FPS V-Sync and 100% Uncapped GPU Stress Mode
- `[Q]` or `[ESC]` : Exit

### 8. Resolution Presets (Direct CLI)
```bash
# Datacenter 16K Preset (15360x8640)
./dyno_term --preset datacenter

# 4K UHD (Default: 3840x2160)
./dyno_term --res 4k

# 8K UHD (7680x4320)
./dyno_term --res 8k

# 1440p QHD (2560x1440)
./dyno_term --res 1440p

# 1080p FHD (1920x1080)
./dyno_term --res 1080p

# 720p APU / Mobile (1280x720)
./dyno_term --res 720p
```

### 9. Comprehensive Self-Test & System Diagnostics
```bash
./dyno_term --test
```

### 10. Historical Runs & Reference Baseline Comparison
```bash
# View chronological history table
./dyno_term --history

# Pin latest run as reference baseline
./dyno_term --set-baseline

# Compare any report against reference baseline
./dyno_term --compare dyno_benchmark_results.json
```

### 11. Multi-Iteration Consistency & Thermal Stability Suite
```bash
# Run 5-iteration consistency suite with 60s cooldown pauses
./dyno_term --consistency

# 3 iterations with 30s cooldown
./dyno_term -c 3 --cooldown 30

# Back-to-back thermal heat-soak (0s cooldown)
./dyno_term -c 5 --cooldown 0
```

---

## Process Exit Codes

`dyno_term` returns standardized Linux exit codes:

| Exit Code | Identifier | Semantic Description |
| :---: | :--- | :--- |
| **`0`** | `Ok` | Workload executed and validated successfully. |
| **`1`** | `WorkloadFailure` | Workload failure (e.g. thermal tripwire breached, frame checksum mismatch). |
| **`2`** | `InvalidArgs` | Invalid command line flags or arguments. |
| **`3`** | `NoUsableGpu` | No compatible GPU adapter or backend found on system. |
| **`4`** | `Cancelled` | Execution terminated cleanly by `SIGINT` or `SIGTERM` within 2 seconds. |
| **`5`** | `InternalError` | Unexpected internal runtime panic or error. |

---

## Running Tests

To run all unit and integration test suites:
```bash
cargo test --all-targets
```

To run both the full Cargo test suite and binary self-tests in one step:
```bash
./scripts/run_tests.sh
```
