# PrettyDynoTerm

> **Cross-Platform Terminal GPU & CPU Benchmark Engine**  
> High-performance 4K offscreen 3D raymarching, fractal calculus, relativistic astrophysics, atmospheric vortex dynamics, and cardiac electrophysiology rendered directly inside your terminal in 24-bit TrueColor.

---

## Features

- **Scientific Computational Physics & Biophysics Engine**:
  - **Act I: Atmospheric Ocean Waves**: 14 Gerstner wave octaves, trochoidal crests, 48-step adiabatic cumulus cloud deck, and Henyey-Greenstein Mie radiative transfer.
  - **Act II: The Mandelbulb Core**: Infinite recursive 3D fractal caverns ($z \mapsto z^8 + c$) testing raw GPU arithmetic logic and warp/wavefront branch divergence across 160 ray steps and 24 polynomial iterations.
  - **Act III: Relativistic Black Hole**: 80 4th-order Runge-Kutta (RK4) curved spacetime geodesic integration steps around a Kerr black hole with frame-dragging and Novikov-Thorne accretion disk.
  - **Act IV: Tornadic Mesocyclone (Weather Model)**: Burgers-Rott vortex dynamics, cyclostrophic pressure deficit balance ($\frac{dp}{dr} = \rho \frac{v_\theta^2}{r}$), adiabatic condensation funnel cloud, and surface debris friction layer.
  - **Act V: Cardiac Electrophysiology (Medical Model)**: Aliev-Panfilov 2-variable reaction-diffusion PDE simulating 3D scroll-wave re-entry rotors (ventricular fibrillation) across anisotropic myocardial fibers with optical mapping fluorescence.
  - **Act VI: Kinetic N-Body Supernova (CPU)**: 32,000 gravitational celestial bodies simulated across all CPU threads via Rayon, with velocity-based thermal heat gradient coloring.
- **High-Performance 4K UHD Offscreen Compute**:
  - Defaults to computing at **3840×2160 (4K UHD)**—computing over **8.29 million pixels** per frame!
  - VRAM-native GPU compute downsampling to your terminal character grid using UTF-8 half-blocks (`▀`), providing extreme supersampling anti-aliasing (SSAA).
- **Cybernetic Telemetry HUD**:
  - Live GPU core clock (MHz), junction temperature (°C), board power (Watts), and VRAM utilization.
  - Real-time CPU multi-core utilization sparkline and thread count.
  - Frametime analysis with instantaneous FPS, average frametime, and 1% lows.
- **Deterministic Finite Benchmark**:
  - Fixed-workload execution (not an imprecise timed run).
  - Standardized score badge strictly comparable across systems.
  - Outputs detailed telemetry JSON logs (`dyno_benchmark_results.json`).

---

## Installation & Building

```bash
git clone git@github.com:flavanoids/PrettyDynoTerm.git
cd PrettyDynoTerm
cargo build --release
```

The compiled binary will be located at `./target/release/dyno_term`.

---

## Usage

### 1. Interactive CLI Launcher Menu (Default)
Simply running the binary without mode flags opens the full interactive terminal launcher:
```bash
./target/release/dyno_term
```
- **Menu Features**:
  - **Launch Interactive Demo**: 60 FPS smooth preview with live telemetry HUD.
  - **Launch 100% Uncapped Stress Test**: Uncapped frame pipeline to peg GPU to 100% power and boost clock.
  - **Run Full Benchmark Suite**: Deterministic workload scoring with instant scorecard report.
  - **Select Starting Scene**: Direct jump to Acts I through VI.
  - **Compute Resolution Preset**: Switch on-the-fly between 16K, 8K, 4K UHD, 1440p, 1080p, 720p, Native Viewport, or Custom pixel dimensions.
  - **Configure Benchmark Frames**: Preset or custom iterations per act.
  - **Toggle Stress Mode**: Instant hotkey switch (`[U]`) between 60 FPS capped and uncapped.
  - **Switch GPU Adapter**: Switch WebGPU adapter between discrete GPUs, integrated GPUs, and CPU rasterizers.
  - **Live Hardware Telemetry & Diagnostics**: 150ms real-time monitoring of GPU temp, board power draw, core clock, VRAM, and CPU load.
- **Menu Controls**:
  - `[↑ / ↓]` or `[k / j]` : Navigate menu options
  - `[ENTER]` or `[SPACE]` : Select option / open submenu
  - `[1 - 9, 0]` : Direct numeric hotkeys
  - `[U]` : Toggle uncapped GPU stress mode
  - `[Q]` or `[ESC]` : Exit menu

### 2. Direct Interactive Demo Mode
```bash
./target/release/dyno_term --demo
```
- **In-Demo Controls**:
  - `[SPACE]` : Switch between the 6 scenes
  - `[U]` : Toggle between 60 FPS V-Sync and 100% Uncapped GPU Stress Mode
  - `[Q]` or `[ESC]` : Exit

### 3. Direct Benchmark Suite
Runs deterministic frames across all six acts and generates a scorecard:
```bash
./target/release/dyno_term --benchmark
```

### 4. Resolution Presets (Direct CLI)
```bash
# 4K UHD (Default: 3840x2160)
./target/release/dyno_term --res 4k

# Extreme 16K UHD Stress (15360x8640)
./target/release/dyno_term --res 16k

# 8K UHD (7680x4320)
./target/release/dyno_term --res 8k

# 1440p QHD (2560x1440)
./target/release/dyno_term --res 1440p

# 1080p FHD (1920x1080)
./target/release/dyno_term --res 1080p

# 720p APU / Mobile (1280x720)
./target/release/dyno_term --res 720p
```
