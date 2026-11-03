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

### 1. Interactive Demo Mode (4K UHD Default)
```bash
./target/release/dyno_term --demo
```
- **Controls**:
  - `[SPACE]` : Switch between the 6 scenes
  - `[U]` : Toggle between 60 FPS V-Sync and 100% Uncapped GPU Stress Mode
  - `[Q]` or `[ESC]` : Exit

### 2. Standard 4K Benchmark
Runs deterministic 4K frames per scene across all six acts:
```bash
./target/release/dyno_term --benchmark
```

### 3. Resolution Presets
```bash
# 4K UHD (Default: 3840x2160)
./target/release/dyno_term --demo --res 4k

# Extreme 16K UHD Stress (15360x8640)
./target/release/dyno_term --demo --res 16k

# 8K UHD (7680x4320)
./target/release/dyno_term --demo --res 8k

# 1440p QHD (2560x1440)
./target/release/dyno_term --demo --res 1440p

# 1080p FHD (1920x1080)
./target/release/dyno_term --demo --res 1080p

# 720p APU / Mobile (1280x720)
./target/release/dyno_term --demo --res 720p
```
