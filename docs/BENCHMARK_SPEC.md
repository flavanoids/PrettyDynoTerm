# Benchmark Methodology & Scoring Standard

`dyno_term` uses a **finite, deterministic workload model** rather than an imprecise timed loop (e.g. "run for 60 seconds"). This ensures that results are consistent, repeatable, and comparable across different architectures and thermal form factors.

---

## 1. Why Finite Workloads Beat Timed Tests

In traditional benchmarks that run for an arbitrary duration (e.g., "render for 1 minute"):
- As a thin laptop heats up, thermal throttling slows clock speeds midway through the test, meaning the device renders *fewer frames in the second half than in the first*.
- Background operating system scheduler interrupts introduce variance.
- Two identical GPUs in different chassis configurations produce erratic, non-reproducible frame counts.

In `dyno_term`'s **Deterministic Model**:
- Every test runs a **fixed, identical mathematical budget** (e.g., exactly 100 frames of 16K raymarching).
- Every single machine computes the exact same number of ray steps, trigonometric evaluations, and gravitational particle pairs.
- The metric evaluated is **time-to-completion, frametime consistency, and arithmetic throughput**.
- **Hardware Pre-flight Validation**: Run `./target/release/dyno_term --test` before executing official scoring runs to verify GPU compute queues, shader bindings, and telemetry sensors are fully operational.

---

## 2. Standardized Resolution Tiers

To ensure benchmark scores are universally comparable, the compute engine operates on fixed canonical resolutions:

| Preset | Resolution | Total Pixels / Frame | Primary Hardware Target |
| :--- | :--- | :--- | :--- |
| **`16k` (Default)** | **15360 × 8640** | **132,710,400** | High-End Desktop GPUs & Workstations |
| **`8k`** | **7680 × 4320** | **33,177,600** | Performance Discrete GPUs |
| **`4k`** | **3840 × 2160** | **8,294,400** | Mid-Range Desktop & Mobile GPUs |
| **`1440p`** | **2560 × 1440** | **3,686,400** | Laptops & Entry Discrete GPUs |
| **`1080p`** | **1920 × 1080** | **2,073,600** | Budget Laptops & Modern APUs |
| **`720p`** | **1280 × 720** | **921,600** | Handhelds, Tablets & Phones |
| **`native`** | Dynamic | $W_{term} \times H_{term} \cdot 2$ | Terminal Viewport Native |

---

## 3. Scoring Formula

Each scene possesses a micro-architectural weighting factor reflecting its arithmetic complexity:

$$\text{Raw Workload} = \left( N_{\text{frames}} \times \text{Pixels}_{\text{frame}} \right) \times W_{\text{scene}}$$

| Act | Scene Name | Domain | Arithmetic Weight ($W_{\text{scene}}$) |
| :--- | :--- | :--- | :---: |
| **Act I** | Atmospheric Ocean Waves | Hydrodynamics | $1.0\times$ (14 Gerstner octaves + Mie clouds + reflection) |
| **Act II** | The Mandelbulb Core | Fractal Algebra | $1.8\times$ (Order-8 polynomial unrolled fractal loops + cone AO) |
| **Act III** | Relativistic Black Hole | General Relativity | $1.4\times$ (80 RK4 curved geodesic spacetime integrations) |
| **Act IV** | Tornadic Mesocyclone | Severe Meteorology | $1.2\times$ (Burgers-Rott vortex + cyclostrophic pressure deficit) |
| **Act V** | Cardiac Electrophysiology | Biophysics / Medicine | $1.3\times$ (Aliev-Panfilov reaction-diffusion PDE + rotors) |
| **Act VI** | Lattice Boltzmann Flow | Computational Fluid Dynamics | $1.5\times$ (D2Q9 BGK collision relaxation + von Kármán street) |
| **Act VII** | Cybernetic Gait Kinematics | 3D Character Animation | $1.4\times$ (Hierarchical forward kinematics + smin skinning + mirror reflection) |
| **Act VIII** | Kinetic Supernova | Celestial Gravitation | $1.0\times$ (32,000 $N$-Body particle gravitational forces) |

The final score is computed as throughput normalized to completion time using a standardized divisor ($20,000,000$ operations/sec = $1.00\text{ pt}$):

$$\text{Score}_{\text{scene}} = \frac{\text{Raw Workload}}{\text{Elapsed Time (seconds)} \times 20,000,000}$$

$$\text{Dyno Benchmark Score} = \sum_{i=1}^{8} \text{Score}_{\text{scene } i}$$

### Score Scaling & Calibration
Scores are formatted to two decimal places (`f64`, e.g. `898.66 pts`) to provide an intuitive, memorable decimal scale:
- **Baseline Scale**: $20.0\text{ Giga-Pixel-Ops/s} = 1,000.00\text{ pts}$ ($1.0\text{ Giga-Pixel-Ops/s} = 50.00\text{ pts}$).
- **Mid-Range Baseline**: A modern mainstream GPU (e.g. AMD Radeon RX 6650 XT) scores $\approx 850.00 - 950.00\text{ pts}$ at 4K UHD.
- **Enthusiast / Flagship**: Modern high-end GPUs (e.g. NVIDIA RTX 4080 / 4090) score $\approx 1,500.00 - 2,500.00\text{ pts}$.
- **Entry-Level / Integrated**: Modern APUs and entry GPUs score $\approx 100.00 - 250.00\text{ pts}$.

### Explicit Silicon Hardware Performance Metrics
In addition to normalized benchmark points, `dyno_term` computes and exports silicon compute and efficiency metrics grounded in instruction and memory profiling:
1. **Estimated Floating-Point Throughput (TFLOPS)**:
   Calculated from the weighted instruction count across all 8 Acts (averaging $\sim 2,250$ FLOPs per pixel across RK4 integrations, transcendental wave superpositions, and polynomial unrolling):
   $$\text{TFLOPS} = \frac{N_{\text{frames}} \times \text{Pixels}_{\text{frame}} \times 2,250}{\text{Elapsed Seconds} \times 10^{12}}$$
2. **Effective Memory Bandwidth (GB/s)**:
   Measures high-resolution compute buffer write transactions (4 bytes/pixel) plus downsample texture reads and writes:
   $$\text{Bandwidth (GB/s)} = \frac{N_{\text{frames}} \times \left( \text{Pixels} \times 4.0 + 160 \times 60 \times 8.0 \right)}{\text{Elapsed Seconds} \times 10^9}$$
3. **Energy Efficiency (Joules per Million Rays / Mray-Joules)**:
   Measures total electrical energy consumed per unit of computational output using live sysfs/hwmon telemetry power integration:
   $$\text{Total Joules} = \bar{P}_{\text{board}} \times t_{\text{elapsed}}, \quad \text{Efficiency} = \frac{\text{Total Joules}}{\text{Total Pixels} / 10^6}$$

---

## 4. Historical Baseline & Comparison Engine

`dyno_term` features a built-in persistent benchmark history ledger and delta comparison engine:
- **Baseline Reference**: Users can pin any run as the reference baseline (`dyno_term --set-baseline <FILE>`), creating `dyno_baseline.json`.
- **Chronological History**: Every benchmark run is appended to `dyno_benchmark_history.jsonl` with timestamps, hardware specs, and per-act scores.
- **Automated Comparison Table**: Post-benchmark runs automatically diff against the pinned baseline, reporting:
  - Overall score delta ($\pm \%$ and points).
  - Per-act FPS, frametime, and score deltas.
  - Hardware clock and thermal deltas.
  - **Thermal Throttling Heuristics**: Flags regressions if GPU frequency drops $> 4.5\%$ while temperatures spike.
- **CLI Management**:
  - `dyno_term --history`: Prints the chronological table of all past runs with `[BASELINE]` tags.
  - `dyno_term --compare <REPORT_PATH>`: Diffs a specific report against the active baseline.
  - `dyno_term --set-baseline [PATH]`: Pins the latest or specified report as active baseline.

---

## 5. Frametime Consistency & 1% Lows

Average FPS alone can hide stuttering, thread stalls, and pipeline bubbles. `dyno_term` tracks the entire distribution of frame execution intervals:
- **Average Frametime**: Mean latency per frame ($\mu$).
- **1% Lows (99th Percentile Frametime)**: Identifies worst-case hitching caused by memory bus contention, OS scheduling delays, or shader recompilation.

---

## 5. Automated GPU Sensor Assessment & Telemetry Architecture

`dyno_term` features a continuous, low-overhead hardware telemetry engine that polls kernel sysfs interfaces (`/sys/class/hwmon` and `/sys/class/drm/card*/device/`) every 15ms.

### Hardware Metrics Collected
- **Voltages & Power**: GPU core voltage ($V_{\text{core}} / V_{\text{ddgfx}}$ in mV), instantaneous board power (Watts), and OEM board power limit / PPT cap ($P_{\text{cap}}$).
- **Frequencies & Bus**: Core clock frequency (MHz), Memory clock (MHz), Memory controller load percentage (`mem_busy_percent`), VRAM vendor architecture (e.g. Samsung GDDR6, Micron, SK Hynix), and negotiated PCIe link speed and width (`pp_dpm_pcie`, e.g. `PCIe 16.0 GT/s (x8)`).
- **Thermals & Acoustics**: Standard package/edge temperature, internal localized junction/hotspot peak, VRAM IC sensor temperature, hardware critical thermal throttle thresholds ($T_{\text{crit}}$), acoustic fan RPM, and fan PWM duty cycle percentage.

### Automated Assessment Heuristics
1. **Thermal Coupling Delta ($\Delta T_{\text{hotspot}}$)**:
   $$\Delta T_{\text{hotspot}} = T_{\text{hotspot}} - T_{\text{edge}}$$
   - $\le 18^\circ\text{C}$: **`OPTIMAL`** (Exceptional vapor chamber coupling and uniform thermal contact)
   - $\le 28^\circ\text{C}$: **`NOMINAL`** (Healthy thermal interface material and mounting pressure)
   - $29-35^\circ\text{C}$: **`ELEVATED`** (Warm thermal gradient, heat dissipation constrained)
   - $> 35^\circ\text{C}$: **`CRITICAL`** (Degraded thermal interface, pump-out, or uneven coldplate mounting)

2. **Thermal Headroom**:
   $$\text{Headroom} = T_{\text{crit}} - T_{\text{hotspot}}$$
   Evaluates remaining temperature margin before hardware thermal trip points (typically $110^\circ\text{C}$ on AMD Navi and $105^\circ\text{C}$ on NVIDIA architectures).

3. **Throttling Verdict Engine**:
   Flags throttling if:
   - Thermal headroom $\le 2.0^\circ\text{C}$ with hotspot $\ge 95.0^\circ\text{C}$, OR
   - Hotspot $\ge T_{\text{crit}}$, OR
   - Core clock drops $> 22\%$ below peak under heavy continuous GPU load ($\ge 85\%$).

4. **Power Limit Detection**:
   Detects whether the benchmark reached the maximum allowable board PPT cap:
   $$P_{\text{peak}} \ge 0.98 \times P_{\text{cap}}$$

---

## 6. JSON Telemetry Schema (`dyno_benchmark_results.json`)

Every benchmark run exports a structured JSON report including top-level peak metrics, per-act results, automated GPU assessment scorecard, and continuous periodic time-series telemetry snapshots (sampled at 4 Hz):

```json
{
  "timestamp": "UTC_Epoch_1793814430",
  "system_info": {
    "gpu_name": "AMD Radeon RX 6650 XT (RADV NAVI23)",
    "gpu_backend": "Vulkan",
    "gpu_driver": "radv",
    "cpu_name": "12th Gen Intel(R) Core(TM) i5-12600K",
    "cpu_threads": 16
  },
  "total_elapsed_seconds": 6.82,
  "total_frames": 700,
  "overall_score": 898.66,
  "peak_gpu_temp_c": 66.0,
  "peak_gpu_hotspot_c": 99.0,
  "peak_gpu_util_pct: 100.0,
  "peak_gpu_power_w": 156.0,
  "peak_gpu_clock_mhz": 2765,
  "peak_vram_mb": 5558,
  "gpu_telemetry": {
    "vram_vendor": "Samsung GDDR6",
    "pcie_link": "PCIe 16.0GT/s (x8)",
    "power_cap_w": 157.0,
    "temp_crit_c": 100.0,
    "hotspot_crit_c": 110.0,
    "edge_temp_c": { "min": 42.0, "max": 65.0, "avg": 61.2, "p95": 65.0 },
    "hotspot_temp_c": { "min": 42.0, "max": 99.0, "avg": 91.4, "p95": 99.0 },
    "mem_temp_c": { "min": 40.0, "max": 46.0, "avg": 42.1, "p95": 46.0 },
    "delta_hotspot_edge_c": 34.0,
    "thermal_headroom_c": 11.0,
    "core_clock_mhz": { "min": 77.0, "max": 2765.0, "avg": 2553.0, "p95": 2661.0 },
    "mem_clock_mhz": { "min": 541.0, "max": 1094.0, "avg": 641.0, "p95": 675.0 },
    "core_voltage_mv": { "min": 6.0, "max": 1150.0, "avg": 1084.0, "p95": 1143.0 },
    "board_power_w": { "min": 14.0, "max": 156.0, "avg": 139.9, "p95": 156.0 },
    "fan_rpm": { "min": 0.0, "max": 1431.0, "avg": 1162.0, "p95": 1395.0 },
    "peak_fan_pwm_pct": 43.1,
    "thermal_verdict": "HEALTHY — 11.0°C thermal headroom, sustained boost",
    "cooling_efficiency_rating": "ELEVATED (Die Δ 29-35°C, Warm Thermal Gradient)",
    "throttling_detected": false,
    "power_limit_pegged": true
  },
  "telemetry_samples": [
    {
      "elapsed_sec": 0.012,
      "act_id": 0,
      "fps": 89.2,
      "frametime_ms": 11.2,
      "core_clock_mhz": 77,
      "mem_clock_mhz": 541,
      "core_voltage_mv": 725,
      "gpu_util_pct": 2.0,
      "mem_util_pct": 1.0,
      "edge_temp_c": 42.0,
      "hotspot_temp_c": 42.0,
      "mem_temp_c": 40.0,
      "fan_rpm": 0,
      "fan_pwm_pct": 0.0,
      "board_power_w": 14.0,
      "vram_used_mb": 5555
    }
  ],
  "scene_results": [
    {
      "scene_id": 0,
      "scene_name": "ATMOSPHERIC OCEAN WAVES (GPU)",
      "frames_rendered": 100,
      "elapsed_seconds": 1.055,
      "avg_fps": 94.7,
      "avg_frametime_ms": 10.55,
      "p1_low_frametime_ms": 10.65,
      "score": 7887323
    }
  ]
}
```

---

## Multi-Iteration Consistency & Thermal Stability Specification

To verify determinism, measure hardware throttling, and quantify statistical run-to-run variance, `dyno_term` includes a native consistency test harness via `-c, --consistency [RUNS]` and `--cooldown <SECONDS>`.

### Statistical Metrics Evaluated
1. **Mean ($\mu$)**:
   $$\mu = \frac{1}{N}\sum_{i=1}^N x_i$$
2. **Sample Standard Deviation ($s$)**:
   $$s = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (x_i - \mu)^2}$$
3. **Coefficient of Variation ($CV\%$)**:
   $$CV\% = \left(\frac{s}{\mu}\right) \times 100\%$$
4. **Spread / Range ($\Delta$)**:
   $$\text{Range} = \max(x) - \min(x), \quad \text{Spread } \% = \frac{\text{Range}}{\mu} \times 100\%$$

### Consistency Rating Thresholds
- **$CV \le 0.25\%$**: `EXEMPLARY` (Top-tier determinism, negligible run-to-run noise)
- **$CV \le 0.75\%$**: `EXCELLENT` (Industry-grade benchmark consistency)
- **$CV \le 1.50\%$**: `VERY GOOD` (Minor background OS / scheduler variance)
- **$CV \le 3.00\%$**: `STABLE` (Acceptable consistency)
- **$CV > 3.00\%$**: `HIGH VARIANCE` (Noticeable thermal throttling or background contention)

### Consistency JSON Schema (`dyno_consistency_results.json`)
The resulting report includes:
- Top-level `score_stats`, `duration_stats`, `peak_gpu_temp_stats`, `peak_hotspot_stats`, and `peak_clock_stats` summaries.
- Per-act `scene_stats` arrays documenting Mean FPS, Standard Deviation, and CV% across all iterations.
- Full nested `runs` array containing the raw `BenchmarkReport` data of every iteration.

