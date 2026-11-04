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
| **Act VII** | Kinetic Supernova | Celestial Gravitation | $1.0\times$ (32,000 $N$-Body particle gravitational forces) |

The final score is computed as throughput normalized to completion time:

$$\text{Score}_{\text{scene}} = \frac{\text{Raw Workload}}{\text{Elapsed Time (seconds)} \times 100}$$

$$\text{Dyno Benchmark Score} = \sum_{i=1}^{7} \text{Score}_{\text{scene } i}$$

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

## 5. JSON Telemetry Schema (`dyno_benchmark_results.json`)

Every benchmark run exports a structured JSON report for logging and leaderboard integration:

```json
{
  "timestamp": "UTC_Epoch_1793666272",
  "system_info": {
    "gpu_name": "AMD Radeon RX 6650 XT (RADV NAVI23)",
    "gpu_backend": "Vulkan",
    "gpu_driver": "radv",
    "cpu_name": "12th Gen Intel(R) Core(TM) i5-12600K",
    "cpu_threads": 16
  },
  "total_elapsed_seconds": 1.90,
  "total_frames": 16,
  "overall_score": 77487233,
  "peak_gpu_temp_c": 53.0,
  "peak_gpu_power_w": 64.0,
  "peak_vram_mb": 6064,
  "scene_results": [
    {
      "scene_id": 0,
      "scene_name": "BIOLUMINESCENT WAVES (GPU)",
      "frames_rendered": 4,
      "elapsed_seconds": 0.608,
      "avg_fps": 6.6,
      "avg_frametime_ms": 152.07,
      "p1_low_frametime_ms": 166.17,
      "score": 8727182
    }
  ]
}
```
