# Cross-Platform GPU & Compute Engine

One of `dyno_term`'s primary design goals is unified, high-performance execution across **Linux (AMD, Nvidia, Intel)** and **macOS (Apple Silicon M1–M4 & Intel Mac)** without vendor lock-in or display server dependencies.

---

## 1. Unified Graphics Backend: WebGPU (`wgpu`)

Rather than maintaining separate CUDA, ROCm, Vulkan, and Metal implementations, `dyno_term` uses **`wgpu`** (the safe, portable WebGPU implementation in Rust).

```
                      dyno_term (WGSL Shaders & Compute Code)
                                       │
                                   wgpu 30.0
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
     Vulkan 1.3 / 1.4                                         Apple Metal
      (Linux Platforms)                                     (macOS Platforms)
            │                                                     │
   ┌────────┴────────┬────────┐                          ┌────────┴────────┐
   ▼                 ▼        ▼                          ▼                 ▼
AMD (RADV)      Nvidia (NVK) Intel (ANV)            Apple Silicon      Intel Mac
(RX 6650 XT)    (RTX 4090)   (Arc B580)               (M1-M4)
```

### Why This Architecture Works:
1. **Zero Runtime Overhead**: On Linux, `wgpu` translates WGSL compute shaders directly into native Vulkan SPIR-V. On macOS, it compiles directly to native Metal Shading Language (MSL).
2. **Headless Execution**: Computes completely offscreen using `BufferUsages::STORAGE` and `BufferUsages::MAP_READ`. It does **not require an X11 Display, Wayland Compositor, or macOS Quartz window surface**. It runs flawlessly inside remote cloud GPU instances, CI/CD runners, and headless SSH sessions.
3. **Multi-Adapter Enumeration**: Uses `instance.enumerate_adapters()` to detect all physical GPUs on the machine. Users with multiple GPUs (e.g. an integrated AMD APU + discrete Radeon 6650 XT) can select which GPU to benchmark via `--gpu <INDEX>`.
4. **Automated Pipeline Diagnostics**: Running `./target/release/dyno_term --test` executes automated test dispatches across all compute shaders and verifies preview readback on the selected or fallback GPU.

---

## 2. Hardware Telemetry Architecture

Accurate hardware monitoring is critical for thermal and power profiling during a benchmark run.

### Linux Telemetry Engine (`src/telemetry.rs`):
- **GPU Core Frequency**: Read directly from `/sys/class/drm/card*/device/hwmon/hwmon*/freq1_input` (Hz $\to$ MHz).
- **GPU Temperatures**: Edge, junction, and memory temperatures parsed from `temp1_input`, `temp2_input` (millidegrees C $\to$ °C).
- **GPU Board Power**: Average board power consumption read from `power1_average` (microwatts $\to$ Watts).
- **VRAM Utilization**: Exact memory allocations parsed from `/sys/class/drm/card*/device/mem_info_vram_used` and `mem_info_vram_total`.
- **CPU Utilization**: CPU jiffies computed dynamically by sampling `/proc/stat` delta times between frames.

### macOS Telemetry Engine:
- **CPU Model & Core Counts**: Retrieved via `libc::sysctlbyname("machdep.cpu.brand_string", ...)`.
- **Unified Memory & SOC**: Queryable via `IOKit` and `mach` host statistics.

---

## 3. Terminal TrueColor & Half-Block Rasterizer

`dyno_term` avoids font-rendering degradation by utilizing **Unicode Upper Half-Blocks (`▀` U+2280)**:

```
Character Cell:
┌─────────────────────────┐
│ Foreground RGB (Top)    │ <- Pixel (x, y*2)
├─────────────────────────┤
│ Background RGB (Bottom) │ <- Pixel (x, y*2 + 1)
└─────────────────────────┘
```

- Each character cell transmits two vertical subpixels simultaneously.
- A standard 120×40 character terminal renders **120×80 full 24-bit TrueColor pixels**.
- Double-buffered memory updates write the entire frame in a single atomic buffer flush (`stdout.write_all()`), eliminating tearing and terminal stutter.
