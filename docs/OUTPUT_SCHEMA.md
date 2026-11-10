# DynoTerm Output Schema Specification

**Schema Version:** `1.0.0`

PrettyDynoTerm (`dyno_term`) supports three output modes via the `--output` CLI flag:
- `human`: (Default) Interactive styled terminal output (HUD and ANSI cursor controls in TTY mode, plain text in headless mode).
- `json`: Final indented JSON benchmark report emitted to `stdout` upon completion.
- `jsonl`: Real-time streaming JSON Lines emitted to `stdout`. Every line is an independent, valid JSON document containing a `"schema_version"` field.

---

## 1. Exit Codes

DynoTerm returns standardized Linux exit codes to integrate cleanly into workflow supervisors (such as systemd and Project Senna):

| Exit Code | Identifier | Description |
| :---: | :--- | :--- |
| **`0`** | `Ok` | Successful execution and validation. |
| **`1`** | `WorkloadFailure` | Workload failure (e.g. thermal failsafe tripped, frame checksum mismatch). |
| **`2`** | `InvalidArgs` | Invalid command line arguments (e.g. unknown preset, or headless mode without a runnable subcommand). |
| **`3`** | `NoUsableGpu` | No compatible GPU adapter or backend found on the system. |
| **`4`** | `Cancelled` | Execution cleanly aborted by `SIGINT` or `SIGTERM` within 2 seconds. |
| **`5`** | `InternalError` | Unexpected internal runtime failure or panic. |

---

## 2. JSONL Streaming Records (`--output jsonl`)

In `jsonl` mode, every line emitted to `stdout` contains:
- `schema_version`: String (e.g. `"1.0.0"`).
- `type`: String denoting record type (`start`, `progress`, `gpu_result`, `summary`, `error`, `cancelled`).
- `timestamp`: ISO 8601 UTC timestamp string (e.g. `"2026-11-10T14:30:00.123Z"`).

### 2.1 Start Record (`type: "start"`)
Emitted immediately once initialization is complete and workload execution begins.

```json
{
  "schema_version": "1.0.0",
  "type": "start",
  "timestamp": "2026-11-10T14:30:00.123Z",
  "pid": 54321,
  "mode": "benchmark",
  "preset": "4k",
  "frames_per_scene": 100,
  "total_scenes": 8,
  "gpus": [
    {
      "index": 0,
      "name": "AMD Radeon RX 6700 XT",
      "backend": "wgpu",
      "driver": "Mesa 23.2.1",
      "pci_bus_id": "0000:03:00.0"
    }
  ],
  "failsafe_enabled": true
}
```

### 2.2 Progress Record (`type: "progress"`)
Emitted at least once per second ($\ge 1\text{/s}$) during active GPU execution:

```json
{
  "schema_version": "1.0.0",
  "type": "progress",
  "timestamp": "2026-11-10T14:30:01.450Z",
  "scene_index": 1,
  "scene_name": "Mandelbulb Raymarching",
  "frame": 45,
  "total_scene_frames": 100,
  "progress_pct": 45.0,
  "fps": 182.4,
  "gpu_index": 0,
  "gpu_temp_c": 58.0,
  "gpu_power_w": 115.0
}
```

### 2.3 GPU Result Record (`type: "gpu_result"`)
Emitted per GPU upon workload completion:

```json
{
  "schema_version": "1.0.0",
  "type": "gpu_result",
  "timestamp": "2026-11-10T14:30:15.890Z",
  "gpu_index": 0,
  "gpu_name": "AMD Radeon RX 6700 XT",
  "score": 14250.5,
  "avg_fps": 178.2,
  "one_percent_low_fps": 132.0,
  "scenes": [
    {
      "scene_id": 0,
      "name": "Waves & Ocean Dynamics",
      "fps": 210.5,
      "score": 1850.0
    }
  ],
  "peak_temp_c": 64.5,
  "peak_power_w": 145.2
}
```

### 2.4 Summary Record (`type: "summary"`)
Emitted as the final record when execution succeeds:

```json
{
  "schema_version": "1.0.0",
  "type": "summary",
  "timestamp": "2026-11-10T14:30:16.010Z",
  "status": "ok",
  "overall_score": 14250.5,
  "elapsed_seconds": 15.887,
  "exit_code": 0
}
```

### 2.5 Error Record (`type: "error"`)
Emitted when an operational failure occurs:

```json
{
  "schema_version": "1.0.0",
  "type": "error",
  "timestamp": "2026-11-10T14:30:05.112Z",
  "error_type": "ThermalTrip",
  "message": "Silicon junction hotspot reached 106.0°C exceeding safety ceiling of 105.0°C",
  "exit_code": 1
}
```

### 2.6 Cancelled Record (`type: "cancelled"`)
Emitted when the application is cancelled via `SIGINT` or `SIGTERM`:

```json
{
  "schema_version": "1.0.0",
  "type": "cancelled",
  "timestamp": "2026-11-10T14:30:07.456Z",
  "reason": "SIGINT or SIGTERM signal received",
  "elapsed_seconds": 7.333,
  "exit_code": 4
}
```
