# PrettyDynoTerm Embedding & Host Integration Contract

This document specifies the official integration contract between **PrettyDynoTerm** (`dyno_term`) and host workload supervisors, orchestrators, and job runners.

---

## 1. Process Lifecycle & Execution Model

Host orchestrators launch `dyno_term` as an unprivileged, headless subprocess inside datacenter compute nodes.

### 1.1 Non-Interactive & Headless Guarantees
- When launched without a pseudo-terminal (TTY) or with `--headless`, `dyno_term` operates strictly non-interactively:
  - Disables raw mode and alternate screen buffer switching.
  - Suppresses all ANSI cursor and clearing sequences.
  - Reads nothing from `stdin` (`/dev/null` safe).
  - Emits machine-parsable JSON or JSONL records directly to `stdout`.

### 1.2 Read-Only Working Directory Guarantee
- `dyno_term` executes cleanly when the current working directory (`CWD`) is read-only (e.g. `chmod 555` or read-only container root `/`).
- **Disk Isolation Rules:**
  - If neither `--report-file` nor `--report-dir` is specified, `dyno_term` **writes nothing to disk**.
  - To persist benchmark JSON artifacts, the host caller must provide `--report-dir <DIR>` or `--report-file <PATH>`. `dyno_term` creates target parent directories and writes exclusively inside that target.

### 1.3 Signal Teardown & 2-Second Watchdog Contract
- `dyno_term` installs POSIX signal handlers for `SIGINT` (Ctrl+C) and `SIGTERM`.
- **Guarantees upon receiving `SIGINT` / `SIGTERM`:**
  1. The running compute pipeline immediately halts active compute dispatch.
  2. Allocations on GPU devices are released.
  3. A final `cancelled` JSONL record is emitted to `stdout`:
     ```json
     {"schema_version":"1.0.0","type":"cancelled","timestamp":"...","reason":"SIGINT or SIGTERM signal received","elapsed_seconds":1.234}
     ```
  4. The process exits within $\le 2.0$ seconds with **Exit Code `4`**.
  5. A dedicated background watchdog thread enforces an unconditional process kill (`SIGKILL`) if any GPU driver call deadlocks for longer than 2.0 seconds.

---

## 2. Linux Exit Codes

The host supervisor inspects process exit codes to determine failure classification:

| Exit Code | Name | Semantic Meaning |
| :---: | :--- | :--- |
| **`0`** | `Ok` | Workload executed and validated successfully. |
| **`1`** | `WorkloadFailure` | Workload failure (e.g., thermal limit breached, hardware verification mismatch). |
| **`2`** | `InvalidArgs` | Invalid command line flags or parameters. |
| **`3`** | `NoUsableGpu` | No compatible GPU adapter or backend found on the system. |
| **`4`** | `Cancelled` | Execution terminated cleanly by `SIGINT` or `SIGTERM` within 2 seconds. |
| **`5`** | `InternalError` | Unexpected runtime panic or unhandled internal error. |

---

## 3. Metadata & Build Introspection (`--version`)

Host orchestrators query binary capabilities, git commit lineage, and embedded compute shader digests using:

```bash
dyno_term --version --output json
```

### JSON Response Schema:
```json
{
  "version": "1.0.0",
  "git_commit": "1b04503eba283f13d7bb5f59c8cfdbb69fb6d207",
  "embedded_kernels": {
    "cuda_ptx_sm90_sha256": "752449575a4c064f92f036a476beb93dbf353bc333c26d273a7b9f690941a57b",
    "hip_co_gfx_sha256": "ae1eae69ccd8159bb569df23a0d8b1d90384d9daf982c9912a417e55c5710b9a",
    "wgsl_shader_sha256": "051eee8f1bce54d8f81fae6b03140424daf14c619ec6b959ee8f844d734915e9"
  },
  "supported_backends": [
    "wgpu",
    "cuda",
    "hip"
  ]
}
```

- **`git_commit`**: Full 40-character SHA-1 hash of the git commit used to compile the binary.
- **`embedded_kernels`**: Cryptographic SHA-256 digests of the embedded PTX (`sm_90`), HIP fat binary (`gfx906`, `gfx908`, `gfx90a`, `gfx942`), and unified WGSL shader.
- **`supported_backends`**: Backends supported by this binary build.

---

## 4. Multi-GPU Device Selection & Isolation

### 4.1 CLI Flags
- `--gpus all` (default): Executes concurrent benchmark across all detected discrete GPU adapters.
- `--gpus 0,2,5`: Selects specific zero-indexed GPU adapters by system ordinal.

### 4.2 Transparent Environment Filtering
When `--gpus all` is set, `dyno_term` automatically honours vendor environment isolation variables:
1. `CUDA_VISIBLE_DEVICES` (e.g. `CUDA_VISIBLE_DEVICES=0,1`)
2. `HIP_VISIBLE_DEVICES` (e.g. `HIP_VISIBLE_DEVICES=2,3`)

If either variable is exported in the environment, `dyno_term` filters its target adapter set to match the visible indices. If the variable is empty or masks all available devices, `dyno_term` exits cleanly with code `3` (`NoUsableGpu`).

---

## 5. Workload Modes

### 5.1 Fixed Duration Mode (`--duration SECONDS`)
Runs benchmark scenes cyclically (0 through 7, repeating) continuously until `SECONDS` have elapsed:
```bash
dyno_term --headless --benchmark --preset datacenter --duration 60 --output jsonl
```
- Emits real-time progress records ($\ge 1\text{/s}$).
- Reports completed frame throughput (FPS), overall work units (Score), estimated TFLOPS, effective VRAM bandwidth, and telemetry upon completion.

### 5.2 Fixed Frame Mode (`--frames N`)
Runs exactly `N` frames per scene across all 8 scenes (total $8 \times N$ frames).

### 5.3 Datacenter Preset (`--preset datacenter`)
- Configures 16K compute canvas ($15360 \times 8640 = 132.7\text{ Mpix/frame}$).
- Validates VRAM allocation tier against hardware limits:
  - Enforces $\le 75\%$ total VRAM ceiling and $\le 90\%$ live free VRAM rule across 16 GB, 32 GB, 80 GB, 94 GB, and 141 GB tiers.
- Employs headless uncapped compute with 8-byte checksum verification to eliminate PCIe readback bottlenecks.

---

## 6. Output Streaming Specification (`--output jsonl`)

In `jsonl` mode, `dyno_term` writes one self-contained JSON line per event to `stdout`:

1. **`start`**: Workload parameters, detected GPUs, PCI bus IDs, and safety limits.
2. **`progress`**: Emitted $\ge 1\text{/s}$ per GPU with current scene, frame index, FPS, junction temperature, and board power.
3. **`thermal_trip`** (if tripped): Breach details, sensor reading, limit, and debounced sample count.
4. **`gpu_result`**: Per-GPU benchmark score, average FPS, 1% low frametime, scene breakdown, and fleet outlier flags.
5. **`summary`**: Overall node benchmark score, total elapsed time, active GPU count, list of outlier GPU indices, and exit code.
6. **`cancelled`** (if interrupted): Reason string and total elapsed seconds.
7. **`error`** (if failed): Error type, message string, and exit code.

Refer to `docs/OUTPUT_SCHEMA.md` for full schema type definitions.

---

## 7. glibc Compatibility ($\le 2.31$) & Portability

Target datacenter nodes run Debian 11 (glibc 2.31) or custom enterprise Linux distributions without development tools or internet access.

### 7.1 Dynamic glibc $\le 2.28$ Container Build
Build in a Rocky Linux 8 (glibc 2.28) container:
```bash
./scripts/build_rocky8_binary.sh
```
This guarantees all dynamic libc bindings (`pthread`, `dlopen`, `stat`, `math`) resolve against `GLIBC_2.28` or lower, running out-of-the-box on Debian 11 (2.31), Ubuntu 20.04/22.04, RHEL/Rocky 8/9, and newer.

### 7.2 Static MUSL Build & Zero-Dependency Fleet Packaging
Compile a static PIE binary with zero glibc dependencies and package it for fleet deployment (no Docker, Podman, or Alpine required):
```bash
./scripts/package_release.sh
```
This runs `cargo build --target x86_64-unknown-linux-musl --release`, verifies dynamic isolation via `check_glibc_symbols.sh`, strips symbols (~7.6 MB binary), and archives it into `dist/dyno_term-v<VERSION>-linux-x86_64.tar.gz` (~3.1 MB) with SHA-256 checksums.

Or compile manually:
```bash
cargo build --target x86_64-unknown-linux-musl --release
```

### 7.3 Verification Script
Run the automated symbol verification script to ensure the binary strictly satisfies the compatibility contract:
```bash
./scripts/check_glibc_symbols.sh target/release/dyno_term
```
Verification succeeds when the highest required glibc version is $\le 2.31$ (or when static with 0 glibc symbols).
