use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::benchmark::{
    BenchmarkReport, BenchmarkScorer, SystemReport, ThermalFailsafe, ThermalTripReport,
};
use crate::cpu::NBodySimulation;
use crate::gpu::GpuContext;
use crate::gpu::pipeline::{ComputePipelineManager, Uniforms};
use crate::output::{
    ExitCode, GpuResultRecord, GpuStartInfo, OutputEmitter, ProgressRecord, SceneResultRecord,
    StartRecord, SummaryRecord,
};
use crate::runner::demo::SCENE_NAMES;
use crate::runner::outliers::{GpuRunStats, detect_gpu_outliers};
use crate::signal;
use crate::telemetry::{PciBusId, TelemetryCollector};

#[derive(Debug, Clone)]
pub struct MultiGpuBenchmarkOptions {
    pub selected_gpus: Vec<usize>,
    pub frames_per_scene: u32,
    pub duration_secs: Option<u64>,
    pub compute_width: u32,
    pub compute_height: u32,
    pub res_tag: String,
    pub is_datacenter_preset: bool,
    pub max_temp_c: f32,
    pub failsafe_enabled: bool,
    pub report_file: Option<String>,
}

#[derive(Debug)]
pub enum WorkerOutcome {
    Success(Box<GpuRunResult>),
    ThermalTrip(ThermalTripReport),
    Cancelled,
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct GpuRunResult {
    pub gpu_index: usize,
    pub pci_bus_id: PciBusId,
    pub gpu_name: String,
    pub duration_s: f64,
    pub report: BenchmarkReport,
}

/// Runs a concurrent multi-GPU benchmark across all selected adapters.
pub fn run_multi_gpu_benchmark(
    opts: &MultiGpuBenchmarkOptions,
    telemetry: Arc<TelemetryCollector>,
    output_emitter: Arc<OutputEmitter>,
) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let all_adapters = GpuContext::list_adapters();
    if all_adapters.is_empty() {
        output_emitter.emit_error(
            "NoUsableGpu",
            "No primary GPU adapters found on system",
            ExitCode::NoUsableGpu,
        );
        return Ok(ExitCode::NoUsableGpu);
    }

    let num_gpus = opts.selected_gpus.len();
    if num_gpus == 0 {
        output_emitter.emit_error(
            "InvalidArgs",
            "No GPUs selected for multi-GPU execution",
            ExitCode::InvalidArgs,
        );
        return Ok(ExitCode::InvalidArgs);
    }

    // 1. Prepare start record metadata
    let start_gpus: Vec<GpuStartInfo> = opts
        .selected_gpus
        .iter()
        .map(|&idx| {
            if idx < all_adapters.len() {
                let info = &all_adapters[idx];
                GpuStartInfo {
                    index: idx,
                    name: info.name.clone(),
                    backend: format!("{:?}", info.backend).to_lowercase(),
                    driver: info.driver.clone(),
                    pci_bus_id: if info.pci_bus_id.is_empty() {
                        None
                    } else {
                        Some(info.pci_bus_id.as_str().to_string())
                    },
                }
            } else {
                GpuStartInfo {
                    index: idx,
                    name: format!("GPU Adapter {}", idx),
                    backend: "wgpu".to_string(),
                    driver: "unknown".to_string(),
                    pci_bus_id: None,
                }
            }
        })
        .collect();

    let start_record = StartRecord::new(
        "benchmark",
        &opts.res_tag,
        opts.frames_per_scene,
        8,
        start_gpus,
        opts.failsafe_enabled,
    );
    output_emitter.emit_jsonl(&start_record);

    let start_time = Instant::now();

    // 2. Concurrency barrier to guarantee all GPUs begin execution simultaneously
    let barrier = Arc::new(Barrier::new(num_gpus));
    let mut handles = Vec::with_capacity(num_gpus);

    for &gpu_index in &opts.selected_gpus {
        let barrier_clone = Arc::clone(&barrier);
        let telemetry_clone = Arc::clone(&telemetry);
        let emitter_clone = Arc::clone(&output_emitter);
        let opts_clone = opts.clone();

        let handle = thread::spawn(move || {
            run_worker_thread(
                gpu_index,
                barrier_clone,
                telemetry_clone,
                emitter_clone,
                opts_clone,
            )
        });
        handles.push(handle);
    }

    // 3. Collect outcomes from all worker threads
    let mut results = Vec::with_capacity(num_gpus);
    let mut any_cancelled = false;
    let mut trip_error: Option<ThermalTripReport> = None;
    let mut thread_failure: Option<String> = None;

    for handle in handles {
        match handle.join() {
            Ok(outcome) => match outcome {
                WorkerOutcome::Success(res) => results.push(*res),
                WorkerOutcome::ThermalTrip(trip) => trip_error = Some(trip),
                WorkerOutcome::Cancelled => any_cancelled = true,
                WorkerOutcome::Failed(err) => thread_failure = Some(err),
            },
            Err(_) => {
                thread_failure = Some("Worker thread panicked".to_string());
            }
        }
    }

    let elapsed_total = start_time.elapsed().as_secs_f64();

    // Check cancellation
    if any_cancelled || signal::is_termination_requested() {
        output_emitter.emit_cancelled(
            "Process terminated by SIGINT / SIGTERM signal",
            elapsed_total,
        );
        return Ok(ExitCode::Cancelled);
    }

    // Check thermal tripwire
    if let Some(trip) = trip_error {
        output_emitter.emit_thermal_trip(
            trip.gpu_index,
            &trip.sensor_type,
            trip.current_temp_c,
            trip.ceiling_temp_c,
            trip.crit_limit_c,
        );
        let msg = format!(
            "Emergency thermal failsafe triggered: {} temp {:.1}°C exceeded ceiling {:.1}°C ({})",
            trip.sensor_type, trip.current_temp_c, trip.ceiling_temp_c, trip.message
        );
        output_emitter.emit_error("ThermalTrip", &msg, ExitCode::WorkloadFailure);
        return Ok(ExitCode::WorkloadFailure);
    }

    // Check general failure
    if let Some(err) = thread_failure {
        output_emitter.emit_error("InternalError", &err, ExitCode::InternalError);
        return Ok(ExitCode::InternalError);
    }

    // 4. Outlier Detection across completed GPUs
    let run_stats: Vec<GpuRunStats> = results
        .iter()
        .map(|r| {
            let avg_fps = r.report.total_frames as f64 / (r.duration_s + 1e-6);
            GpuRunStats {
                gpu_index: r.gpu_index,
                pci_bus_id: r.pci_bus_id.clone(),
                gpu_name: r.gpu_name.clone(),
                duration_s: r.duration_s,
                score: r.report.overall_score,
                avg_fps,
                peak_temp_c: if r.report.peak_gpu_temp_c > 0.0 {
                    Some(r.report.peak_gpu_temp_c)
                } else {
                    None
                },
                peak_power_w: if r.report.peak_gpu_power_w > 0.0 {
                    Some(r.report.peak_gpu_power_w)
                } else {
                    None
                },
            }
        })
        .collect();

    let outlier_flags = detect_gpu_outliers(&run_stats);

    let mut outlier_gpu_indices = Vec::new();
    let mut total_node_score = 0.0f64;

    // 5. Emit per-GPU results
    for (res, flag) in results.iter().zip(outlier_flags.iter()) {
        total_node_score += res.report.overall_score;
        if flag.is_outlier {
            outlier_gpu_indices.push(res.gpu_index);
        }

        let scenes: Vec<SceneResultRecord> = res
            .report
            .scene_results
            .iter()
            .map(|s| SceneResultRecord {
                scene_id: s.scene_id,
                name: s.scene_name.clone(),
                fps: s.avg_fps,
                score: s.score,
            })
            .collect();

        let avg_fps = res.report.total_frames as f64 / (res.duration_s + 1e-6);
        let mut gpu_rec = GpuResultRecord::new(
            res.gpu_index,
            &res.gpu_name,
            res.report.overall_score,
            avg_fps,
            0.0,
            scenes,
            if res.report.peak_gpu_temp_c > 0.0 {
                Some(res.report.peak_gpu_temp_c)
            } else {
                None
            },
            if res.report.peak_gpu_power_w > 0.0 {
                Some(res.report.peak_gpu_power_w)
            } else {
                None
            },
        );

        if !res.pci_bus_id.is_empty() {
            gpu_rec = gpu_rec.with_pci_bus_id(res.pci_bus_id.as_str());
        }

        if flag.is_outlier {
            gpu_rec = gpu_rec.with_outliers(true, flag.reasons.clone());
        }

        output_emitter.emit_jsonl(&gpu_rec);
    }

    // 6. Emit Node Summary Record
    let summary = SummaryRecord::new_with_preset(
        "ok",
        &opts.res_tag,
        total_node_score,
        elapsed_total,
        num_gpus,
        outlier_gpu_indices.clone(),
        ExitCode::Ok.as_i32(),
    );
    output_emitter.emit_jsonl(&summary);

    // 7. Human Mode Console Output
    if output_emitter.format().is_human() {
        println!();
        println!(
            "\x1b[38;2;0;220;255;1m╔══════════════════════════════════════════════════════════════════════════╗"
        );
        println!("║                  DYNO_TERM MULTI-GPU NODE BENCHMARK                      ║");
        println!(
            "╠══════════════════════════════════════════════════════════════════════════╣\x1b[0m"
        );
        println!(
            "\x1b[38;2;140;150;170m║ Concurrent GPUs: \x1b[38;2;255;255;255;1m{:<53}\x1b[38;2;0;220;255m║",
            format!("{} Active Workers", num_gpus)
        );
        println!(
            "\x1b[38;2;140;150;170m║ Compute Preset:  \x1b[38;2;0;255;200;1m{:<53}\x1b[38;2;0;220;255m║",
            opts.res_tag
        );
        println!(
            "\x1b[38;2;140;150;170m║ Total Node Score:\x1b[38;2;255;180;0;1m{:<53}\x1b[38;2;0;220;255m║",
            format!("{:.2} POINTS (Node Sum)", total_node_score)
        );
        println!(
            "\x1b[38;2;0;220;255m╠══════════════════════════════════════════════════════════════════════════╣\x1b[0m"
        );

        for (res, flag) in results.iter().zip(outlier_flags.iter()) {
            let outlier_tag = if flag.is_outlier {
                format!(
                    " \x1b[38;2;255;80;80;1m[OUTLIER: {}]\x1b[0m",
                    flag.reasons.join(", ")
                )
            } else {
                String::new()
            };
            println!(
                "\x1b[38;2;140;150;170m║ GPU {}: \x1b[38;2;255;255;255;1m{} ({})\x1b[0m{}",
                res.gpu_index,
                res.gpu_name,
                res.pci_bus_id.as_str(),
                outlier_tag
            );
            println!(
                "\x1b[38;2;140;150;170m║   Score: \x1b[38;2;255;180;0;1m{:.2}\x1b[38;2;140;150;170m pts │ Duration: \x1b[38;2;0;255;200;1m{:.2}s\x1b[38;2;140;150;170m │ Peak Temp: \x1b[38;2;0;255;200;1m{:.0}°C\x1b[0m",
                res.report.overall_score, res.duration_s, res.report.peak_gpu_temp_c
            );
        }

        println!(
            "\x1b[38;2;0;220;255m╚══════════════════════════════════════════════════════════════════════════╝\x1b[0m"
        );

        if !outlier_gpu_indices.is_empty() {
            println!(
                "\r\n\x1b[38;2;255;100;100;1m⚠ Fleet Outlier Warning: GPUs {:?} deviated significantly from peer median.\x1b[0m",
                outlier_gpu_indices
            );
        }
    } else if output_emitter.format().is_json()
        && let Some(first) = results.first()
        && let Ok(json) = serde_json::to_string_pretty(&first.report)
    {
        println!("{}", json);
    }

    // Save report file if requested
    if let Some(ref path) = opts.report_file
        && let Some(first) = results.first()
    {
        let _ = BenchmarkScorer::save_report(&first.report, path);
    }

    Ok(ExitCode::Ok)
}

fn run_worker_thread(
    gpu_index: usize,
    barrier: Arc<Barrier>,
    telemetry: Arc<TelemetryCollector>,
    output_emitter: Arc<OutputEmitter>,
    opts: MultiGpuBenchmarkOptions,
) -> WorkerOutcome {
    // 1. Initialize GPU context for this specific adapter index
    let ctx = match pollster::block_on(GpuContext::new_for_adapter_index(gpu_index)) {
        Ok(c) => c,
        Err(e) => return WorkerOutcome::Failed(format!("GPU {} init error: {}", gpu_index, e)),
    };

    let pci_id = ctx.adapter_info.pci_bus_id.clone();
    let gpu_name = ctx.adapter_info.name.clone();

    // 2. Initialize Compute Pipeline for this worker's device/queue
    let pipeline = match ComputePipelineManager::new(
        Arc::clone(&ctx.device),
        Arc::clone(&ctx.queue),
        crate::shaders::UNIFIED_SHADER,
        opts.compute_width,
        opts.compute_height,
    ) {
        Ok(p) => p,
        Err(e) => return WorkerOutcome::Failed(format!("GPU {} pipeline error: {}", gpu_index, e)),
    };

    let mut failsafe = ThermalFailsafe::new(opts.max_temp_c, opts.failsafe_enabled)
        .with_gpu_index(gpu_index)
        .with_datacenter_preset(opts.is_datacenter_preset);
    let mut scorer = BenchmarkScorer::new();
    let compute_pixels = (opts.compute_width * opts.compute_height) as u64;

    let mut nbody = NBodySimulation::new(
        32_000,
        opts.compute_width as usize,
        opts.compute_height as usize,
    );

    // 3. Coordinate start with peer workers
    barrier.wait();
    let run_start = Instant::now();
    let mut accumulated_scenes: std::collections::BTreeMap<u32, crate::benchmark::SceneResult> =
        std::collections::BTreeMap::new();
    let duration_limit = opts.duration_secs.map(Duration::from_secs);
    let mut total_rendered_frames: u32 = 0;
    let mut scene_cycle = 0u32;

    loop {
        if signal::is_termination_requested() {
            return WorkerOutcome::Cancelled;
        }

        if let Some(limit) = duration_limit {
            if run_start.elapsed() >= limit && !accumulated_scenes.is_empty() {
                break;
            }
        } else if scene_cycle >= 8 {
            break;
        }

        if opts.duration_secs.is_none() {
            barrier.wait();
        }

        let scene_id = scene_cycle % 8;
        scene_cycle += 1;

        let scene_start = Instant::now();
        let mut last_progress_emit = Instant::now();
        let mut frames_completed_in_pass = 0u32;

        for frame_idx in 0..opts.frames_per_scene {
            if signal::is_termination_requested() {
                return WorkerOutcome::Cancelled;
            }
            if let Some(limit) = duration_limit
                && run_start.elapsed() >= limit
                && (frames_completed_in_pass > 0 || !accumulated_scenes.is_empty())
            {
                break;
            }

            let frame_start = Instant::now();
            let elapsed = scene_start.elapsed().as_secs_f32();

            if scene_id < 7 {
                let uniforms = Uniforms {
                    time: elapsed,
                    frame: frame_idx,
                    width: opts.compute_width,
                    height: opts.compute_height,
                    cam_pos: [0.0, 1.8, -4.5],
                    scene_id,
                    cam_dir: [0.0, -0.2, 1.0],
                    benchmark_mode: 1,
                };

                // In headless mode: run compute uncapped with 8-byte checksum (avoids ~530MB readback bottleneck)
                if output_emitter.is_headless() {
                    let _checksum = pipeline.render_compute_sync_with_checksum(&uniforms);
                } else {
                    pipeline.render_compute_sync(&uniforms);
                }

                let frame_dur = frame_start.elapsed();
                let stats = telemetry.sample_or_default(&pci_id, &gpu_name);

                if let Some(trip) = failsafe.check(&stats) {
                    return WorkerOutcome::ThermalTrip(trip);
                }

                scorer.record_frame_with_stats(frame_dur, scene_id, &stats);
                frames_completed_in_pass += 1;
                total_rendered_frames += 1;

                // Stream jsonl progress
                if output_emitter.format().is_jsonl()
                    && (last_progress_emit.elapsed() >= Duration::from_secs(1)
                        || frame_idx == opts.frames_per_scene - 1)
                {
                    last_progress_emit = Instant::now();
                    let current_fps = if frame_dur.as_secs_f32() > 0.0 {
                        1.0 / frame_dur.as_secs_f32()
                    } else {
                        0.0
                    };
                    let mut prog = ProgressRecord::new(
                        scene_id,
                        SCENE_NAMES[scene_id as usize],
                        frame_idx + 1,
                        opts.frames_per_scene,
                        current_fps,
                        gpu_index,
                        if stats.gpu_temp_c > 0.0 {
                            Some(stats.gpu_temp_c)
                        } else {
                            None
                        },
                        if stats.gpu_power_w > 0.0 {
                            Some(stats.gpu_power_w)
                        } else {
                            None
                        },
                    );
                    if !pci_id.is_empty() {
                        prog = prog.with_pci_bus_id(pci_id.as_str());
                    }
                    output_emitter.emit_jsonl(&prog);
                }
            } else {
                // Scene 7: Multi-threaded CPU N-Body
                nbody.step_multithreaded(0.005);
                let frame_dur = frame_start.elapsed();
                let stats = telemetry.sample_or_default(&pci_id, &gpu_name);

                if let Some(trip) = failsafe.check(&stats) {
                    return WorkerOutcome::ThermalTrip(trip);
                }

                scorer.record_frame_with_stats(frame_dur, scene_id, &stats);
                frames_completed_in_pass += 1;
                total_rendered_frames += 1;

                if output_emitter.format().is_jsonl()
                    && (last_progress_emit.elapsed() >= Duration::from_secs(1)
                        || frame_idx == opts.frames_per_scene - 1)
                {
                    last_progress_emit = Instant::now();
                    let current_fps = if frame_dur.as_secs_f32() > 0.0 {
                        1.0 / frame_dur.as_secs_f32()
                    } else {
                        0.0
                    };
                    let mut prog = ProgressRecord::new(
                        scene_id,
                        SCENE_NAMES[scene_id as usize],
                        frame_idx + 1,
                        opts.frames_per_scene,
                        current_fps,
                        gpu_index,
                        if stats.gpu_temp_c > 0.0 {
                            Some(stats.gpu_temp_c)
                        } else {
                            None
                        },
                        if stats.gpu_power_w > 0.0 {
                            Some(stats.gpu_power_w)
                        } else {
                            None
                        },
                    );
                    if !pci_id.is_empty() {
                        prog = prog.with_pci_bus_id(pci_id.as_str());
                    }
                    output_emitter.emit_jsonl(&prog);
                }
            }
        }

        if frames_completed_in_pass > 0 {
            let scene_result = scorer.compute_scene_result(
                scene_id,
                SCENE_NAMES[scene_id as usize],
                compute_pixels,
            );
            let stats_end = telemetry.sample_or_default(&pci_id, &gpu_name);
            scorer.record_telemetry_snapshot(scene_id, &stats_end, true);

            accumulated_scenes
                .entry(scene_id)
                .and_modify(|existing| {
                    existing.frames_rendered += scene_result.frames_rendered;
                    existing.elapsed_seconds += scene_result.elapsed_seconds;
                    if existing.elapsed_seconds > 0.0 {
                        existing.avg_fps =
                            existing.frames_rendered as f64 / existing.elapsed_seconds;
                        existing.avg_frametime_ms =
                            (existing.elapsed_seconds / existing.frames_rendered as f64) * 1000.0;
                    }
                    existing.p1_low_frametime_ms = existing
                        .p1_low_frametime_ms
                        .max(scene_result.p1_low_frametime_ms);
                    existing.score += scene_result.score;
                })
                .or_insert(scene_result);
        }
    }

    let scene_results: Vec<crate::benchmark::SceneResult> =
        accumulated_scenes.into_values().collect();
    let duration_s = run_start.elapsed().as_secs_f64();
    let total_score: f64 =
        ((scene_results.iter().map(|s| s.score).sum::<f64>()) * 100.0).round() / 100.0;

    let stats = telemetry.sample_or_default(&pci_id, &gpu_name);
    let assessment = scorer.generate_telemetry_assessment(&stats);
    let (tflops, bandwidth_gbps, energy_eff) = BenchmarkScorer::compute_hardware_metrics(
        total_rendered_frames,
        compute_pixels,
        duration_s,
        stats.gpu_power_w,
    );

    let report = BenchmarkReport {
        timestamp: {
            let secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            format!("UTC_Epoch_{}", secs)
        },
        system_info: SystemReport {
            gpu_name: gpu_name.clone(),
            gpu_backend: format!("{:?}", ctx.adapter_info.backend),
            gpu_driver: ctx.adapter_info.driver.clone(),
            cpu_name: stats.cpu_name,
            cpu_threads: stats.cpu_threads,
        },
        preset: opts.res_tag.clone(),
        total_elapsed_seconds: duration_s,
        total_frames: total_rendered_frames,
        overall_score: total_score,
        peak_gpu_temp_c: scorer.peak_temp,
        peak_gpu_hotspot_c: scorer.peak_hotspot,
        peak_gpu_util_pct: scorer.peak_util,
        peak_gpu_power_w: scorer.peak_power,
        peak_gpu_clock_mhz: scorer.peak_clock.max(stats.peak_gpu_clock_mhz),
        peak_vram_mb: scorer.peak_vram,
        scene_results,
        estimated_tflops: tflops,
        effective_bandwidth_gbps: bandwidth_gbps,
        energy_joules_per_mray: energy_eff,
        gpu_telemetry: Some(assessment),
        telemetry_samples: scorer.snapshots.clone(),
    };

    WorkerOutcome::Success(Box::new(GpuRunResult {
        gpu_index,
        pci_bus_id: pci_id,
        gpu_name,
        duration_s,
        report,
    }))
}
