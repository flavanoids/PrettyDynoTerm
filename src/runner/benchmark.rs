use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::benchmark::{
    BenchmarkReport, BenchmarkScorer, DEFAULT_BASELINE_FILE, DEFAULT_HISTORY_FILE, SystemReport,
    ThermalFailsafe, compare_reports, format_comparison_table, format_gpu_assessment_box,
    load_baseline, print_emergency_thermal_box, sanitize_terminal_string,
};
use crate::cpu::NBodySimulation;
use crate::gpu::pipeline::Uniforms;
use crate::runner::demo::SCENE_NAMES;
use crate::runner::session::{
    BenchmarkOptions, BenchmarkPassOptions, BenchmarkPassResult, TerminalSession,
};
use crate::terminal::{Hud, get_scene_math_model, visible_len};

pub fn execute_benchmark_pass(
    session: &mut TerminalSession,
    opts: &BenchmarkPassOptions,
) -> Result<BenchmarkPassResult, Box<dyn std::error::Error>> {
    let mut failsafe = ThermalFailsafe::new(opts.max_temp_c, opts.failsafe_enabled);
    let mut scorer = BenchmarkScorer::new();
    let total_benchmark_start = Instant::now();
    let compute_pixels = (session.pipeline.width * session.pipeline.height) as u64;
    let total_bench_frames = (opts.frames_per_scene * 8).max(1);
    let mut total_frames_completed: u32 = 0;
    let mut cumulative_score: f64 = 0.0;

    let mut term_preview_buf = vec![0u8; session.term_width * session.term_height * 4];
    let mut nbody = NBodySimulation::new(
        32_000,
        session.pipeline.width as usize,
        session.pipeline.height as usize,
    );

    let duration_limit = opts.duration_secs.map(Duration::from_secs);
    let mut accumulated_scenes: std::collections::BTreeMap<u32, crate::benchmark::SceneResult> =
        std::collections::BTreeMap::new();
    let mut scene_cycle = 0u32;

    loop {
        if crate::signal::is_termination_requested() {
            return Ok(BenchmarkPassResult::Cancelled);
        }

        if let Some(limit) = duration_limit {
            if total_benchmark_start.elapsed() >= limit && !accumulated_scenes.is_empty() {
                break;
            }
        } else if scene_cycle >= 8 {
            break;
        }

        let scene_id = scene_cycle % 8;
        scene_cycle += 1;

        let mut hud = Hud::new(
            session.term_width,
            SCENE_NAMES[scene_id as usize].to_string(),
        );
        hud.resolution_label = session.res_tag.to_string();
        hud.math_model = get_scene_math_model(scene_id).to_string();
        hud.is_benchmark = true;
        hud.act_index = (scene_id + 1) as usize;
        hud.act_total = 8;
        hud.iteration_info = opts.iteration_info;
        hud.current_score = cumulative_score;

        let scene_start = Instant::now();
        let mut last_terminal_draw = Instant::now();
        let mut last_progress_emit = Instant::now();
        let mut frames_in_this_pass = 0u32;

        for frame_idx in 0..opts.frames_per_scene {
            // Check for process termination signal (SIGINT / SIGTERM)
            if crate::signal::is_termination_requested() {
                return Ok(BenchmarkPassResult::Cancelled);
            }

            if let Some(limit) = duration_limit
                && total_benchmark_start.elapsed() >= limit
                && (frames_in_this_pass > 0 || !accumulated_scenes.is_empty())
            {
                break;
            }

            // Check for exit abort key or HUD toggle (interactive mode only)
            if !session.is_headless()
                && event::poll(Duration::from_millis(0))?
                && let Event::Key(key) = event::read()?
            {
                if (key.modifiers.contains(KeyModifiers::CONTROL)
                    && (key.code == KeyCode::Char('c') || key.code == KeyCode::Char('C')))
                    || key.code == KeyCode::Char('q')
                    || key.code == KeyCode::Esc
                {
                    return Ok(BenchmarkPassResult::AbortedByUser);
                }
                match key.code {
                    KeyCode::Char('h') | KeyCode::Char('H') => {
                        hud.toggle_mode();
                    }
                    _ => {}
                }
            }

            let frame_start = Instant::now();
            let elapsed = scene_start.elapsed().as_secs_f32();
            total_frames_completed += 1;
            frames_in_this_pass += 1;

            // Decoupled 60 FPS Terminal preview (only drawn when not headless)
            let should_draw_terminal = !session.is_headless()
                && (last_terminal_draw.elapsed() >= Duration::from_millis(16)
                    || frame_idx == opts.frames_per_scene - 1);

            if scene_id < 7 {
                // GPU Compute Scenes (Full Uncapped Stress)
                let uniforms = Uniforms {
                    time: elapsed,
                    frame: frame_idx,
                    width: session.pipeline.width,
                    height: session.pipeline.height,
                    cam_pos: [0.0, 1.8, -4.5],
                    scene_id,
                    cam_dir: [0.0, -0.2, 1.0],
                    benchmark_mode: 1,
                };

                if should_draw_terminal {
                    session.pipeline.render_with_preview(
                        &uniforms,
                        session.term_width as u32,
                        session.term_height as u32,
                        &mut term_preview_buf,
                    );
                } else {
                    session.pipeline.render_compute_sync(&uniforms);
                }
                let frame_dur = frame_start.elapsed();

                let stats = session.telemetry.sample(&session.adapter_info.name);

                // Real-time thermal tripwire guard
                if let Some(trip) = failsafe.check(&stats) {
                    return Ok(BenchmarkPassResult::ThermalTrip(trip));
                }

                scorer.record_frame_with_stats(frame_dur, scene_id, &stats);

                // Emit streaming progress in jsonl mode (>= 1/s or scene boundary)
                if session.output_emitter.format().is_jsonl()
                    && (last_progress_emit.elapsed() >= Duration::from_secs(1)
                        || frame_idx == opts.frames_per_scene - 1)
                {
                    last_progress_emit = Instant::now();
                    let current_fps = if frame_dur.as_secs_f32() > 0.0 {
                        1.0 / frame_dur.as_secs_f32()
                    } else {
                        0.0
                    };
                    let prog = crate::output::ProgressRecord::new(
                        scene_id,
                        SCENE_NAMES[scene_id as usize],
                        frame_idx + 1,
                        opts.frames_per_scene,
                        current_fps,
                        0,
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
                    session.output_emitter.emit_jsonl(&prog);
                }

                if should_draw_terminal {
                    last_terminal_draw = Instant::now();
                    let frame_ms = frame_dur.as_secs_f32() * 1000.0;
                    hud.record_frametime(frame_ms);
                    let live_scene_score = scorer.current_scene_score(scene_id, compute_pixels);
                    hud.current_score = cumulative_score + live_scene_score;
                    hud.act_progress_pct =
                        ((frame_idx + 1) as f32 / opts.frames_per_scene as f32) * 100.0;
                    hud.overall_progress_pct =
                        (total_frames_completed as f32 / total_bench_frames as f32) * 100.0;
                    hud.rays_per_sec =
                        (compute_pixels as f32 / (frame_dur.as_secs_f32() + 1e-6)) / 1e9;

                    let top_lines = hud.render_top();
                    let bottom_lines = hud.render_bottom(&stats);
                    session.renderer.render(
                        &term_preview_buf,
                        session.term_width,
                        session.term_height,
                        &top_lines,
                        &bottom_lines,
                    )?;
                }
            } else {
                // Act VII: Multi-Threaded CPU N-Body Scene (32,000 celestial bodies)
                nbody.step_multithreaded(0.005);
                let frame_dur = frame_start.elapsed();

                let stats = session.telemetry.sample(&session.adapter_info.name);

                // Real-time thermal tripwire guard
                if let Some(trip) = failsafe.check(&stats) {
                    return Ok(BenchmarkPassResult::ThermalTrip(trip));
                }

                scorer.record_frame_with_stats(frame_dur, scene_id, &stats);

                // Emit streaming progress in jsonl mode (>= 1/s or scene boundary)
                if session.output_emitter.format().is_jsonl()
                    && (last_progress_emit.elapsed() >= Duration::from_secs(1)
                        || frame_idx == opts.frames_per_scene - 1)
                {
                    last_progress_emit = Instant::now();
                    let current_fps = if frame_dur.as_secs_f32() > 0.0 {
                        1.0 / frame_dur.as_secs_f32()
                    } else {
                        0.0
                    };
                    let prog = crate::output::ProgressRecord::new(
                        scene_id,
                        SCENE_NAMES[scene_id as usize],
                        frame_idx + 1,
                        opts.frames_per_scene,
                        current_fps,
                        0,
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
                    session.output_emitter.emit_jsonl(&prog);
                }

                if should_draw_terminal {
                    last_terminal_draw = Instant::now();
                    let frame_ms = frame_dur.as_secs_f32() * 1000.0;
                    hud.record_frametime(frame_ms);
                    let live_scene_score = scorer.current_scene_score(scene_id, compute_pixels);
                    hud.current_score = cumulative_score + live_scene_score;
                    hud.act_progress_pct =
                        ((frame_idx + 1) as f32 / opts.frames_per_scene as f32) * 100.0;
                    hud.overall_progress_pct =
                        (total_frames_completed as f32 / total_bench_frames as f32) * 100.0;
                    hud.rays_per_sec =
                        (compute_pixels as f32 / (frame_dur.as_secs_f32() + 1e-6)) / 1e9;

                    nbody.rasterize_to_preview(
                        &mut term_preview_buf,
                        session.term_width,
                        session.term_height,
                    );

                    let top_lines = hud.render_top();
                    let bottom_lines = hud.render_bottom(&stats);
                    session.renderer.render(
                        &term_preview_buf,
                        session.term_width,
                        session.term_height,
                        &top_lines,
                        &bottom_lines,
                    )?;
                }
            }
        }

        if frames_in_this_pass > 0 {
            let scene_result = scorer.compute_scene_result(
                scene_id,
                SCENE_NAMES[scene_id as usize],
                compute_pixels,
            );
            let stats_end = session.telemetry.sample(&session.adapter_info.name);
            scorer.record_telemetry_snapshot(scene_id, &stats_end, true);

            cumulative_score += scene_result.score;
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
    let total_elapsed = total_benchmark_start.elapsed().as_secs_f64();
    let total_score: f64 =
        ((scene_results.iter().map(|s| s.score).sum::<f64>()) * 100.0).round() / 100.0;

    let stats = session.telemetry.sample(&session.adapter_info.name);
    let assessment = scorer.generate_telemetry_assessment(&stats);
    let (tflops, bandwidth_gbps, energy_eff) = BenchmarkScorer::compute_hardware_metrics(
        total_frames_completed,
        compute_pixels,
        total_elapsed,
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
            gpu_name: session.adapter_info.name.clone(),
            gpu_backend: format!("{:?}", session.adapter_info.backend),
            gpu_driver: session.adapter_info.driver.clone(),
            cpu_name: stats.cpu_name,
            cpu_threads: stats.cpu_threads,
        },
        preset: session.res_tag.to_string(),
        total_elapsed_seconds: total_elapsed,
        total_frames: total_frames_completed,
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

    Ok(BenchmarkPassResult::Completed(Box::new(report)))
}

pub fn run_benchmark(
    session: &mut TerminalSession,
    opts: &BenchmarkOptions,
) -> Result<crate::output::ExitCode, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    // In jsonl mode, emit StartRecord at launch
    if session.output_emitter.format().is_jsonl() {
        let start_rec = crate::output::StartRecord::new(
            "benchmark",
            session.res_tag,
            opts.frames_per_scene,
            8,
            vec![crate::output::GpuStartInfo {
                index: 0,
                name: session.adapter_info.name.clone(),
                backend: format!("{:?}", session.adapter_info.backend),
                driver: session.adapter_info.driver.clone(),
                pci_bus_id: None,
            }],
            opts.failsafe_enabled,
        );
        session.output_emitter.emit_jsonl(&start_rec);
    }

    if !session.is_headless() {
        session.renderer.enter_alternate_screen()?;
        crossterm::terminal::enable_raw_mode()?;
    }

    let pass_opts = BenchmarkPassOptions {
        frames_per_scene: opts.frames_per_scene,
        duration_secs: opts.duration_secs,
        iteration_info: None,
        max_temp_c: opts.max_temp_c,
        failsafe_enabled: opts.failsafe_enabled,
    };

    let pass_res = execute_benchmark_pass(session, &pass_opts);

    if !session.is_headless() {
        let _ = crossterm::terminal::disable_raw_mode();
        let _ = session.renderer.leave_alternate_screen();
    }

    let total_elapsed = start_time.elapsed().as_secs_f64();

    let report = match pass_res? {
        BenchmarkPassResult::Completed(r) => *r,
        BenchmarkPassResult::AbortedByUser | BenchmarkPassResult::Cancelled => {
            session
                .output_emitter
                .emit_cancelled("SIGINT or SIGTERM signal received", total_elapsed);
            return Ok(crate::output::ExitCode::Cancelled);
        }
        BenchmarkPassResult::ThermalTrip(trip) => {
            session.output_emitter.emit_error(
                "ThermalTrip",
                &trip.message,
                crate::output::ExitCode::WorkloadFailure,
            );
            if session.output_emitter.format().is_human() && !session.is_headless() {
                print_emergency_thermal_box(&trip);
            }
            return Ok(crate::output::ExitCode::WorkloadFailure);
        }
    };

    if let Some(ref report_path) = opts.report_file
        && let Err(e) = BenchmarkScorer::save_report(&report, report_path)
    {
        eprintln!(
            "Warning: Failed to save benchmark report to '{}': {}",
            report_path, e
        );
    }

    match session.output_emitter.format() {
        crate::output::OutputFormat::Jsonl => {
            // Emit GpuResultRecord
            let scenes = report
                .scene_results
                .iter()
                .map(|s| crate::output::SceneResultRecord {
                    scene_id: s.scene_id,
                    name: s.scene_name.clone(),
                    fps: s.avg_fps,
                    score: s.score,
                })
                .collect();

            let avg_fps = report.total_frames as f64 / (report.total_elapsed_seconds + 1e-6);
            let gpu_res = crate::output::GpuResultRecord::new(
                0,
                &report.system_info.gpu_name,
                report.overall_score,
                avg_fps,
                0.0,
                scenes,
                if report.peak_gpu_temp_c > 0.0 {
                    Some(report.peak_gpu_temp_c)
                } else {
                    None
                },
                if report.peak_gpu_power_w > 0.0 {
                    Some(report.peak_gpu_power_w)
                } else {
                    None
                },
            );
            session.output_emitter.emit_jsonl(&gpu_res);

            // Emit SummaryRecord
            let summary = crate::output::SummaryRecord::single(
                "ok",
                report.overall_score,
                report.total_elapsed_seconds,
                crate::output::ExitCode::Ok.as_i32(),
            );
            session.output_emitter.emit_jsonl(&summary);
        }
        crate::output::OutputFormat::Json => {
            if let Ok(json) = serde_json::to_string_pretty(&report) {
                println!("{}", json);
            }
        }
        crate::output::OutputFormat::Human => {
            let _ = crate::benchmark::append_to_history(&report, DEFAULT_HISTORY_FILE);
            if !session.is_headless() {
                let report_dest = opts.report_file.as_deref().unwrap_or("stdout");
                print_scorecard(&report, report_dest, session.res_tag);
                if let Some(baseline) = load_baseline(DEFAULT_BASELINE_FILE) {
                    let comparison = compare_reports(&report, &baseline);
                    println!();
                    print!("{}", format_comparison_table(&comparison));
                } else {
                    println!(
                        "\r\n\x1b[38;2;120;140;160m💡 Tip: Set this run as your reference baseline with:\x1b[0m \x1b[1mdyno_term --set-baseline\x1b[0m"
                    );
                }
            } else {
                print_plain_scorecard(&report, session.res_tag);
            }
        }
    }

    Ok(crate::output::ExitCode::Ok)
}

pub fn print_plain_scorecard(report: &BenchmarkReport, resolution: &str) {
    println!("=== DYNO_TERM BENCHMARK REPORT ===");
    println!(
        "GPU: {} ({})",
        report.system_info.gpu_name, report.system_info.gpu_backend
    );
    println!(
        "CPU: {} ({} Threads)",
        report.system_info.cpu_name, report.system_info.cpu_threads
    );
    println!("Resolution: {}", resolution);
    println!("Overall Score: {:.2}", report.overall_score);
    println!("Total Elapsed Time: {:.2}s", report.total_elapsed_seconds);
    println!("Scenes:");
    for s in &report.scene_results {
        println!(
            "  - {}: {:.1} FPS, {:.2} pts",
            s.scene_name, s.avg_fps, s.score
        );
    }
}

pub fn print_scorecard(report: &BenchmarkReport, output_path: &str, resolution: &str) {
    let clean_gpu = sanitize_terminal_string(&report.system_info.gpu_name);
    let clean_backend = sanitize_terminal_string(&report.system_info.gpu_backend);
    let clean_cpu = sanitize_terminal_string(&report.system_info.cpu_name);

    println!();
    println!(
        "\x1b[38;2;0;220;255;1m┌──────────────────────────────────────────────────────────────────────────┐"
    );
    println!("│                       DYNO_TERM BENCHMARK SCORECARD                      │");
    println!("├──────────────────────────────────────────────────────────────────────────┤\x1b[0m");
    println!(
        "\x1b[38;2;140;150;170m│ GPU: \x1b[38;2;255;255;255;1m{:<67}\x1b[38;2;0;220;255m│",
        format!("{} ({})", clean_gpu, clean_backend)
    );
    println!(
        "\x1b[38;2;140;150;170m│ CPU: \x1b[38;2;255;255;255;1m{:<67}\x1b[38;2;0;220;255m│",
        format!("{} ({} Threads)", clean_cpu, report.system_info.cpu_threads)
    );
    println!(
        "\x1b[38;2;140;150;170m│ Mode: \x1b[38;2;0;255;200;1m{:<29}\x1b[38;2;140;150;170m│ Target: \x1b[38;2;255;180;0;1m{:<27}\x1b[38;2;0;220;255m│",
        "Deterministic Workload", resolution
    );
    println!(
        "\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m"
    );

    for res in &report.scene_results {
        let name_trimmed: String = res.scene_name.chars().take(28).collect();
        let frame_stat = format!(
            "{:>5.1}ms ({:>4.1})",
            res.avg_frametime_ms, res.p1_low_frametime_ms
        );
        let score_stat = format!("{:>6.2} pts", res.score);
        println!(
            "\x1b[38;2;140;150;170m│ \x1b[38;2;255;255;255;1m{:<28} \x1b[38;2;0;220;255m│ \x1b[38;2;0;255;200;1m{:>5.1} FPS \x1b[38;2;0;220;255m│ \x1b[38;2;255;255;255m{:<16} \x1b[38;2;0;220;255m│ \x1b[38;2;255;180;0;1m{:<10} \x1b[38;2;0;220;255m│",
            name_trimmed, res.avg_fps, frame_stat, score_stat
        );
    }

    println!(
        "\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m"
    );
    let peak_str = format!(
        "Peak: \x1b[38;2;0;255;200;1m{:.0}°C\x1b[38;2;140;150;170m │ \x1b[38;2;0;255;200;1m{:>3.0}%\x1b[38;2;140;150;170m GPU Load │ \x1b[38;2;0;255;200;1m{:.0}W\x1b[38;2;140;150;170m │ \x1b[38;2;255;180;0;1m{}MHz\x1b[38;2;140;150;170m │ \x1b[38;2;0;255;200;1m{}MB VRAM\x1b[38;2;140;150;170m │ \x1b[38;2;255;255;255;1m{:.2}s\x1b[38;2;140;150;170m",
        report.peak_gpu_temp_c,
        report.peak_gpu_util_pct,
        report.peak_gpu_power_w,
        report.peak_gpu_clock_mhz,
        report.peak_vram_mb,
        report.total_elapsed_seconds
    );
    let vis_len = visible_len(&peak_str);
    let pad_len = 72usize.saturating_sub(vis_len);
    println!(
        "\x1b[38;2;140;150;170m│ {}{}\x1b[38;2;0;220;255m│",
        peak_str,
        " ".repeat(pad_len)
    );
    let hw_metrics_str = format!(
        "Compute: \x1b[38;2;0;255;200;1m{:.2} TFLOPS\x1b[38;2;140;150;170m │ Bandwidth: \x1b[38;2;0;255;200;1m{:.1} GB/s\x1b[38;2;140;150;170m │ Energy: \x1b[38;2;255;180;0;1m{:.2} J/Mray\x1b[38;2;140;150;170m",
        report.estimated_tflops, report.effective_bandwidth_gbps, report.energy_joules_per_mray,
    );
    let vis_len_hw = visible_len(&hw_metrics_str);
    let pad_len_hw = 72usize.saturating_sub(vis_len_hw);
    println!(
        "\x1b[38;2;140;150;170m│ {}{}\x1b[38;2;0;220;255m│",
        hw_metrics_str,
        " ".repeat(pad_len_hw)
    );
    if let Some(ref assessment) = report.gpu_telemetry {
        print!("{}", format_gpu_assessment_box(assessment));
    }
    println!(
        "\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m"
    );
    let score_str = format!("{:>10.2} POINTS ", report.overall_score);
    println!(
        "\x1b[38;2;255;255;255;1m│ OVERALL DYNO BENCHMARK SCORE:                          \x1b[38;2;255;180;0;1m{}\x1b[38;2;0;220;255m│",
        score_str
    );
    println!(
        "\x1b[38;2;0;220;255m└──────────────────────────────────────────────────────────────────────────┘\x1b[0m"
    );
    println!(
        "\x1b[38;2;140;150;170mReport saved to: \x1b[38;2;0;255;200m{}\x1b[0m\n",
        output_path
    );
}
