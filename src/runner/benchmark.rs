use crossterm::event::{self, Event, KeyCode};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::benchmark::{
    BenchmarkReport, BenchmarkScorer, DEFAULT_BASELINE_FILE, DEFAULT_HISTORY_FILE, SystemReport,
    ThermalFailsafe, compare_reports, format_comparison_table, format_gpu_assessment_box,
    load_baseline, print_emergency_thermal_box,
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
    let mut scene_results = Vec::new();
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

    for scene_id in 0..8u32 {
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

        for frame_idx in 0..opts.frames_per_scene {
            // Check for exit abort key or HUD toggle
            if event::poll(Duration::from_millis(0))?
                && let Event::Key(key) = event::read()?
            {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        return Ok(BenchmarkPassResult::AbortedByUser);
                    }
                    KeyCode::Char('h') | KeyCode::Char('H') => {
                        hud.toggle_mode();
                    }
                    _ => {}
                }
            }

            let frame_start = Instant::now();
            let elapsed = scene_start.elapsed().as_secs_f32();
            total_frames_completed += 1;

            // Decoupled 60 FPS Terminal preview
            let should_draw_terminal = last_terminal_draw.elapsed() >= Duration::from_millis(16)
                || frame_idx == opts.frames_per_scene - 1;

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

        let scene_result =
            scorer.compute_scene_result(scene_id, SCENE_NAMES[scene_id as usize], compute_pixels);
        let stats_end = session.telemetry.sample(&session.adapter_info.name);
        scorer.record_telemetry_snapshot(scene_id, &stats_end, true);

        cumulative_score += scene_result.score;
        scene_results.push(scene_result);
    }

    let total_elapsed = total_benchmark_start.elapsed().as_secs_f64();
    let total_score: f64 =
        ((scene_results.iter().map(|s| s.score).sum::<f64>()) * 100.0).round() / 100.0;

    let stats = session.telemetry.sample(&session.adapter_info.name);
    let assessment = scorer.generate_telemetry_assessment(&stats);
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
        total_elapsed_seconds: total_elapsed,
        total_frames: opts.frames_per_scene * 8,
        overall_score: total_score,
        peak_gpu_temp_c: scorer.peak_temp,
        peak_gpu_hotspot_c: scorer.peak_hotspot,
        peak_gpu_util_pct: scorer.peak_util,
        peak_gpu_power_w: scorer.peak_power,
        peak_gpu_clock_mhz: scorer.peak_clock.max(stats.peak_gpu_clock_mhz),
        peak_vram_mb: scorer.peak_vram,
        scene_results,
        gpu_telemetry: Some(assessment),
        telemetry_samples: scorer.snapshots.clone(),
    };

    Ok(BenchmarkPassResult::Completed(Box::new(report)))
}

pub fn run_benchmark(
    session: &mut TerminalSession,
    opts: &BenchmarkOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    session.renderer.enter_alternate_screen()?;
    crossterm::terminal::enable_raw_mode()?;

    let pass_opts = BenchmarkPassOptions {
        frames_per_scene: opts.frames_per_scene,
        iteration_info: None,
        max_temp_c: opts.max_temp_c,
        failsafe_enabled: opts.failsafe_enabled,
    };

    let pass_res = execute_benchmark_pass(session, &pass_opts);

    crossterm::terminal::disable_raw_mode()?;
    let _ = session.renderer.leave_alternate_screen();

    let report = match pass_res? {
        BenchmarkPassResult::Completed(r) => *r,
        BenchmarkPassResult::AbortedByUser => {
            println!("Benchmark aborted by user.");
            return Ok(());
        }
        BenchmarkPassResult::ThermalTrip(trip) => {
            print_emergency_thermal_box(&trip);
            return Ok(());
        }
    };

    let _ = BenchmarkScorer::save_report(&report, &opts.output_path);
    let _ = crate::benchmark::append_to_history(&report, DEFAULT_HISTORY_FILE);
    print_scorecard(&report, &opts.output_path, session.res_tag);

    // If baseline exists, display comparison table
    if let Some(baseline) = load_baseline(DEFAULT_BASELINE_FILE) {
        let comparison = compare_reports(&report, &baseline);
        println!();
        print!("{}", format_comparison_table(&comparison));
    } else {
        println!(
            "\r\n\x1b[38;2;120;140;160m💡 Tip: Set this run as your reference baseline with:\x1b[0m \x1b[1mdyno_term --set-baseline\x1b[0m"
        );
    }

    Ok(())
}

pub fn print_scorecard(report: &BenchmarkReport, output_path: &str, resolution: &str) {
    println!();
    println!(
        "\x1b[38;2;0;220;255;1m┌──────────────────────────────────────────────────────────────────────────┐"
    );
    println!("│                       DYNO_TERM BENCHMARK SCORECARD                      │");
    println!("├──────────────────────────────────────────────────────────────────────────┤\x1b[0m");
    println!(
        "\x1b[38;2;140;150;170m│ GPU: \x1b[38;2;255;255;255;1m{:<67}\x1b[38;2;0;220;255m│",
        format!(
            "{} ({})",
            report.system_info.gpu_name, report.system_info.gpu_backend
        )
    );
    println!(
        "\x1b[38;2;140;150;170m│ CPU: \x1b[38;2;255;255;255;1m{:<67}\x1b[38;2;0;220;255m│",
        format!(
            "{} ({} Threads)",
            report.system_info.cpu_name, report.system_info.cpu_threads
        )
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
