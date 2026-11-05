use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use dyno_term::benchmark::{
    compute_consistency_report, format_consistency_scorecard, format_gpu_assessment_box,
    save_consistency_report, BenchmarkReport, BenchmarkScorer, SystemReport,
    DEFAULT_CONSISTENCY_FILE,
};
use dyno_term::cli::{parse_resolution, Args};
use dyno_term::cpu::NBodySimulation;
use dyno_term::gpu::{self, ComputePipelineManager, GpuContext};
use dyno_term::self_test;
use dyno_term::shaders;
use dyno_term::telemetry::TelemetryCollector;
use dyno_term::terminal::{
    format_resolution_tag, get_scene_math_model, visible_len, CliMenu, Hud, HudMode, MenuAction, MenuConfig,
    TerminalRenderer,
};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.test {
        self_test::run_self_tests(args.gpu)?;
        return Ok(());
    }

    let baseline_path = args
        .baseline
        .as_deref()
        .unwrap_or(dyno_term::benchmark::DEFAULT_BASELINE_FILE);

    if args.history {
        let history = dyno_term::benchmark::load_history(dyno_term::benchmark::DEFAULT_HISTORY_FILE)?;
        let baseline = dyno_term::benchmark::load_baseline(baseline_path);
        let table = dyno_term::benchmark::format_history_table(
            &history,
            baseline.as_ref().map(|b| b.timestamp.as_str()),
        );
        print!("{}", table);
        return Ok(());
    }

    if let Some(ref report_path) = args.set_baseline {
        let content = std::fs::read_to_string(report_path)
            .map_err(|e| format!("Failed to read benchmark report at '{}': {}", report_path, e))?;
        let report: BenchmarkReport = serde_json::from_str(&content)
            .map_err(|e| format!("Invalid benchmark JSON in '{}': {}", report_path, e))?;
        dyno_term::benchmark::save_baseline(&report, baseline_path)?;
        println!(
            "\x1b[38;2;0;255;128;1m✓ Pinned benchmark from '{}' ({:.2} pts, {}) as active reference baseline ('{}')\x1b[0m",
            report_path, report.overall_score, report.timestamp, baseline_path
        );
        return Ok(());
    }

    if let Some(ref report_path) = args.compare {
        let content = std::fs::read_to_string(report_path)
            .map_err(|e| format!("Failed to read benchmark report at '{}': {}", report_path, e))?;
        let cur_report: BenchmarkReport = serde_json::from_str(&content)
            .map_err(|e| format!("Invalid benchmark JSON in '{}': {}", report_path, e))?;
        let base_report = dyno_term::benchmark::load_baseline(baseline_path).ok_or_else(|| {
            format!(
                "No reference baseline found at '{}'. Set one with --set-baseline <FILE>",
                baseline_path
            )
        })?;
        let comparison = dyno_term::benchmark::compare_reports(&cur_report, &base_report);
        print!("{}", dyno_term::benchmark::format_comparison_table(&comparison));
        return Ok(());
    }

    if args.list_gpus {
        println!("Available GPU Adapters:");
        for (i, adapter) in GpuContext::list_adapters().iter().enumerate() {
            println!(
                "  [{}] {} | Backend: {:?} | Type: {:?} | Driver: {}",
                i, adapter.name, adapter.backend, adapter.device_type, adapter.driver
            );
        }
        return Ok(());
    }

    // Initialize GPU Context
    let mut gpu_ctx = pollster::block_on(GpuContext::new(args.gpu))
        .map_err(|e| format!("GPU initialization failed: {}", e))?;

    let mut device = Arc::clone(&gpu_ctx.device);
    let mut queue = Arc::clone(&gpu_ctx.queue);
    let mut adapter_info = gpu_ctx.adapter_info.clone();

    // Terminal preview dimensions
    let (term_cols, term_rows) = crossterm::terminal::size().unwrap_or((120, 40));
    let term_width = args.width.unwrap_or(term_cols as u32).max(40);
    let hud_rows = 7u32;
    let char_height = args
        .height
        .map(|h| h / 2)
        .unwrap_or(term_rows.saturating_sub(hud_rows as u16) as u32)
        .max(8);
    let term_pixel_height = char_height * 2;

    let run_demo = args.demo;
    let run_benchmark_mode = args.benchmark;
    let run_consistency_mode = args.consistency.is_some();
    let run_menu = args.menu || (!run_demo && !run_benchmark_mode && !run_consistency_mode);

    // Canonical internal compute resolution: Defaults to 4K UHD (3840x2160) for balanced high-performance GPU compute
    let (compute_w, compute_h) = parse_resolution(&args.res, term_width, term_pixel_height);

    let mut res_tag = format_resolution_tag(compute_w, compute_h);

    // Initialize Compute Pipeline with chosen compute resolution (with pre-flight limit guard & fallback)
    let mut pipeline = match ComputePipelineManager::new(
        Arc::clone(&device),
        Arc::clone(&queue),
        shaders::UNIFIED_SHADER,
        compute_w,
        compute_h,
    ) {
        Ok(p) => p,
        Err(err) => {
            eprintln!("\x1b[38;2;255;100;100;1m⚠ Resolution {}x{} exceeds hardware limits:\x1b[0m {}", compute_w, compute_h, err);
            let (fb_w, fb_h) = dyno_term::benchmark::GpuMemoryInfo::detect(&device)
                .recommend_resolution(dyno_term::benchmark::VramTestTier::Light);
            eprintln!("\x1b[38;2;0;255;200;1m▶ Falling back safely to dynamically adapted {}x{} preset...\x1b[0m", fb_w, fb_h);
            let fallback_p = ComputePipelineManager::new(
                Arc::clone(&device),
                Arc::clone(&queue),
                shaders::UNIFIED_SHADER,
                fb_w,
                fb_h,
            ).map_err(|e| format!("Failed to create fallback pipeline: {}", e))?;
            res_tag = format_resolution_tag(fallback_p.width, fallback_p.height);
            fallback_p
        }
    };

    // Initialize Telemetry & Renderer
    let telemetry = TelemetryCollector::new(adapter_info.name.clone());
    let mut renderer = TerminalRenderer::new();

    if let Some(ref out_dir) = args.capture {
        capture_screenshots(&mut pipeline, out_dir)?;
        return Ok(());
    }

    if run_menu {
        let all_adapters = GpuContext::list_adapters();
        let mut config = MenuConfig {
            resolution_name: res_tag.clone(),
            compute_width: pipeline.width,
            compute_height: pipeline.height,
            frames_per_scene: args.frames,
            selected_scene: args.scene,
            uncapped: args.uncapped,
            selected_gpu_index: args.gpu.unwrap_or(0),
            output_path: args.output.clone(),
            consistency_iterations: 5,
            cooldown_seconds: args.cooldown,
            max_temp_c: args.max_temp,
            target_temp_c: args.target_temp,
            failsafe_enabled: !args.no_failsafe,
        };

        let mut menu = CliMenu::new();
        loop {
            let (cols, rows) = crossterm::terminal::size().unwrap_or((120, 40));
            let cur_term_w = args.width.unwrap_or(cols as u32).max(40) as usize;
            let cur_term_h = args.height.map(|h| h / 2).unwrap_or(rows.saturating_sub(7) as u32).max(8) as usize * 2;

            match menu.run(&mut config, &telemetry, &adapter_info, &all_adapters)? {
                MenuAction::LaunchDemo => {
                    let cur_res_tag = format_resolution_tag(config.compute_width, config.compute_height);
                    run_interactive_demo(
                        &mut pipeline,
                        &mut renderer,
                        &telemetry,
                        &adapter_info,
                        cur_term_w,
                        cur_term_h,
                        &cur_res_tag,
                        config.selected_scene,
                        config.uncapped,
                        config.max_temp_c,
                        config.failsafe_enabled,
                    )?;
                }
                MenuAction::LaunchUncapped => {
                    let cur_res_tag = format_resolution_tag(config.compute_width, config.compute_height);
                    run_interactive_demo(
                        &mut pipeline,
                        &mut renderer,
                        &telemetry,
                        &adapter_info,
                        cur_term_w,
                        cur_term_h,
                        &cur_res_tag,
                        config.selected_scene,
                        true,
                        config.max_temp_c,
                        config.failsafe_enabled,
                    )?;
                }
                MenuAction::RunBenchmark => {
                    let cur_res_tag = format_resolution_tag(config.compute_width, config.compute_height);
                    run_benchmark(
                        &mut pipeline,
                        &mut renderer,
                        &telemetry,
                        &adapter_info,
                        config.frames_per_scene,
                        cur_term_w,
                        cur_term_h,
                        &cur_res_tag,
                        &config.output_path,
                        config.max_temp_c,
                        config.failsafe_enabled,
                    )?;
                    println!("\r\n\x1b[38;2;0;255;200;1m▶ Benchmark finished. Press [Enter] or [Space] to return to Main Menu...\x1b[0m");
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                    wait_for_menu_return()?;
                }
                MenuAction::RunConsistency => {
                    let cur_res_tag = format_resolution_tag(config.compute_width, config.compute_height);
                    let out_file = if config.output_path == "dyno_benchmark_results.json" {
                        DEFAULT_CONSISTENCY_FILE
                    } else {
                        &config.output_path
                    };
                    run_consistency_suite(
                        &mut pipeline,
                        &mut renderer,
                        &telemetry,
                        &adapter_info,
                        config.consistency_iterations,
                        config.cooldown_seconds,
                        config.target_temp_c,
                        config.max_temp_c,
                        config.failsafe_enabled,
                        config.frames_per_scene,
                        cur_term_w,
                        cur_term_h,
                        &cur_res_tag,
                        out_file,
                    )?;
                    println!("\r\n\x1b[38;2;0;255;200;1m▶ Consistency suite finished. Press [Enter] or [Space] to return to Main Menu...\x1b[0m");
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                    wait_for_menu_return()?;
                }
                MenuAction::RecreatePipeline => {
                    let prev_w = pipeline.width;
                    let prev_h = pipeline.height;
                    match ComputePipelineManager::new(
                        Arc::clone(&device),
                        Arc::clone(&queue),
                        shaders::UNIFIED_SHADER,
                        config.compute_width,
                        config.compute_height,
                    ) {
                        Ok(new_p) => {
                            config.resolution_name = format_resolution_tag(config.compute_width, config.compute_height);
                            pipeline = new_p;
                        }
                        Err(err) => {
                            eprintln!("\r\n\x1b[38;2;255;100;100;1m⚠ Hardware Buffer Limit Exceeded: {}\x1b[0m", err);
                            eprintln!("\x1b[38;2;255;180;0;1m▶ Reverting safely to previous resolution ({}x{})...\x1b[0m", prev_w, prev_h);
                            std::thread::sleep(std::time::Duration::from_millis(2000));
                            config.compute_width = prev_w;
                            config.compute_height = prev_h;
                            config.resolution_name = format_resolution_tag(prev_w, prev_h);
                        }
                    }
                }
                MenuAction::SwitchGpu(idx) => {
                    config.selected_gpu_index = idx;
                    gpu_ctx = pollster::block_on(GpuContext::new(Some(idx)))
                        .map_err(|e| format!("Failed to switch GPU adapter: {}", e))?;
                    device = Arc::clone(&gpu_ctx.device);
                    queue = Arc::clone(&gpu_ctx.queue);
                    adapter_info = gpu_ctx.adapter_info.clone();
                    config.resolution_name = format_resolution_tag(config.compute_width, config.compute_height);
                    match ComputePipelineManager::new(
                        Arc::clone(&device),
                        Arc::clone(&queue),
                        shaders::UNIFIED_SHADER,
                        config.compute_width,
                        config.compute_height,
                    ) {
                        Ok(new_p) => {
                            pipeline = new_p;
                        }
                        Err(err) => {
                            eprintln!("\r\n\x1b[38;2;255;100;100;1m⚠ Resolution {}x{} exceeds limits on adapter {}: {}\x1b[0m", config.compute_width, config.compute_height, adapter_info.name, err);
                            eprintln!("\x1b[38;2;0;255;200;1m▶ Falling back to 1080p FHD safe preset...\x1b[0m");
                            std::thread::sleep(std::time::Duration::from_millis(2000));
                            config.compute_width = 1920;
                            config.compute_height = 1080;
                            config.resolution_name = format_resolution_tag(1920, 1080);
                            pipeline = ComputePipelineManager::new(
                                Arc::clone(&device),
                                Arc::clone(&queue),
                                shaders::UNIFIED_SHADER,
                                1920,
                                1080,
                            )?;
                        }
                    }
                }
                MenuAction::Exit => {
                    break;
                }
            }
        }
    } else if let Some(iterations) = args.consistency {
        let out_file = if args.output == "dyno_benchmark_results.json" {
            DEFAULT_CONSISTENCY_FILE
        } else {
            &args.output
        };
        run_consistency_suite(
            &mut pipeline,
            &mut renderer,
            &telemetry,
            &adapter_info,
            iterations,
            args.cooldown,
            args.target_temp,
            args.max_temp,
            !args.no_failsafe,
            args.frames,
            term_width as usize,
            term_pixel_height as usize,
            &res_tag,
            out_file,
        )?;
    } else if run_benchmark_mode {
        run_benchmark(
            &mut pipeline,
            &mut renderer,
            &telemetry,
            &adapter_info,
            args.frames,
            term_width as usize,
            term_pixel_height as usize,
            &res_tag,
            &args.output,
            args.max_temp,
            !args.no_failsafe,
        )?;
    } else {
        run_interactive_demo(
            &mut pipeline,
            &mut renderer,
            &telemetry,
            &adapter_info,
            term_width as usize,
            term_pixel_height as usize,
            &res_tag,
            args.scene,
            args.uncapped,
            args.max_temp,
            !args.no_failsafe,
        )?;
    }

    Ok(())
}

fn wait_for_menu_return() -> Result<(), Box<dyn std::error::Error>> {
    crossterm::terminal::enable_raw_mode()?;
    loop {
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Enter | KeyCode::Char(' ') | KeyCode::Esc | KeyCode::Char('q') => break,
                    _ => {}
                }
            }
        }
    }
    crossterm::terminal::disable_raw_mode()?;
    Ok(())
}

fn run_interactive_demo(
    pipeline: &mut ComputePipelineManager,
    renderer: &mut TerminalRenderer,
    telemetry: &TelemetryCollector,
    adapter_info: &gpu::GpuAdapterInfo,
    mut term_width: usize,
    mut term_height: usize,
    res_tag: &str,
    initial_scene: u32,
    initial_uncapped: bool,
    max_temp_c: f32,
    failsafe_enabled: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    renderer.enter_alternate_screen()?;
    crossterm::terminal::enable_raw_mode()?;

    let scene_names = [
        "ATMOSPHERIC OCEAN WAVES (GPU)",
        "THE MANDELBULB CORE (GPU)",
        "RELATIVISTIC BLACK HOLE (GPU)",
        "TORNADIC MESOCYCLONE (GPU)",
        "CARDIAC ELECTROPHYSIOLOGY (GPU)",
        "LATTICE BOLTZMANN VORTEX (GPU)",
        "CYBERNETIC GAIT KINEMATICS (GPU)",
        "KINETIC N-BODY SUPERNOVA (CPU)",
    ];

    let mut scene_id = initial_scene % 8;
    let mut uncapped = initial_uncapped;
    let mut failsafe = dyno_term::benchmark::ThermalFailsafe::new(max_temp_c, failsafe_enabled);
    let mut hud = Hud::new(term_width, scene_names[scene_id as usize].to_string());
    hud.resolution_label = res_tag.to_string();
    hud.math_model = get_scene_math_model(scene_id).to_string();
    hud.is_benchmark = false;
    hud.is_uncapped = uncapped;
    let start_time = Instant::now();
    let mut frame_count: u32 = 0;

    let mut nbody = NBodySimulation::new(24_000, pipeline.width as usize, pipeline.height as usize);
    let mut term_preview_buf = vec![0u8; term_width * term_height * 4];

    let target_frame_duration = Duration::from_micros(16_667);
    let mut last_terminal_draw = Instant::now();

    loop {
        let frame_start = Instant::now();

        // Handle user input
        if event::poll(Duration::from_millis(0))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => break,
                    KeyCode::Char(' ') => {
                        scene_id = (scene_id + 1) % 8;
                        hud.scene_title = scene_names[scene_id as usize].to_string();
                        hud.math_model = get_scene_math_model(scene_id).to_string();
                    }
                    KeyCode::Char('u') | KeyCode::Char('U') => {
                        uncapped = !uncapped;
                        hud.is_uncapped = uncapped;
                    }
                    KeyCode::Char('h') | KeyCode::Char('H') => {
                        let new_mode = hud.toggle_mode();
                        let hud_rows = match new_mode {
                            HudMode::Standard => 7,
                            HudMode::Minimal => 2,
                            HudMode::Hidden => 0,
                        };
                        if let Ok((_cols, rows)) = crossterm::terminal::size() {
                            term_height = (rows.saturating_sub(hud_rows as u16) as usize * 2).max(16);
                            term_preview_buf.resize(term_width * term_height * 4, 0);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Check for terminal resize
        if let Ok((cols, rows)) = crossterm::terminal::size() {
            let hud_rows = match hud.mode {
                HudMode::Standard => 7,
                HudMode::Minimal => 2,
                HudMode::Hidden => 0,
            };
            let target_w = cols as usize;
            let target_h = (rows.saturating_sub(hud_rows as u16) as usize * 2).max(16);
            if target_w != term_width || target_h != term_height {
                term_width = target_w;
                term_height = target_h;
                term_preview_buf.resize(term_width * term_height * 4, 0);
                hud.width = term_width;
            }
        }

        let elapsed = start_time.elapsed().as_secs_f32();

        // In uncapped mode, pace terminal stdout redraw to ~60 FPS (16ms) so stdout bandwidth doesn't bottleneck GPU
        let should_draw_terminal = !uncapped || last_terminal_draw.elapsed() >= Duration::from_millis(16);

        if scene_id < 7 {
            let uniforms = gpu::pipeline::Uniforms {
                time: elapsed,
                frame: frame_count,
                width: pipeline.width,
                height: pipeline.height,
                cam_pos: [0.0, 1.8, -4.5],
                scene_id,
                cam_dir: [0.0, -0.2, 1.0],
                benchmark_mode: if uncapped { 1 } else { 0 },
            };

            if should_draw_terminal {
                pipeline.render_with_preview(
                    &uniforms,
                    term_width as u32,
                    term_height as u32,
                    &mut term_preview_buf,
                );
            } else {
                pipeline.render_compute_async(&uniforms);
            }
        } else {
            nbody.step_multithreaded(0.005);
            if should_draw_terminal {
                nbody.rasterize_to_preview(&mut term_preview_buf, term_width, term_height);
            }
        }

        let stats = telemetry.sample(&adapter_info.name);

        // In-flight active emergency thermal tripwire check
        if let Some(trip) = failsafe.check(&stats) {
            if uncapped {
                // Tripwire breach under uncapped stress: disengage uncapped mode immediately to allow thermal recovery
                uncapped = false;
                hud.is_uncapped = false;
            } else {
                // Tripwire hit even while running capped at 60 FPS: emergency shutdown to protect silicon
                crossterm::terminal::disable_raw_mode()?;
                let _ = renderer.leave_alternate_screen();
                dyno_term::benchmark::print_emergency_thermal_box(&trip);
                eprintln!("\x1b[38;2;255;180;0;1m[!] Interactive demo terminated to protect hardware silicon from thermal exhaustion.\x1b[0m\r\n");
                return Ok(());
            }
        }

        let render_dur = frame_start.elapsed();

        if should_draw_terminal {
            last_terminal_draw = Instant::now();
            let frametime_ms = render_dur.as_secs_f32() * 1000.0;
            hud.record_frametime(frametime_ms);
            hud.demo_elapsed_secs = elapsed;
            hud.demo_frame_count = frame_count as u64;
            hud.is_uncapped = uncapped;
            let compute_pixels = (pipeline.width * pipeline.height) as f32;
            hud.rays_per_sec = (compute_pixels / (render_dur.as_secs_f32() + 1e-6)) / 1e9;

            let top_lines = hud.render_top();
            let bottom_lines = hud.render_bottom(&stats);

            renderer.render(
                &term_preview_buf,
                term_width,
                term_height,
                &top_lines,
                &bottom_lines,
            )?;
        }

        frame_count += 1;

        // V-Sync Sleep: only if NOT in uncapped 100% stress mode
        if !uncapped {
            let total_frame_dur = frame_start.elapsed();
            if total_frame_dur < target_frame_duration {
                std::thread::sleep(target_frame_duration - total_frame_dur);
            }
        }
    }

    crossterm::terminal::disable_raw_mode()?;
    renderer.leave_alternate_screen()?;
    Ok(())
}

#[derive(Debug, Clone)]
pub enum BenchmarkPassResult {
    Completed(BenchmarkReport),
    AbortedByUser,
    ThermalTrip(dyno_term::benchmark::ThermalTripReport),
}

fn execute_benchmark_pass(
    pipeline: &mut ComputePipelineManager,
    renderer: &mut TerminalRenderer,
    telemetry: &TelemetryCollector,
    adapter_info: &gpu::GpuAdapterInfo,
    frames_per_scene: u32,
    term_width: usize,
    term_height: usize,
    res_tag: &str,
    iteration_info: Option<(usize, usize)>,
    max_temp_c: f32,
    failsafe_enabled: bool,
) -> Result<BenchmarkPassResult, Box<dyn std::error::Error>> {
    let scene_names = [
        "ATMOSPHERIC OCEAN WAVES (GPU)",
        "THE MANDELBULB CORE (GPU)",
        "RELATIVISTIC BLACK HOLE (GPU)",
        "TORNADIC MESOCYCLONE (GPU)",
        "CARDIAC ELECTROPHYSIOLOGY (GPU)",
        "LATTICE BOLTZMANN VORTEX (GPU)",
        "CYBERNETIC GAIT KINEMATICS (GPU)",
        "KINETIC N-BODY SUPERNOVA (CPU)",
    ];

    let mut failsafe = dyno_term::benchmark::ThermalFailsafe::new(max_temp_c, failsafe_enabled);
    let mut scorer = BenchmarkScorer::new();
    let mut scene_results = Vec::new();
    let total_benchmark_start = Instant::now();
    let compute_pixels = (pipeline.width * pipeline.height) as u64;
    let total_bench_frames = (frames_per_scene * 8).max(1);
    let mut total_frames_completed: u32 = 0;
    let mut cumulative_score: f64 = 0.0;

    let mut term_preview_buf = vec![0u8; term_width * term_height * 4];
    let mut nbody = NBodySimulation::new(32_000, pipeline.width as usize, pipeline.height as usize);

    for scene_id in 0..8u32 {
        let mut hud = Hud::new(term_width, scene_names[scene_id as usize].to_string());
        hud.resolution_label = res_tag.to_string();
        hud.math_model = get_scene_math_model(scene_id).to_string();
        hud.is_benchmark = true;
        hud.act_index = (scene_id + 1) as usize;
        hud.act_total = 8;
        hud.iteration_info = iteration_info;
        hud.current_score = cumulative_score;

        let scene_start = Instant::now();
        let mut last_terminal_draw = Instant::now();

        for frame_idx in 0..frames_per_scene {
            // Check for exit abort key or HUD toggle
            if event::poll(Duration::from_millis(0))? {
                if let Event::Key(key) = event::read()? {
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
            }

            let frame_start = Instant::now();
            let elapsed = scene_start.elapsed().as_secs_f32();
            total_frames_completed += 1;

            // Decoupled 60 FPS Terminal preview
            let should_draw_terminal = last_terminal_draw.elapsed() >= Duration::from_millis(16)
                || frame_idx == frames_per_scene - 1;

            if scene_id < 7 {
                // GPU Compute Scenes (Full 16K Uncapped Stress)
                let uniforms = gpu::pipeline::Uniforms {
                    time: elapsed,
                    frame: frame_idx,
                    width: pipeline.width,
                    height: pipeline.height,
                    cam_pos: [0.0, 1.8, -4.5],
                    scene_id,
                    cam_dir: [0.0, -0.2, 1.0],
                    benchmark_mode: 1,
                };

                if should_draw_terminal {
                    pipeline.render_with_preview(
                        &uniforms,
                        term_width as u32,
                        term_height as u32,
                        &mut term_preview_buf,
                    );
                } else {
                    pipeline.render_compute_sync(&uniforms);
                }
                let frame_dur = frame_start.elapsed();

                let stats = telemetry.sample(&adapter_info.name);

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
                    hud.act_progress_pct = ((frame_idx + 1) as f32 / frames_per_scene as f32) * 100.0;
                    hud.overall_progress_pct = (total_frames_completed as f32 / total_bench_frames as f32) * 100.0;
                    hud.rays_per_sec = (compute_pixels as f32 / (frame_dur.as_secs_f32() + 1e-6)) / 1e9;

                    let top_lines = hud.render_top();
                    let bottom_lines = hud.render_bottom(&stats);
                    renderer.render(
                        &term_preview_buf,
                        term_width,
                        term_height,
                        &top_lines,
                        &bottom_lines,
                    )?;
                }
            } else {
                // Act VII: Multi-Threaded CPU N-Body Scene (32,000 celestial bodies)
                nbody.step_multithreaded(0.005);
                let frame_dur = frame_start.elapsed();

                let stats = telemetry.sample(&adapter_info.name);

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
                    hud.act_progress_pct = ((frame_idx + 1) as f32 / frames_per_scene as f32) * 100.0;
                    hud.overall_progress_pct = (total_frames_completed as f32 / total_bench_frames as f32) * 100.0;
                    hud.rays_per_sec = (compute_pixels as f32 / (frame_dur.as_secs_f32() + 1e-6)) / 1e9;

                    nbody.rasterize_to_preview(&mut term_preview_buf, term_width, term_height);

                    let top_lines = hud.render_top();
                    let bottom_lines = hud.render_bottom(&stats);
                    renderer.render(
                        &term_preview_buf,
                        term_width,
                        term_height,
                        &top_lines,
                        &bottom_lines,
                    )?;
                }
            }
        }

        let scene_result = scorer.compute_scene_result(
            scene_id,
            scene_names[scene_id as usize],
            compute_pixels,
        );
        let stats_end = telemetry.sample(&adapter_info.name);
        scorer.record_telemetry_snapshot(scene_id, &stats_end, true);

        cumulative_score += scene_result.score;
        scene_results.push(scene_result);
    }

    let total_elapsed = total_benchmark_start.elapsed().as_secs_f64();
    let total_score: f64 = ((scene_results.iter().map(|s| s.score).sum::<f64>()) * 100.0).round() / 100.0;

    let stats = telemetry.sample(&adapter_info.name);
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
            gpu_name: adapter_info.name.clone(),
            gpu_backend: format!("{:?}", adapter_info.backend),
            gpu_driver: adapter_info.driver.clone(),
            cpu_name: stats.cpu_name,
            cpu_threads: stats.cpu_threads,
        },
        total_elapsed_seconds: total_elapsed,
        total_frames: frames_per_scene * 8,
        overall_score: total_score,
        peak_gpu_temp_c: scorer.peak_temp,
        peak_gpu_hotspot_c: scorer.peak_hotspot,
        peak_gpu_util_pct: scorer.peak_util,
        peak_gpu_power_w: scorer.peak_power,
        peak_gpu_clock_mhz: scorer.peak_clock.max(stats.peak_gpu_clock_mhz),
        peak_vram_mb: scorer.peak_vram,
        scene_results: scene_results.clone(),
        gpu_telemetry: Some(assessment),
        telemetry_samples: scorer.snapshots.clone(),
    };

    Ok(BenchmarkPassResult::Completed(report))
}

fn format_box_line(content: &str, inner_width: usize) -> String {
    let vis = visible_len(content);
    let pad = inner_width.saturating_sub(vis);
    format!(
        "\x1b[38;2;0;220;255m│\x1b[0m \x1b[38;2;255;255;255m{}{}\x1b[0m \x1b[38;2;0;220;255m│\x1b[0m\r\n",
        content,
        " ".repeat(pad)
    )
}

fn run_cooldown_screen(
    renderer: &mut TerminalRenderer,
    telemetry: &TelemetryCollector,
    adapter_info: &gpu::GpuAdapterInfo,
    _term_width: usize,
    _term_height: usize,
    completed_run: usize,
    total_runs: usize,
    cooldown_seconds: u64,
    target_temp_c: f32,
    last_report: &BenchmarkReport,
) -> Result<bool, Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let cooldown_dur = Duration::from_secs(cooldown_seconds);

    loop {
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char(' ') | KeyCode::Enter => {
                            return Ok(true);
                        }
                        KeyCode::Char('q') | KeyCode::Esc => {
                            return Ok(false);
                        }
                        _ => {}
                    }
                }
            }
        }

        let elapsed = start_time.elapsed();
        let stats = telemetry.sample(&adapter_info.name);
        let cur_temp = stats.gpu_temp_c;

        let is_thermally_cooled = cur_temp > 0.0 && cur_temp <= target_temp_c;
        let is_heat_soaked = cur_temp >= 72.0;

        let time_expired = elapsed >= cooldown_dur;

        // Auto-advance when cooldown timer is finished AND GPU is not heat-soaked
        if time_expired && !is_heat_soaked {
            return Ok(true);
        }

        // Failsafe cap: if heat-soaked, wait up to an extra 30s max before auto-advancing
        if time_expired && elapsed >= cooldown_dur + Duration::from_secs(30) {
            return Ok(true);
        }

        let remaining_secs = cooldown_seconds.saturating_sub(elapsed.as_secs());
        let progress_ratio = (elapsed.as_secs_f32() / cooldown_seconds.max(1) as f32).clamp(0.0, 1.0);

        let bar_width: usize = 28;
        let filled_chars = (progress_ratio * bar_width as f32).round() as usize;
        let empty_chars = bar_width.saturating_sub(filled_chars);
        let progress_bar = format!(
            "\x1b[38;2;0;255;200;1m{}\x1b[38;2;60;80;100m{}\x1b[0m",
            "█".repeat(filled_chars),
            "░".repeat(empty_chars)
        );

        let temp_delta = cur_temp - last_report.peak_gpu_temp_c;
        let delta_sign = if temp_delta >= 0.0 { "+" } else { "" };

        let mut buf = String::with_capacity(2048);
        buf.push_str("\r\n");
        buf.push_str("\x1b[38;2;0;220;255;1m┌──────────────────────────────────────────────────────────────────────────┐\r\n");
        buf.push_str("│                   THERMAL STABILIZATION COOLDOWN PAUSE                   │\r\n");
        buf.push_str("├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        let line1 = format!(
            "Completed Run     : \x1b[38;2;255;255;255;1m{} of {}\x1b[38;2;140;150;170m (\x1b[38;2;255;180;0;1m{:>8.2} pts\x1b[38;2;140;150;170m, \x1b[38;2;255;255;255m{:.2}s\x1b[38;2;140;150;170m)",
            completed_run, total_runs, last_report.overall_score, last_report.total_elapsed_seconds
        );
        buf.push_str(&format_box_line(&line1, 72));

        let line2 = format!(
            "Next Iteration    : \x1b[38;2;0;255;200;1mRun {} of {}\x1b[38;2;140;150;170m",
            completed_run + 1, total_runs
        );
        buf.push_str(&format_box_line(&line2, 72));

        let line3 = format!(
            "Cooldown Progress : [{}] \x1b[38;2;255;180;0;1m{:>2}s\x1b[38;2;140;150;170m / {}s remaining",
            progress_bar, remaining_secs, cooldown_seconds
        );
        buf.push_str(&format_box_line(&line3, 72));

        buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        let status_badge = if is_thermally_cooled {
            format!("\x1b[38;2;0;255;200;1m[●] COOLED (≤ {:.1}°C)\x1b[0m", target_temp_c)
        } else if is_heat_soaked {
            "\x1b[38;2;255;60;60;1m[⚠] HEAT-SOAK EXHAUSTION\x1b[0m".to_string()
        } else {
            let diff = cur_temp - target_temp_c;
            format!("\x1b[38;2;0;220;255;1m[○] DISSIPATING (+{:.1}°C to target)\x1b[0m", diff.max(0.0))
        };

        let line_head = format!("THERMAL HYSTERESIS: {}", status_badge);
        buf.push_str(&format_box_line(&line_head, 72));

        let line4 = format!(
            "GPU Core Temp     : \x1b[38;2;0;255;200;1m{:.1}°C\x1b[38;2;140;150;170m (Peak: {:.1}°C, Δ: {}{:.1}°C, Goal: ≤{:.1}°C)",
            cur_temp, last_report.peak_gpu_temp_c, delta_sign, temp_delta, target_temp_c
        );
        buf.push_str(&format_box_line(&line4, 72));

        let line5 = format!(
            "GPU Hotspot Temp  : \x1b[38;2;0;255;200;1m{:.1}°C\x1b[38;2;140;150;170m (Run Peak: {:.1}°C)",
            stats.gpu_hotspot_c, last_report.peak_gpu_hotspot_c
        );
        buf.push_str(&format_box_line(&line5, 72));

        let line6 = format!(
            "GPU Clock / Power : \x1b[38;2;255;255;255;1m{} MHz\x1b[38;2;140;150;170m │ \x1b[38;2;255;255;255;1m{:.1} W\x1b[38;2;140;150;170m",
            stats.gpu_clock_mhz, stats.gpu_power_w
        );
        buf.push_str(&format_box_line(&line6, 72));

        buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        let line_foot = if is_heat_soaked && time_expired {
            "\x1b[38;2;255;180;0;1m[Space / Enter]\x1b[0m Force Continue  │  \x1b[1m[Esc / Q]\x1b[0m Finish Early"
        } else {
            "\x1b[1m[Space / Enter]\x1b[0m Skip Cooldown   │  \x1b[1m[Esc / Q]\x1b[0m Finish Early"
        };
        buf.push_str(&format_box_line(line_foot, 72));

        buf.push_str("\x1b[38;2;0;220;255;1m└──────────────────────────────────────────────────────────────────────────┘\x1b[0m\r\n");

        renderer.render_raw_text(&buf)?;
    }
}

fn run_benchmark(
    pipeline: &mut ComputePipelineManager,
    renderer: &mut TerminalRenderer,
    telemetry: &TelemetryCollector,
    adapter_info: &gpu::GpuAdapterInfo,
    frames_per_scene: u32,
    term_width: usize,
    term_height: usize,
    res_tag: &str,
    output_path: &str,
    max_temp_c: f32,
    failsafe_enabled: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    renderer.enter_alternate_screen()?;
    crossterm::terminal::enable_raw_mode()?;

    let pass_res = execute_benchmark_pass(
        pipeline,
        renderer,
        telemetry,
        adapter_info,
        frames_per_scene,
        term_width,
        term_height,
        res_tag,
        None,
        max_temp_c,
        failsafe_enabled,
    );

    crossterm::terminal::disable_raw_mode()?;
    let _ = renderer.leave_alternate_screen();

    let report = match pass_res? {
        BenchmarkPassResult::Completed(r) => r,
        BenchmarkPassResult::AbortedByUser => {
            println!("Benchmark aborted by user.");
            return Ok(());
        }
        BenchmarkPassResult::ThermalTrip(trip) => {
            dyno_term::benchmark::print_emergency_thermal_box(&trip);
            return Ok(());
        }
    };

    let _ = BenchmarkScorer::save_report(&report, output_path);
    let _ = dyno_term::benchmark::append_to_history(&report, dyno_term::benchmark::DEFAULT_HISTORY_FILE);
    print_scorecard(&report, output_path, res_tag);

    // If baseline exists, display comparison table
    if let Some(baseline) = dyno_term::benchmark::load_baseline(dyno_term::benchmark::DEFAULT_BASELINE_FILE) {
        let comparison = dyno_term::benchmark::compare_reports(&report, &baseline);
        println!();
        print!("{}", dyno_term::benchmark::format_comparison_table(&comparison));
    } else {
        println!("\r\n\x1b[38;2;120;140;160m💡 Tip: Set this run as your reference baseline with:\x1b[0m \x1b[1mdyno_term --set-baseline\x1b[0m");
    }

    Ok(())
}

fn run_consistency_suite(
    pipeline: &mut ComputePipelineManager,
    renderer: &mut TerminalRenderer,
    telemetry: &TelemetryCollector,
    adapter_info: &gpu::GpuAdapterInfo,
    iterations: usize,
    cooldown_seconds: u64,
    target_temp_c: f32,
    max_temp_c: f32,
    failsafe_enabled: bool,
    frames_per_scene: u32,
    term_width: usize,
    term_height: usize,
    res_tag: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let iterations = iterations.max(1);
    renderer.enter_alternate_screen()?;
    crossterm::terminal::enable_raw_mode()?;

    let mut completed_runs = Vec::with_capacity(iterations);
    let mut halted_by_thermal = false;

    for i in 1..=iterations {
        let pass_res = execute_benchmark_pass(
            pipeline,
            renderer,
            telemetry,
            adapter_info,
            frames_per_scene,
            term_width,
            term_height,
            res_tag,
            Some((i, iterations)),
            max_temp_c,
            failsafe_enabled,
        );

        let report = match pass_res {
            Ok(BenchmarkPassResult::Completed(r)) => r,
            Ok(BenchmarkPassResult::AbortedByUser) => {
                break;
            }
            Ok(BenchmarkPassResult::ThermalTrip(trip)) => {
                crossterm::terminal::disable_raw_mode()?;
                let _ = renderer.leave_alternate_screen();
                dyno_term::benchmark::print_emergency_thermal_box(&trip);
                halted_by_thermal = true;
                break;
            }
            Err(e) => {
                crossterm::terminal::disable_raw_mode()?;
                let _ = renderer.leave_alternate_screen();
                return Err(e);
            }
        };

        let _ = dyno_term::benchmark::append_to_history(&report, dyno_term::benchmark::DEFAULT_HISTORY_FILE);
        completed_runs.push(report);

        if i < iterations && cooldown_seconds > 0 {
            let last = completed_runs.last().unwrap();
            let keep_going = run_cooldown_screen(
                renderer,
                telemetry,
                adapter_info,
                term_width,
                term_height,
                i,
                iterations,
                cooldown_seconds,
                target_temp_c,
                last,
            )?;
            if !keep_going {
                break;
            }
        }
    }

    crossterm::terminal::disable_raw_mode()?;
    let _ = renderer.leave_alternate_screen();

    if halted_by_thermal {
        eprintln!("\x1b[38;2;255;180;0;1m⚠ Consistency testing halted prematurely by thermal tripwire after {} run(s).\x1b[0m", completed_runs.len());
    }

    if completed_runs.is_empty() {
        println!("Consistency suite aborted before any iterations completed.");
        return Ok(());
    }

    let report = compute_consistency_report(
        &completed_runs,
        cooldown_seconds,
        frames_per_scene,
        res_tag,
    );

    let _ = save_consistency_report(&report, output_path);
    print!("{}", format_consistency_scorecard(&report));
    println!("Consistency report saved to: {}", output_path);

    Ok(())
}

fn print_scorecard(report: &BenchmarkReport, output_path: &str, resolution: &str) {
    println!();
    println!("\x1b[38;2;0;220;255;1m┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│                       DYNO_TERM BENCHMARK SCORECARD                      │");
    println!("├──────────────────────────────────────────────────────────────────────────┤\x1b[0m");
    println!(
        "\x1b[38;2;140;150;170m│ GPU: \x1b[38;2;255;255;255;1m{:<67}\x1b[38;2;0;220;255m│",
        format!("{} ({})", report.system_info.gpu_name, report.system_info.gpu_backend)
    );
    println!(
        "\x1b[38;2;140;150;170m│ CPU: \x1b[38;2;255;255;255;1m{:<67}\x1b[38;2;0;220;255m│",
        format!("{} ({} Threads)", report.system_info.cpu_name, report.system_info.cpu_threads)
    );
    println!(
        "\x1b[38;2;140;150;170m│ Mode: \x1b[38;2;0;255;200;1m{:<29}\x1b[38;2;140;150;170m│ Target: \x1b[38;2;255;180;0;1m{:<27}\x1b[38;2;0;220;255m│",
        "Deterministic Workload", resolution
    );
    println!("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m");

    for res in &report.scene_results {
        let name_trimmed: String = res.scene_name.chars().take(28).collect();
        let frame_stat = format!("{:>5.1}ms ({:>4.1})", res.avg_frametime_ms, res.p1_low_frametime_ms);
        let score_stat = format!("{:>6.2} pts", res.score);
        println!(
            "\x1b[38;2;140;150;170m│ \x1b[38;2;255;255;255;1m{:<28} \x1b[38;2;0;220;255m│ \x1b[38;2;0;255;200;1m{:>5.1} FPS \x1b[38;2;0;220;255m│ \x1b[38;2;255;255;255m{:<16} \x1b[38;2;0;220;255m│ \x1b[38;2;255;180;0;1m{:<10} \x1b[38;2;0;220;255m│",
            name_trimmed, res.avg_fps, frame_stat, score_stat
        );
    }

    println!("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m");
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
    println!("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m");
    let score_str = format!("{:>10.2} POINTS ", report.overall_score);
    println!(
        "\x1b[38;2;255;255;255;1m│ OVERALL DYNO BENCHMARK SCORE:                          \x1b[38;2;255;180;0;1m{}\x1b[38;2;0;220;255m│",
        score_str
    );
    println!("\x1b[38;2;0;220;255m└──────────────────────────────────────────────────────────────────────────┘\x1b[0m");
    println!("\x1b[38;2;140;150;170mReport saved to: \x1b[38;2;0;255;200m{}\x1b[0m\n", output_path);
}

fn capture_screenshots(
    pipeline: &mut ComputePipelineManager,
    out_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(out_dir)?;

    let capture_w = 960u32;
    let capture_h = 540u32;
    let mut buf = vec![0u8; (capture_w * capture_h * 4) as usize];

    let scene_files = [
        "act1_ocean_waves",
        "act2_mandelbulb",
        "act3_black_hole",
        "act4_tornado",
        "act5_cardiac",
        "act6_lbm_vortex",
        "act7_kinematics",
        "act8_nbody",
    ];

    for scene_id in 0..8u32 {
        println!("Rendering Act {}/8...", scene_id + 1);
        if scene_id < 7 {
            for frame in 0..40 {
                let uniforms = gpu::pipeline::Uniforms {
                    time: 3.5,
                    frame,
                    width: pipeline.width,
                    height: pipeline.height,
                    cam_pos: [0.0, 1.8, -4.5],
                    scene_id,
                    cam_dir: [0.0, -0.2, 1.0],
                    benchmark_mode: 0,
                };
                pipeline.render_with_preview(&uniforms, capture_w, capture_h, &mut buf);
            }
        } else {
            let mut nbody = NBodySimulation::new(24_000, capture_w as usize, capture_h as usize);
            for _ in 0..80 {
                nbody.step_multithreaded(0.016);
            }
            nbody.rasterize_to_preview(&mut buf, capture_w as usize, capture_h as usize);
        }

        let ppm_path = format!("{}/{}.ppm", out_dir, scene_files[scene_id as usize]);
        save_ppm(&ppm_path, capture_w, capture_h, &buf)?;
    }

    println!("All 8 scenes captured successfully to {}", out_dir);
    Ok(())
}

fn save_ppm(path: &str, width: u32, height: u32, rgba: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    write!(file, "P6\n{} {}\n255\n", width, height)?;
    for chunk in rgba.chunks_exact(4) {
        file.write_all(&[chunk[0], chunk[1], chunk[2]])?;
    }
    Ok(())
}
