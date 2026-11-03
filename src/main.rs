mod benchmark;
mod cpu;
mod gpu;
mod shaders;
mod telemetry;
mod terminal;

use benchmark::{BenchmarkReport, BenchmarkScorer, SystemReport};
use clap::Parser;
use cpu::NBodySimulation;
use crossterm::event::{self, Event, KeyCode};
use gpu::{ComputePipelineManager, GpuContext};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use telemetry::TelemetryCollector;
use terminal::{format_resolution_tag, CliMenu, Hud, MenuAction, MenuConfig, TerminalRenderer};

#[derive(Parser, Debug)]
#[command(
    name = "dyno_term",
    about = "Cross-platform terminal GPU/CPU benchmark & artistic demoparty renderer",
    version = "1.0.0"
)]
struct Args {
    /// Open the interactive CLI launcher menu (default when no mode flag is passed)
    #[arg(short, long)]
    menu: bool,

    /// Run the deterministic benchmark suite directly
    #[arg(short, long)]
    benchmark: bool,

    /// Run in interactive demo preview mode directly
    #[arg(short, long)]
    demo: bool,

    /// Run demo mode uncapped (100% GPU stress test, no 60 FPS limiter)
    #[arg(short, long)]
    uncapped: bool,

    /// Number of frames per scene in benchmark mode
    #[arg(short, long, default_value_t = 100)]
    frames: u32,

    /// Compute resolution preset ("16k", "8k", "4k", "1440p", "1080p", "720p", "native")
    #[arg(long, default_value = "4k")]
    res: String,

    /// Specific scene to display in demo mode (0: Waves, 1: Mandelbulb, 2: Black Hole, 3: Tornado, 4: Cardiac, 5: CPU N-Body)
    #[arg(short, long, default_value_t = 0)]
    scene: u32,

    /// Select GPU adapter by index
    #[arg(short, long)]
    gpu: Option<usize>,

    /// List all available GPU adapters and exit
    #[arg(long)]
    list_gpus: bool,

    /// Custom render width in pixels
    #[arg(long)]
    width: Option<u32>,

    /// Custom render height in pixels
    #[arg(long)]
    height: Option<u32>,

    /// Output JSON path for benchmark report
    #[arg(short, long, default_value = "dyno_benchmark_results.json")]
    output: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

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
    let run_menu = args.menu || (!run_demo && !run_benchmark_mode);

    // Canonical internal compute resolution: Defaults to 4K UHD (3840x2160) for balanced high-performance GPU compute
    let (compute_w, compute_h) = match args.res.to_lowercase().as_str() {
        "16k" => (15360, 8640),
        "8k" => (7680, 4320),
        "4k" | "2160p" | "uhd" => (3840, 2160),
        "1440p" | "2k" | "qhd" => (2560, 1440),
        "1080p" | "fhd" => (1920, 1080),
        "720p" | "hd" => (1280, 720),
        "native" => (term_width, term_pixel_height),
        _ => (3840, 2160),
    };

    let res_tag = format_resolution_tag(compute_w, compute_h);

    // Initialize Compute Pipeline with chosen compute resolution
    let mut pipeline = ComputePipelineManager::new(
        Arc::clone(&device),
        Arc::clone(&queue),
        shaders::UNIFIED_SHADER,
        compute_w,
        compute_h,
    )?;

    // Initialize Telemetry & Renderer
    let telemetry = TelemetryCollector::new(adapter_info.name.clone());
    let mut renderer = TerminalRenderer::new();

    if run_menu {
        let all_adapters = GpuContext::list_adapters();
        let mut config = MenuConfig {
            resolution_name: res_tag.clone(),
            compute_width: compute_w,
            compute_height: compute_h,
            frames_per_scene: args.frames,
            selected_scene: args.scene,
            uncapped: args.uncapped,
            selected_gpu_index: args.gpu.unwrap_or(0),
            output_path: args.output.clone(),
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
                    )?;
                    println!("\r\n\x1b[38;2;0;255;200;1m▶ Benchmark finished. Press [Enter] or [Space] to return to Main Menu...\x1b[0m");
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                    wait_for_menu_return()?;
                }
                MenuAction::RecreatePipeline => {
                    config.resolution_name = format_resolution_tag(config.compute_width, config.compute_height);
                    pipeline = ComputePipelineManager::new(
                        Arc::clone(&device),
                        Arc::clone(&queue),
                        shaders::UNIFIED_SHADER,
                        config.compute_width,
                        config.compute_height,
                    )?;
                }
                MenuAction::SwitchGpu(idx) => {
                    config.selected_gpu_index = idx;
                    gpu_ctx = pollster::block_on(GpuContext::new(Some(idx)))
                        .map_err(|e| format!("Failed to switch GPU adapter: {}", e))?;
                    device = Arc::clone(&gpu_ctx.device);
                    queue = Arc::clone(&gpu_ctx.queue);
                    adapter_info = gpu_ctx.adapter_info.clone();
                    config.resolution_name = format_resolution_tag(config.compute_width, config.compute_height);
                    pipeline = ComputePipelineManager::new(
                        Arc::clone(&device),
                        Arc::clone(&queue),
                        shaders::UNIFIED_SHADER,
                        config.compute_width,
                        config.compute_height,
                    )?;
                }
                MenuAction::Exit => {
                    break;
                }
            }
        }
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
        )?;
    }

    Ok(())
}

fn wait_for_menu_return() -> Result<(), Box<dyn std::error::Error>> {
    crossterm::terminal::enable_raw_mode()?;
    loop {
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
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
) -> Result<(), Box<dyn std::error::Error>> {
    renderer.enter_alternate_screen()?;
    crossterm::terminal::enable_raw_mode()?;

    let scene_names = [
        "ATMOSPHERIC OCEAN WAVES (GPU)",
        "THE MANDELBULB CORE (GPU)",
        "RELATIVISTIC BLACK HOLE (GPU)",
        "TORNADIC MESOCYCLONE (GPU)",
        "CARDIAC ELECTROPHYSIOLOGY (GPU)",
        "KINETIC N-BODY SUPERNOVA (CPU)",
    ];

    let mut scene_id = initial_scene % 6;
    let mut uncapped = initial_uncapped;
    let mut hud = Hud::new(term_width, scene_names[scene_id as usize].to_string());
    hud.resolution_label = res_tag.to_string();
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
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char(' ') => {
                        scene_id = (scene_id + 1) % 6;
                        hud.scene_title = scene_names[scene_id as usize].to_string();
                    }
                    KeyCode::Char('u') | KeyCode::Char('U') => {
                        uncapped = !uncapped;
                    }
                    _ => {}
                }
            }
        }

        // Check for terminal resize
        if let Ok((cols, rows)) = crossterm::terminal::size() {
            let target_w = cols as usize;
            let target_h = (rows.saturating_sub(7) as usize * 2).max(16);
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

        if scene_id < 5 {
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
        let render_dur = frame_start.elapsed();

        if should_draw_terminal {
            last_terminal_draw = Instant::now();
            let frametime_ms = render_dur.as_secs_f32() * 1000.0;
            hud.frametime_ms = frametime_ms;
            hud.fps = if frametime_ms > 0.0 {
                1000.0 / frametime_ms
            } else {
                60.0
            };
            hud.progress_pct = ((frame_count % 300) as f32 / 300.0) * 100.0;
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
) -> Result<(), Box<dyn std::error::Error>> {
    renderer.enter_alternate_screen()?;
    crossterm::terminal::enable_raw_mode()?;

    let scene_names = [
        "ATMOSPHERIC OCEAN WAVES (GPU)",
        "THE MANDELBULB CORE (GPU)",
        "RELATIVISTIC BLACK HOLE (GPU)",
        "TORNADIC MESOCYCLONE (GPU)",
        "CARDIAC ELECTROPHYSIOLOGY (GPU)",
        "KINETIC N-BODY SUPERNOVA (CPU)",
    ];

    let mut scorer = BenchmarkScorer::new();
    let mut scene_results = Vec::new();
    let total_benchmark_start = Instant::now();
    let compute_pixels = (pipeline.width * pipeline.height) as u64;

    let mut term_preview_buf = vec![0u8; term_width * term_height * 4];
    let mut nbody = NBodySimulation::new(32_000, pipeline.width as usize, pipeline.height as usize);

    for scene_id in 0..6u32 {
        let mut hud = Hud::new(term_width, scene_names[scene_id as usize].to_string());
        hud.resolution_label = res_tag.to_string();

        let scene_start = Instant::now();
        let mut last_terminal_draw = Instant::now();

        for frame_idx in 0..frames_per_scene {
            // Check for exit abort key
            if event::poll(Duration::from_millis(0))? {
                if let Event::Key(key) = event::read()? {
                    if key.code == KeyCode::Char('q') || key.code == KeyCode::Esc {
                        crossterm::terminal::disable_raw_mode()?;
                        renderer.leave_alternate_screen()?;
                        println!("Benchmark aborted by user.");
                        return Ok(());
                    }
                }
            }

            let frame_start = Instant::now();
            let elapsed = scene_start.elapsed().as_secs_f32();

            // Decoupled 60 FPS Terminal preview
            let should_draw_terminal = last_terminal_draw.elapsed() >= Duration::from_millis(16)
                || frame_idx == frames_per_scene - 1;

            if scene_id < 5 {
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
                scorer.record_frame(
                    frame_dur,
                    stats.gpu_temp_c,
                    stats.gpu_power_w,
                    stats.vram_used_mb,
                    stats.gpu_clock_mhz,
                );

                if should_draw_terminal {
                    last_terminal_draw = Instant::now();
                    let frame_ms = frame_dur.as_secs_f32() * 1000.0;
                    hud.frametime_ms = frame_ms;
                    hud.fps = if frame_ms > 0.0 { 1000.0 / frame_ms } else { 0.0 };
                    hud.progress_pct = ((frame_idx + 1) as f32 / frames_per_scene as f32) * 100.0;
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
                // Act IV: Multi-Threaded CPU N-Body Scene (32,000 celestial bodies)
                nbody.step_multithreaded(0.005);
                let frame_dur = frame_start.elapsed();

                let stats = telemetry.sample(&adapter_info.name);
                scorer.record_frame(
                    frame_dur,
                    stats.gpu_temp_c,
                    stats.gpu_power_w,
                    stats.vram_used_mb,
                    stats.gpu_clock_mhz,
                );

                if should_draw_terminal {
                    last_terminal_draw = Instant::now();
                    let frame_ms = frame_dur.as_secs_f32() * 1000.0;
                    hud.frametime_ms = frame_ms;
                    hud.fps = if frame_ms > 0.0 { 1000.0 / frame_ms } else { 0.0 };
                    hud.progress_pct = ((frame_idx + 1) as f32 / frames_per_scene as f32) * 100.0;
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
        scene_results.push(scene_result);
    }

    let total_elapsed = total_benchmark_start.elapsed().as_secs_f64();
    let total_score: u64 = scene_results.iter().map(|s| s.score).sum();

    crossterm::terminal::disable_raw_mode()?;
    renderer.leave_alternate_screen()?;

    // Generate Final Report
    let stats = telemetry.sample(&adapter_info.name);
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
        total_frames: frames_per_scene * 6,
        overall_score: total_score,
        peak_gpu_temp_c: scorer.peak_temp,
        peak_gpu_power_w: scorer.peak_power,
        peak_gpu_clock_mhz: scorer.peak_clock.max(stats.peak_gpu_clock_mhz),
        peak_vram_mb: scorer.peak_vram,
        scene_results: scene_results.clone(),
    };

    let _ = BenchmarkScorer::save_report(&report, output_path);
    print_scorecard(&report, output_path, res_tag);

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
        let score_stat = format!("{:>7} pts", res.score);
        println!(
            "\x1b[38;2;140;150;170m│ \x1b[38;2;255;255;255;1m{:<28} \x1b[38;2;0;220;255m│ \x1b[38;2;0;255;200;1m{:>5.1} FPS \x1b[38;2;0;220;255m│ \x1b[38;2;255;255;255m{:<16} \x1b[38;2;0;220;255m│ \x1b[38;2;255;180;0;1m{:<10} \x1b[38;2;0;220;255m│",
            name_trimmed, res.avg_fps, frame_stat, score_stat
        );
    }

    println!("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m");
    println!(
        "\x1b[38;2;140;150;170m│ Peak: \x1b[38;2;0;255;200;1m{:.0}°C\x1b[38;2;140;150;170m | \x1b[38;2;0;255;200;1m{:.0}W\x1b[38;2;140;150;170m | \x1b[38;2;255;180;0;1m{}MHz\x1b[38;2;140;150;170m | \x1b[38;2;0;255;200;1m{}MB VRAM\x1b[38;2;140;150;170m | Frames: \x1b[38;2;255;255;255;1m{:<3}\x1b[38;2;140;150;170m | Time: \x1b[38;2;255;255;255;1m{:.2}s\x1b[38;2;0;220;255m│",
        report.peak_gpu_temp_c, report.peak_gpu_power_w, report.peak_gpu_clock_mhz, report.peak_vram_mb, report.total_frames, report.total_elapsed_seconds
    );
    println!("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m");
    println!(
        "\x1b[38;2;255;255;255;1m│ OVERALL DYNO BENCHMARK SCORE:                 \x1b[38;2;255;180;0;1m{:>12} POINTS         \x1b[38;2;0;220;255m│",
        report.overall_score
    );
    println!("\x1b[38;2;0;220;255m└────────────────────────────────────────────────────────────────────────┘\x1b[0m");
    println!("\x1b[38;2;140;150;170mReport saved to: \x1b[38;2;0;255;200m{}\x1b[0m\n", output_path);
}
