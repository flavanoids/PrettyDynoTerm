use clap::Parser;
use dyno_term::benchmark::{
    BenchmarkReport, DEFAULT_BASELINE_FILE, DEFAULT_HISTORY_FILE, compare_reports,
    format_comparison_table, format_history_table, load_baseline, load_history, save_baseline,
};
use dyno_term::cli::{Args, parse_gpu_selection, parse_resolution};
use dyno_term::gpu::{ComputePipelineManager, GpuContext};
use dyno_term::output::{ExitCode, GpuStartInfo, OutputEmitter};
use dyno_term::runner::{
    BenchmarkOptions, ConsistencyOptions, DemoOptions, TerminalSession, capture_screenshots,
    run_benchmark, run_consistency_suite, run_interactive_demo, wait_for_menu_return,
};
use dyno_term::self_test;
use dyno_term::shaders;
use dyno_term::telemetry::TelemetryCollector;
use dyno_term::terminal::{
    CliMenu, MenuAction, MenuConfig, TerminalRenderer, format_resolution_tag,
};
use std::io::IsTerminal;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Install signal handlers for clean SIGINT / SIGTERM teardown within 2s
    let _ = dyno_term::signal::install_signal_handlers();

    let args = Args::parse();

    if args.version {
        if args.output.is_json() || args.output.is_jsonl() {
            let meta = dyno_term::output::VersionMetadata::current();
            println!("{}", serde_json::to_string_pretty(&meta)?);
        } else {
            println!("dyno_term {}", env!("CARGO_PKG_VERSION"));
        }
        ExitCode::Ok.exit();
    }

    if args.test_internal_error {
        ExitCode::InternalError.exit();
    }

    // Auto-detect headless mode: explicit flag, or non-TTY stdout, or JSON/JSONL output formats
    let is_stdout_tty = std::io::stdout().is_terminal();
    let headless =
        args.headless || !is_stdout_tty || args.output.is_jsonl() || args.output.is_json();

    let output_emitter = OutputEmitter::new(args.output, headless);

    // Install global terminal restoration panic hook (only reset terminal if interactive TTY)
    let default_panic_hook = std::panic::take_hook();
    let is_interactive = !headless;
    std::panic::set_hook(Box::new(move |panic_info| {
        if is_interactive {
            let _ = crossterm::terminal::disable_raw_mode();
            use std::io::Write;
            let _ = write!(std::io::stdout(), "\x1b[0m\x1b[?25h\x1b[?1049l");
            let _ = std::io::stdout().flush();
        }
        default_panic_hook(panic_info);
    }));

    if args.test {
        match self_test::run_self_tests(args.gpu) {
            Ok(_) => ExitCode::Ok.exit(),
            Err(e) => {
                output_emitter.emit_error(
                    "SelfTestFailure",
                    &e.to_string(),
                    ExitCode::WorkloadFailure,
                );
                ExitCode::WorkloadFailure.exit();
            }
        }
    }

    let baseline_path = args.baseline.as_deref().unwrap_or(DEFAULT_BASELINE_FILE);

    if args.history {
        let history = load_history(DEFAULT_HISTORY_FILE)?;
        let baseline = load_baseline(baseline_path);
        let table = format_history_table(&history, baseline.as_ref().map(|b| b.timestamp.as_str()));
        print!("{}", table);
        ExitCode::Ok.exit();
    }

    if let Some(ref report_path) = args.set_baseline {
        let content = std::fs::read_to_string(report_path).map_err(|e| {
            format!(
                "Failed to read benchmark report at '{}': {}",
                report_path, e
            )
        })?;
        let report: BenchmarkReport = serde_json::from_str(&content)
            .map_err(|e| format!("Invalid benchmark JSON in '{}': {}", report_path, e))?;
        save_baseline(&report, baseline_path)?;
        if !headless {
            println!(
                "\x1b[38;2;0;255;128;1m✓ Pinned benchmark from '{}' ({:.2} pts, {}) as active reference baseline ('{}')\x1b[0m",
                report_path, report.overall_score, report.timestamp, baseline_path
            );
        } else {
            println!(
                "Pinned benchmark from '{}' ({:.2} pts, {}) as active reference baseline ('{}')",
                report_path, report.overall_score, report.timestamp, baseline_path
            );
        }
        ExitCode::Ok.exit();
    }

    if let Some(ref report_path) = args.compare {
        let content = std::fs::read_to_string(report_path).map_err(|e| {
            format!(
                "Failed to read benchmark report at '{}': {}",
                report_path, e
            )
        })?;
        let cur_report: BenchmarkReport = serde_json::from_str(&content)
            .map_err(|e| format!("Invalid benchmark JSON in '{}': {}", report_path, e))?;
        let base_report = load_baseline(baseline_path).ok_or_else(|| {
            format!(
                "No reference baseline found at '{}'. Set one with --set-baseline <FILE>",
                baseline_path
            )
        })?;
        let comparison = compare_reports(&cur_report, &base_report);
        print!("{}", format_comparison_table(&comparison));
        ExitCode::Ok.exit();
    }

    if args.list_gpus {
        let adapters = GpuContext::list_adapters();
        if args.output.is_json() || args.output.is_jsonl() {
            let list: Vec<GpuStartInfo> = adapters
                .iter()
                .enumerate()
                .map(|(i, a)| GpuStartInfo {
                    index: i,
                    name: a.name.clone(),
                    backend: format!("{:?}", a.backend),
                    driver: a.driver.clone(),
                    pci_bus_id: None,
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&list)?);
        } else {
            println!("Available GPU Adapters:");
            for (i, adapter) in adapters.iter().enumerate() {
                println!(
                    "  [{}] {} | Backend: {:?} | Type: {:?} | Driver: {}",
                    i, adapter.name, adapter.backend, adapter.device_type, adapter.driver
                );
            }
        }
        ExitCode::Ok.exit();
    }

    let run_demo = args.demo;
    let run_benchmark_mode = args.benchmark;
    let run_consistency_mode = args.consistency.is_some();
    let run_menu = args.menu || (!run_demo && !run_benchmark_mode && !run_consistency_mode);

    // Disallow interactive menu when running in headless mode
    if headless && run_menu {
        output_emitter.emit_error(
            "InvalidArgs",
            "Interactive CLI menu cannot run in headless mode. Please specify a subcommand such as --benchmark or --demo.",
            ExitCode::InvalidArgs,
        );
        ExitCode::InvalidArgs.exit();
    }

    // Resolve compute backend (auto, wgpu, cuda, hip)
    let _compute_backend = match dyno_term::backend::resolve_backend(args.backend) {
        Ok(b) => b,
        Err(e) => {
            output_emitter.emit_error("NoUsableGpu", &e.to_string(), ExitCode::NoUsableGpu);
            ExitCode::NoUsableGpu.exit();
        }
    };

    // Enumerate available adapters and parse target GPU selection
    let all_adapters = GpuContext::list_adapters();
    if all_adapters.is_empty() {
        output_emitter.emit_error(
            "NoUsableGpu",
            "No compatible GPU adapter found on system",
            ExitCode::NoUsableGpu,
        );
        ExitCode::NoUsableGpu.exit();
    }

    let selected_gpus = match parse_gpu_selection(&args.gpus, args.gpu, all_adapters.len()) {
        Ok(gpus) => gpus,
        Err(err) => {
            if err.contains("out of bounds") || err.contains("No compatible GPU") {
                output_emitter.emit_error("NoUsableGpu", &err, ExitCode::NoUsableGpu);
                ExitCode::NoUsableGpu.exit();
            } else {
                output_emitter.emit_error("InvalidArgs", &err, ExitCode::InvalidArgs);
                ExitCode::InvalidArgs.exit();
            }
        }
    };

    // Terminal preview dimensions
    let (term_cols, term_rows) = if headless {
        (120, 40)
    } else {
        crossterm::terminal::size().unwrap_or((120, 40))
    };
    let term_width = args.width.unwrap_or(term_cols as u32).max(40);
    let hud_rows = 7u32;
    let char_height = args
        .height
        .map(|h| h / 2)
        .unwrap_or(term_rows.saturating_sub(hud_rows as u16) as u32)
        .max(8);
    let term_pixel_height = char_height * 2;

    // Canonical internal compute resolution
    let (mut compute_w, mut compute_h) = parse_resolution(&args.res, term_width, term_pixel_height);
    let mut res_tag = format_resolution_tag(compute_w, compute_h);
    let mut is_datacenter_preset = false;
    let mut benchmark_frames = args.frames;

    if let Some(ref preset) = args.preset {
        match preset.to_lowercase().as_str() {
            "datacenter" => {
                compute_w = 15360;
                compute_h = 8640;
                res_tag = "datacenter".to_string();
                is_datacenter_preset = true;
                // Size so MI50 takes 60-90s: default 15 frames per scene (120 total)
                if args.frames == 100 {
                    benchmark_frames = 15;
                }
            }
            other => {
                output_emitter.emit_error(
                    "InvalidArgs",
                    &format!(
                        "Unknown preset '{}'. Supported presets: 'datacenter'",
                        other
                    ),
                    ExitCode::InvalidArgs,
                );
                ExitCode::InvalidArgs.exit();
            }
        }
    }

    let safe_max_temp = if args.max_temp.is_finite() && args.max_temp > 0.0 {
        args.max_temp
    } else {
        105.0
    };
    let safe_target_temp =
        if args.target_temp.is_finite() && (30.0..=90.0).contains(&args.target_temp) {
            args.target_temp
        } else {
            55.0
        };

    let resolved_report_file = if let Some(ref dir) = args.report_dir {
        std::fs::create_dir_all(dir)?;
        Some(format!(
            "{}/dyno_benchmark_results.json",
            dir.trim_end_matches('/')
        ))
    } else {
        args.report_file.clone()
    };

    // Fast-path: Multi-GPU Benchmark execution (spawns dedicated worker threads per GPU)
    if run_benchmark_mode {
        let telemetry = Arc::new(TelemetryCollector::new(
            all_adapters[selected_gpus[0]].name.clone(),
        ));
        let multi_opts = dyno_term::runner::MultiGpuBenchmarkOptions {
            selected_gpus,
            frames_per_scene: benchmark_frames,
            duration_secs: args.duration,
            compute_width: compute_w,
            compute_height: compute_h,
            res_tag: res_tag.clone(),
            is_datacenter_preset,
            max_temp_c: safe_max_temp,
            failsafe_enabled: !args.no_failsafe,
            report_file: resolved_report_file.clone(),
        };
        let code = dyno_term::runner::run_multi_gpu_benchmark(
            &multi_opts,
            telemetry,
            Arc::new(output_emitter),
        )?;
        code.exit();
    }

    // Initialize primary GPU Context for interactive modes (menu / demo / consistency / screenshot)
    let primary_gpu_index = selected_gpus[0];
    let mut gpu_ctx = match pollster::block_on(GpuContext::new_for_adapter_index(primary_gpu_index))
    {
        Ok(ctx) => ctx,
        Err(e) => {
            output_emitter.emit_error("NoUsableGpu", &e.to_string(), ExitCode::NoUsableGpu);
            ExitCode::NoUsableGpu.exit();
        }
    };

    let mut device = Arc::clone(&gpu_ctx.device);
    let mut queue = Arc::clone(&gpu_ctx.queue);
    let mut adapter_info = gpu_ctx.adapter_info.clone();

    // Initialize Compute Pipeline with chosen compute resolution
    let mut pipeline = match ComputePipelineManager::new(
        Arc::clone(&device),
        Arc::clone(&queue),
        shaders::UNIFIED_SHADER,
        compute_w,
        compute_h,
    ) {
        Ok(p) => p,
        Err(err) => {
            if !headless {
                eprintln!(
                    "\x1b[38;2;255;100;100;1m⚠ Resolution {}x{} exceeds hardware limits:\x1b[0m {}",
                    compute_w, compute_h, err
                );
            }
            let (fb_w, fb_h) = dyno_term::benchmark::GpuMemoryInfo::detect(&device)
                .recommend_resolution(dyno_term::benchmark::VramTestTier::Light);
            if !headless {
                eprintln!(
                    "\x1b[38;2;0;255;200;1m▶ Falling back safely to dynamically adapted {}x{} preset...\x1b[0m",
                    fb_w, fb_h
                );
            }
            let fallback_p = ComputePipelineManager::new(
                Arc::clone(&device),
                Arc::clone(&queue),
                shaders::UNIFIED_SHADER,
                fb_w,
                fb_h,
            )
            .map_err(|e| format!("Failed to create fallback pipeline: {}", e))?;
            res_tag = format_resolution_tag(fallback_p.width, fallback_p.height);
            fallback_p
        }
    };

    // Initialize Telemetry & Renderer
    let telemetry = TelemetryCollector::new(adapter_info.name.clone());
    let mut renderer = TerminalRenderer::new();

    if let Some(ref out_dir) = args.capture {
        capture_screenshots(&mut pipeline, out_dir)?;
        ExitCode::Ok.exit();
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
            output_path: args
                .report_file
                .clone()
                .unwrap_or_else(|| "dyno_benchmark_results.json".to_string()),
            consistency_iterations: 5,
            cooldown_seconds: args.cooldown,
            max_temp_c: safe_max_temp,
            target_temp_c: safe_target_temp,
            failsafe_enabled: !args.no_failsafe,
        };

        let mut menu = CliMenu::new();
        loop {
            let (cols, rows) = crossterm::terminal::size().unwrap_or((120, 40));
            let cur_term_w = args.width.unwrap_or(cols as u32).max(40) as usize;
            let cur_term_h = args
                .height
                .map(|h| h / 2)
                .unwrap_or(rows.saturating_sub(7) as u32)
                .max(8) as usize
                * 2;

            match menu.run(&mut config, &telemetry, &adapter_info, &all_adapters)? {
                MenuAction::LaunchDemo => {
                    let cur_res_tag =
                        format_resolution_tag(config.compute_width, config.compute_height);
                    let mut session = TerminalSession {
                        pipeline: &mut pipeline,
                        renderer: &mut renderer,
                        telemetry: &telemetry,
                        adapter_info: &adapter_info,
                        output_emitter: &output_emitter,
                        term_width: cur_term_w,
                        term_height: cur_term_h,
                        res_tag: &cur_res_tag,
                    };
                    let opts = DemoOptions {
                        initial_scene: config.selected_scene,
                        initial_uncapped: config.uncapped,
                        max_temp_c: config.max_temp_c,
                        failsafe_enabled: config.failsafe_enabled,
                    };
                    run_interactive_demo(&mut session, &opts)?;
                }
                MenuAction::LaunchUncapped => {
                    let cur_res_tag =
                        format_resolution_tag(config.compute_width, config.compute_height);
                    let mut session = TerminalSession {
                        pipeline: &mut pipeline,
                        renderer: &mut renderer,
                        telemetry: &telemetry,
                        adapter_info: &adapter_info,
                        output_emitter: &output_emitter,
                        term_width: cur_term_w,
                        term_height: cur_term_h,
                        res_tag: &cur_res_tag,
                    };
                    let opts = DemoOptions {
                        initial_scene: config.selected_scene,
                        initial_uncapped: true,
                        max_temp_c: config.max_temp_c,
                        failsafe_enabled: config.failsafe_enabled,
                    };
                    run_interactive_demo(&mut session, &opts)?;
                }
                MenuAction::RunBenchmark => {
                    let cur_res_tag =
                        format_resolution_tag(config.compute_width, config.compute_height);
                    let mut session = TerminalSession {
                        pipeline: &mut pipeline,
                        renderer: &mut renderer,
                        telemetry: &telemetry,
                        adapter_info: &adapter_info,
                        output_emitter: &output_emitter,
                        term_width: cur_term_w,
                        term_height: cur_term_h,
                        res_tag: &cur_res_tag,
                    };
                    let opts = BenchmarkOptions {
                        frames_per_scene: config.frames_per_scene,
                        duration_secs: args.duration,
                        report_file: resolved_report_file
                            .clone()
                            .or_else(|| Some(config.output_path.clone())),
                        max_temp_c: config.max_temp_c,
                        failsafe_enabled: config.failsafe_enabled,
                    };
                    let _ = run_benchmark(&mut session, &opts);
                    println!(
                        "\r\n\x1b[38;2;0;255;200;1m▶ Benchmark finished. Press [Enter] or [Space] to return to Main Menu...\x1b[0m"
                    );
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                    wait_for_menu_return()?;
                }
                MenuAction::RunConsistency => {
                    let cur_res_tag =
                        format_resolution_tag(config.compute_width, config.compute_height);
                    let mut session = TerminalSession {
                        pipeline: &mut pipeline,
                        renderer: &mut renderer,
                        telemetry: &telemetry,
                        adapter_info: &adapter_info,
                        output_emitter: &output_emitter,
                        term_width: cur_term_w,
                        term_height: cur_term_h,
                        res_tag: &cur_res_tag,
                    };
                    let opts = ConsistencyOptions {
                        iterations: config.consistency_iterations,
                        cooldown_seconds: config.cooldown_seconds,
                        target_temp_c: config.target_temp_c,
                        max_temp_c: config.max_temp_c,
                        failsafe_enabled: config.failsafe_enabled,
                        frames_per_scene: config.frames_per_scene,
                        report_file: Some(config.output_path.clone()),
                    };
                    run_consistency_suite(&mut session, &opts)?;
                    println!(
                        "\r\n\x1b[38;2;0;255;200;1m▶ Consistency suite finished. Press [Enter] or [Space] to return to Main Menu...\x1b[0m"
                    );
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
                            config.resolution_name =
                                format_resolution_tag(config.compute_width, config.compute_height);
                            pipeline = new_p;
                        }
                        Err(err) => {
                            eprintln!(
                                "\r\n\x1b[38;2;255;100;100;1m⚠ Hardware Buffer Limit Exceeded: {}\x1b[0m",
                                err
                            );
                            eprintln!(
                                "\x1b[38;2;255;180;0;1m▶ Reverting safely to previous resolution ({}x{})...\x1b[0m",
                                prev_w, prev_h
                            );
                            std::thread::sleep(std::time::Duration::from_millis(2000));
                            config.compute_width = prev_w;
                            config.compute_height = prev_h;
                            config.resolution_name = format_resolution_tag(prev_w, prev_h);
                        }
                    }
                }
                MenuAction::SwitchGpu(idx) => {
                    config.selected_gpu_index = idx;
                    gpu_ctx = match pollster::block_on(GpuContext::new(Some(idx))) {
                        Ok(ctx) => ctx,
                        Err(e) => {
                            eprintln!("Failed to switch GPU adapter: {}", e);
                            continue;
                        }
                    };
                    device = Arc::clone(&gpu_ctx.device);
                    queue = Arc::clone(&gpu_ctx.queue);
                    adapter_info = gpu_ctx.adapter_info.clone();
                    config.resolution_name =
                        format_resolution_tag(config.compute_width, config.compute_height);
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
                            eprintln!(
                                "\r\n\x1b[38;2;255;100;100;1m⚠ Resolution {}x{} exceeds limits on adapter {}: {}\x1b[0m",
                                config.compute_width, config.compute_height, adapter_info.name, err
                            );
                            eprintln!(
                                "\x1b[38;2;0;255;200;1m▶ Falling back to 1080p FHD safe preset...\x1b[0m"
                            );
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
        ExitCode::Ok.exit();
    } else if let Some(iterations) = args.consistency {
        let mut session = TerminalSession {
            pipeline: &mut pipeline,
            renderer: &mut renderer,
            telemetry: &telemetry,
            adapter_info: &adapter_info,
            output_emitter: &output_emitter,
            term_width: term_width as usize,
            term_height: term_pixel_height as usize,
            res_tag: &res_tag,
        };
        let opts = ConsistencyOptions {
            iterations,
            cooldown_seconds: args.cooldown,
            target_temp_c: safe_target_temp,
            max_temp_c: safe_max_temp,
            failsafe_enabled: !args.no_failsafe,
            frames_per_scene: args.frames,
            report_file: resolved_report_file.clone(),
        };
        run_consistency_suite(&mut session, &opts)?;
        ExitCode::Ok.exit();
    } else {
        let mut session = TerminalSession {
            pipeline: &mut pipeline,
            renderer: &mut renderer,
            telemetry: &telemetry,
            adapter_info: &adapter_info,
            output_emitter: &output_emitter,
            term_width: term_width as usize,
            term_height: term_pixel_height as usize,
            res_tag: &res_tag,
        };
        let opts = DemoOptions {
            initial_scene: args.scene,
            initial_uncapped: args.uncapped,
            max_temp_c: safe_max_temp,
            failsafe_enabled: !args.no_failsafe,
        };
        run_interactive_demo(&mut session, &opts)?;
        ExitCode::Ok.exit();
    }
}
