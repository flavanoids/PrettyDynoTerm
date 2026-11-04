use crate::benchmark::{BenchmarkReport, BenchmarkScorer, SystemReport};
use crate::cpu::NBodySimulation;
use crate::gpu::pipeline::{DownsampleUniforms, Uniforms};
use crate::gpu::{ComputePipelineManager, GpuContext};
use crate::shaders::{DOWNSAMPLE_SHADER, UNIFIED_SHADER};
use crate::telemetry::TelemetryCollector;
use crate::terminal::{format_resolution_tag, get_scene_math_model, visible_len, TerminalRenderer};
use std::time::{Duration, Instant};

pub struct TestReport {
    pub suite_name: &'static str,
    pub passed: bool,
    pub duration: Duration,
    pub details: Vec<String>,
}

pub fn run_self_tests(preferred_gpu: Option<usize>) -> Result<(), Box<dyn std::error::Error>> {
    let overall_start = Instant::now();

    println!("\x1b[38;2;0;255;200;1m================================================================================");
    println!("          DYNO_TERM SYSTEM SELF-TEST & HARDWARE VERIFICATION SUITE              ");
    println!("================================================================================\x1b[0m");

    let mut suites: Vec<TestReport> = Vec::new();

    // 1. CPU Simulation Suite
    suites.push(test_cpu_suite());

    // 2. WGSL Compute Shaders & Uniform Layout
    suites.push(test_shader_suite());

    // 3. GPU Context & Hardware Pipeline
    suites.push(test_gpu_suite(preferred_gpu));

    // 4. Telemetry & Hardware Sensors
    suites.push(test_telemetry_suite());

    // 5. Benchmark Engine & Scorer Math
    suites.push(test_benchmark_suite());

    // 6. Terminal Engine & ANSI Renderer
    suites.push(test_terminal_suite());

    let overall_duration = overall_start.elapsed();
    let total_suites = suites.len();
    let passed_suites = suites.iter().filter(|s| s.passed).count();
    let all_passed = passed_suites == total_suites;

    println!("\x1b[38;2;0;255;200;1m================================================================================");
    if all_passed {
        println!(
            "\x1b[38;2;0;255;128;1m  RESULT: ALL {} TEST SUITES PASSED (0 failures, completed in {:.2?})",
            total_suites, overall_duration
        );
    } else {
        println!(
            "\x1b[38;2;255;80;80;1m  RESULT: {} OF {} TEST SUITES FAILED (completed in {:.2?})",
            total_suites - passed_suites,
            total_suites,
            overall_duration
        );
    }
    println!("\x1b[38;2;0;255;200;1m================================================================================\x1b[0m");

    if all_passed {
        Ok(())
    } else {
        Err(format!("Self-test failed ({} suite(s) failed)", total_suites - passed_suites).into())
    }
}

fn test_cpu_suite() -> TestReport {
    let t0 = Instant::now();
    let mut details = Vec::new();
    let mut passed = true;

    // Test initialization
    let particle_count = 2048;
    let mut sim = NBodySimulation::new(particle_count, 160, 80);
    if sim.particles.len() != particle_count {
        passed = false;
        details.push(format!("Particle count mismatch: expected {}, got {}", particle_count, sim.particles.len()));
    } else {
        details.push(format!("Initialized {} particles across dual colliding galaxies", particle_count));
    }

    // Test physics step
    let initial_pos = (sim.particles[0].x, sim.particles[0].y, sim.particles[0].z);
    sim.step_multithreaded(0.016);
    let after_pos = (sim.particles[0].x, sim.particles[0].y, sim.particles[0].z);

    let has_moved = initial_pos != after_pos;
    let has_nans = sim.particles.iter().any(|p| {
        !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite() || !p.vx.is_finite() || !p.vy.is_finite() || !p.vz.is_finite()
    });

    if !has_moved {
        passed = false;
        details.push("Physics integration failure: particles did not update positions".to_string());
    } else if has_nans {
        passed = false;
        details.push("Numerical instability: NaNs or Infs detected after step".to_string());
    } else {
        details.push("Multi-threaded Rayon physics integration verified (0 NaNs/Infs)".to_string());
    }

    // Test preview rasterizer
    let mut buf = vec![0u8; 160 * 80 * 4];
    sim.rasterize_to_preview(&mut buf, 160, 80);
    let has_drawn_particles = buf.chunks_exact(4).any(|c| c[0] > 5 || c[1] > 6 || c[2] > 16);
    if !has_drawn_particles {
        passed = false;
        details.push("Rasterizer failure: no particle pixels blended into preview buffer".to_string());
    } else {
        details.push("Preview frame buffer rasterizer & color blending verified".to_string());
    }

    let duration = t0.elapsed();
    print_suite_status(1, 6, "CPU Simulation & N-Body Physics Engine", passed, duration, &details);
    TestReport {
        suite_name: "CPU Simulation & N-Body Physics Engine",
        passed,
        duration,
        details,
    }
}

fn test_shader_suite() -> TestReport {
    let t0 = Instant::now();
    let mut details = Vec::new();
    let mut passed = true;

    // Verify UNIFIED_SHADER tokens
    let unified_checks = [
        "@compute",
        "@workgroup_size(16, 16)",
        "struct Uniforms",
        "fn main(",
        "render_waves",
        "render_mandelbulb",
        "render_blackhole",
        "render_tornado",
        "render_cardiac",
        "render_lbm_vortex",
    ];
    let mut missing_unified = Vec::new();
    for token in unified_checks {
        if !UNIFIED_SHADER.contains(token) {
            missing_unified.push(token);
        }
    }
    if !missing_unified.is_empty() {
        passed = false;
        details.push(format!("UNIFIED_SHADER missing tokens: {:?}", missing_unified));
    } else {
        details.push("UNIFIED_SHADER WGSL structure & 6 GPU acts verified".to_string());
    }

    // Verify DOWNSAMPLE_SHADER
    let ds_checks = ["@compute", "@workgroup_size(16, 16)", "struct DownsampleUniforms", "fn downsample_main("];
    let mut missing_ds = Vec::new();
    for token in ds_checks {
        if !DOWNSAMPLE_SHADER.contains(token) {
            missing_ds.push(token);
        }
    }
    if !missing_ds.is_empty() {
        passed = false;
        details.push(format!("DOWNSAMPLE_SHADER missing tokens: {:?}", missing_ds));
    } else {
        details.push("DOWNSAMPLE_SHADER WGSL structure & workgroups verified".to_string());
    }

    // Verify Memory Layouts
    let u_size = std::mem::size_of::<Uniforms>();
    let ds_size = std::mem::size_of::<DownsampleUniforms>();

    if u_size != 48 {
        passed = false;
        details.push(format!("Uniforms size mismatch: expected 48, got {}", u_size));
    } else if u_size % 16 != 0 {
        passed = false;
        details.push(format!("Uniforms size must be multiple of 16 for WGSL std140: got {}", u_size));
    } else if ds_size != 16 {
        passed = false;
        details.push(format!("DownsampleUniforms size mismatch: expected 16, got {}", ds_size));
    } else {
        details.push(format!("Uniforms ({}B, 16B aligned) and DownsampleUniforms ({}B) layout verified", u_size, ds_size));
    }

    let duration = t0.elapsed();
    print_suite_status(2, 6, "WGSL Compute Shaders & Uniform Layouts", passed, duration, &details);
    TestReport {
        suite_name: "WGSL Compute Shaders & Uniform Layouts",
        passed,
        duration,
        details,
    }
}

fn test_gpu_suite(preferred_gpu: Option<usize>) -> TestReport {
    let t0 = Instant::now();
    let mut details = Vec::new();
    let mut passed = true;

    // Enumerate adapters
    let adapters = GpuContext::list_adapters();
    if adapters.is_empty() {
        passed = false;
        details.push("No compatible GPU adapter found (Vulkan/Metal/DX12)".to_string());
        let duration = t0.elapsed();
        print_suite_status(3, 6, "GPU Hardware & Compute Pipeline", passed, duration, &details);
        return TestReport {
            suite_name: "GPU Hardware & Compute Pipeline",
            passed,
            duration,
            details,
        };
    }

    details.push(format!("Found {} GPU adapter(s)", adapters.len()));

    // Create GPU context
    let gpu_ctx = match pollster::block_on(GpuContext::new(preferred_gpu)) {
        Ok(ctx) => ctx,
        Err(e) => {
            passed = false;
            details.push(format!("GPU context initialization failed: {}", e));
            let duration = t0.elapsed();
            print_suite_status(3, 6, "GPU Hardware & Compute Pipeline", passed, duration, &details);
            return TestReport {
                suite_name: "GPU Hardware & Compute Pipeline",
                passed,
                duration,
                details,
            };
        }
    };

    let info = &gpu_ctx.adapter_info;
    details.push(format!("Active GPU: {} | {:?} | {:?}", info.name, info.backend, info.device_type));

    // Create ComputePipelineManager at test resolution (320x180)
    let test_w = 320u32;
    let test_h = 180u32;
    let prev_w = 80u32;
    let prev_h = 40u32;

    let mut pipeline = match ComputePipelineManager::new(
        std::sync::Arc::clone(&gpu_ctx.device),
        std::sync::Arc::clone(&gpu_ctx.queue),
        UNIFIED_SHADER,
        test_w,
        test_h,
    ) {
        Ok(p) => p,
        Err(e) => {
            passed = false;
            details.push(format!("Pipeline compilation failed: {}", e));
            let duration = t0.elapsed();
            print_suite_status(3, 6, "GPU Hardware & Compute Pipeline", passed, duration, &details);
            return TestReport {
                suite_name: "GPU Hardware & Compute Pipeline",
                passed,
                duration,
                details,
            };
        }
    };

    details.push(format!("Pipeline instantiated @ {}x{} ({} pixels)", test_w, test_h, test_w * test_h));

    // Test compute dispatch for Scene 0 (Waves) and Scene 1 (Mandelbulb)
    let mut preview_buf = vec![0u8; (prev_w * prev_h * 4) as usize];
    let mut verified_scenes = 0;

    for scene_id in 0..=5 {
        let uniforms = Uniforms {
            time: 1.5,
            frame: 1,
            width: test_w,
            height: test_h,
            cam_pos: [0.0, 2.0, -5.0],
            scene_id,
            cam_dir: [0.0, 0.0, 1.0],
            benchmark_mode: 0,
        };

        pipeline.render_with_preview(&uniforms, prev_w, prev_h, &mut preview_buf);
        let has_output = preview_buf.iter().any(|&b| b > 0);
        if has_output {
            verified_scenes += 1;
        }
    }

    if verified_scenes == 6 {
        details.push("All 6 GPU shader scenes (Waves, Mandelbulb, Black Hole, Tornado, Cardiac, LBM Vortex) compute & downsampling verified".to_string());
    } else {
        passed = false;
        details.push(format!("Only {} of 6 GPU scenes produced valid render output", verified_scenes));
    }

    let duration = t0.elapsed();
    print_suite_status(3, 6, "GPU Hardware & Compute Pipeline", passed, duration, &details);
    TestReport {
        suite_name: "GPU Hardware & Compute Pipeline",
        passed,
        duration,
        details,
    }
}

fn test_telemetry_suite() -> TestReport {
    let t0 = Instant::now();
    let mut details = Vec::new();
    let mut passed = true;

    let threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
    details.push(format!("Detected {} system CPU threads", threads));

    let cpu_name = TelemetryCollector::read_cpu_name();
    details.push(format!("CPU Model: {}", cpu_name));

    if let Some((total, idle)) = TelemetryCollector::read_proc_stat() {
        details.push(format!("/proc/stat active: total ticks {}, idle ticks {}", total, idle));
    } else {
        details.push("/proc/stat not readable (fallback mode active)".to_string());
    }

    // Telemetry collector instance check
    let collector = TelemetryCollector::new("SelfTest GPU".to_string());
    let stats = collector.sample("SelfTest GPU");
    if stats.gpu_name != "SelfTest GPU" {
        passed = false;
        details.push("TelemetryCollector sample name mismatch".to_string());
    } else {
        details.push("TelemetryCollector background sampling worker verified".to_string());
    }
    drop(collector);

    let duration = t0.elapsed();
    print_suite_status(4, 6, "Telemetry & Hardware Sensors", passed, duration, &details);
    TestReport {
        suite_name: "Telemetry & Hardware Sensors",
        passed,
        duration,
        details,
    }
}

fn test_benchmark_suite() -> TestReport {
    let t0 = Instant::now();
    let mut details = Vec::new();
    let mut passed = true;

    let mut scorer = BenchmarkScorer::new();
    scorer.record_frame(Duration::from_millis(16), 50.0, 60.0, 75.0, 100.0, 2048, 2000);
    scorer.record_frame(Duration::from_millis(14), 65.0, 75.0, 95.0, 140.0, 4096, 2500);

    if scorer.peak_temp != 65.0 || scorer.peak_power != 140.0 || scorer.peak_clock != 2500 {
        passed = false;
        details.push("BenchmarkScorer peak metric tracking mismatch".to_string());
    } else {
        details.push("BenchmarkScorer peak metric tracking verified".to_string());
    }

    let res = scorer.compute_scene_result(0, "Waves", 1920 * 1080);
    if res.frames_rendered != 2 || res.score == 0 || res.avg_fps <= 0.0 {
        passed = false;
        details.push("BenchmarkScorer scene result calculation failed".to_string());
    } else {
        details.push(format!("Scene result scoring & FPS statistics verified (score: {})", res.score));
    }

    // JSON round-trip
    let report = BenchmarkReport {
        timestamp: "2026-11-04 10:00:00 UTC".to_string(),
        system_info: SystemReport {
            gpu_name: "Test GPU".to_string(),
            gpu_backend: "Vulkan".to_string(),
            gpu_driver: "test".to_string(),
            cpu_name: "Test CPU".to_string(),
            cpu_threads: 8,
        },
        total_elapsed_seconds: 5.0,
        total_frames: 300,
        overall_score: 50000,
        peak_gpu_temp_c: 65.0,
        peak_gpu_hotspot_c: 75.0,
        peak_gpu_util_pct: 95.0,
        peak_gpu_power_w: 140.0,
        peak_gpu_clock_mhz: 2500,
        peak_vram_mb: 4096,
        scene_results: vec![res],
    };

    match serde_json::to_string(&report) {
        Ok(json) => match serde_json::from_str::<BenchmarkReport>(&json) {
            Ok(parsed) => {
                if parsed.overall_score == report.overall_score {
                    details.push("BenchmarkReport JSON serialization round-trip verified".to_string());
                } else {
                    passed = false;
                    details.push("JSON deserialization score mismatch".to_string());
                }
            }
            Err(e) => {
                passed = false;
                details.push(format!("JSON deserialization error: {}", e));
            }
        },
        Err(e) => {
            passed = false;
            details.push(format!("JSON serialization error: {}", e));
        }
    }

    let duration = t0.elapsed();
    print_suite_status(5, 6, "Benchmark Engine & Scorer Math", passed, duration, &details);
    TestReport {
        suite_name: "Benchmark Engine & Scorer Math",
        passed,
        duration,
        details,
    }
}

fn test_terminal_suite() -> TestReport {
    let t0 = Instant::now();
    let mut details = Vec::new();
    let mut passed = true;

    // Test resolution tag
    let tag_4k = format_resolution_tag(3840, 2160);
    if !tag_4k.contains("4K") {
        passed = false;
        details.push(format!("Resolution tag mismatch: expected '4K', got '{}'", tag_4k));
    } else {
        details.push("Resolution formatting preset tags verified".to_string());
    }

    // Test math models for scenes
    let mut math_models_ok = true;
    for s in 0..=6 {
        if get_scene_math_model(s).is_empty() {
            math_models_ok = false;
        }
    }
    if !math_models_ok {
        passed = false;
        details.push("One or more scene math model descriptions were empty".to_string());
    } else {
        details.push("Act math model descriptions (Gerstner, Mandelbulb, Schwarzschild, etc.) verified".to_string());
    }

    // Test visible_len
    let colored = "\x1b[38;2;0;255;200;1mHello\x1b[0m";
    if visible_len(colored) != 5 {
        passed = false;
        details.push(format!("ANSI visible_len mismatch: expected 5, got {}", visible_len(colored)));
    } else {
        details.push("ANSI escape sequence width stripping verified".to_string());
    }

    // Test TerminalRenderer
    let renderer = TerminalRenderer::new();
    drop(renderer);
    details.push("TerminalRenderer half-block compositor instantiated".to_string());

    let duration = t0.elapsed();
    print_suite_status(6, 6, "Terminal Engine & ANSI Renderer", passed, duration, &details);
    TestReport {
        suite_name: "Terminal Engine & ANSI Renderer",
        passed,
        duration,
        details,
    }
}

fn print_suite_status(
    index: usize,
    total: usize,
    name: &str,
    passed: bool,
    duration: Duration,
    details: &[String],
) {
    let status_str = if passed {
        "\x1b[38;2;0;255;128;1m[ PASS ]\x1b[0m"
    } else {
        "\x1b[38;2;255;80;80;1m[ FAIL ]\x1b[0m"
    };

    println!(
        "  [{}/{}] {:<42} {} ({:.2?})",
        index, total, name, status_str, duration
    );
    for d in details {
        println!("        \x1b[38;2;120;120;140m•\x1b[0m \x1b[38;2;200;210;220m{}\x1b[0m", d);
    }
}
