use dyno_term::benchmark::failsafe::{GpuMemoryInfo, VramHeadroomCheck, VramTestTier};
use dyno_term::gpu::pipeline::Uniforms;
use dyno_term::gpu::{ComputePipelineManager, GpuContext};
use dyno_term::shaders::UNIFIED_SHADER;
use std::sync::Arc;

#[test]
fn test_gpu_context_and_compute_dispatch() {
    let adapters = GpuContext::list_adapters();
    if adapters.is_empty() {
        println!("Skipping GPU integration test: no compatible GPU adapter found");
        return;
    }

    let gpu_ctx = match pollster::block_on(GpuContext::new(None)) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping GPU test: failed to initialize device ({})", e);
            return;
        }
    };

    let test_w = 160u32;
    let test_h = 90u32;
    let prev_w = 40u32;
    let prev_h = 20u32;

    let mut pipeline = ComputePipelineManager::new(
        Arc::clone(&gpu_ctx.device),
        Arc::clone(&gpu_ctx.queue),
        UNIFIED_SHADER,
        test_w,
        test_h,
    )
    .expect("Failed to create compute pipeline");

    let uniforms = Uniforms {
        time: 1.0,
        frame: 1,
        width: test_w,
        height: test_h,
        cam_pos: [0.0, 1.8, -4.5],
        scene_id: 0,
        cam_dir: [0.0, -0.2, 1.0],
        benchmark_mode: 0,
    };

    let mut preview_buf = vec![0u8; (prev_w * prev_h * 4) as usize];
    pipeline.render_with_preview(&uniforms, prev_w, prev_h, &mut preview_buf);

    let has_nonzero = preview_buf.iter().any(|&b| b > 0);
    assert!(
        has_nonzero,
        "GPU compute pass should write non-zero pixel data to preview buffer"
    );
}

#[test]
fn test_vram_headroom_guard_rejection() {
    let adapters = GpuContext::list_adapters();
    if adapters.is_empty() {
        println!("Skipping GPU test: no compatible GPU adapter found");
        return;
    }

    let gpu_ctx = match pollster::block_on(GpuContext::new(None)) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping GPU test: failed to initialize device ({})", e);
            return;
        }
    };

    // An impossibly large resolution (65536 x 65536 = 17.1 GB single buffer = ~34GB total pipeline)
    let err = ComputePipelineManager::new(
        Arc::clone(&gpu_ctx.device),
        Arc::clone(&gpu_ctx.queue),
        UNIFIED_SHADER,
        65536,
        65536,
    );
    assert!(
        err.is_err(),
        "Extreme buffer allocation should be rejected safely by pre-flight guard before allocation"
    );
    let err_msg = err.err().unwrap();
    assert!(
        err_msg.contains("exceeding GPU hardware")
            || err_msg.contains("storage binding")
            || err_msg.contains("Out-of-memory failsafe")
            || err_msg.contains("workgroups"),
        "Error message should explain buffer or memory limit breach, got: {}",
        err_msg
    );
}

#[test]
fn test_dynamic_vram_adaptation_and_allocation() {
    let adapters = GpuContext::list_adapters();
    if adapters.is_empty() {
        println!("Skipping GPU test: no compatible GPU adapter found");
        return;
    }

    let gpu_ctx = match pollster::block_on(GpuContext::new(None)) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping GPU test: failed to initialize device ({})", e);
            return;
        }
    };

    let mem_info = GpuMemoryInfo::detect(&gpu_ctx.device);
    println!(
        "Detected GPU VRAM: {} MB total ({} MB free, {} MB used)",
        mem_info.total_vram_mb, mem_info.free_vram_mb, mem_info.used_vram_mb
    );

    // Dynamically calculate adapted resolution for Balanced testing
    let (dyn_w, dyn_h) = mem_info.recommend_resolution(VramTestTier::Balanced);
    println!(
        "Dynamically adapted test resolution: {}x{} ({:.2} Mpix)",
        dyn_w,
        dyn_h,
        (dyn_w as f64 * dyn_h as f64) / 1_000_000.0
    );

    let mut pipeline = ComputePipelineManager::new(
        Arc::clone(&gpu_ctx.device),
        Arc::clone(&gpu_ctx.queue),
        UNIFIED_SHADER,
        dyn_w,
        dyn_h,
    )
    .expect("Pipeline instantiation at dynamically adapted resolution should succeed");

    let uniforms = Uniforms {
        time: 2.0,
        frame: 1,
        width: dyn_w,
        height: dyn_h,
        cam_pos: [0.0, 1.8, -4.5],
        scene_id: 0,
        cam_dir: [0.0, -0.2, 1.0],
        benchmark_mode: 0,
    };

    let prev_w = 80u32;
    let prev_h = 40u32;
    let mut preview_buf = vec![0u8; (prev_w * prev_h * 4) as usize];
    pipeline.render_with_preview(&uniforms, prev_w, prev_h, &mut preview_buf);

    assert!(
        preview_buf.iter().any(|&b| b > 0),
        "Render output at dynamic resolution should contain rendered pixels"
    );

    // Verify safe dynamic resizing down to Light tier
    let (light_w, light_h) = mem_info.recommend_resolution(VramTestTier::Light);
    assert!(pipeline.try_resize(light_w, light_h).is_ok());
    assert_eq!(pipeline.width, light_w);
    assert_eq!(pipeline.height, light_h);
}

#[test]
fn test_out_of_bounds_fail_safes() {
    let adapters = GpuContext::list_adapters();
    if adapters.is_empty() {
        println!("Skipping GPU test: no compatible GPU adapter found");
        return;
    }

    let gpu_ctx = match pollster::block_on(GpuContext::new(None)) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping GPU test: failed to initialize device ({})", e);
            return;
        }
    };

    // 1. Zero dimensions rejection (0x0, 0x1080, 1920x0)
    let res_0x0 = ComputePipelineManager::new(
        Arc::clone(&gpu_ctx.device),
        Arc::clone(&gpu_ctx.queue),
        UNIFIED_SHADER,
        0,
        0,
    );
    assert!(res_0x0.is_err(), "0x0 resolution must be rejected");

    let res_0x1080 = ComputePipelineManager::new(
        Arc::clone(&gpu_ctx.device),
        Arc::clone(&gpu_ctx.queue),
        UNIFIED_SHADER,
        0,
        1080,
    );
    assert!(res_0x1080.is_err(), "0x1080 resolution must be rejected");

    let res_1920x0 = ComputePipelineManager::new(
        Arc::clone(&gpu_ctx.device),
        Arc::clone(&gpu_ctx.queue),
        UNIFIED_SHADER,
        1920,
        0,
    );
    assert!(res_1920x0.is_err(), "1920x0 resolution must be rejected");

    // 2. Workgroup limit out of bounds (width exceeds max_compute_workgroups_per_dimension * 16)
    let max_wg = gpu_ctx.device.limits().max_compute_workgroups_per_dimension;
    let oob_width = ((max_wg as u64 + 10) * 16) as u32;
    let res_oob_wg = ComputePipelineManager::new(
        Arc::clone(&gpu_ctx.device),
        Arc::clone(&gpu_ctx.queue),
        UNIFIED_SHADER,
        oob_width,
        16,
    );
    assert!(
        res_oob_wg.is_err(),
        "Workgroups exceeding max_compute_workgroups_per_dimension must be rejected safely"
    );

    // 3. Valid pipeline resize to 0x0 must be safely rejected without crashing or corrupting pipeline
    let mut valid_pipeline = ComputePipelineManager::new(
        Arc::clone(&gpu_ctx.device),
        Arc::clone(&gpu_ctx.queue),
        UNIFIED_SHADER,
        160,
        90,
    )
    .expect("Small valid pipeline must succeed");

    assert!(valid_pipeline.try_resize(0, 0).is_err());
    assert_eq!(valid_pipeline.width, 160);
    assert_eq!(valid_pipeline.height, 90);

    // 4. render_with_preview with 0 dimensions or empty buffer must not panic or divide by zero
    let uniforms = Uniforms {
        time: 1.0,
        frame: 1,
        width: 160,
        height: 90,
        cam_pos: [0.0, 0.0, 0.0],
        scene_id: 0,
        cam_dir: [0.0, 0.0, 1.0],
        benchmark_mode: 0,
    };
    let mut empty_buf: Vec<u8> = Vec::new();
    valid_pipeline.render_with_preview(&uniforms, 0, 0, &mut empty_buf);
    valid_pipeline.render_with_preview(&uniforms, 40, 20, &mut empty_buf);
}

#[test]
fn test_out_of_memory_fail_safes() {
    let adapters = GpuContext::list_adapters();
    if adapters.is_empty() {
        println!("Skipping GPU test: no compatible GPU adapter found");
        return;
    }

    let gpu_ctx = match pollster::block_on(GpuContext::new(None)) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping GPU test: failed to initialize device ({})", e);
            return;
        }
    };

    let mem_info = GpuMemoryInfo::detect(&gpu_ctx.device);

    // If free VRAM was detected, evaluate an allocation that demands more memory than available
    if mem_info.free_vram_mb > 0 {
        // Calculate resolution where estimated pipeline memory > free VRAM
        let target_bytes = (mem_info.free_vram_mb + 2048) * 1024 * 1024;
        let single_buf = (target_bytes.saturating_sub(64 * 1024 * 1024)) / 2;
        let pixels = single_buf / 4;
        let k = ((pixels as f64 / 144.0).sqrt() as u32).max(10);
        let test_w = ((16 * k) / 16) * 16;
        let test_h = ((9 * k) / 16) * 16;

        let eval_result = VramHeadroomCheck::evaluate(&gpu_ctx.device, test_w, test_h, None);
        assert!(
            eval_result.is_err(),
            "Workload requiring {} MB must be rejected on GPU with {} MB free",
            target_bytes / (1024 * 1024),
            mem_info.free_vram_mb
        );
        let msg = eval_result.err().unwrap();
        assert!(
            msg.contains("Out-of-memory failsafe")
                || msg.contains("exceeding GPU hardware")
                || msg.contains("storage binding"),
            "Error message should explain memory headroom breach, got: {}",
            msg
        );
    }
}

#[test]
fn test_vram_scaling_model_8gb_to_200gb() {
    // Test across diverse synthetic profiles spanning 4GB to 200GB VRAM
    let profiles = [
        (
            "4GB Entry Discrete",
            GpuMemoryInfo::synthetic(
                4096,
                2500,
                2 * 1024 * 1024 * 1024,
                2 * 1024 * 1024 * 1024,
                65535,
            ),
        ),
        (
            "8GB Discrete (e.g. RX 6650 XT)",
            GpuMemoryInfo::synthetic(
                8192,
                3500,
                4 * 1024 * 1024 * 1024,
                4 * 1024 * 1024 * 1024,
                65535,
            ),
        ),
        (
            "16GB Enthusiast",
            GpuMemoryInfo::synthetic(
                16384,
                12000,
                8 * 1024 * 1024 * 1024,
                8 * 1024 * 1024 * 1024,
                65535,
            ),
        ),
        (
            "24GB Pro / Creator",
            GpuMemoryInfo::synthetic(
                24576,
                18000,
                12 * 1024 * 1024 * 1024,
                12 * 1024 * 1024 * 1024,
                65535,
            ),
        ),
        (
            "80GB AI Workstation",
            GpuMemoryInfo::synthetic(
                81920,
                70000,
                32 * 1024 * 1024 * 1024,
                32 * 1024 * 1024 * 1024,
                65535,
            ),
        ),
        (
            "200GB Datacenter Scale",
            GpuMemoryInfo::synthetic(
                204800,
                180000,
                64 * 1024 * 1024 * 1024,
                64 * 1024 * 1024 * 1024,
                65535,
            ),
        ),
    ];

    let mut last_balanced_pixels = 0u64;
    let mut last_stress_pixels = 0u64;

    for (label, profile) in &profiles {
        let (light_w, light_h) = profile.recommend_resolution(VramTestTier::Light);
        let (bal_w, bal_h) = profile.recommend_resolution(VramTestTier::Balanced);
        let (stress_w, stress_h) = profile.recommend_resolution(VramTestTier::Stress);
        let (max_w, max_h) = profile.recommend_resolution(VramTestTier::MaximumSafe);

        let bal_pix = (bal_w as u64) * (bal_h as u64);
        let stress_pix = (stress_w as u64) * (stress_h as u64);
        let max_pix = (max_w as u64) * (max_h as u64);

        println!(
            "Profile [{}]: Light={}x{}, Balanced={}x{} ({:.1} Mpix), Stress={}x{} ({:.1} Mpix), MaxSafe={}x{} ({:.1} Mpix)",
            label,
            light_w,
            light_h,
            bal_w,
            bal_h,
            bal_pix as f64 / 1e6,
            stress_w,
            stress_h,
            stress_pix as f64 / 1e6,
            max_w,
            max_h,
            max_pix as f64 / 1e6
        );

        // Verification 1: Within each profile, tiers must scale monotonically
        assert!(bal_pix >= (light_w as u64 * light_h as u64));
        assert!(stress_pix >= bal_pix);
        assert!(max_pix >= stress_pix);

        // Verification 2: Across profiles, higher VRAM capacity allows progressively larger workloads
        assert!(bal_pix >= last_balanced_pixels);
        assert!(stress_pix >= last_stress_pixels);
        last_balanced_pixels = bal_pix;
        last_stress_pixels = stress_pix;

        // Verification 3: Every recommended resolution must strictly respect memory headroom
        let single_buf_bytes = max_pix * 4;
        let total_est_mb = (single_buf_bytes * 2 + 64 * 1024 * 1024) / (1024 * 1024);
        assert!(
            total_est_mb <= profile.free_vram_mb,
            "MaxSafe resolution on {} must not exceed free VRAM (got {} MB, free {} MB)",
            label,
            total_est_mb,
            profile.free_vram_mb
        );
        assert!(single_buf_bytes <= profile.max_buffer_size);
    }
}
