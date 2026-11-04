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

    // An impossibly large resolution (65536 x 65536 = 17.1 GB single buffer)
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
        err_msg.contains("exceeding GPU hardware limit") || err_msg.contains("storage binding"),
        "Error message should explain buffer limit breach, got: {}",
        err_msg
    );
}
