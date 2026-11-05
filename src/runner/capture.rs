use crate::cpu::NBodySimulation;
use crate::gpu::{ComputePipelineManager, pipeline::Uniforms};
use std::io::Write;

pub fn capture_screenshots(
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
                let uniforms = Uniforms {
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
    let mut file = std::fs::File::create(path)?;
    write!(file, "P6\n{} {}\n255\n", width, height)?;
    for chunk in rgba.as_chunks::<4>().0 {
        file.write_all(&[chunk[0], chunk[1], chunk[2]])?;
    }
    Ok(())
}
