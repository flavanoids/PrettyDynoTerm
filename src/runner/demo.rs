use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use std::time::{Duration, Instant};

use crate::benchmark::ThermalFailsafe;
use crate::cpu::NBodySimulation;
use crate::gpu::pipeline::Uniforms;
use crate::runner::session::{DemoOptions, TerminalSession};
use crate::terminal::{Hud, HudMode, get_scene_math_model};

pub const SCENE_NAMES: [&str; 8] = [
    "ATMOSPHERIC OCEAN WAVES (GPU)",
    "THE MANDELBULB CORE (GPU)",
    "RELATIVISTIC BLACK HOLE (GPU)",
    "TORNADIC MESOCYCLONE (GPU)",
    "CARDIAC ELECTROPHYSIOLOGY (GPU)",
    "LATTICE BOLTZMANN VORTEX (GPU)",
    "CYBERNETIC GAIT KINEMATICS (GPU)",
    "KINETIC N-BODY SUPERNOVA (CPU)",
];

pub fn run_interactive_demo(
    session: &mut TerminalSession,
    opts: &DemoOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    session.renderer.enter_alternate_screen()?;
    crossterm::terminal::enable_raw_mode()?;

    let mut term_width = session.term_width;
    let mut term_height = session.term_height;
    let mut scene_id = opts.initial_scene % 8;
    let mut uncapped = opts.initial_uncapped;
    let mut failsafe = ThermalFailsafe::new(opts.max_temp_c, opts.failsafe_enabled);
    let mut hud = Hud::new(term_width, SCENE_NAMES[scene_id as usize].to_string());
    hud.resolution_label = session.res_tag.to_string();
    hud.math_model = get_scene_math_model(scene_id).to_string();
    hud.is_benchmark = false;
    hud.is_uncapped = uncapped;
    let start_time = Instant::now();
    let mut frame_count: u32 = 0;

    let mut nbody = NBodySimulation::new(
        24_000,
        session.pipeline.width as usize,
        session.pipeline.height as usize,
    );
    let mut term_preview_buf = vec![0u8; term_width * term_height * 4];

    let target_frame_duration = Duration::from_micros(16_667);
    let mut last_terminal_draw = Instant::now();

    loop {
        let frame_start = Instant::now();

        // Handle user input
        if event::poll(Duration::from_millis(0))?
            && let Event::Key(key) = event::read()?
        {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            if key.modifiers.contains(KeyModifiers::CONTROL)
                && (key.code == KeyCode::Char('c') || key.code == KeyCode::Char('C'))
            {
                break;
            }
            match key.code {
                KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => break,
                KeyCode::Char(' ') => {
                    scene_id = (scene_id + 1) % 8;
                    hud.scene_title = SCENE_NAMES[scene_id as usize].to_string();
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
        let should_draw_terminal =
            !uncapped || last_terminal_draw.elapsed() >= Duration::from_millis(16);

        if scene_id < 7 {
            let uniforms = Uniforms {
                time: elapsed,
                frame: frame_count,
                width: session.pipeline.width,
                height: session.pipeline.height,
                cam_pos: [0.0, 1.8, -4.5],
                scene_id,
                cam_dir: [0.0, -0.2, 1.0],
                benchmark_mode: if uncapped { 1 } else { 0 },
            };

            if should_draw_terminal {
                session.pipeline.render_with_preview(
                    &uniforms,
                    term_width as u32,
                    term_height as u32,
                    &mut term_preview_buf,
                );
            } else {
                session.pipeline.render_compute_async(&uniforms);
            }
        } else {
            nbody.step_multithreaded(0.005);
            if should_draw_terminal {
                nbody.rasterize_to_preview(&mut term_preview_buf, term_width, term_height);
            }
        }

        let stats = session.telemetry.sample(&session.adapter_info.name);

        // In-flight active emergency thermal tripwire check
        if let Some(trip) = failsafe.check(&stats) {
            if uncapped {
                // Tripwire breach under uncapped stress: disengage uncapped mode immediately to allow thermal recovery
                uncapped = false;
                hud.is_uncapped = false;
            } else {
                // Tripwire hit even while running capped at 60 FPS: emergency shutdown to protect silicon
                crossterm::terminal::disable_raw_mode()?;
                let _ = session.renderer.leave_alternate_screen();
                crate::benchmark::print_emergency_thermal_box(&trip);
                eprintln!(
                    "\x1b[38;2;255;180;0;1m[!] Interactive demo terminated to protect hardware silicon from thermal exhaustion.\x1b[0m\r\n"
                );
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
            let compute_pixels = (session.pipeline.width * session.pipeline.height) as f32;
            hud.rays_per_sec = (compute_pixels / (render_dur.as_secs_f32() + 1e-6)) / 1e9;

            let top_lines = hud.render_top();
            let bottom_lines = hud.render_bottom(&stats);

            session.renderer.render(
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
    session.renderer.leave_alternate_screen()?;
    Ok(())
}
