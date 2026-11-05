use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use std::time::{Duration, Instant};

use crate::benchmark::{
    BenchmarkReport, DEFAULT_HISTORY_FILE, append_to_history, compute_consistency_report,
    format_consistency_scorecard, print_emergency_thermal_box, save_consistency_report,
};
use crate::runner::benchmark::execute_benchmark_pass;
use crate::runner::session::{
    BenchmarkPassOptions, BenchmarkPassResult, ConsistencyOptions, CooldownOptions, TerminalSession,
};
use crate::terminal::visible_len;

pub fn format_box_line(content: &str, inner_width: usize) -> String {
    let vis = visible_len(content);
    let pad = inner_width.saturating_sub(vis);
    format!(
        "\x1b[38;2;0;220;255m│\x1b[0m \x1b[38;2;255;255;255m{}{}\x1b[0m \x1b[38;2;0;220;255m│\x1b[0m\r\n",
        content,
        " ".repeat(pad)
    )
}

pub fn run_cooldown_screen(
    session: &mut TerminalSession,
    opts: &CooldownOptions,
) -> Result<bool, Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let cooldown_dur = Duration::from_secs(opts.cooldown_seconds);

    loop {
        if event::poll(Duration::from_millis(100))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            if key.modifiers.contains(KeyModifiers::CONTROL)
                && (key.code == KeyCode::Char('c') || key.code == KeyCode::Char('C'))
            {
                return Ok(false);
            }
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

        let elapsed = start_time.elapsed();
        let stats = session.telemetry.sample(&session.adapter_info.name);
        let cur_temp = stats.gpu_temp_c;

        let is_thermally_cooled = cur_temp > 0.0 && cur_temp <= opts.target_temp_c;
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

        let remaining_secs = opts.cooldown_seconds.saturating_sub(elapsed.as_secs());
        let progress_ratio =
            (elapsed.as_secs_f32() / opts.cooldown_seconds.max(1) as f32).clamp(0.0, 1.0);

        let bar_width: usize = 28;
        let filled_chars = (progress_ratio * bar_width as f32).round() as usize;
        let empty_chars = bar_width.saturating_sub(filled_chars);
        let progress_bar = format!(
            "\x1b[38;2;0;255;200;1m{}\x1b[38;2;60;80;100m{}\x1b[0m",
            "█".repeat(filled_chars),
            "░".repeat(empty_chars)
        );

        let temp_delta = cur_temp - opts.last_report.peak_gpu_temp_c;
        let delta_sign = if temp_delta >= 0.0 { "+" } else { "" };

        let mut buf = String::with_capacity(2048);
        buf.push_str("\r\n");
        buf.push_str("\x1b[38;2;0;220;255;1m┌──────────────────────────────────────────────────────────────────────────┐\r\n");
        buf.push_str(
            "│                   THERMAL STABILIZATION COOLDOWN PAUSE                   │\r\n",
        );
        buf.push_str("├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        let line1 = format!(
            "Completed Run     : \x1b[38;2;255;255;255;1m{} of {}\x1b[38;2;140;150;170m (\x1b[38;2;255;180;0;1m{:>8.2} pts\x1b[38;2;140;150;170m, \x1b[38;2;255;255;255m{:.2}s\x1b[38;2;140;150;170m)",
            opts.completed_run,
            opts.total_runs,
            opts.last_report.overall_score,
            opts.last_report.total_elapsed_seconds
        );
        buf.push_str(&format_box_line(&line1, 72));

        let line2 = format!(
            "Next Iteration    : \x1b[38;2;0;255;200;1mRun {} of {}\x1b[38;2;140;150;170m",
            opts.completed_run + 1,
            opts.total_runs
        );
        buf.push_str(&format_box_line(&line2, 72));

        let line3 = format!(
            "Cooldown Progress : [{}] \x1b[38;2;255;180;0;1m{:>2}s\x1b[38;2;140;150;170m / {}s remaining",
            progress_bar, remaining_secs, opts.cooldown_seconds
        );
        buf.push_str(&format_box_line(&line3, 72));

        buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        let status_badge = if is_thermally_cooled {
            format!(
                "\x1b[38;2;0;255;200;1m[●] COOLED (≤ {:.1}°C)\x1b[0m",
                opts.target_temp_c
            )
        } else if is_heat_soaked {
            "\x1b[38;2;255;60;60;1m[⚠] HEAT-SOAK EXHAUSTION\x1b[0m".to_string()
        } else {
            let diff = cur_temp - opts.target_temp_c;
            format!(
                "\x1b[38;2;0;220;255;1m[○] DISSIPATING (+{:.1}°C to target)\x1b[0m",
                diff.max(0.0)
            )
        };

        let line_head = format!("THERMAL HYSTERESIS: {}", status_badge);
        buf.push_str(&format_box_line(&line_head, 72));

        let line4 = format!(
            "GPU Core Temp     : \x1b[38;2;0;255;200;1m{:.1}°C\x1b[38;2;140;150;170m (Peak: {:.1}°C, Δ: {}{:.1}°C, Goal: ≤{:.1}°C)",
            cur_temp, opts.last_report.peak_gpu_temp_c, delta_sign, temp_delta, opts.target_temp_c
        );
        buf.push_str(&format_box_line(&line4, 72));

        let line5 = format!(
            "GPU Hotspot Temp  : \x1b[38;2;0;255;200;1m{:.1}°C\x1b[38;2;140;150;170m (Run Peak: {:.1}°C)",
            stats.gpu_hotspot_c, opts.last_report.peak_gpu_hotspot_c
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

        session.renderer.render_raw_text(&buf)?;
    }
}

pub fn run_consistency_suite(
    session: &mut TerminalSession,
    opts: &ConsistencyOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let iterations = opts.iterations.max(1);
    session.renderer.enter_alternate_screen()?;
    crossterm::terminal::enable_raw_mode()?;

    let mut completed_runs: Vec<BenchmarkReport> = Vec::with_capacity(iterations);
    let mut halted_by_thermal = false;

    for i in 1..=iterations {
        let pass_opts = BenchmarkPassOptions {
            frames_per_scene: opts.frames_per_scene,
            iteration_info: Some((i, iterations)),
            max_temp_c: opts.max_temp_c,
            failsafe_enabled: opts.failsafe_enabled,
        };

        let pass_res = execute_benchmark_pass(session, &pass_opts);

        let report = match pass_res {
            Ok(BenchmarkPassResult::Completed(r)) => *r,
            Ok(BenchmarkPassResult::AbortedByUser) => {
                break;
            }
            Ok(BenchmarkPassResult::ThermalTrip(trip)) => {
                crossterm::terminal::disable_raw_mode()?;
                let _ = session.renderer.leave_alternate_screen();
                print_emergency_thermal_box(&trip);
                halted_by_thermal = true;
                break;
            }
            Err(e) => {
                crossterm::terminal::disable_raw_mode()?;
                let _ = session.renderer.leave_alternate_screen();
                return Err(e);
            }
        };

        let _ = append_to_history(&report, DEFAULT_HISTORY_FILE);
        completed_runs.push(report);

        if i < iterations && opts.cooldown_seconds > 0 {
            let last = completed_runs.last().unwrap();
            let cooldown_opts = CooldownOptions {
                completed_run: i,
                total_runs: iterations,
                cooldown_seconds: opts.cooldown_seconds,
                target_temp_c: opts.target_temp_c,
                last_report: last,
            };
            let keep_going = run_cooldown_screen(session, &cooldown_opts)?;
            if !keep_going {
                break;
            }
        }
    }

    crossterm::terminal::disable_raw_mode()?;
    let _ = session.renderer.leave_alternate_screen();

    if halted_by_thermal {
        eprintln!(
            "\x1b[38;2;255;180;0;1m⚠ Consistency testing halted prematurely by thermal tripwire after {} run(s).\x1b[0m",
            completed_runs.len()
        );
    }

    if completed_runs.is_empty() {
        println!("Consistency suite aborted before any iterations completed.");
        return Ok(());
    }

    let report = compute_consistency_report(
        &completed_runs,
        opts.cooldown_seconds,
        opts.frames_per_scene,
        session.res_tag,
    );

    print!("{}", format_consistency_scorecard(&report));
    if let Err(e) = save_consistency_report(&report, &opts.output_path) {
        eprintln!(
            "\r\n\x1b[38;2;255;80;80;1m⚠ Warning: Failed to save consistency report to '{}': {}\x1b[0m",
            opts.output_path, e
        );
    } else {
        println!("Consistency report saved to: {}", opts.output_path);
    }

    Ok(())
}
