pub mod benchmark;
pub mod capture;
pub mod consistency;
pub mod demo;
pub mod session;

pub use benchmark::{execute_benchmark_pass, print_scorecard, run_benchmark};
pub use capture::capture_screenshots;
pub use consistency::{format_box_line, run_consistency_suite, run_cooldown_screen};
pub use demo::{SCENE_NAMES, run_interactive_demo};
pub use session::{
    BenchmarkOptions, BenchmarkPassOptions, BenchmarkPassResult, ConsistencyOptions,
    CooldownOptions, DemoOptions, TerminalSession,
};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use std::time::Duration;

pub fn wait_for_menu_return() -> Result<(), Box<dyn std::error::Error>> {
    crossterm::terminal::enable_raw_mode()?;
    loop {
        if event::poll(Duration::from_millis(50))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            match key.code {
                KeyCode::Enter | KeyCode::Char(' ') | KeyCode::Esc | KeyCode::Char('q') => break,
                _ => {}
            }
        }
    }
    crossterm::terminal::disable_raw_mode()?;
    Ok(())
}
