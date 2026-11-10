use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

static TERMINATION_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Initializes POSIX signal handling for SIGINT (Ctrl-C) and SIGTERM.
/// Spawns a 2.0s watchdog thread that ensures termination within 2.0s of signal receipt.
pub fn install_signal_handlers() -> Result<(), Box<dyn std::error::Error>> {
    let flag = Arc::new(AtomicBool::new(false));

    signal_hook::flag::register(signal_hook::consts::SIGINT, Arc::clone(&flag))?;
    signal_hook::flag::register(signal_hook::consts::SIGTERM, Arc::clone(&flag))?;

    let watcher_flag = Arc::clone(&flag);
    thread::Builder::new()
        .name("signal-watchdog".to_string())
        .spawn(move || {
            // Poll for signal flag
            while !watcher_flag.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(20));
            }

            // Signal received! Mark global static
            TERMINATION_REQUESTED.store(true, Ordering::SeqCst);

            // Give the main thread up to 2.0 seconds to perform graceful GPU cleanup and flush records
            let deadline = Instant::now() + Duration::from_millis(2000);
            while Instant::now() < deadline {
                thread::sleep(Duration::from_millis(50));
            }

            // Hard deadline exceeded (2s): force exit with code 4
            std::process::exit(crate::output::ExitCode::Cancelled.as_i32());
        })?;

    Ok(())
}

/// Returns true if SIGINT or SIGTERM has been received.
#[inline]
pub fn is_termination_requested() -> bool {
    TERMINATION_REQUESTED.load(Ordering::Relaxed)
}

/// Explicitly triggers termination state (e.g. from tests or UI abort).
pub fn request_termination() {
    TERMINATION_REQUESTED.store(true, Ordering::SeqCst);
}

/// Resets termination state (for unit testing).
pub fn reset_termination_for_test() {
    TERMINATION_REQUESTED.store(false, Ordering::SeqCst);
}
