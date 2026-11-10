use crate::backend::BackendChoice;
use crate::output::OutputFormat;
use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "dyno_term",
    about = "Cross-platform terminal GPU/CPU benchmark & artistic demoparty renderer",
    version = "1.0.0"
)]
pub struct Args {
    /// Open the interactive CLI launcher menu (default when no mode flag is passed)
    #[arg(short, long)]
    pub menu: bool,

    /// Run in headless mode: no TTY, no raw mode, no terminal control sequences
    #[arg(long)]
    pub headless: bool,

    /// Output format (human, json, jsonl)
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Human)]
    pub output: OutputFormat,

    /// Optional path to save benchmark JSON report file
    #[arg(long)]
    pub report_file: Option<String>,

    /// Internal test flag: trigger intentional internal error (exit code 5)
    #[arg(long, hide = true)]
    pub test_internal_error: bool,

    /// Run the deterministic benchmark suite directly
    #[arg(short, long)]
    pub benchmark: bool,

    /// Run multi-iteration consistency & thermal stability suite (e.g. -c 5 or --consistency 5)
    #[arg(short = 'c', long, alias = "stability", num_args = 0..=1, default_missing_value = "5")]
    pub consistency: Option<usize>,

    /// Cooldown pause duration in seconds between consistency iterations (default: 60)
    #[arg(long, default_value_t = 60)]
    pub cooldown: u64,

    /// Target temperature in Celsius for dynamic consistency cooldown (default: 55.0)
    #[arg(long, default_value_t = 55.0)]
    pub target_temp: f32,

    /// Maximum safe thermal limit in Celsius before triggering emergency tripwire (default: 105.0)
    #[arg(long, default_value_t = 105.0)]
    pub max_temp: f32,

    /// Disable emergency thermal tripwire and hardware safeguards (use with caution)
    #[arg(long)]
    pub no_failsafe: bool,

    /// Run in interactive demo preview mode directly
    #[arg(short, long)]
    pub demo: bool,

    /// Run demo mode uncapped (100% GPU stress test, no 60 FPS limiter)
    #[arg(short, long)]
    pub uncapped: bool,

    /// Run comprehensive self-test and system diagnostics suite and exit
    #[arg(short = 't', long, alias = "self-test")]
    pub test: bool,

    /// Number of frames per scene in benchmark mode
    #[arg(short, long, default_value_t = 100)]
    pub frames: u32,

    /// Compute resolution preset ("16k", "8k", "4k", "1440p", "1080p", "720p", "native")
    #[arg(long, default_value = "4k")]
    pub res: String,

    /// Specific scene to display in demo mode (0: Waves, 1: Mandelbulb, 2: Black Hole, 3: Tornado, 4: Cardiac, 5: LBM Vortex, 6: Kinematics, 7: CPU N-Body)
    #[arg(short, long, default_value_t = 0)]
    pub scene: u32,

    /// Select GPU adapter by index
    #[arg(short, long)]
    pub gpu: Option<usize>,

    /// Compute backend selection: "auto" (default), "wgpu", "cuda", or "hip"
    #[arg(long, value_enum, default_value_t = BackendChoice::Auto)]
    pub backend: BackendChoice,

    /// Target GPU selection: "all" (default) or comma-separated indices (e.g. "0,2,5")
    #[arg(long, default_value = "all")]
    pub gpus: String,

    /// List all available GPU adapters and exit
    #[arg(long)]
    pub list_gpus: bool,

    /// Custom render width in pixels
    #[arg(long)]
    pub width: Option<u32>,

    /// Custom render height in pixels
    #[arg(long)]
    pub height: Option<u32>,

    /// Show chronological table of all past benchmark runs and exit
    #[arg(long)]
    pub history: bool,

    /// Set reference baseline report from specified file (or default latest report) and exit
    #[arg(long, num_args = 0..=1, default_missing_value = "dyno_benchmark_results.json")]
    pub set_baseline: Option<String>,

    /// Path to reference baseline JSON file (default: "dyno_baseline.json")
    #[arg(long)]
    pub baseline: Option<String>,

    /// Compare specified report file against reference baseline and exit
    #[arg(long)]
    pub compare: Option<String>,

    /// Capture raw frame screenshots of all 6 acts to output directory and exit
    #[arg(long)]
    pub capture: Option<String>,
}

/// Resolves compute resolution preset string to (width, height) pixel dimensions.
pub fn parse_resolution(res: &str, fallback_w: u32, fallback_h: u32) -> (u32, u32) {
    match res.to_lowercase().as_str() {
        "16k" => (15360, 8640),
        "8k" => (7680, 4320),
        "4k" | "2160p" | "uhd" => (3840, 2160),
        "1440p" | "2k" | "qhd" => (2560, 1440),
        "1080p" | "fhd" => (1920, 1080),
        "720p" | "hd" => (1280, 720),
        "native" => (fallback_w, fallback_h),
        _ => (3840, 2160),
    }
}

/// Parses the `--gpus` argument string or single `--gpu` flag against the total number of available adapters.
pub fn parse_gpu_selection(
    gpus_str: &str,
    single_gpu: Option<usize>,
    total_available: usize,
) -> Result<Vec<usize>, String> {
    if total_available == 0 {
        return Err("No compatible GPU adapters found on system".to_string());
    }

    if let Some(single) = single_gpu {
        if single >= total_available {
            return Err(format!(
                "Requested GPU index {} out of bounds (found {} adapters)",
                single, total_available
            ));
        }
        return Ok(vec![single]);
    }

    let trimmed = gpus_str.trim();
    if trimmed.eq_ignore_ascii_case("all") {
        return Ok((0..total_available).collect());
    }

    let mut indices = Vec::new();
    for part in trimmed.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let idx = p
            .parse::<usize>()
            .map_err(|_| format!("Invalid GPU index '{}' in --gpus", p))?;
        if idx >= total_available {
            return Err(format!(
                "GPU index {} in --gpus is out of bounds (found {} adapters)",
                idx, total_available
            ));
        }
        if !indices.contains(&idx) {
            indices.push(idx);
        }
    }

    if indices.is_empty() {
        return Err("No GPU indices specified in --gpus".to_string());
    }

    Ok(indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gpu_selection_all() {
        let sel = parse_gpu_selection("all", None, 8).unwrap();
        assert_eq!(sel, vec![0, 1, 2, 3, 4, 5, 6, 7]);

        let sel_one = parse_gpu_selection("all", None, 1).unwrap();
        assert_eq!(sel_one, vec![0]);
    }

    #[test]
    fn test_parse_gpu_selection_comma_list() {
        let sel = parse_gpu_selection("0, 2, 5", None, 8).unwrap();
        assert_eq!(sel, vec![0, 2, 5]);
    }

    #[test]
    fn test_parse_gpu_selection_single_override() {
        let sel = parse_gpu_selection("all", Some(3), 8).unwrap();
        assert_eq!(sel, vec![3]);
    }

    #[test]
    fn test_parse_gpu_selection_errors() {
        assert!(parse_gpu_selection("9", None, 8).is_err());
        assert!(parse_gpu_selection("foo", None, 8).is_err());
        assert!(parse_gpu_selection("", None, 8).is_err());
        assert!(parse_gpu_selection("all", None, 0).is_err());
    }
}
