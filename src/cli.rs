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

    /// List all available GPU adapters and exit
    #[arg(long)]
    pub list_gpus: bool,

    /// Custom render width in pixels
    #[arg(long)]
    pub width: Option<u32>,

    /// Custom render height in pixels
    #[arg(long)]
    pub height: Option<u32>,

    /// Output JSON path for benchmark report
    #[arg(short, long, default_value = "dyno_benchmark_results.json")]
    pub output: String,

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
