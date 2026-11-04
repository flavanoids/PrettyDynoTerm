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

    /// Specific scene to display in demo mode (0: Waves, 1: Mandelbulb, 2: Black Hole, 3: Tornado, 4: Cardiac, 5: CPU N-Body)
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
