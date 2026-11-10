use crate::telemetry::HardwareStats;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HudMode {
    Standard,
    Minimal,
    Hidden,
}

pub struct Hud {
    pub width: usize,
    pub scene_title: String,
    pub math_model: String,
    pub resolution_label: String,
    pub mode: HudMode,
    pub is_benchmark: bool,
    pub is_uncapped: bool,

    // Benchmark progress
    pub act_index: usize,
    pub act_total: usize,
    pub act_progress_pct: f32,
    pub overall_progress_pct: f32,
    pub current_score: f64,
    pub iteration_info: Option<(usize, usize)>,

    // Demo session telemetry
    pub demo_elapsed_secs: f32,
    pub demo_frame_count: u64,

    // Rolling frametimes & performance
    pub frametime_ms: f32,
    pub avg_frametime_ms: f32,
    pub p1_low_ms: f32,
    pub fps: f32,
    pub rays_per_sec: f32,
    pub frametime_history: VecDeque<f32>,
}

impl Hud {
    pub fn new(width: usize, scene_title: String) -> Self {
        Self {
            width,
            scene_title,
            math_model: "Clausius-Clapeyron Vapor Thermodynamics".to_string(),
            resolution_label: "4K UHD (3840x2160)".to_string(),
            mode: HudMode::Standard,
            is_benchmark: false,
            is_uncapped: false,
            act_index: 1,
            act_total: 7,
            act_progress_pct: 0.0,
            overall_progress_pct: 0.0,
            current_score: 0.0,
            iteration_info: None,
            demo_elapsed_secs: 0.0,
            demo_frame_count: 0,
            frametime_ms: 0.0,
            avg_frametime_ms: 0.0,
            p1_low_ms: 0.0,
            fps: 0.0,
            rays_per_sec: 0.0,
            frametime_history: VecDeque::with_capacity(32),
        }
    }

    pub fn toggle_mode(&mut self) -> HudMode {
        self.mode = match self.mode {
            HudMode::Standard => HudMode::Minimal,
            HudMode::Minimal => HudMode::Hidden,
            HudMode::Hidden => HudMode::Standard,
        };
        self.mode
    }

    pub fn record_frametime(&mut self, ms: f32) {
        if self.frametime_history.len() >= 30 {
            self.frametime_history.pop_front();
        }
        self.frametime_history.push_back(ms);

        self.frametime_ms = ms;
        let sum: f32 = self.frametime_history.iter().sum();
        let count = self.frametime_history.len() as f32;
        self.avg_frametime_ms = if count > 0.0 { sum / count } else { ms };
        self.fps = if self.avg_frametime_ms > 0.0 {
            1000.0 / self.avg_frametime_ms
        } else {
            0.0
        };

        // Calculate 95th-99th percentile (1% low) over window
        let mut sorted: Vec<f32> = self.frametime_history.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p_idx = ((sorted.len() as f32 * 0.95) as usize).min(sorted.len() - 1);
        self.p1_low_ms = sorted[p_idx];
    }

    pub fn generate_sparkline(&self, width: usize) -> String {
        const SPARK_CHARS: [char; 8] = [' ', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        if self.frametime_history.is_empty() {
            return "─".repeat(width);
        }

        let count = self.frametime_history.len().min(width);
        let recent: Vec<f32> = self
            .frametime_history
            .iter()
            .rev()
            .take(count)
            .rev()
            .copied()
            .collect();
        let min_v = recent.iter().copied().fold(f32::INFINITY, f32::min);
        let max_v = recent.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_v - min_v).max(1.0);
        let avg = self.avg_frametime_ms.max(1.0);

        let mut out = String::new();
        let pad = width.saturating_sub(recent.len());
        out.push_str(&" ".repeat(pad));

        for &val in &recent {
            let norm = ((val - min_v) / range).clamp(0.0, 0.99);
            let idx = (norm * 8.0) as usize;
            let c = SPARK_CHARS[idx.min(7)];

            // Color-code sparkline: red for severe spike, gold for moderate, mint for normal
            if val > avg * 1.35 {
                out.push_str("\x1b[38;2;255;60;60;1m");
            } else if val > avg * 1.15 {
                out.push_str("\x1b[38;2;255;180;0;1m");
            } else {
                out.push_str("\x1b[38;2;0;255;200m");
            }
            out.push(c);
        }
        out.push_str("\x1b[0m");
        out
    }

    pub fn render_top(&self) -> Vec<String> {
        if self.mode == HudMode::Hidden {
            return vec![];
        }

        let border_color = "\x1b[38;2;0;220;255m"; // Cyber cyan
        let text_color = "\x1b[38;2;255;255;255;1m";
        let res_color = "\x1b[38;2;255;180;0;1m"; // Amber gold
        let math_color = "\x1b[38;2;0;255;200m"; // Neon mint
        let reset = "\x1b[0m";

        let w = self.width;
        let overhead = 18;
        let budget = w.saturating_sub(overhead);

        let (title_left, title_mid, title_right) = if budget >= 82 {
            (
                " DYNO_TERM v1.0 ".to_string(),
                format!(" {} ", self.resolution_label),
                format!(" {} │ {} ", self.scene_title, self.math_model),
            )
        } else if budget >= 55 {
            let res_short = self
                .resolution_label
                .split_whitespace()
                .next()
                .unwrap_or("4K");
            (
                " DYNO_TERM ".to_string(),
                format!(" {} ", res_short),
                format!(" {} │ {} ", self.scene_title, self.math_model),
            )
        } else if budget >= 36 {
            let res_short = self
                .resolution_label
                .split_whitespace()
                .next()
                .unwrap_or("4K");
            let scene_short = self
                .scene_title
                .replace(" (GPU)", "")
                .replace(" (CPU)", "")
                .replace("THE ", "");
            (
                " DYNO ".to_string(),
                format!(" {} ", res_short),
                format!(" {} ", scene_short),
            )
        } else {
            let res_short = self
                .resolution_label
                .split_whitespace()
                .next()
                .unwrap_or("4K");
            let scene_words: Vec<&str> = self.scene_title.split_whitespace().collect();
            let scene_word = scene_words.first().unwrap_or(&"BENCH");
            (
                " D ".to_string(),
                format!(" {} ", res_short),
                format!(" {} ", scene_word),
            )
        };

        let mut right = title_right;
        let left_mid_len = title_left.chars().count() + title_mid.chars().count();
        if left_mid_len + right.chars().count() > budget {
            let max_r = budget.saturating_sub(left_mid_len);
            right = right.chars().take(max_r).collect();
        }

        let total_text_len =
            title_left.chars().count() + title_mid.chars().count() + right.chars().count();
        let spare = budget.saturating_sub(total_text_len);
        let fill_left = "─".repeat(spare / 2);
        let fill_right = "─".repeat(spare.saturating_sub(spare / 2));

        let formatted_right = if right.contains('│') {
            let parts: Vec<&str> = right.split('│').collect();
            format!(
                "{text_color}{}{border_color}│{math_color}{}",
                parts[0], parts[1]
            )
        } else {
            format!("{text_color}{right}")
        };

        let line = format!(
            "{border_color}┌── [{text_color}{title_left}{border_color}] {fill_left} [{res_color}{title_mid}{border_color}] {fill_right} [{formatted_right}{border_color}] ──┐{reset}"
        );

        vec![line]
    }

    pub fn render_bottom(&self, stats: &HardwareStats) -> Vec<String> {
        if self.mode != HudMode::Standard {
            return vec![];
        }

        let border_color = "\x1b[38;2;0;220;255m"; // Cyan border
        let label_color = "\x1b[38;2;140;150;170m"; // Muted slate
        let val_color = "\x1b[38;2;0;255;200;1m"; // Neon mint
        let highlight = "\x1b[38;2;255;180;0;1m"; // Amber
        let reset = "\x1b[0m";

        let w = self.width;

        // Line 0: Separator
        let sep_overhead = 14;
        let sep_budget = w.saturating_sub(sep_overhead);

        let (mut telemetry_title, mut progress_title) = if self.is_benchmark {
            let iter_prefix = if let Some((cur, total)) = self.iteration_info {
                format!("[RUN {}/{}] ", cur, total)
            } else {
                String::new()
            };
            if sep_budget >= 60 {
                (
                    " TELEMETRY ".to_string(),
                    format!(
                        " {}ACT {}/{}: {:>3.0}% │ OVERALL: {:>3.0}% ",
                        iter_prefix,
                        self.act_index,
                        self.act_total,
                        self.act_progress_pct,
                        self.overall_progress_pct
                    ),
                )
            } else if sep_budget >= 35 {
                (
                    " TELEMETRY ".to_string(),
                    format!(
                        " {}ACT {}/{}: {:>3.0}% ",
                        iter_prefix, self.act_index, self.act_total, self.act_progress_pct
                    ),
                )
            } else {
                (
                    " STATS ".to_string(),
                    format!(" {:>3.0}% ", self.act_progress_pct),
                )
            }
        } else {
            let mins = (self.demo_elapsed_secs / 60.0) as u32;
            let secs = (self.demo_elapsed_secs % 60.0) as u32;
            if sep_budget >= 48 {
                (
                    " LIVE TELEMETRY ".to_string(),
                    format!(
                        " DEMO: {:02}:{:02} │ {} FRAMES ",
                        mins, secs, self.demo_frame_count
                    ),
                )
            } else if sep_budget >= 28 {
                (
                    " TELEMETRY ".to_string(),
                    format!(" DEMO: {:02}:{:02} ", mins, secs),
                )
            } else {
                (" LIVE ".to_string(), format!(" {:02}:{:02} ", mins, secs))
            }
        };

        if telemetry_title.chars().count() + progress_title.chars().count() > sep_budget {
            let avail = sep_budget.saturating_sub(telemetry_title.chars().count());
            progress_title = progress_title.chars().take(avail).collect();
        }
        if telemetry_title.chars().count() + progress_title.chars().count() > sep_budget {
            telemetry_title = telemetry_title.chars().take(sep_budget).collect();
            progress_title.clear();
        }

        let sep_text_len = telemetry_title.chars().count() + progress_title.chars().count();
        let sep_fill_len = sep_budget.saturating_sub(sep_text_len);
        let sep_fill = "─".repeat(sep_fill_len);

        let separator = format!(
            "{border_color}├── [{label_color}{telemetry_title}{border_color}] {sep_fill} [{highlight}{progress_title}{border_color}] ──┤{reset}"
        );

        // Columns space: "│ " (2) + " │ " (3) + " │ " (3) + " │" (2) = 10 chars
        let col_space = w.saturating_sub(10);
        let c1_w = (col_space * 32) / 100;
        let c2_w = (col_space * 38) / 100;
        let c3_w = col_space.saturating_sub(c1_w + c2_w);

        // ─────────────────────────────────────────────
        // Column 1: GPU Hardware, Clocks, Thermals, VRAM
        // ─────────────────────────────────────────────
        let c1_l1 = format!("{label_color}GPU: {val_color}{}", stats.gpu_name);

        let temp_col = if stats.gpu_temp_c >= 80.0 {
            "\x1b[38;2;255;60;60;1m" // Hot Red warning
        } else if stats.gpu_temp_c >= 65.0 {
            "\x1b[38;2;255;180;0;1m" // Warm Amber
        } else {
            val_color // Cool Mint
        };
        let pwr_col = if stats.gpu_power_w >= 140.0 {
            highlight
        } else {
            val_color
        };
        let util_col = if stats.gpu_util_pct >= 90.0 {
            val_color // Neon mint - saturated
        } else if stats.gpu_util_pct >= 50.0 {
            highlight // Amber
        } else {
            label_color // Slate
        };

        let c1_l2 = if c1_w >= 36 {
            format!(
                "{val_color}{:>4}MHz {border_color}│ {temp_col}{:>2.0}°C {border_color}│ {util_col}{:>3.0}% Load {border_color}│ {pwr_col}{:>3.0}W",
                stats.gpu_clock_mhz, stats.gpu_temp_c, stats.gpu_util_pct, stats.gpu_power_w
            )
        } else if c1_w >= 28 {
            format!(
                "{val_color}{:>4}MHz {border_color}│ {temp_col}{:>2.0}°C {border_color}│ {pwr_col}{:>3.0}W",
                stats.gpu_clock_mhz, stats.gpu_temp_c, stats.gpu_power_w
            )
        } else if c1_w >= 20 {
            format!(
                "{val_color}{}M {temp_col}{:.0}°C {pwr_col}{:.0}W",
                stats.gpu_clock_mhz, stats.gpu_temp_c, stats.gpu_power_w
            )
        } else {
            format!(
                "{val_color}{}M {temp_col}{:.0}°C",
                stats.gpu_clock_mhz, stats.gpu_temp_c
            )
        };

        let vram_pct = (stats.vram_used_mb * 100)
            .checked_div(stats.vram_total_mb)
            .unwrap_or(0)
            .min(100);
        let gpu_bar_len = if c1_w >= 32 {
            5
        } else if c1_w >= 26 {
            4
        } else {
            3
        };
        let gpu_bar = make_bar(stats.gpu_util_pct, gpu_bar_len);

        let vram_bar_len = if c1_w >= 32 {
            5
        } else if c1_w >= 26 {
            4
        } else {
            3
        };
        let vram_bar = make_bar(vram_pct as f32, vram_bar_len);

        let c1_l3 = if c1_w >= 36 {
            format!(
                "{label_color}GPU: {util_col}[{gpu_bar}] {:>3.0}% {border_color}│ {label_color}VRAM: {highlight}[{vram_bar}] {:>2.0}%",
                stats.gpu_util_pct, vram_pct
            )
        } else if c1_w >= 28 {
            format!(
                "{label_color}GPU: {util_col}[{gpu_bar}] {:>3.0}% {border_color}│ {label_color}VR: {val_color}{:>2.0}%",
                stats.gpu_util_pct, vram_pct
            )
        } else if c1_w >= 20 {
            format!(
                "{label_color}GPU: {util_col}{:>3.0}% {border_color}│ {label_color}VR: {val_color}{:>2.0}%",
                stats.gpu_util_pct, vram_pct
            )
        } else {
            format!("{label_color}GPU: {util_col}{:>3.0}%", stats.gpu_util_pct)
        };

        // ─────────────────────────────────────────────
        // Column 2: Frametime, Live Sparkline, Pacing, CPU
        // ─────────────────────────────────────────────
        let spark_w = if c2_w >= 32 {
            10
        } else if c2_w >= 24 {
            6
        } else {
            3
        };
        let spark = self.generate_sparkline(spark_w);

        let c2_l1 = if c2_w >= 26 {
            format!(
                "{label_color}Frame: {val_color}{:>5.1}ms {border_color}[{spark}{border_color}]",
                self.frametime_ms
            )
        } else {
            format!("{label_color}F: {val_color}{:>5.1}ms", self.frametime_ms)
        };

        let c2_l2 = if c2_w >= 30 {
            format!(
                "{label_color}Avg: {val_color}{:>4.1}ms {border_color}│ {label_color}1%: {highlight}{:>4.1}ms {border_color}│ {val_color}{:>3.0} FPS",
                self.avg_frametime_ms, self.p1_low_ms, self.fps
            )
        } else if c2_w >= 20 {
            format!(
                "{label_color}1%: {highlight}{:>4.1}ms {border_color}│ {val_color}{:>4.0} FPS",
                self.p1_low_ms, self.fps
            )
        } else {
            format!("{val_color}{:>4.0} FPS", self.fps)
        };

        let cpu_bar_w = if c2_w >= 28 {
            8
        } else if c2_w >= 20 {
            5
        } else {
            3
        };
        let cpu_bar = make_bar(stats.cpu_usage_pct, cpu_bar_w);
        let c2_l3 = if c2_w >= 26 {
            format!(
                "{label_color}CPU: {val_color}[{cpu_bar}] {:>4.1}% {label_color}({}T)",
                stats.cpu_usage_pct, stats.cpu_threads
            )
        } else if c2_w >= 18 {
            format!(
                "{label_color}CPU: {val_color}[{cpu_bar}] {:>3.0}%",
                stats.cpu_usage_pct
            )
        } else {
            format!("{label_color}CPU: {val_color}{:>3.0}%", stats.cpu_usage_pct)
        };

        // ─────────────────────────────────────────────
        // Column 3: Progress & Score Ticker / Demo Status
        // ─────────────────────────────────────────────
        let (c3_l1, c3_l2, c3_l3) = if self.is_benchmark {
            let bar_w = if c3_w >= 24 {
                10
            } else if c3_w >= 16 {
                6
            } else {
                3
            };
            let p_bar = make_bar(self.act_progress_pct, bar_w);
            let l1 = if c3_w >= 18 {
                format!(
                    "{label_color}Act: {highlight}[{p_bar}] {val_color}{:>3.0}%",
                    self.act_progress_pct
                )
            } else {
                format!("{highlight}[{p_bar}]")
            };

            let l2 = if c3_w >= 26 {
                let est_tflops = self.rays_per_sec * 2.25;
                format!(
                    "{label_color}Rays: {val_color}{:>4.1}G {border_color}│ {val_color}{:>4.1}TF",
                    self.rays_per_sec, est_tflops
                )
            } else if c3_w >= 18 {
                format!(
                    "{label_color}Rays: {val_color}{:>5.2} Giga/s",
                    self.rays_per_sec
                )
            } else if c3_w >= 12 {
                format!("{val_color}{:>5.2} G/s", self.rays_per_sec)
            } else {
                format!("{val_color}{:>4.1}G", self.rays_per_sec)
            };

            let l3 = if c3_w >= 18 {
                format!(
                    "{label_color}Score: {highlight}{:>7.2} pts",
                    self.current_score
                )
            } else {
                format!("{highlight}{:>6.1}", self.current_score)
            };
            (l1, l2, l3)
        } else {
            let mins = (self.demo_elapsed_secs / 60.0) as u32;
            let secs = (self.demo_elapsed_secs % 60.0) as u32;
            let l1 = if c3_w >= 20 {
                format!(
                    "{label_color}Time: {val_color}{:02}:{:02} {label_color}({:04}f)",
                    mins,
                    secs,
                    self.demo_frame_count % 10000
                )
            } else {
                format!("{label_color}T: {val_color}{:02}:{:02}", mins, secs)
            };

            let l2 = if c3_w >= 26 {
                let est_tflops = self.rays_per_sec * 2.25;
                format!(
                    "{label_color}Rays: {val_color}{:>4.1}G {border_color}│ {val_color}{:>4.1}TF",
                    self.rays_per_sec, est_tflops
                )
            } else if c3_w >= 18 {
                format!(
                    "{label_color}Rays: {val_color}{:>5.2} Giga/s",
                    self.rays_per_sec
                )
            } else {
                format!("{val_color}{:>5.2} G/s", self.rays_per_sec)
            };

            let l3 = if self.is_uncapped {
                format!("{label_color}Mode: \x1b[38;2;255;60;60;1m[🔥UNCAPPED]")
            } else {
                format!("{label_color}Mode: {val_color}[60FPS V-SYNC]")
            };
            (l1, l2, l3)
        };

        // Assemble columns with pixel-exact visible width padding
        let p_c1_1 = fit_to_width(&c1_l1, c1_w);
        let p_c2_1 = fit_to_width(&c2_l1, c2_w);
        let p_c3_1 = fit_to_width(&c3_l1, c3_w);

        let p_c1_2 = fit_to_width(&c1_l2, c1_w);
        let p_c2_2 = fit_to_width(&c2_l2, c2_w);
        let p_c3_2 = fit_to_width(&c3_l2, c3_w);

        let p_c1_3 = fit_to_width(&c1_l3, c1_w);
        let p_c2_3 = fit_to_width(&c2_l3, c2_w);
        let p_c3_3 = fit_to_width(&c3_l3, c3_w);

        let line1 = format!(
            "{border_color}│ {p_c1_1} {border_color}│ {p_c2_1} {border_color}│ {p_c3_1} {border_color}│{reset}"
        );
        let line2 = format!(
            "{border_color}│ {p_c1_2} {border_color}│ {p_c2_2} {border_color}│ {p_c3_2} {border_color}│{reset}"
        );
        let line3 = format!(
            "{border_color}│ {p_c1_3} {border_color}│ {p_c2_3} {border_color}│ {p_c3_3} {border_color}│{reset}"
        );

        let footer_fill = "─".repeat(w.saturating_sub(2));
        let footer = format!("{border_color}└{footer_fill}┘{reset}");

        vec![separator, line1, line2, line3, footer]
    }
}

pub fn get_scene_math_model(scene_id: u32) -> &'static str {
    match scene_id % 8 {
        0 => "Clausius-Clapeyron Thermodynamics & Gerstner Wave Spectrum",
        1 => "Power-8 3D Fractal Distance Estimator: z ↦ z⁸ + c",
        2 => "Kerr Spacetime Geodesic Deflection & Doppler Beaming",
        3 => "Navier-Stokes Burgers-Rott Vortex Dynamics",
        4 => "Aliev-Panfilov Reaction-Diffusion Electrophysiology",
        5 => "Lattice Boltzmann D2Q9 & von Kármán Vortex Shedding",
        6 => "Hierarchical Forward Kinematics & smin Muscle Skinning",
        7 => "Multithreaded 32K Celestial Particle Gravitation",
        _ => "Computational Benchmark",
    }
}

fn make_bar(pct: f32, width: usize) -> String {
    let filled = (((pct.clamp(0.0, 100.0) / 100.0) * width as f32).round() as usize).min(width);
    let empty = width.saturating_sub(filled);
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

fn fit_to_width(s: &str, target_w: usize) -> String {
    let vlen = crate::terminal::menu::visible_len(s);
    if vlen <= target_w {
        let pad = target_w - vlen;
        format!("{}{}\x1b[0m", s, " ".repeat(pad))
    } else {
        // Truncate visible chars to target_w
        let mut out = String::new();
        let mut count = 0;
        let mut in_escape = false;
        for c in s.chars() {
            if c == '\x1b' {
                in_escape = true;
                out.push(c);
            } else if in_escape {
                out.push(c);
                if c.is_ascii_alphabetic() {
                    in_escape = false;
                }
            } else if count < target_w {
                out.push(c);
                count += 1;
            }
        }
        out.push_str("\x1b[0m");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip_ansi(s: &str) -> String {
        let mut out = String::new();
        let mut in_escape = false;
        for c in s.chars() {
            if c == '\x1b' {
                in_escape = true;
            } else if in_escape {
                if c.is_ascii_alphabetic() {
                    in_escape = false;
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn test_hud_exact_widths() {
        let stats = HardwareStats {
            gpu_name: "AMD Radeon RX 6650 XT (RADV NAVI23)".to_string(),
            gpu_clock_mhz: 2650,
            gpu_mem_clock_mhz: 2248,
            gpu_voltage_mv: 1137,
            gpu_util_pct: 98.0,
            mem_util_pct: 80.0,
            gpu_temp_c: 62.0,
            gpu_hotspot_c: 94.0,
            gpu_mem_temp_c: 78.0,
            gpu_temp_crit_c: 100.0,
            gpu_hotspot_crit_c: 110.0,
            gpu_fan_rpm: Some(1450),
            gpu_fan_max_rpm: Some(3200),
            gpu_fan_pwm: Some(115),
            gpu_power_w: 148.0,
            gpu_power_cap_w: 157.0,
            vram_used_mb: 5500,
            vram_total_mb: 8176,
            vram_vendor: "Samsung GDDR6".to_string(),
            pcie_link: "16.0 GT/s, x8".to_string(),
            cpu_name: "12th Gen Intel Core i5-12600K".to_string(),
            cpu_usage_pct: 45.2,
            cpu_threads: 16,
            peak_gpu_clock_mhz: 2668,
            peak_gpu_util_pct: 99.0,
            ..Default::default()
        };

        for width in 50..=200 {
            let mut hud = Hud::new(width, "ATMOSPHERIC OCEAN WAVES (GPU)".to_string());
            hud.is_benchmark = true;
            hud.act_index = 3;
            hud.act_total = 6;
            hud.act_progress_pct = 42.5;
            hud.overall_progress_pct = 47.0;
            hud.fps = 94.2;
            hud.frametime_ms = 10.62;
            hud.avg_frametime_ms = 10.58;
            hud.p1_low_ms = 11.12;
            hud.rays_per_sec = 0.79;
            hud.current_score = 781.32;
            for ft in [10.2, 10.5, 10.6, 11.1, 10.4, 10.5, 12.0, 10.6] {
                hud.record_frametime(ft);
            }

            let top_lines = hud.render_top();
            for (idx, line) in top_lines.iter().enumerate() {
                let clean = strip_ansi(line);
                assert_eq!(
                    clean.chars().count(),
                    width,
                    "Top HUD line {} length mismatch at width {}: '{}' (len {})",
                    idx,
                    width,
                    clean,
                    clean.chars().count()
                );
            }

            let bottom_lines = hud.render_bottom(&stats);
            for (idx, line) in bottom_lines.iter().enumerate() {
                let clean = strip_ansi(line);
                assert_eq!(
                    clean.chars().count(),
                    width,
                    "Bottom HUD line {} length mismatch at width {}: '{}' (len {})",
                    idx,
                    width,
                    clean,
                    clean.chars().count()
                );
            }

            // Also test in Demo Mode
            hud.is_benchmark = false;
            hud.is_uncapped = true;
            hud.demo_elapsed_secs = 85.0;
            hud.demo_frame_count = 5420;

            let demo_bottom = hud.render_bottom(&stats);
            for (idx, line) in demo_bottom.iter().enumerate() {
                let clean = strip_ansi(line);
                assert_eq!(
                    clean.chars().count(),
                    width,
                    "Demo Bottom HUD line {} length mismatch at width {}: '{}' (len {})",
                    idx,
                    width,
                    clean,
                    clean.chars().count()
                );
            }
        }
    }

    #[test]
    fn test_get_scene_math_models() {
        for scene_id in 0..=7 {
            let desc = get_scene_math_model(scene_id);
            assert!(
                !desc.is_empty(),
                "Scene {} math model should not be empty",
                scene_id
            );
        }
    }
}
