use crate::telemetry::HardwareStats;

pub struct Hud {
    pub width: usize,
    pub scene_title: String,
    pub resolution_label: String,
    pub progress_pct: f32,
    pub fps: f32,
    pub frametime_ms: f32,
    pub avg_frametime_ms: f32,
    pub p1_low_ms: f32,
    pub rays_per_sec: f32,
    pub current_score: u64,
}

impl Hud {
    pub fn new(width: usize, scene_title: String) -> Self {
        Self {
            width,
            scene_title,
            resolution_label: "4K UHD (3840x2160)".to_string(),
            progress_pct: 0.0,
            fps: 0.0,
            frametime_ms: 0.0,
            avg_frametime_ms: 0.0,
            p1_low_ms: 0.0,
            rays_per_sec: 0.0,
            current_score: 0,
        }
    }

    pub fn render_top(&self) -> Vec<String> {
        let border_color = "\x1b[38;2;0;220;255m"; // Cyber cyan
        let text_color = "\x1b[38;2;255;255;255;1m";
        let res_color = "\x1b[38;2;255;180;0;1m"; // Amber gold
        let reset = "\x1b[0m";

        let w = self.width;
        // Total structural border overhead:
        // "┌── [" (5) + "] " (2) + " [" (2) + "] " (2) + " [" (2) + "] ──┐" (5) = 18 chars
        let overhead = 18;
        let budget = w.saturating_sub(overhead);

        let (title_left, title_mid, title_right) = if budget >= 68 {
            (
                " DYNO_TERM v1.0 ".to_string(),
                format!(" {} ", self.resolution_label),
                format!(" SCENE: {} ", self.scene_title),
            )
        } else if budget >= 42 {
            let res_short = self.resolution_label.split_whitespace().next().unwrap_or("4K");
            let scene_short = self
                .scene_title
                .replace(" (GPU)", "")
                .replace(" (CPU)", "")
                .replace("THE ", "");
            (
                " DYNO_TERM ".to_string(),
                format!(" {} ", res_short),
                format!(" {} ", scene_short),
            )
        } else {
            let res_short = self.resolution_label.split_whitespace().next().unwrap_or("4K");
            let scene_words: Vec<&str> = self.scene_title.split_whitespace().collect();
            let scene_word = scene_words.first().unwrap_or(&"BENCH");
            (
                " DYNO ".to_string(),
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

        let line = format!(
            "{border_color}┌── [{text_color}{title_left}{border_color}] {fill_left} [{res_color}{title_mid}{border_color}] {fill_right} [{text_color}{right}{border_color}] ──┐{reset}"
        );

        vec![line]
    }

    pub fn render_bottom(&self, stats: &HardwareStats) -> Vec<String> {
        let border_color = "\x1b[38;2;0;220;255m"; // Cyan border
        let label_color = "\x1b[38;2;140;150;170m"; // Muted slate
        let val_color = "\x1b[38;2;0;255;200;1m"; // Neon mint
        let highlight = "\x1b[38;2;255;180;0;1m"; // Amber
        let reset = "\x1b[0m";

        let w = self.width;

        // Line 0: Separator
        let sep_overhead = 14; // "├── [" (5) + "] " (2) + " [" (2) + "] ──┤" (5) = 14 chars
        let sep_budget = w.saturating_sub(sep_overhead);

        let (telemetry_title, progress_title) = if sep_budget >= 28 {
            (
                " TELEMETRY ".to_string(),
                format!(" PROGRESS: {:>5.1}% ", self.progress_pct),
            )
        } else {
            (
                " STATS ".to_string(),
                format!(" {:>4.0}% ", self.progress_pct),
            )
        };

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

        // Column 1
        let max_gpu = c1_w.saturating_sub(5);
        let gpu_name_part: String = stats.gpu_name.chars().take(max_gpu).collect();
        let c1_raw1 = format!("GPU: {}", gpu_name_part);
        let c1_l1 = format!(
            "{label_color}GPU: {val_color}{}{}",
            gpu_name_part,
            " ".repeat(c1_w.saturating_sub(c1_raw1.chars().count()))
        );

        let c1_raw2 = if c1_w >= 28 {
            format!(
                "Clk: {:>4}MHz | {:>2.0}°C | {:>3.0}W",
                stats.gpu_clock_mhz, stats.gpu_temp_c, stats.gpu_power_w
            )
        } else if c1_w >= 20 {
            format!(
                "{}MHz | {:.0}°C | {:.0}W",
                stats.gpu_clock_mhz, stats.gpu_temp_c, stats.gpu_power_w
            )
        } else {
            format!("{}M | {:.0}W", stats.gpu_clock_mhz, stats.gpu_power_w)
        };
        let c1_r2_trimmed: String = c1_raw2.chars().take(c1_w).collect();
        let c1_l2 = format!(
            "{val_color}{}{}",
            c1_r2_trimmed,
            " ".repeat(c1_w.saturating_sub(c1_r2_trimmed.chars().count()))
        );

        let vram_pct = if stats.vram_total_mb > 0 {
            (stats.vram_used_mb * 100) / stats.vram_total_mb
        } else {
            0
        };
        let c1_raw3 = if c1_w >= 26 {
            format!(
                "VRAM: {:>4}/{:>4}MB ({}%)",
                stats.vram_used_mb, stats.vram_total_mb, vram_pct
            )
        } else if c1_w >= 18 {
            format!(
                "VRAM: {}/{}M",
                stats.vram_used_mb, stats.vram_total_mb
            )
        } else {
            format!("VRAM: {}%", vram_pct)
        };
        let c1_r3_trimmed: String = c1_raw3.chars().take(c1_w).collect();
        let c1_l3 = format!(
            "{val_color}{}{}",
            c1_r3_trimmed,
            " ".repeat(c1_w.saturating_sub(c1_r3_trimmed.chars().count()))
        );

        // Column 2
        let c2_raw1 = if c2_w >= 30 {
            format!(
                "Frame: {:>5.2} ms (Avg {:>5.2} ms)",
                self.frametime_ms, self.avg_frametime_ms
            )
        } else if c2_w >= 20 {
            format!(
                "Frame: {:>5.1}ms ({:>5.1}ms)",
                self.frametime_ms, self.avg_frametime_ms
            )
        } else {
            format!("F: {:>5.1}ms", self.frametime_ms)
        };
        let c2_r1_trimmed: String = c2_raw1.chars().take(c2_w).collect();
        let c2_l1 = format!(
            "{val_color}{}{}",
            c2_r1_trimmed,
            " ".repeat(c2_w.saturating_sub(c2_r1_trimmed.chars().count()))
        );

        let c2_raw2 = if c2_w >= 28 {
            format!("1% Low: {:>5.2} ms | FPS: {:>5.1}", self.p1_low_ms, self.fps)
        } else if c2_w >= 20 {
            format!("1%: {:>4.1}ms | {:>4.0} FPS", self.p1_low_ms, self.fps)
        } else {
            format!("{:.0} FPS", self.fps)
        };
        let c2_r2_trimmed: String = c2_raw2.chars().take(c2_w).collect();
        let c2_l2 = format!(
            "{val_color}{}{}",
            c2_r2_trimmed,
            " ".repeat(c2_w.saturating_sub(c2_r2_trimmed.chars().count()))
        );

        let cpu_bar_w = if c2_w >= 28 { 8 } else if c2_w >= 20 { 5 } else { 3 };
        let cpu_filled = (((stats.cpu_usage_pct / 100.0) * cpu_bar_w as f32).round() as usize)
            .min(cpu_bar_w);
        let cpu_empty = cpu_bar_w.saturating_sub(cpu_filled);
        let cpu_bar = format!("{}{}", "█".repeat(cpu_filled), "░".repeat(cpu_empty));
        let c2_raw3 = if c2_w >= 26 {
            format!(
                "CPU: [{}] {:>4.1}% ({}T)",
                cpu_bar, stats.cpu_usage_pct, stats.cpu_threads
            )
        } else if c2_w >= 18 {
            format!("CPU: [{}] {:>3.0}%", cpu_bar, stats.cpu_usage_pct)
        } else {
            format!("CPU: {:>3.0}%", stats.cpu_usage_pct)
        };
        let c2_r3_trimmed: String = c2_raw3.chars().take(c2_w).collect();
        let c2_l3 = format!(
            "{val_color}{}{}",
            c2_r3_trimmed,
            " ".repeat(c2_w.saturating_sub(c2_r3_trimmed.chars().count()))
        );

        // Column 3
        let bar_w = if c3_w >= 24 { 12 } else if c3_w >= 16 { 8 } else { 4 };
        let filled = (((self.progress_pct / 100.0) * bar_w as f32).round() as usize).min(bar_w);
        let empty = bar_w.saturating_sub(filled);
        let p_bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
        let c3_raw1 = if c3_w >= 16 {
            format!("Prog: [{}]", p_bar)
        } else {
            format!("[{}]", p_bar)
        };
        let c3_r1_trimmed: String = c3_raw1.chars().take(c3_w).collect();
        let c3_l1 = format!(
            "{highlight}{}{}",
            c3_r1_trimmed,
            " ".repeat(c3_w.saturating_sub(c3_r1_trimmed.chars().count()))
        );

        let c3_raw2 = if c3_w >= 18 {
            format!("Rays: {:>5.2} Giga/s", self.rays_per_sec)
        } else if c3_w >= 12 {
            format!("{:>5.2} G/s", self.rays_per_sec)
        } else {
            format!("{:>4.1}G", self.rays_per_sec)
        };
        let c3_r2_trimmed: String = c3_raw2.chars().take(c3_w).collect();
        let c3_l2 = format!(
            "{val_color}{}{}",
            c3_r2_trimmed,
            " ".repeat(c3_w.saturating_sub(c3_r2_trimmed.chars().count()))
        );

        let c3_raw3 = if c3_w >= 18 {
            format!("Score: {:>7} pts", self.current_score)
        } else if c3_w >= 12 {
            format!("{:>7} pts", self.current_score)
        } else {
            format!("{:>6}", self.current_score)
        };
        let c3_r3_trimmed: String = c3_raw3.chars().take(c3_w).collect();
        let c3_l3 = format!(
            "{highlight}{}{}",
            c3_r3_trimmed,
            " ".repeat(c3_w.saturating_sub(c3_r3_trimmed.chars().count()))
        );

        // Lines assembly
        let line1 = format!(
            "{border_color}│ {c1_l1} {border_color}│ {c2_l1} {border_color}│ {c3_l1} {border_color}│{reset}"
        );
        let line2 = format!(
            "{border_color}│ {c1_l2} {border_color}│ {c2_l2} {border_color}│ {c3_l2} {border_color}│{reset}"
        );
        let line3 = format!(
            "{border_color}│ {c1_l3} {border_color}│ {c2_l3} {border_color}│ {c3_l3} {border_color}│{reset}"
        );

        let footer_fill = "─".repeat(w.saturating_sub(2));
        let footer = format!("{border_color}└{footer_fill}┘{reset}");

        vec![separator, line1, line2, line3, footer]
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
            gpu_temp_c: 55.0,
            gpu_power_w: 145.0,
            gpu_clock_mhz: 2650,
            peak_gpu_clock_mhz: 2668,
            vram_used_mb: 5500,
            vram_total_mb: 8176,
            cpu_name: "12th Gen Intel Core i5-12600K".to_string(),
            cpu_usage_pct: 45.2,
            cpu_threads: 16,
        };

        for width in 50..=200 {
            let mut hud = Hud::new(width, "THE MANDELBULB CORE (GPU)".to_string());
            hud.progress_pct = 42.5;
            hud.fps = 85.0;
            hud.frametime_ms = 11.76;
            hud.avg_frametime_ms = 12.00;
            hud.p1_low_ms = 12.50;
            hud.rays_per_sec = 0.75;
            hud.current_score = 150000;

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
        }
    }
}
