use std::io::{self, Write};

pub struct TerminalRenderer {
    stdout: io::Stdout,
    output_buffer: String,
    last_fg: (u8, u8, u8),
    last_bg: (u8, u8, u8),
    has_last_colors: bool,
}

impl TerminalRenderer {
    pub fn new() -> Self {
        Self {
            stdout: io::stdout(),
            output_buffer: String::with_capacity(256 * 1024),
            last_fg: (0, 0, 0),
            last_bg: (0, 0, 0),
            has_last_colors: false,
        }
    }

    pub fn enter_alternate_screen(&mut self) -> io::Result<()> {
        write!(self.stdout, "\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H")?;
        self.stdout.flush()
    }

    pub fn leave_alternate_screen(&mut self) -> io::Result<()> {
        write!(self.stdout, "\x1b[0m\x1b[?25h\x1b[?1049l")?;
        self.stdout.flush()
    }

    /// Renders an RGBA8 buffer to terminal using UTF-8 half-blocks (`▀`).
    pub fn render(
        &mut self,
        rgba: &[u8],
        width: usize,
        height: usize,
        hud_lines_top: &[String],
        hud_lines_bottom: &[String],
    ) -> io::Result<()> {
        self.output_buffer.clear();
        self.has_last_colors = false;

        // Move cursor to top-left (home)
        self.output_buffer.push_str("\x1b[H");

        // 1. Render Top HUD Lines
        for line in hud_lines_top {
            self.output_buffer.push_str("\x1b[0m");
            self.output_buffer.push_str(line);
            self.output_buffer.push_str("\x1b[K\r\n");
        }

        // 2. Render Visual Canvas using Half-Blocks
        let char_rows = height / 2;
        let stride = width * 4;

        for cy in 0..char_rows {
            let y_top = cy * 2;
            let y_bot = y_top + 1;

            let row_top_offset = y_top * stride;
            let row_bot_offset = y_bot * stride;

            for x in 0..width {
                let px_top_idx = row_top_offset + (x * 4);
                let px_bot_idx = row_bot_offset + (x * 4);

                let r_top = rgba[px_top_idx];
                let g_top = rgba[px_top_idx + 1];
                let b_top = rgba[px_top_idx + 2];

                let r_bot = rgba[px_bot_idx];
                let g_bot = rgba[px_bot_idx + 1];
                let b_bot = rgba[px_bot_idx + 2];

                // Optimize escape sequences: only emit if colors changed
                if !self.has_last_colors
                    || self.last_fg != (r_top, g_top, b_top)
                    || self.last_bg != (r_bot, g_bot, b_bot)
                {
                    use std::fmt::Write;
                    let _ = write!(
                        self.output_buffer,
                        "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m",
                        r_top, g_top, b_top, r_bot, g_bot, b_bot
                    );
                    self.last_fg = (r_top, g_top, b_top);
                    self.last_bg = (r_bot, g_bot, b_bot);
                    self.has_last_colors = true;
                }

                // Upper half-block: foreground is top, background is bottom
                self.output_buffer.push('▀');
            }

            // End of canvas line
            self.output_buffer.push_str("\x1b[0m\x1b[K\r\n");
            self.has_last_colors = false;
        }

        // 3. Render Bottom HUD Lines
        let num_bottom = hud_lines_bottom.len();
        for (i, line) in hud_lines_bottom.iter().enumerate() {
            self.output_buffer.push_str("\x1b[0m");
            self.output_buffer.push_str(line);
            self.output_buffer.push_str("\x1b[K");
            if i + 1 < num_bottom {
                self.output_buffer.push_str("\r\n");
            }
        }

        // Write entire frame in a single atomic buffer flush
        self.stdout.write_all(self.output_buffer.as_bytes())?;
        self.stdout.flush()
    }
}

/// Downsamples an offscreen RGBA buffer to match terminal preview dimensions
#[allow(dead_code)]
pub fn downsample_rgba(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
) {
    if src_w == dst_w && src_h == dst_h {
        dst.copy_from_slice(src);
        return;
    }

    for dy in 0..dst_h {
        let sy = (dy * src_h) / dst_h;
        let src_row = sy * src_w * 4;
        let dst_row = dy * dst_w * 4;

        for dx in 0..dst_w {
            let sx = (dx * src_w) / dst_w;
            let src_idx = src_row + sx * 4;
            let dst_idx = dst_row + dx * 4;

            dst[dst_idx] = src[src_idx];
            dst[dst_idx + 1] = src[src_idx + 1];
            dst[dst_idx + 2] = src[src_idx + 2];
            dst[dst_idx + 3] = 255;
        }
    }
}
