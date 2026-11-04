use crate::gpu::GpuAdapterInfo;
use crate::telemetry::TelemetryCollector;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use std::io::{self, Write};
use std::time::{Duration, Instant};

pub const SCENE_NAMES: [&str; 8] = [
    "Act I: Atmospheric Ocean Waves (GPU)",
    "Act II: The Mandelbulb Core (GPU)",
    "Act III: Relativistic Black Hole (GPU)",
    "Act IV: Tornadic Mesocyclone (GPU - Weather Model)",
    "Act V: Cardiac Electrophysiology (GPU - Medical Model)",
    "Act VI: Lattice Boltzmann Flow (GPU - CFD Vortex Street)",
    "Act VII: Cybernetic Gait Kinematics (GPU - 3D Animation)",
    "Act VIII: Kinetic N-Body Supernova (CPU)",
];

pub const RESOLUTION_PRESETS: [(&str, &str, u32, u32); 7] = [
    ("16K Extreme", "15360 x 8640  (132.7 Mpix) [Extreme GPU Stress]", 15360, 8640),
    ("8K Ultra",    " 7680 x 4320  (33.2 Mpix)  [Heavy Workload]", 7680, 4320),
    ("4K UHD",      " 3840 x 2160  (8.29 Mpix)  [Default / Balanced]", 3840, 2160),
    ("1440p QHD",   " 2560 x 1440  (3.69 Mpix)  [High Performance]", 2560, 1440),
    ("1080p FHD",   " 1920 x 1080  (2.07 Mpix)  [Standard HD]", 1920, 1080),
    ("720p HD",     " 1280 x 720   (0.92 Mpix)  [Fast / Low Power]", 1280, 720),
    ("Native Grid", "Viewport      (Matches terminal character grid 1:1)", 0, 0),
];

pub const FRAME_PRESETS: [(&str, &str, u32); 4] = [
    ("Quick Check",    "  30 frames / act   (240 frames total,  ~3.5 sec)", 30),
    ("Standard Bench", " 100 frames / act   (800 frames total,  ~12 sec)", 100),
    ("Stress Test",    " 300 frames / act   (2400 frames total, ~35 sec)", 300),
    ("Endurance Soak", "1000 frames / act   (8000 frames total, ~120 sec)", 1000),
];

#[derive(Debug, Clone)]
pub struct MenuConfig {
    pub resolution_name: String,
    pub compute_width: u32,
    pub compute_height: u32,
    pub frames_per_scene: u32,
    pub selected_scene: u32,
    pub uncapped: bool,
    pub selected_gpu_index: usize,
    pub output_path: String,
    pub consistency_iterations: usize,
    pub cooldown_seconds: u64,
    pub max_temp_c: f32,
    pub target_temp_c: f32,
    pub failsafe_enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MenuAction {
    LaunchDemo,
    LaunchUncapped,
    RunBenchmark,
    RunConsistency,
    RecreatePipeline,
    SwitchGpu(usize),
    Exit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MenuView {
    Main,
    SelectResolution,
    SelectScene,
    SelectFrames,
    SelectGpu,
    TelemetryInfo,
}

pub struct CliMenu {
    view: MenuView,
    selected_idx: usize,
    buf: String,
}

impl CliMenu {
    pub fn new() -> Self {
        Self {
            view: MenuView::Main,
            selected_idx: 0,
            buf: String::with_capacity(16 * 1024),
        }
    }

    pub fn run(
        &mut self,
        config: &mut MenuConfig,
        telemetry: &TelemetryCollector,
        adapter_info: &GpuAdapterInfo,
        all_adapters: &[GpuAdapterInfo],
    ) -> io::Result<MenuAction> {
        let mut stdout = io::stdout();
        write!(stdout, "\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H")?;
        stdout.flush()?;
        crossterm::terminal::enable_raw_mode()?;

        let mut last_draw = Instant::now() - Duration::from_secs(1);

        let action = loop {
            let (term_cols, term_rows) = crossterm::terminal::size().unwrap_or((80, 25));

            // In telemetry view, refresh every 150ms; in other views, refresh every 500ms for clock/temp
            let refresh_dur = if self.view == MenuView::TelemetryInfo {
                Duration::from_millis(150)
            } else {
                Duration::from_millis(500)
            };

            if last_draw.elapsed() >= refresh_dur {
                self.render_current_view(config, telemetry, adapter_info, all_adapters, term_cols, term_rows)?;
                last_draw = Instant::now();
            }

            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }

                    // Global quit on Ctrl+C
                    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                        break MenuAction::Exit;
                    }

                    match self.view {
                        MenuView::Main => {
                            if let Some(act) = self.handle_main_input(key.code, config, all_adapters.len())? {
                                break act;
                            }
                        }
                        MenuView::SelectResolution => {
                            if let Some(act) = self.handle_resolution_input(key.code, config, term_cols as u32, term_rows as u32)? {
                                break act;
                            }
                        }
                        MenuView::SelectScene => {
                            self.handle_scene_input(key.code, config);
                        }
                        MenuView::SelectFrames => {
                            if let Some(act) = self.handle_frames_input(key.code, config)? {
                                break act;
                            }
                        }
                        MenuView::SelectGpu => {
                            if let Some(act) = self.handle_gpu_input(key.code, all_adapters.len()) {
                                break act;
                            }
                        }
                        MenuView::TelemetryInfo => {
                            match key.code {
                                KeyCode::Enter
                                | KeyCode::Esc
                                | KeyCode::Backspace
                                | KeyCode::Char('q')
                                | KeyCode::Char('Q')
                                | KeyCode::Char(' ') => {
                                    self.view = MenuView::Main;
                                    self.selected_idx = 9;
                                }
                                _ => {}
                            }
                        }
                    }
                    self.render_current_view(config, telemetry, adapter_info, all_adapters, term_cols, term_rows)?;
                    last_draw = Instant::now();
                }
            }
        };

        crossterm::terminal::disable_raw_mode()?;
        write!(stdout, "\x1b[0m\x1b[?25h\x1b[?1049l")?;
        stdout.flush()?;

        Ok(action)
    }

    fn handle_main_input(
        &mut self,
        code: KeyCode,
        config: &mut MenuConfig,
        _adapter_count: usize,
    ) -> io::Result<Option<MenuAction>> {
        const MAIN_ITEMS_COUNT: usize = 11;

        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected_idx = (self.selected_idx + MAIN_ITEMS_COUNT - 1) % MAIN_ITEMS_COUNT;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.selected_idx = (self.selected_idx + 1) % MAIN_ITEMS_COUNT;
            }
            KeyCode::Home => {
                self.selected_idx = 0;
            }
            KeyCode::End => {
                self.selected_idx = MAIN_ITEMS_COUNT - 1;
            }
            KeyCode::Char('1') => return Ok(Some(MenuAction::LaunchDemo)),
            KeyCode::Char('2') => return Ok(Some(MenuAction::LaunchUncapped)),
            KeyCode::Char('3') => return Ok(Some(MenuAction::RunBenchmark)),
            KeyCode::Char('4') => return Ok(Some(MenuAction::RunConsistency)),
            KeyCode::Char('5') => {
                self.view = MenuView::SelectScene;
                self.selected_idx = (config.selected_scene as usize).min(SCENE_NAMES.len() - 1);
            }
            KeyCode::Char('6') => {
                self.view = MenuView::SelectResolution;
                self.selected_idx = 2; // Default 4K
            }
            KeyCode::Char('7') => {
                self.view = MenuView::SelectFrames;
                self.selected_idx = 1; // Default Standard 100
            }
            KeyCode::Char('8') | KeyCode::Char('u') | KeyCode::Char('U') => {
                config.uncapped = !config.uncapped;
            }
            KeyCode::Char('9') => {
                self.view = MenuView::SelectGpu;
                self.selected_idx = config.selected_gpu_index;
            }
            KeyCode::Char('t') | KeyCode::Char('T') => {
                self.view = MenuView::TelemetryInfo;
            }
            KeyCode::Char('0') | KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
                return Ok(Some(MenuAction::Exit));
            }
            KeyCode::Enter | KeyCode::Char(' ') => match self.selected_idx {
                0 => return Ok(Some(MenuAction::LaunchDemo)),
                1 => return Ok(Some(MenuAction::LaunchUncapped)),
                2 => return Ok(Some(MenuAction::RunBenchmark)),
                3 => return Ok(Some(MenuAction::RunConsistency)),
                4 => {
                    self.view = MenuView::SelectScene;
                    self.selected_idx = (config.selected_scene as usize).min(SCENE_NAMES.len() - 1);
                }
                5 => {
                    self.view = MenuView::SelectResolution;
                    self.selected_idx = 2;
                }
                6 => {
                    self.view = MenuView::SelectFrames;
                    self.selected_idx = 1;
                }
                7 => {
                    config.uncapped = !config.uncapped;
                }
                8 => {
                    self.view = MenuView::SelectGpu;
                    self.selected_idx = config.selected_gpu_index;
                }
                9 => {
                    self.view = MenuView::TelemetryInfo;
                }
                10 => return Ok(Some(MenuAction::Exit)),
                _ => {}
            },
            _ => {}
        }
        Ok(None)
    }

    fn handle_resolution_input(
        &mut self,
        code: KeyCode,
        config: &mut MenuConfig,
        term_cols: u32,
        term_rows: u32,
    ) -> io::Result<Option<MenuAction>> {
        const TOTAL_RES_OPTIONS: usize = 9; // 0..=6 presets, 7 custom, 8 back

        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected_idx = (self.selected_idx + TOTAL_RES_OPTIONS - 1) % TOTAL_RES_OPTIONS;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.selected_idx = (self.selected_idx + 1) % TOTAL_RES_OPTIONS;
            }
            KeyCode::Home => self.selected_idx = 0,
            KeyCode::End => self.selected_idx = TOTAL_RES_OPTIONS - 1,
            KeyCode::Char('1') => return Ok(self.apply_res_preset(0, config)),
            KeyCode::Char('2') => return Ok(self.apply_res_preset(1, config)),
            KeyCode::Char('3') => return Ok(self.apply_res_preset(2, config)),
            KeyCode::Char('4') => return Ok(self.apply_res_preset(3, config)),
            KeyCode::Char('5') => return Ok(self.apply_res_preset(4, config)),
            KeyCode::Char('6') => return Ok(self.apply_res_preset(5, config)),
            KeyCode::Char('7') => return Ok(self.apply_native_res(config, term_cols, term_rows)),
            KeyCode::Char('8') => return self.prompt_custom_res(config),
            KeyCode::Char('0') | KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc | KeyCode::Backspace => {
                self.view = MenuView::Main;
                self.selected_idx = 5;
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                if self.selected_idx < 6 {
                    return Ok(self.apply_res_preset(self.selected_idx, config));
                } else if self.selected_idx == 6 {
                    return Ok(self.apply_native_res(config, term_cols, term_rows));
                } else if self.selected_idx == 7 {
                    return self.prompt_custom_res(config);
                } else {
                    self.view = MenuView::Main;
                    self.selected_idx = 5;
                }
            }
            _ => {}
        }
        Ok(None)
    }

    fn apply_res_preset(&mut self, idx: usize, config: &mut MenuConfig) -> Option<MenuAction> {
        let preset = &RESOLUTION_PRESETS[idx];
        config.compute_width = preset.2;
        config.compute_height = preset.3;
        config.resolution_name = format_resolution_tag(config.compute_width, config.compute_height);
        self.view = MenuView::Main;
        self.selected_idx = 5;
        Some(MenuAction::RecreatePipeline)
    }

    fn apply_native_res(&mut self, config: &mut MenuConfig, cols: u32, rows: u32) -> Option<MenuAction> {
        let term_w = cols.max(40);
        let char_h = rows.saturating_sub(7).max(8);
        config.compute_width = term_w;
        config.compute_height = char_h * 2;
        config.resolution_name = format_resolution_tag(config.compute_width, config.compute_height);
        self.view = MenuView::Main;
        self.selected_idx = 5;
        Some(MenuAction::RecreatePipeline)
    }

    fn prompt_custom_res(&mut self, config: &mut MenuConfig) -> io::Result<Option<MenuAction>> {
        let mut stdout = io::stdout();
        crossterm::terminal::disable_raw_mode()?;
        write!(stdout, "\x1b[0m\x1b[?25h\x1b[?1049l")?;
        stdout.flush()?;

        println!("\r\n\x1b[38;2;0;220;255;1m─── CUSTOM COMPUTE RESOLUTION ───\x1b[0m");
        print!("\x1b[38;2;255;255;255;1mEnter compute width in pixels (e.g. 2560): \x1b[38;2;0;255;200m");
        stdout.flush()?;
        let mut w_str = String::new();
        let _ = io::stdin().read_line(&mut w_str);

        print!("\x1b[38;2;255;255;255;1mEnter compute height in pixels (e.g. 1440): \x1b[38;2;0;255;200m");
        stdout.flush()?;
        let mut h_str = String::new();
        let _ = io::stdin().read_line(&mut h_str);
        print!("\x1b[0m");
        stdout.flush()?;

        let mut changed = false;
        if let (Ok(w), Ok(h)) = (w_str.trim().parse::<u32>(), h_str.trim().parse::<u32>()) {
            if (64..=16384).contains(&w) && (64..=16384).contains(&h) {
                config.compute_width = w;
                config.compute_height = h;
                config.resolution_name = format_resolution_tag(w, h);
                changed = true;
            }
        }

        write!(stdout, "\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H")?;
        stdout.flush()?;
        crossterm::terminal::enable_raw_mode()?;

        self.view = MenuView::Main;
        self.selected_idx = 5;

        if changed {
            Ok(Some(MenuAction::RecreatePipeline))
        } else {
            Ok(None)
        }
    }

    fn handle_scene_input(&mut self, code: KeyCode, config: &mut MenuConfig) {
        let total_scene_items = SCENE_NAMES.len() + 1;
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected_idx = (self.selected_idx + total_scene_items - 1) % total_scene_items;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.selected_idx = (self.selected_idx + 1) % total_scene_items;
            }
            KeyCode::Home => self.selected_idx = 0,
            KeyCode::End => self.selected_idx = total_scene_items - 1,
            KeyCode::Char('0') | KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc | KeyCode::Backspace => {
                self.view = MenuView::Main;
                self.selected_idx = 4;
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                if self.selected_idx < SCENE_NAMES.len() {
                    config.selected_scene = self.selected_idx as u32;
                }
                self.view = MenuView::Main;
                self.selected_idx = 4;
            }
            _ => {
                if let KeyCode::Char(c) = code {
                    if let Some(digit) = c.to_digit(10) {
                        if digit >= 1 && (digit as usize) <= SCENE_NAMES.len() {
                            config.selected_scene = (digit - 1) as u32;
                            self.view = MenuView::Main;
                            self.selected_idx = 4;
                        }
                    }
                }
            }
        }
    }

    fn handle_frames_input(&mut self, code: KeyCode, config: &mut MenuConfig) -> io::Result<Option<MenuAction>> {
        const TOTAL_FRAME_ITEMS: usize = 6; // 4 presets + custom + back
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected_idx = (self.selected_idx + TOTAL_FRAME_ITEMS - 1) % TOTAL_FRAME_ITEMS;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.selected_idx = (self.selected_idx + 1) % TOTAL_FRAME_ITEMS;
            }
            KeyCode::Home => self.selected_idx = 0,
            KeyCode::End => self.selected_idx = TOTAL_FRAME_ITEMS - 1,
            KeyCode::Char('1') => {
                config.frames_per_scene = FRAME_PRESETS[0].2;
                self.view = MenuView::Main;
                self.selected_idx = 6;
            }
            KeyCode::Char('2') => {
                config.frames_per_scene = FRAME_PRESETS[1].2;
                self.view = MenuView::Main;
                self.selected_idx = 6;
            }
            KeyCode::Char('3') => {
                config.frames_per_scene = FRAME_PRESETS[2].2;
                self.view = MenuView::Main;
                self.selected_idx = 6;
            }
            KeyCode::Char('4') => {
                config.frames_per_scene = FRAME_PRESETS[3].2;
                self.view = MenuView::Main;
                self.selected_idx = 6;
            }
            KeyCode::Char('5') => return self.prompt_custom_frames(config),
            KeyCode::Char('0') | KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc | KeyCode::Backspace => {
                self.view = MenuView::Main;
                self.selected_idx = 6;
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                if self.selected_idx < 4 {
                    config.frames_per_scene = FRAME_PRESETS[self.selected_idx].2;
                    self.view = MenuView::Main;
                    self.selected_idx = 6;
                } else if self.selected_idx == 4 {
                    return self.prompt_custom_frames(config);
                } else {
                    self.view = MenuView::Main;
                    self.selected_idx = 6;
                }
            }
            _ => {}
        }
        Ok(None)
    }

    fn prompt_custom_frames(&mut self, config: &mut MenuConfig) -> io::Result<Option<MenuAction>> {
        let mut stdout = io::stdout();
        crossterm::terminal::disable_raw_mode()?;
        write!(stdout, "\x1b[0m\x1b[?25h\x1b[?1049l")?;
        stdout.flush()?;

        println!("\r\n\x1b[38;2;0;220;255;1m─── CUSTOM BENCHMARK FRAMES ───\x1b[0m");
        print!("\x1b[38;2;255;255;255;1mEnter frames per scene (1 - 50,000): \x1b[38;2;0;255;200m");
        stdout.flush()?;
        let mut f_str = String::new();
        let _ = io::stdin().read_line(&mut f_str);
        print!("\x1b[0m");
        stdout.flush()?;

        if let Ok(frames) = f_str.trim().parse::<u32>() {
            if (1..=50_000).contains(&frames) {
                config.frames_per_scene = frames;
            }
        }

        write!(stdout, "\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H")?;
        stdout.flush()?;
        crossterm::terminal::enable_raw_mode()?;

        self.view = MenuView::Main;
        self.selected_idx = 6;
        Ok(None)
    }

    fn handle_gpu_input(&mut self, code: KeyCode, adapter_count: usize) -> Option<MenuAction> {
        let total_items = adapter_count + 1;
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected_idx = (self.selected_idx + total_items - 1) % total_items;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.selected_idx = (self.selected_idx + 1) % total_items;
            }
            KeyCode::Home => self.selected_idx = 0,
            KeyCode::End => self.selected_idx = total_items - 1,
            KeyCode::Char('0') | KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc | KeyCode::Backspace => {
                self.view = MenuView::Main;
                self.selected_idx = 8;
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                if self.selected_idx < adapter_count {
                    let idx = self.selected_idx;
                    self.view = MenuView::Main;
                    self.selected_idx = 8;
                    return Some(MenuAction::SwitchGpu(idx));
                } else {
                    self.view = MenuView::Main;
                    self.selected_idx = 8;
                }
            }
            _ => {
                if let KeyCode::Char(c) = code {
                    if let Some(digit) = c.to_digit(10) {
                        let idx = digit as usize;
                        if idx < adapter_count {
                            self.view = MenuView::Main;
                            self.selected_idx = 8;
                            return Some(MenuAction::SwitchGpu(idx));
                        }
                    }
                }
            }
        }
        None
    }

    fn render_current_view(
        &mut self,
        config: &MenuConfig,
        telemetry: &TelemetryCollector,
        adapter_info: &GpuAdapterInfo,
        all_adapters: &[GpuAdapterInfo],
        cols: u16,
        rows: u16,
    ) -> io::Result<()> {
        self.buf.clear();
        self.buf.push_str("\x1b[H");

        let left_pad_len = (cols.saturating_sub(76) / 2) as usize;
        let left_pad = " ".repeat(left_pad_len);

        let top_pad_count = (rows.saturating_sub(24) / 2).min(4);
        for _ in 0..top_pad_count {
            self.buf.push_str("\r\n");
        }

        match self.view {
            MenuView::Main => {
                self.render_main_menu(&left_pad, config, telemetry, adapter_info);
            }
            MenuView::SelectResolution => {
                self.render_resolution_menu(&left_pad, config);
            }
            MenuView::SelectScene => {
                self.render_scene_menu(&left_pad, config);
            }
            MenuView::SelectFrames => {
                self.render_frames_menu(&left_pad, config);
            }
            MenuView::SelectGpu => {
                self.render_gpu_menu(&left_pad, config, all_adapters);
            }
            MenuView::TelemetryInfo => {
                self.render_telemetry_view(&left_pad, config, telemetry, adapter_info);
            }
        }

        let mut stdout = io::stdout();
        stdout.write_all(self.buf.as_bytes())?;
        stdout.flush()
    }

    fn render_main_menu(
        &mut self,
        pad: &str,
        config: &MenuConfig,
        telemetry: &TelemetryCollector,
        adapter_info: &GpuAdapterInfo,
    ) {
        let stats = telemetry.sample(&adapter_info.name);

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255;1m┌──────────────────────────────────────────────────────────────────────────┐\x1b[0m\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;0;255;200;1m◆ \x1b[38;2;255;255;255;1mPRETTY DYNO TERM\x1b[0m \x1b[38;2;140;150;170mv1.0\x1b[0m \x1b[38;2;0;220;255m│ \x1b[38;2;255;180;0;1mGPU COMPUTE & DEMOPARTY BENCHMARK\x1b[38;2;0;255;200;1m ◆\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;140;150;170m8 Biophysical & Math Acts  \x1b[38;2;0;220;255m•\x1b[38;2;140;150;170m  WebGPU Shaders  \x1b[38;2;0;220;255m•\x1b[38;2;140;150;170m  Calibrated 1K Pt Metric\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_section_header("SYSTEM HARDWARE & SENSOR TELEMETRY"));
        self.buf.push_str("\r\n");

        // Live GPU Card line
        let gpu_temp_str = if stats.gpu_temp_c > 0.0 {
            format!("{:.1}°C", stats.gpu_temp_c)
        } else {
            "Sensors OK".to_string()
        };
        let gpu_pwr_str = if stats.gpu_power_w > 0.0 {
            format!("{:.1}W", stats.gpu_power_w)
        } else {
            "Active".to_string()
        };
        let gpu_vram_str = if stats.vram_used_mb > 0 {
            format!("{} MB VRAM", stats.vram_used_mb)
        } else {
            "WebGPU".to_string()
        };
        let gpu_line = format!(
            "\x1b[38;2;0;220;255;1mGPU\x1b[0m  \x1b[38;2;255;255;255;1m{}\x1b[0m \x1b[38;2;140;150;170m({:?})\x1b[0m  \x1b[38;2;0;220;255m│\x1b[0m \x1b[38;2;0;255;200;1m{}\x1b[0m \x1b[38;2;140;150;170m•\x1b[0m \x1b[38;2;255;180;0;1m{}\x1b[0m \x1b[38;2;140;150;170m•\x1b[0m \x1b[38;2;255;255;255m{}\x1b[0m",
            trim_to(&adapter_info.name, 20),
            adapter_info.backend,
            gpu_temp_str,
            gpu_pwr_str,
            gpu_vram_str
        );
        self.buf.push_str(pad);
        self.buf.push_str(&format_line(&gpu_line));
        self.buf.push_str("\r\n");

        // Live CPU Card line
        let cpu_line = format!(
            "\x1b[38;2;0;220;255;1mCPU\x1b[0m  \x1b[38;2;255;255;255;1m{}\x1b[0m \x1b[38;2;140;150;170m({} Th)\x1b[0m  \x1b[38;2;0;220;255m│\x1b[0m \x1b[38;2;0;255;200;1m{:.1}% Load\x1b[0m \x1b[38;2;140;150;170m•\x1b[0m \x1b[38;2;255;180;0;1mMulti-Core\x1b[0m \x1b[38;2;140;150;170m•\x1b[0m \x1b[38;2;255;255;255m8 Acts\x1b[0m",
            trim_to(&stats.cpu_name, 20),
            stats.cpu_threads,
            stats.cpu_usage_pct
        );
        self.buf.push_str(pad);
        self.buf.push_str(&format_line(&cpu_line));
        self.buf.push_str("\r\n");

        // Section 1: Execution Modes
        self.buf.push_str(pad);
        self.buf.push_str(&format_section_header("BENCHMARK & EXECUTION MODES"));
        self.buf.push_str("\r\n");

        let row0 = format_menu_row(self.selected_idx == 0, "1", "Launch Interactive Demo", "[ 60 FPS V-Sync Preview ]");
        self.buf.push_str(pad);
        self.buf.push_str(&row0);
        self.buf.push_str("\r\n");

        let row1 = format_menu_row(self.selected_idx == 1, "2", "Launch GPU Stress Test", "[ 100% Uncapped Max Load ]");
        self.buf.push_str(pad);
        self.buf.push_str(&row1);
        self.buf.push_str("\r\n");

        let row2 = format_menu_row(self.selected_idx == 2, "3", "Run Full Benchmark Suite", "[ 8 Acts · Scorecard ]");
        self.buf.push_str(pad);
        self.buf.push_str(&row2);
        self.buf.push_str("\r\n");

        let row3 = format_menu_row(self.selected_idx == 3, "4", "Run Consistency Suite", "[ 5x Soak + Cooldown ]");
        self.buf.push_str(pad);
        self.buf.push_str(&row3);
        self.buf.push_str("\r\n");

        // Section 2: Workload Configuration
        self.buf.push_str(pad);
        self.buf.push_str(&format_section_header("WORKLOAD CONFIGURATION"));
        self.buf.push_str("\r\n");

        let scene_badge = format_scene_badge(config.selected_scene);
        let row4 = format_menu_row(self.selected_idx == 4, "5", "Select Starting Scene", &scene_badge);
        self.buf.push_str(pad);
        self.buf.push_str(&row4);
        self.buf.push_str("\r\n");

        let res_badge = format_resolution_badge(config.compute_width, config.compute_height);
        let row5 = format_menu_row(self.selected_idx == 5, "6", "Compute Resolution", &res_badge);
        self.buf.push_str(pad);
        self.buf.push_str(&row5);
        self.buf.push_str("\r\n");

        let total_bench_frames = config.frames_per_scene * (SCENE_NAMES.len() as u32);
        let frames_badge = format!("[ {} f/act ({} tot) ]", config.frames_per_scene, total_bench_frames);
        let row6 = format_menu_row(self.selected_idx == 6, "7", "Benchmark Frame Duration", &frames_badge);
        self.buf.push_str(pad);
        self.buf.push_str(&row6);
        self.buf.push_str("\r\n");

        let stress_badge = if config.uncapped {
            "[ UNCAPPED 100% ]"
        } else {
            "[ 60 FPS V-SYNC ]"
        };
        let row7 = format_menu_row(self.selected_idx == 7, "8", "Framerate & V-Sync Mode", stress_badge);
        self.buf.push_str(pad);
        self.buf.push_str(&row7);
        self.buf.push_str("\r\n");

        // Section 3: Hardware & System
        self.buf.push_str(pad);
        self.buf.push_str(&format_section_header("SYSTEM & HARDWARE"));
        self.buf.push_str("\r\n");

        let gpu_badge = format!("[ #{}: {} ]", config.selected_gpu_index, trim_to(&adapter_info.name, 18));
        let row8 = format_menu_row(self.selected_idx == 8, "9", "Switch GPU Adapter", &gpu_badge);
        self.buf.push_str(pad);
        self.buf.push_str(&row8);
        self.buf.push_str("\r\n");

        let telem_badge = if stats.gpu_temp_c > 0.0 {
            format!("[ {:.0}°C · {:.0}W · {} RPM ]", stats.gpu_temp_c, stats.gpu_power_w, stats.gpu_fan_rpm)
        } else {
            "[ Sensor Diagnostics ]".to_string()
        };
        let row9 = format_menu_row(self.selected_idx == 9, "T", "Live Hardware Telemetry", &telem_badge);
        self.buf.push_str(pad);
        self.buf.push_str(&row9);
        self.buf.push_str("\r\n");

        let row10 = format_menu_row(self.selected_idx == 10, "0", "Exit DynoTerm Launcher", "[ Return to Shell ]");
        self.buf.push_str(pad);
        self.buf.push_str(&row10);
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;140;150;170m[↑/↓] Navigate  [Enter] Select  [1-9/T/0] Jump  [U] Uncap  [Q] Exit\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255;1m└──────────────────────────────────────────────────────────────────────────┘\x1b[0m\r\n");
    }

    fn render_resolution_menu(&mut self, pad: &str, config: &MenuConfig) {
        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255;1m┌──────────────────────────────────────────────────────────────────────────┐\x1b[0m\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;255;255;255;1mSELECT COMPUTE RESOLUTION\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;140;150;170mMain Menu > Workload Configuration > Compute Resolution\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_section_header("RESOLUTION PRESETS"));
        self.buf.push_str("\r\n");

        let cur_w = config.compute_width;
        let cur_h = config.compute_height;

        let res_items = [
            ("1", RESOLUTION_PRESETS[0].0, "[ 15360x8640 · 132.7 Mpix · Extreme Load ]", RESOLUTION_PRESETS[0].2, RESOLUTION_PRESETS[0].3),
            ("2", RESOLUTION_PRESETS[1].0, "[ 7680x4320 · 33.2 Mpix · Heavy Workload ]", RESOLUTION_PRESETS[1].2, RESOLUTION_PRESETS[1].3),
            ("3", RESOLUTION_PRESETS[2].0, "[ 3840x2160 · 8.29 Mpix · Balanced Default ]", RESOLUTION_PRESETS[2].2, RESOLUTION_PRESETS[2].3),
            ("4", RESOLUTION_PRESETS[3].0, "[ 2560x1440 · 3.69 Mpix · High Performance ]", RESOLUTION_PRESETS[3].2, RESOLUTION_PRESETS[3].3),
            ("5", RESOLUTION_PRESETS[4].0, "[ 1920x1080 · 2.07 Mpix · Standard HD ]", RESOLUTION_PRESETS[4].2, RESOLUTION_PRESETS[4].3),
            ("6", RESOLUTION_PRESETS[5].0, "[ 1280x720 · 0.92 Mpix · Fast Preview ]", RESOLUTION_PRESETS[5].2, RESOLUTION_PRESETS[5].3),
        ];

        for (idx, (num, name, badge, w, h)) in res_items.iter().enumerate() {
            let is_sel = idx == self.selected_idx;
            let is_active = *w == cur_w && *h == cur_h;
            let row = format_choice_row(is_sel, num, is_active, name, badge);
            self.buf.push_str(pad);
            self.buf.push_str(&row);
            self.buf.push_str("\r\n");
        }

        self.buf.push_str(pad);
        self.buf.push_str(&format_section_header("SPECIAL MODES & RETURN"));
        self.buf.push_str("\r\n");

        // Native grid (idx 6)
        let is_native_active = cur_w < 300 && cur_w > 0;
        let row_native = format_choice_row(self.selected_idx == 6, "7", is_native_active, "Native Grid", "[ Viewport · 1:1 Character Grid ]");
        self.buf.push_str(pad);
        self.buf.push_str(&row_native);
        self.buf.push_str("\r\n");

        // Custom (idx 7)
        let is_custom_active = !res_items.iter().any(|p| p.3 == cur_w && p.4 == cur_h) && !is_native_active;
        let custom_badge = if is_custom_active {
            format!("[ Active: {}x{} ]", cur_w, cur_h)
        } else {
            "[ Prompt Width x Height ]".to_string()
        };
        let row_custom = format_choice_row(self.selected_idx == 7, "8", is_custom_active, "Custom Pixels", &custom_badge);
        self.buf.push_str(pad);
        self.buf.push_str(&row_custom);
        self.buf.push_str("\r\n");

        // Back item (idx 8)
        let row_back = format_choice_row(self.selected_idx == 8, "0", false, "Return to Main Menu", "[ Keep Current Selection ]");
        self.buf.push_str(pad);
        self.buf.push_str(&row_back);
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;140;150;170m[↑/↓/k/j] Navigate   [Enter] Select   [1-8/0] Quick Pick   [Esc/Q] Back\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255;1m└──────────────────────────────────────────────────────────────────────────┘\x1b[0m\r\n");
    }

    fn render_scene_menu(&mut self, pad: &str, config: &MenuConfig) {
        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255;1m┌──────────────────────────────────────────────────────────────────────────┐\x1b[0m\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;255;255;255;1mSELECT STARTING ACT / DEMO SCENE\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;140;150;170mMain Menu > Workload Configuration > Starting Act\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_section_header("AVAILABLE COMPUTATIONAL ACTS"));
        self.buf.push_str("\r\n");

        let scene_details = [
            ("Act I: Ocean Waves",         "[ Fluid Dynamics / Gravity Waves ]"),
            ("Act II: Mandelbulb Core",    "[ 3D Fractal Geometry Raymarch ]"),
            ("Act III: Black Hole Core",   "[ General Relativity Geodesics ]"),
            ("Act IV: Mesocyclone",        "[ Atmospheric Navier-Stokes ]"),
            ("Act V: Electrophysiology",   "[ Cardiac FitzHugh-Nagumo ]"),
            ("Act VI: Boltzmann Flow",     "[ CFD D2Q9 Vortex Shedding ]"),
            ("Act VII: Gait Kinematics",   "[ Forward Kinematics 3D Rig ]"),
            ("Act VIII: N-Body Supernova", "[ Gravitational Astrophysics ]"),
        ];

        for (idx, (name, badge)) in scene_details.iter().enumerate() {
            let is_sel = idx == self.selected_idx;
            let is_active = idx == (config.selected_scene as usize);
            let hotkey = format!("{}", idx + 1);
            let row = format_choice_row(is_sel, &hotkey, is_active, name, badge);
            self.buf.push_str(pad);
            self.buf.push_str(&row);
            self.buf.push_str("\r\n");
        }

        self.buf.push_str(pad);
        self.buf.push_str(&format_section_header("RETURN"));
        self.buf.push_str("\r\n");

        // Back item (idx 8)
        let is_back_sel = self.selected_idx == SCENE_NAMES.len();
        let row_back = format_choice_row(is_back_sel, "0", false, "Return to Main Menu", "[ Keep Current Selection ]");
        self.buf.push_str(pad);
        self.buf.push_str(&row_back);
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;140;150;170m[↑/↓/k/j] Navigate   [Enter] Select   [1-8/0] Quick Pick   [Esc/Q] Back\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255;1m└──────────────────────────────────────────────────────────────────────────┘\x1b[0m\r\n");
    }

    fn render_frames_menu(&mut self, pad: &str, config: &MenuConfig) {
        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255;1m┌──────────────────────────────────────────────────────────────────────────┐\x1b[0m\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;255;255;255;1mCONFIGURE BENCHMARK FRAME DURATION\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;140;150;170mMain Menu > Workload Configuration > Frame Iterations\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_section_header("BENCHMARK DURATION PRESETS"));
        self.buf.push_str("\r\n");

        let cur_frames = config.frames_per_scene;

        let frame_items = [
            ("1", FRAME_PRESETS[0].0, "[ 30 f/act · 240 tot (~3.5s) ]", FRAME_PRESETS[0].2),
            ("2", FRAME_PRESETS[1].0, "[ 100 f/act · 800 tot (~12s) ]", FRAME_PRESETS[1].2),
            ("3", FRAME_PRESETS[2].0, "[ 300 f/act · 2400 tot (~35s) ]", FRAME_PRESETS[2].2),
            ("4", FRAME_PRESETS[3].0, "[ 1000 f/act · 8000 tot (~120s) ]", FRAME_PRESETS[3].2),
        ];

        for (idx, (num, name, badge, f)) in frame_items.iter().enumerate() {
            let is_sel = idx == self.selected_idx;
            let is_active = *f == cur_frames;
            let row = format_choice_row(is_sel, num, is_active, name, badge);
            self.buf.push_str(pad);
            self.buf.push_str(&row);
            self.buf.push_str("\r\n");
        }

        self.buf.push_str(pad);
        self.buf.push_str(&format_section_header("CUSTOM & RETURN"));
        self.buf.push_str("\r\n");

        // Custom item (idx 4)
        let is_custom_sel = self.selected_idx == 4;
        let is_custom_active = !FRAME_PRESETS.iter().any(|p| p.2 == cur_frames);
        let custom_badge = if is_custom_active {
            format!("[ Active: {} f/act ]", cur_frames)
        } else {
            "[ Enter Arbitrary Count ]".to_string()
        };
        let row_custom = format_choice_row(is_custom_sel, "5", is_custom_active, "Custom Frames", &custom_badge);
        self.buf.push_str(pad);
        self.buf.push_str(&row_custom);
        self.buf.push_str("\r\n");

        // Back item (idx 5)
        let is_back_sel = self.selected_idx == 5;
        let row_back = format_choice_row(is_back_sel, "0", false, "Return to Main Menu", "[ Keep Current Selection ]");
        self.buf.push_str(pad);
        self.buf.push_str(&row_back);
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;140;150;170m[↑/↓/k/j] Navigate   [Enter] Select   [1-5/0] Quick Pick   [Esc/Q] Back\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255;1m└──────────────────────────────────────────────────────────────────────────┘\x1b[0m\r\n");
    }

    fn render_gpu_menu(&mut self, pad: &str, config: &MenuConfig, adapters: &[GpuAdapterInfo]) {
        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255;1m┌──────────────────────────────────────────────────────────────────────────┐\x1b[0m\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;255;255;255;1mSELECT ACTIVE GPU ADAPTER\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;140;150;170mMain Menu > System & Hardware > Target GPU Adapter\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_section_header("DETECTED GRAPHICS HARDWARE"));
        self.buf.push_str("\r\n");

        for (idx, adapter) in adapters.iter().enumerate() {
            let is_sel = idx == self.selected_idx;
            let is_active = idx == config.selected_gpu_index;
            let hotkey = format!("{}", idx);
            let name = trim_to(&adapter.name, 28);
            let badge = format!("[ {:?} · {:?} ]", adapter.backend, adapter.device_type);

            let row = format_choice_row(is_sel, &hotkey, is_active, &name, &badge);
            self.buf.push_str(pad);
            self.buf.push_str(&row);
            self.buf.push_str("\r\n");
        }

        self.buf.push_str(pad);
        self.buf.push_str(&format_section_header("RETURN"));
        self.buf.push_str("\r\n");

        // Back option
        let back_idx = adapters.len();
        let is_back_sel = self.selected_idx == back_idx;
        let row_back = format_choice_row(is_back_sel, "0", false, "Return to Main Menu", "[ Keep Current Selection ]");
        self.buf.push_str(pad);
        self.buf.push_str(&row_back);
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;140;150;170m[↑/↓] Navigate  [Enter] Select  [0-9] Quick Pick  [Esc/Q] Cancel\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255;1m└──────────────────────────────────────────────────────────────────────────┘\x1b[0m\r\n");
    }

    fn render_telemetry_view(
        &mut self,
        pad: &str,
        config: &MenuConfig,
        telemetry: &TelemetryCollector,
        adapter_info: &GpuAdapterInfo,
    ) {
        let stats = telemetry.sample(&adapter_info.name);

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255;1m┌──────────────────────────────────────────────────────────────────────────┐\x1b[0m\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;255;255;255;1mHARDWARE TELEMETRY & SYSTEM MONITOR\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;0;255;200;1m● LIVE SENSORS POLLING (150ms Refresh)\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_section_header("GPU ADAPTER & SENSORS"));
        self.buf.push_str("\r\n");

        let gpu_name_val = format!("\x1b[38;2;255;255;255;1m{}\x1b[0m", trim_to(&adapter_info.name, 44));
        self.buf.push_str(pad);
        self.buf.push_str(&format_telemetry_metric("GPU Device", &gpu_name_val));
        self.buf.push_str("\r\n");

        let driver_val = format!("\x1b[38;2;0;255;200m{} ({:?})\x1b[0m", trim_to(&adapter_info.driver, 28), adapter_info.backend);
        self.buf.push_str(pad);
        self.buf.push_str(&format_telemetry_metric("Driver & Backend", &driver_val));
        self.buf.push_str("\r\n");

        let bus_str = if stats.pcie_link.is_empty() { "PCIe" } else { &stats.pcie_link };
        let arch_val = format!("\x1b[38;2;255;180;0m{:?}\x1b[0m \x1b[38;2;140;150;170m(Bus: \x1b[38;2;255;255;255m{}\x1b[38;2;140;150;170m)\x1b[0m", adapter_info.device_type, bus_str);
        self.buf.push_str(pad);
        self.buf.push_str(&format_telemetry_metric("Architecture", &arch_val));
        self.buf.push_str("\r\n");

        let clocks_val = format!("\x1b[38;2;0;220;255;1m{} MHz\x1b[0m \x1b[38;2;140;150;170m(Peak: \x1b[38;2;255;255;255m{} MHz\x1b[38;2;140;150;170m, Load: \x1b[38;2;0;255;200;1m{:.1}%\x1b[38;2;140;150;170m)\x1b[0m", stats.gpu_clock_mhz, stats.peak_gpu_clock_mhz, stats.gpu_util_pct);
        self.buf.push_str(pad);
        self.buf.push_str(&format_telemetry_metric("GPU Clocks & Load", &clocks_val));
        self.buf.push_str("\r\n");

        let hotspot_val = if stats.gpu_hotspot_c > 0.0 {
            format!("{:.1} °C", stats.gpu_hotspot_c)
        } else {
            "N/A".to_string()
        };
        let temps_val = format!("\x1b[38;2;0;255;200;1m{:.1} °C\x1b[0m \x1b[38;2;140;150;170m(Hotspot: \x1b[38;2;255;180;0;1m{}\x1b[38;2;140;150;170m)\x1b[0m", stats.gpu_temp_c, hotspot_val);
        self.buf.push_str(pad);
        self.buf.push_str(&format_telemetry_metric("GPU Temperatures", &temps_val));
        self.buf.push_str("\r\n");

        let pwr_cap_str = if stats.gpu_power_cap_w > 0.0 {
            format!("{:.0} W", stats.gpu_power_cap_w)
        } else {
            "Board PPT".to_string()
        };
        let fan_str = if stats.gpu_fan_rpm > 0 {
            format!("{} RPM", stats.gpu_fan_rpm)
        } else {
            "0 RPM (Zero-RPM Idle)".to_string()
        };
        let cooling_val = format!("\x1b[38;2;255;255;255m{}\x1b[0m \x1b[38;2;140;150;170m•\x1b[0m \x1b[38;2;255;180;0;1m{:.1} W\x1b[0m \x1b[38;2;140;150;170m(Cap: {})\x1b[0m", fan_str, stats.gpu_power_w, pwr_cap_str);
        self.buf.push_str(pad);
        self.buf.push_str(&format_telemetry_metric("Cooling & Power", &cooling_val));
        self.buf.push_str("\r\n");

        let vram_val = format!("\x1b[38;2;0;255;200;1m{} MB\x1b[0m \x1b[38;2;140;150;170m/ {} MB (\x1b[38;2;255;180;0m{:.1}%\x1b[38;2;140;150;170m Bus Load)\x1b[0m", stats.vram_used_mb, stats.vram_total_mb, stats.mem_util_pct);
        self.buf.push_str(pad);
        self.buf.push_str(&format_telemetry_metric("VRAM Memory Load", &vram_val));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_section_header("CPU PROCESSOR & LOAD"));
        self.buf.push_str("\r\n");

        let cpu_name_val = format!("\x1b[38;2;255;255;255;1m{}\x1b[0m", trim_to(&stats.cpu_name, 44));
        self.buf.push_str(pad);
        self.buf.push_str(&format_telemetry_metric("CPU Processor", &cpu_name_val));
        self.buf.push_str("\r\n");

        let cpu_conc_val = format!("\x1b[38;2;0;255;200;1m{} Threads\x1b[0m \x1b[38;2;140;150;170m•\x1b[0m \x1b[38;2;255;180;0;1m{:.1} %\x1b[0m \x1b[38;2;140;150;170mLoad\x1b[0m", stats.cpu_threads, stats.cpu_usage_pct);
        self.buf.push_str(pad);
        self.buf.push_str(&format_telemetry_metric("CPU Threads & Load", &cpu_conc_val));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_section_header("ACTIVE WORKLOAD STATE"));
        self.buf.push_str("\r\n");

        let res_val = format!("\x1b[38;2;255;255;255;1m{}\x1b[0m", config.resolution_name);
        self.buf.push_str(pad);
        self.buf.push_str(&format_telemetry_metric("Active Resolution", &res_val));
        self.buf.push_str("\r\n");

        let bench_f_val = format!("\x1b[38;2;0;255;200;1m{} f/act\x1b[0m \x1b[38;2;140;150;170m({} total frames across 8 acts)\x1b[0m", config.frames_per_scene, config.frames_per_scene * 8);
        self.buf.push_str(pad);
        self.buf.push_str(&format_telemetry_metric("Benchmark Frames", &bench_f_val));
        self.buf.push_str("\r\n");

        let scene_val = format!("\x1b[38;2;255;255;255;1m{}\x1b[0m", SCENE_NAMES[(config.selected_scene as usize) % SCENE_NAMES.len()]);
        self.buf.push_str(pad);
        self.buf.push_str(&format_telemetry_metric("Active Scene", &scene_val));
        self.buf.push_str("\r\n");

        let mode_val = if config.uncapped {
            "\x1b[38;2;255;60;60;1m100% Uncapped (GPU Full Stress)\x1b[0m"
        } else {
            "\x1b[38;2;0;255;200;1m60 FPS V-Sync (Preview Mode)\x1b[0m"
        };
        self.buf.push_str(pad);
        self.buf.push_str(&format_telemetry_metric("Framerate Mode", mode_val));
        self.buf.push_str("\r\n");

        let failsafe_val = if config.failsafe_enabled {
            format!("\x1b[38;2;0;255;200;1mActive\x1b[0m \x1b[38;2;140;150;170m(Ceiling: {:.0}°C, Cooldown: {:.0}°C)\x1b[0m", config.max_temp_c, config.target_temp_c)
        } else {
            "\x1b[38;2;255;60;60;1mDisabled (No Tripwire)\x1b[0m".to_string()
        };
        self.buf.push_str(pad);
        self.buf.push_str(&format_telemetry_metric("Thermal Tripwire", &failsafe_val));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;0;255;200;1mPress [Enter], [Esc], or [Q] to return to Main Menu...\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255;1m└──────────────────────────────────────────────────────────────────────────┘\x1b[0m\r\n");
    }
}

pub fn format_section_header(title: &str) -> String {
    let tlen = visible_len(title);
    let dashes = 71_usize.saturating_sub(tlen);
    format!(
        "\x1b[38;2;0;220;255m├─ \x1b[38;2;255;180;0;1m{}\x1b[0m\x1b[38;2;0;220;255m {}\x1b[0m\x1b[38;2;0;220;255m{}┤\x1b[0m",
        title,
        "─".repeat(dashes),
        ""
    )
}

pub fn format_menu_row(is_sel: bool, hotkey: &str, label: &str, badge: &str) -> String {
    let cursor = if is_sel { "▶ " } else { "  " };
    let cur_col = if is_sel { "\x1b[38;2;0;255;200;1m" } else { "" };
    let key_col = if is_sel { "\x1b[38;2;255;255;255;1m" } else { "\x1b[38;2;255;180;0;1m" };
    let lbl_col = if is_sel { "\x1b[38;2;255;255;255;1m" } else { "\x1b[38;2;220;230;245m" };
    let bdg_col = if is_sel { "\x1b[38;2;0;255;200;1m" } else { "\x1b[38;2;130;145;165m" };

    let hotkey_box = format!("[{}] ", hotkey);
    let prefix_vlen = 2 + visible_len(&hotkey_box);
    let label_vlen = visible_len(label);
    let badge_vlen = visible_len(badge);
    let gap = 72_usize.saturating_sub(prefix_vlen + label_vlen + badge_vlen);

    let content = if is_sel {
        format!(
            "\x1b[48;2;18;38;58m{}{}{}{}\x1b[0m\x1b[48;2;18;38;58m{}{}\x1b[0m\x1b[48;2;18;38;58m{}{}{}\x1b[0m",
            cur_col, cursor,
            key_col, hotkey_box,
            lbl_col, label,
            " ".repeat(gap),
            bdg_col, badge
        )
    } else {
        format!(
            "{}{}{}{}{}{}{}{}{}\x1b[0m",
            cur_col, cursor,
            key_col, hotkey_box,
            lbl_col, label,
            " ".repeat(gap),
            bdg_col, badge
        )
    };

    format_line(&content)
}

pub fn format_choice_row(
    is_sel: bool,
    hotkey: &str,
    is_active: bool,
    label: &str,
    badge: &str,
) -> String {
    let cursor = if is_sel { "▶ " } else { "  " };
    let cur_col = if is_sel { "\x1b[38;2;0;255;200;1m" } else { "" };
    let key_col = if is_sel { "\x1b[38;2;255;255;255;1m" } else { "\x1b[38;2;255;180;0;1m" };

    let is_back = hotkey == "0" || hotkey == "Esc";
    let (radio_glyph, radio_col) = if is_back {
        ("   ", "")
    } else if is_active {
        ("(●)", "\x1b[38;2;0;255;200;1m")
    } else {
        ("(○)", "\x1b[38;2;100;115;135m")
    };

    let lbl_col = if is_active {
        "\x1b[38;2;255;255;255;1m"
    } else if is_sel {
        "\x1b[38;2;240;245;255;1m"
    } else {
        "\x1b[38;2;200;210;225m"
    };

    let bdg_col = if is_active {
        "\x1b[38;2;0;255;200m"
    } else if is_sel {
        "\x1b[38;2;255;180;0m"
    } else {
        "\x1b[38;2;130;145;165m"
    };

    let hotkey_box = format!("[{}] ", hotkey);
    let radio_box = format!("{} ", radio_glyph);
    let prefix_vlen = 2 + visible_len(&hotkey_box) + visible_len(&radio_box);
    let label_vlen = visible_len(label);
    let badge_vlen = visible_len(badge);
    let gap = 72_usize.saturating_sub(prefix_vlen + label_vlen + badge_vlen);

    let content = if is_sel {
        format!(
            "\x1b[48;2;18;38;58m{}{}{}{}{}{}{}{}{}{}{}\x1b[0m",
            cur_col, cursor,
            key_col, hotkey_box,
            radio_col, radio_box,
            lbl_col, label,
            " ".repeat(gap),
            bdg_col, badge
        )
    } else {
        format!(
            "{}{}{}{}{}{}{}{}{}{}{}\x1b[0m",
            cur_col, cursor,
            key_col, hotkey_box,
            radio_col, radio_box,
            lbl_col, label,
            " ".repeat(gap),
            bdg_col, badge
        )
    };

    format_line(&content)
}

fn format_telemetry_metric(key: &str, val: &str) -> String {
    let key_vlen = visible_len(key);
    let key_pad = 20_usize.saturating_sub(key_vlen);
    let line = format!(
        "\x1b[38;2;140;155;175m{}{}\x1b[38;2;0;220;255m : \x1b[0m{}",
        key,
        " ".repeat(key_pad),
        val
    );
    format_line(&line)
}

pub fn format_resolution_badge(w: u32, h: u32) -> String {
    if w >= 15360 {
        "[ 16K: 15360x8640 ]".to_string()
    } else if w >= 7680 {
        "[ 8K: 7680x4320 ]".to_string()
    } else if w >= 3840 {
        "[ 4K UHD: 3840x2160 ]".to_string()
    } else if w >= 2560 {
        "[ 1440p QHD: 2560x1440 ]".to_string()
    } else if w >= 1920 {
        "[ 1080p FHD: 1920x1080 ]".to_string()
    } else if w >= 1280 {
        "[ 720p HD: 1280x720 ]".to_string()
    } else if w == 0 {
        "[ Viewport 1:1 ]".to_string()
    } else {
        format!("[ {}x{} ]", w, h)
    }
}

pub fn format_scene_badge(scene_idx: u32) -> String {
    match scene_idx % 8 {
        0 => "[ Act I: Atmospheric Waves ]".to_string(),
        1 => "[ Act II: Mandelbulb Core ]".to_string(),
        2 => "[ Act III: Relativistic Black Hole ]".to_string(),
        3 => "[ Act IV: Tornadic Mesocyclone ]".to_string(),
        4 => "[ Act V: Electrophysiology ]".to_string(),
        5 => "[ Act VI: Lattice Boltzmann ]".to_string(),
        6 => "[ Act VII: Gait Kinematics ]".to_string(),
        _ => "[ Act VIII: N-Body Supernova ]".to_string(),
    }
}

pub fn format_resolution_tag(w: u32, h: u32) -> String {
    let mpix = (w as f64 * h as f64) / 1_000_000.0;
    if w >= 15360 {
        format!("16K UHD ({}x{}) [{:.1} Mpix]", w, h, mpix)
    } else if w >= 7680 {
        format!("8K UHD ({}x{}) [{:.1} Mpix]", w, h, mpix)
    } else if w >= 3840 {
        format!("4K UHD ({}x{}) [{:.2} Mpix]", w, h, mpix)
    } else if w >= 2560 {
        format!("1440p QHD ({}x{}) [{:.2} Mpix]", w, h, mpix)
    } else if w >= 1920 {
        format!("1080p FHD ({}x{}) [{:.2} Mpix]", w, h, mpix)
    } else if w >= 1280 {
        format!("720p HD ({}x{}) [{:.2} Mpix]", w, h, mpix)
    } else {
        format!("{}x{} [{:.2} Mpix]", w, h, mpix)
    }
}

pub fn visible_len(s: &str) -> usize {
    let mut in_escape = false;
    let mut count = 0;
    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
        } else if in_escape {
            if c.is_ascii_alphabetic() {
                in_escape = false;
            }
        } else {
            count += 1;
        }
    }
    count
}

pub fn format_line(visible_content: &str) -> String {
    let vlen = visible_len(visible_content);
    let target = 72;
    let pad = if vlen < target { target - vlen } else { 0 };
    format!(
        "\x1b[38;2;0;220;255m│\x1b[0m {}{}\x1b[0m \x1b[38;2;0;220;255m│\x1b[0m",
        visible_content,
        " ".repeat(pad)
    )
}

pub fn format_centered(text: &str) -> String {
    let vlen = visible_len(text);
    let target = 72;
    let total_pad = if vlen < target { target - vlen } else { 0 };
    let left_pad = total_pad / 2;
    let right_pad = total_pad - left_pad;
    format!(
        "\x1b[38;2;0;220;255m│\x1b[0m {}{}{}\x1b[0m \x1b[38;2;0;220;255m│\x1b[0m",
        " ".repeat(left_pad),
        text,
        " ".repeat(right_pad)
    )
}

fn trim_to(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len.saturating_sub(3)).collect();
        format!("{}...", truncated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_menu_box_widths() {
        // Border characters: 1 + 74 + 1 = 76
        assert_eq!("┌──────────────────────────────────────────────────────────────────────────┐".chars().count(), 76);
        assert_eq!("├──────────────────────────────────────────────────────────────────────────┤".chars().count(), 76);
        assert_eq!("└──────────────────────────────────────────────────────────────────────────┘".chars().count(), 76);

        // Test visible_len with and without ANSI escape sequences
        let plain = "Hello World";
        assert_eq!(visible_len(plain), 11);

        let colored = "\x1b[38;2;0;255;200;1mHello World\x1b[0m";
        assert_eq!(visible_len(colored), 11);

        // Test line formatting
        let line = format_line(colored);
        assert_eq!(visible_len(&line), 76);

        let centered = format_centered(colored);
        assert_eq!(visible_len(&centered), 76);

        // Test section headers
        let header = format_section_header("BENCHMARK & EXECUTION MODES");
        assert_eq!(visible_len(&header), 76);

        // Test menu rows
        let row_sel = format_menu_row(true, "1", "Launch Interactive Demo", "[ 60 FPS V-Sync Preview ]");
        assert_eq!(visible_len(&row_sel), 76);

        let row_unsel = format_menu_row(false, "2", "Launch GPU Stress Test", "[ 100% Uncapped Max Load ]");
        assert_eq!(visible_len(&row_unsel), 76);

        // Test choice rows
        let choice_active = format_choice_row(true, "1", true, "4K UHD", "[ 3840x2160 · 8.29 Mpix ]");
        assert_eq!(visible_len(&choice_active), 76);

        let choice_inactive = format_choice_row(false, "2", false, "1080p FHD", "[ 1920x1080 · 2.07 Mpix ]");
        assert_eq!(visible_len(&choice_inactive), 76);

        let choice_back = format_choice_row(false, "0", false, "Return to Main Menu", "[ Keep Current Selection ]");
        assert_eq!(visible_len(&choice_back), 76);

        // Test telemetry metrics
        let telem_row = format_telemetry_metric("GPU Device", "\x1b[38;2;255;255;255;1mNVIDIA GeForce RTX 4090\x1b[0m");
        assert_eq!(visible_len(&telem_row), 76);
    }

    #[test]
    fn test_format_resolution_tags() {
        assert!(format_resolution_tag(15360, 8640).contains("16K UHD (15360x8640)"));
        assert!(format_resolution_tag(7680, 4320).contains("8K UHD (7680x4320)"));
        assert!(format_resolution_tag(3840, 2160).contains("4K UHD (3840x2160)"));
        assert!(format_resolution_tag(2560, 1440).contains("1440p QHD (2560x1440)"));
        assert!(format_resolution_tag(1920, 1080).contains("1080p FHD (1920x1080)"));
        assert!(format_resolution_tag(1280, 720).contains("720p HD (1280x720)"));
        assert!(format_resolution_tag(800, 600).contains("800x600"));
    }

    #[test]
    fn test_format_badges() {
        assert_eq!(format_resolution_badge(3840, 2160), "[ 4K UHD: 3840x2160 ]");
        assert_eq!(format_resolution_badge(1920, 1080), "[ 1080p FHD: 1920x1080 ]");
        assert_eq!(format_resolution_badge(0, 0), "[ Viewport 1:1 ]");

        assert_eq!(format_scene_badge(0), "[ Act I: Atmospheric Waves ]");
        assert_eq!(format_scene_badge(6), "[ Act VII: Gait Kinematics ]");
        assert_eq!(format_scene_badge(7), "[ Act VIII: N-Body Supernova ]");
    }

    #[test]
    fn test_all_menu_views_line_widths() {
        let mut menu = CliMenu::new();
        let config = MenuConfig {
            resolution_name: "4K UHD (3840x2160) [8.29 Mpix]".to_string(),
            compute_width: 3840,
            compute_height: 2160,
            frames_per_scene: 100,
            selected_scene: 0,
            uncapped: false,
            selected_gpu_index: 0,
            output_path: "reports/benchmark.json".to_string(),
            consistency_iterations: 5,
            cooldown_seconds: 60,
            max_temp_c: 105.0,
            target_temp_c: 55.0,
            failsafe_enabled: true,
        };
        let telemetry = TelemetryCollector::new("NVIDIA GeForce RTX 4090".to_string());
        let adapter_info = GpuAdapterInfo {
            name: "NVIDIA GeForce RTX 4090".to_string(),
            backend: wgpu::Backend::Vulkan,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: "550.54.14".to_string(),
        };
        let all_adapters = vec![adapter_info.clone()];

        // 1. Test Main Menu
        menu.buf.clear();
        menu.render_main_menu("", &config, &telemetry, &adapter_info);
        for (idx, line) in menu.buf.lines().enumerate() {
            assert_eq!(
                visible_len(line),
                76,
                "Main menu line {} failed width test: '{}'",
                idx,
                line
            );
        }

        // 2. Test Resolution Menu
        menu.buf.clear();
        menu.render_resolution_menu("", &config);
        for (idx, line) in menu.buf.lines().enumerate() {
            assert_eq!(
                visible_len(line),
                76,
                "Resolution menu line {} failed width test: '{}'",
                idx,
                line
            );
        }

        // 3. Test Scene Menu
        menu.buf.clear();
        menu.render_scene_menu("", &config);
        for (idx, line) in menu.buf.lines().enumerate() {
            assert_eq!(
                visible_len(line),
                76,
                "Scene menu line {} failed width test: '{}'",
                idx,
                line
            );
        }

        // 4. Test Frames Menu
        menu.buf.clear();
        menu.render_frames_menu("", &config);
        for (idx, line) in menu.buf.lines().enumerate() {
            assert_eq!(
                visible_len(line),
                76,
                "Frames menu line {} failed width test: '{}'",
                idx,
                line
            );
        }

        // 5. Test GPU Menu
        menu.buf.clear();
        menu.render_gpu_menu("", &config, &all_adapters);
        for (idx, line) in menu.buf.lines().enumerate() {
            assert_eq!(
                visible_len(line),
                76,
                "GPU menu line {} failed width test: '{}'",
                idx,
                line
            );
        }

        // 6. Test Telemetry View
        menu.buf.clear();
        menu.render_telemetry_view("", &config, &telemetry, &adapter_info);
        for (idx, line) in menu.buf.lines().enumerate() {
            assert_eq!(
                visible_len(line),
                76,
                "Telemetry view line {} failed width test: '{}'",
                idx,
                line
            );
        }
    }
}
