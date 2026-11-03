use crate::gpu::GpuAdapterInfo;
use crate::telemetry::TelemetryCollector;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use std::io::{self, Write};
use std::time::{Duration, Instant};

pub const SCENE_NAMES: [&str; 6] = [
    "Act I: Atmospheric Ocean Waves (GPU)",
    "Act II: The Mandelbulb Core (GPU)",
    "Act III: Relativistic Black Hole (GPU)",
    "Act IV: Tornadic Mesocyclone (GPU - Weather Model)",
    "Act V: Cardiac Electrophysiology (GPU - Medical Model)",
    "Act VI: Kinetic N-Body Supernova (CPU)",
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
    ("Quick Check",    "  30 frames / act   (180 frames total,  ~3 sec)", 30),
    ("Standard Bench", " 100 frames / act   (600 frames total,  ~9 sec)", 100),
    ("Stress Test",    " 300 frames / act   (1800 frames total, ~27 sec)", 300),
    ("Endurance Soak", "1000 frames / act   (6000 frames total, ~90 sec)", 1000),
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MenuAction {
    LaunchDemo,
    LaunchUncapped,
    RunBenchmark,
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
                                    self.selected_idx = 8;
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
        const MAIN_ITEMS_COUNT: usize = 10;

        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected_idx = (self.selected_idx + MAIN_ITEMS_COUNT - 1) % MAIN_ITEMS_COUNT;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.selected_idx = (self.selected_idx + 1) % MAIN_ITEMS_COUNT;
            }
            KeyCode::Char('1') => return Ok(Some(MenuAction::LaunchDemo)),
            KeyCode::Char('2') => return Ok(Some(MenuAction::LaunchUncapped)),
            KeyCode::Char('3') => return Ok(Some(MenuAction::RunBenchmark)),
            KeyCode::Char('4') => {
                self.view = MenuView::SelectScene;
                self.selected_idx = (config.selected_scene as usize).min(SCENE_NAMES.len() - 1);
            }
            KeyCode::Char('5') => {
                self.view = MenuView::SelectResolution;
                self.selected_idx = 2; // Default 4K
            }
            KeyCode::Char('6') => {
                self.view = MenuView::SelectFrames;
                self.selected_idx = 1; // Default Standard 100
            }
            KeyCode::Char('7') | KeyCode::Char('u') | KeyCode::Char('U') => {
                config.uncapped = !config.uncapped;
            }
            KeyCode::Char('8') => {
                self.view = MenuView::SelectGpu;
                self.selected_idx = config.selected_gpu_index;
            }
            KeyCode::Char('9') | KeyCode::Char('t') | KeyCode::Char('T') => {
                self.view = MenuView::TelemetryInfo;
            }
            KeyCode::Char('0') | KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
                return Ok(Some(MenuAction::Exit));
            }
            KeyCode::Enter | KeyCode::Char(' ') => match self.selected_idx {
                0 => return Ok(Some(MenuAction::LaunchDemo)),
                1 => return Ok(Some(MenuAction::LaunchUncapped)),
                2 => return Ok(Some(MenuAction::RunBenchmark)),
                3 => {
                    self.view = MenuView::SelectScene;
                    self.selected_idx = (config.selected_scene as usize).min(SCENE_NAMES.len() - 1);
                }
                4 => {
                    self.view = MenuView::SelectResolution;
                    self.selected_idx = 2;
                }
                5 => {
                    self.view = MenuView::SelectFrames;
                    self.selected_idx = 1;
                }
                6 => {
                    config.uncapped = !config.uncapped;
                }
                7 => {
                    self.view = MenuView::SelectGpu;
                    self.selected_idx = config.selected_gpu_index;
                }
                8 => {
                    self.view = MenuView::TelemetryInfo;
                }
                9 => return Ok(Some(MenuAction::Exit)),
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
            KeyCode::Char('1') => return Ok(self.apply_res_preset(0, config)),
            KeyCode::Char('2') => return Ok(self.apply_res_preset(1, config)),
            KeyCode::Char('3') => return Ok(self.apply_res_preset(2, config)),
            KeyCode::Char('4') => return Ok(self.apply_res_preset(3, config)),
            KeyCode::Char('5') => return Ok(self.apply_res_preset(4, config)),
            KeyCode::Char('6') => return Ok(self.apply_res_preset(5, config)),
            KeyCode::Char('7') => return Ok(self.apply_native_res(config, term_cols, term_rows)),
            KeyCode::Char('8') => return self.prompt_custom_res(config),
            KeyCode::Char('0') | KeyCode::Esc | KeyCode::Backspace => {
                self.view = MenuView::Main;
                self.selected_idx = 4;
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
                    self.selected_idx = 4;
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
        self.selected_idx = 4;
        Some(MenuAction::RecreatePipeline)
    }

    fn apply_native_res(&mut self, config: &mut MenuConfig, cols: u32, rows: u32) -> Option<MenuAction> {
        let term_w = cols.max(40);
        let char_h = rows.saturating_sub(7).max(8);
        config.compute_width = term_w;
        config.compute_height = char_h * 2;
        config.resolution_name = format_resolution_tag(config.compute_width, config.compute_height);
        self.view = MenuView::Main;
        self.selected_idx = 4;
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
        self.selected_idx = 4;

        if changed {
            Ok(Some(MenuAction::RecreatePipeline))
        } else {
            Ok(None)
        }
    }

    fn handle_scene_input(&mut self, code: KeyCode, config: &mut MenuConfig) {
        const TOTAL_SCENE_ITEMS: usize = 7; // 6 scenes + back
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected_idx = (self.selected_idx + TOTAL_SCENE_ITEMS - 1) % TOTAL_SCENE_ITEMS;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.selected_idx = (self.selected_idx + 1) % TOTAL_SCENE_ITEMS;
            }
            KeyCode::Char('1') => {
                config.selected_scene = 0;
                self.view = MenuView::Main;
                self.selected_idx = 3;
            }
            KeyCode::Char('2') => {
                config.selected_scene = 1;
                self.view = MenuView::Main;
                self.selected_idx = 3;
            }
            KeyCode::Char('3') => {
                config.selected_scene = 2;
                self.view = MenuView::Main;
                self.selected_idx = 3;
            }
            KeyCode::Char('4') => {
                config.selected_scene = 3;
                self.view = MenuView::Main;
                self.selected_idx = 3;
            }
            KeyCode::Char('5') => {
                config.selected_scene = 4;
                self.view = MenuView::Main;
                self.selected_idx = 3;
            }
            KeyCode::Char('6') => {
                config.selected_scene = 5;
                self.view = MenuView::Main;
                self.selected_idx = 3;
            }
            KeyCode::Char('0') | KeyCode::Esc | KeyCode::Backspace => {
                self.view = MenuView::Main;
                self.selected_idx = 3;
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                if self.selected_idx < 6 {
                    config.selected_scene = self.selected_idx as u32;
                }
                self.view = MenuView::Main;
                self.selected_idx = 3;
            }
            _ => {}
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
            KeyCode::Char('1') => {
                config.frames_per_scene = FRAME_PRESETS[0].2;
                self.view = MenuView::Main;
                self.selected_idx = 5;
            }
            KeyCode::Char('2') => {
                config.frames_per_scene = FRAME_PRESETS[1].2;
                self.view = MenuView::Main;
                self.selected_idx = 5;
            }
            KeyCode::Char('3') => {
                config.frames_per_scene = FRAME_PRESETS[2].2;
                self.view = MenuView::Main;
                self.selected_idx = 5;
            }
            KeyCode::Char('4') => {
                config.frames_per_scene = FRAME_PRESETS[3].2;
                self.view = MenuView::Main;
                self.selected_idx = 5;
            }
            KeyCode::Char('5') => return self.prompt_custom_frames(config),
            KeyCode::Char('0') | KeyCode::Esc | KeyCode::Backspace => {
                self.view = MenuView::Main;
                self.selected_idx = 5;
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                if self.selected_idx < 4 {
                    config.frames_per_scene = FRAME_PRESETS[self.selected_idx].2;
                    self.view = MenuView::Main;
                    self.selected_idx = 5;
                } else if self.selected_idx == 4 {
                    return self.prompt_custom_frames(config);
                } else {
                    self.view = MenuView::Main;
                    self.selected_idx = 5;
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
        self.selected_idx = 5;
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
            KeyCode::Char('0') | KeyCode::Esc | KeyCode::Backspace => {
                self.view = MenuView::Main;
                self.selected_idx = 7;
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                if self.selected_idx < adapter_count {
                    let idx = self.selected_idx;
                    self.view = MenuView::Main;
                    self.selected_idx = 7;
                    return Some(MenuAction::SwitchGpu(idx));
                } else {
                    self.view = MenuView::Main;
                    self.selected_idx = 7;
                }
            }
            _ => {
                if let KeyCode::Char(c) = code {
                    if let Some(digit) = c.to_digit(10) {
                        let idx = digit as usize;
                        if idx < adapter_count {
                            self.view = MenuView::Main;
                            self.selected_idx = 7;
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

        let top_pad_count = (rows.saturating_sub(26) / 2).min(2);
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
            "\x1b[38;2;255;255;255;1mPRETTY DYNO TERM\x1b[38;2;140;150;170m — \x1b[38;2;255;180;0;1mMAIN MENU & LAUNCHER\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;140;150;170mHigh-Performance 4K GPU Compute & Demoparty Benchmark Suite\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        // System badge
        let gpu_str = format!(
            "\x1b[38;2;140;150;170mGPU: \x1b[38;2;255;255;255;1m{} \x1b[38;2;0;255;200m({:?})\x1b[0m",
            trim_to(&adapter_info.name, 45),
            adapter_info.backend
        );
        self.buf.push_str(pad);
        self.buf.push_str(&format_line(&gpu_str));
        self.buf.push_str("\r\n");

        let cpu_str = format!(
            "\x1b[38;2;140;150;170mCPU: \x1b[38;2;255;255;255;1m{} \x1b[38;2;0;255;200m({} Threads)\x1b[0m",
            trim_to(&stats.cpu_name, 45),
            stats.cpu_threads
        );
        self.buf.push_str(pad);
        self.buf.push_str(&format_line(&cpu_str));
        self.buf.push_str("\r\n");

        // Current Config Summary
        let res_summary = format!(
            "\x1b[38;2;140;150;170mTarget Resolution : \x1b[38;2;255;180;0;1m{}\x1b[0m",
            config.resolution_name
        );
        self.buf.push_str(pad);
        self.buf.push_str(&format_line(&res_summary));
        self.buf.push_str("\r\n");

        let total_bench_frames = config.frames_per_scene * 6;
        let bench_summary = format!(
            "\x1b[38;2;140;150;170mBenchmark Frames  : \x1b[38;2;0;255;200;1m{} frames/act \x1b[38;2;140;150;170m({} frames total)\x1b[0m",
            config.frames_per_scene, total_bench_frames
        );
        self.buf.push_str(pad);
        self.buf.push_str(&format_line(&bench_summary));
        self.buf.push_str("\r\n");

        let scene_summary = format!(
            "\x1b[38;2;140;150;170mActive Scene      : \x1b[38;2;255;255;255;1m{}\x1b[0m",
            SCENE_NAMES[(config.selected_scene as usize) % SCENE_NAMES.len()]
        );
        self.buf.push_str(pad);
        self.buf.push_str(&format_line(&scene_summary));
        self.buf.push_str("\r\n");

        let stress_label = if config.uncapped {
            "\x1b[38;2;255;60;60;1m100% Uncapped (GPU Full Stress)\x1b[0m"
        } else {
            "\x1b[38;2;0;255;200;1m60 FPS V-Sync (Preview Mode)\x1b[0m"
        };
        let stress_summary = format!(
            "\x1b[38;2;140;150;170mDemo Frame Rate   : {}\x1b[0m",
            stress_label
        );
        self.buf.push_str(pad);
        self.buf.push_str(&format_line(&stress_summary));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        // Menu Items
        let items = [
            ("1", "Launch Interactive Demo (60 FPS V-Sync Preview)"),
            ("2", "Launch 100% Uncapped GPU Stress Test (Max Clock/Power)"),
            ("3", "Run Full Benchmark Suite (Deterministic Scorecard)"),
            ("4", "Select Starting Scene (Act I - VI)"),
            ("5", "Change Compute Resolution (16K -> 720p / Native / Custom)"),
            ("6", "Configure Benchmark Frames (30 / 100 / 300 / 1000)"),
            ("7", "Toggle Stress Mode (60 FPS Capped <-> Uncapped)"),
            ("8", "Switch GPU Adapter (Discrete vs Integrated / CPU)"),
            ("9", "View Live Hardware Telemetry & Diagnostics"),
            ("0", "Exit DynoTerm"),
        ];

        for (idx, (num, label)) in items.iter().enumerate() {
            let is_sel = idx == self.selected_idx;
            let line_str = if is_sel {
                format!(
                    "\x1b[48;2;20;40;60m\x1b[38;2;0;255;200;1m ▶ [{}]  >> {:<58}\x1b[0m",
                    num, label
                )
            } else {
                format!(
                    "\x1b[38;2;140;150;170m   \x1b[38;2;255;180;0;1m[{}]\x1b[38;2;255;255;255m  >> {:<58}\x1b[0m",
                    num, label
                )
            };
            self.buf.push_str(pad);
            self.buf.push_str(&format_line(&line_str));
            self.buf.push_str("\r\n");
        }

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;140;150;170m[↑/↓] Navigate  [Enter] Select  [1-9,0] Hotkeys  [U] Uncap  [Q] Exit\x1b[0m",
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
            "\x1b[38;2;140;150;170mHigher internal resolutions significantly scale raymarching GPU load\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        let cur_w = config.compute_width;
        let cur_h = config.compute_height;

        let res_items = [
            ("1", RESOLUTION_PRESETS[0].0, RESOLUTION_PRESETS[0].1, RESOLUTION_PRESETS[0].2, RESOLUTION_PRESETS[0].3),
            ("2", RESOLUTION_PRESETS[1].0, RESOLUTION_PRESETS[1].1, RESOLUTION_PRESETS[1].2, RESOLUTION_PRESETS[1].3),
            ("3", RESOLUTION_PRESETS[2].0, RESOLUTION_PRESETS[2].1, RESOLUTION_PRESETS[2].2, RESOLUTION_PRESETS[2].3),
            ("4", RESOLUTION_PRESETS[3].0, RESOLUTION_PRESETS[3].1, RESOLUTION_PRESETS[3].2, RESOLUTION_PRESETS[3].3),
            ("5", RESOLUTION_PRESETS[4].0, RESOLUTION_PRESETS[4].1, RESOLUTION_PRESETS[4].2, RESOLUTION_PRESETS[4].3),
            ("6", RESOLUTION_PRESETS[5].0, RESOLUTION_PRESETS[5].1, RESOLUTION_PRESETS[5].2, RESOLUTION_PRESETS[5].3),
            ("7", RESOLUTION_PRESETS[6].0, RESOLUTION_PRESETS[6].1, 0, 0),
            ("8", "Custom", "Enter custom pixel width x height", 0, 0),
            ("0", "Back", "Return to Main Menu (Keep Current)", 0, 0),
        ];

        for (idx, (num, name, desc, w, h)) in res_items.iter().enumerate() {
            let is_sel = idx == self.selected_idx;
            let is_active = (*w != 0 && *w == cur_w && *h == cur_h)
                || (idx == 6 && cur_w < 300)
                || (idx == 7 && cur_w != 15360 && cur_w != 7680 && cur_w != 3840 && cur_w != 2560 && cur_w != 1920 && cur_w != 1280 && cur_w >= 300);

            let marker = if is_active { "✔" } else { " " };
            let line_str = if is_sel {
                format!(
                    "\x1b[48;2;20;40;60m\x1b[38;2;0;255;200;1m ▶ [{}]  {:<12} {} {:<45}\x1b[0m",
                    num, name, marker, desc
                )
            } else {
                let col = if is_active { "\x1b[38;2;0;255;200;1m" } else { "\x1b[38;2;255;255;255m" };
                format!(
                    "\x1b[38;2;140;150;170m   \x1b[38;2;255;180;0;1m[{}]\x1b[0m  {}{:<12}\x1b[0m \x1b[38;2;0;255;200m{}\x1b[0m \x1b[38;2;140;150;170m{:<45}\x1b[0m",
                    num, col, name, marker, desc
                )
            };
            self.buf.push_str(pad);
            self.buf.push_str(&format_line(&line_str));
            self.buf.push_str("\r\n");
        }

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;140;150;170m[↑/↓] Navigate   [Enter] Select   [1-8/0] Quick Pick   [Esc] Cancel\x1b[0m",
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
            "\x1b[38;2;140;150;170mAll scenes feature authentic computational physics & biophysics models\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        for (idx, name) in SCENE_NAMES.iter().enumerate() {
            let is_sel = idx == self.selected_idx;
            let is_active = idx == (config.selected_scene as usize);
            let marker = if is_active { "✔" } else { " " };
            let line_str = if is_sel {
                format!(
                    "\x1b[48;2;20;40;60m\x1b[38;2;0;255;200;1m ▶ [{}]  {} {:<59}\x1b[0m",
                    idx + 1, marker, name
                )
            } else {
                let col = if is_active { "\x1b[38;2;0;255;200;1m" } else { "\x1b[38;2;255;255;255m" };
                format!(
                    "\x1b[38;2;140;150;170m   \x1b[38;2;255;180;0;1m[{}]\x1b[0m  \x1b[38;2;0;255;200m{}\x1b[0m {}{:<59}\x1b[0m",
                    idx + 1, marker, col, name
                )
            };
            self.buf.push_str(pad);
            self.buf.push_str(&format_line(&line_str));
            self.buf.push_str("\r\n");
        }

        // Back item
        let is_back_sel = self.selected_idx == SCENE_NAMES.len();
        let back_str = if is_back_sel {
            "\x1b[48;2;20;40;60m\x1b[38;2;0;255;200;1m ▶ [0]    Back to Main Menu                                          \x1b[0m"
        } else {
            "\x1b[38;2;140;150;170m   \x1b[38;2;255;180;0;1m[0]\x1b[0m    Back to Main Menu                                          "
        };
        self.buf.push_str(pad);
        self.buf.push_str(&format_line(back_str));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;140;150;170m[↑/↓] Navigate   [Enter] Select   [1-6/0] Quick Pick   [Esc] Cancel\x1b[0m",
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
            "\x1b[38;2;140;150;170mControls frame iterations per act during deterministic benchmarking\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        let cur_frames = config.frames_per_scene;

        for (idx, (name, desc, f)) in FRAME_PRESETS.iter().enumerate() {
            let is_sel = idx == self.selected_idx;
            let is_active = *f == cur_frames;
            let marker = if is_active { "✔" } else { " " };
            let line_str = if is_sel {
                format!(
                    "\x1b[48;2;20;40;60m\x1b[38;2;0;255;200;1m ▶ [{}]  {:<15} {} {:<43}\x1b[0m",
                    idx + 1, name, marker, desc
                )
            } else {
                let col = if is_active { "\x1b[38;2;0;255;200;1m" } else { "\x1b[38;2;255;255;255m" };
                format!(
                    "\x1b[38;2;140;150;170m   \x1b[38;2;255;180;0;1m[{}]\x1b[0m  {}{:<15}\x1b[0m \x1b[38;2;0;255;200m{}\x1b[0m \x1b[38;2;140;150;170m{:<43}\x1b[0m",
                    idx + 1, col, name, marker, desc
                )
            };
            self.buf.push_str(pad);
            self.buf.push_str(&format_line(&line_str));
            self.buf.push_str("\r\n");
        }

        // Custom item (idx 4)
        let is_custom_sel = self.selected_idx == 4;
        let is_custom_active = !FRAME_PRESETS.iter().any(|p| p.2 == cur_frames);
        let custom_marker = if is_custom_active { "✔" } else { " " };
        let custom_str = if is_custom_sel {
            format!(
                "\x1b[48;2;20;40;60m\x1b[38;2;0;255;200;1m ▶ [5]  {:<15} {} {:<43}\x1b[0m",
                "Custom", custom_marker, format!("Input custom frames (Currently: {} frames)", cur_frames)
            )
        } else {
            let col = if is_custom_active { "\x1b[38;2;0;255;200;1m" } else { "\x1b[38;2;255;255;255m" };
            format!(
                "\x1b[38;2;140;150;170m   \x1b[38;2;255;180;0;1m[5]\x1b[0m  {}{:<15}\x1b[0m \x1b[38;2;0;255;200m{}\x1b[0m \x1b[38;2;140;150;170m{:<43}\x1b[0m",
                col, "Custom", custom_marker, format!("Input custom frames (Currently: {} frames)", cur_frames)
            )
        };
        self.buf.push_str(pad);
        self.buf.push_str(&format_line(&custom_str));
        self.buf.push_str("\r\n");

        // Back item (idx 5)
        let is_back_sel = self.selected_idx == 5;
        let back_str = if is_back_sel {
            "\x1b[48;2;20;40;60m\x1b[38;2;0;255;200;1m ▶ [0]    Back to Main Menu                                          \x1b[0m"
        } else {
            "\x1b[38;2;140;150;170m   \x1b[38;2;255;180;0;1m[0]\x1b[0m    Back to Main Menu                                          "
        };
        self.buf.push_str(pad);
        self.buf.push_str(&format_line(back_str));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;140;150;170m[↑/↓] Navigate   [Enter] Select   [1-5/0] Quick Pick   [Esc] Cancel\x1b[0m",
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
            "\x1b[38;2;140;150;170mRe-initializes hardware context and WebGPU compute pipelines\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        for (idx, adapter) in adapters.iter().enumerate() {
            let is_sel = idx == self.selected_idx;
            let is_active = idx == config.selected_gpu_index;
            let marker = if is_active { "✔" } else { " " };
            let desc = format!("{} | {:?} | {:?}", trim_to(&adapter.name, 36), adapter.backend, adapter.device_type);

            let line_str = if is_sel {
                format!(
                    "\x1b[48;2;20;40;60m\x1b[38;2;0;255;200;1m ▶ [{}]  {} {:<58}\x1b[0m",
                    idx, marker, desc
                )
            } else {
                let col = if is_active { "\x1b[38;2;0;255;200;1m" } else { "\x1b[38;2;255;255;255m" };
                format!(
                    "\x1b[38;2;140;150;170m   \x1b[38;2;255;180;0;1m[{}]\x1b[0m  \x1b[38;2;0;255;200m{}\x1b[0m {}{:<58}\x1b[0m",
                    idx, marker, col, desc
                )
            };
            self.buf.push_str(pad);
            self.buf.push_str(&format_line(&line_str));
            self.buf.push_str("\r\n");
        }

        // Back option
        let back_idx = adapters.len();
        let is_back_sel = self.selected_idx == back_idx;
        let back_str = if is_back_sel {
            "\x1b[48;2;20;40;60m\x1b[38;2;0;255;200;1m ▶ [Esc]  Back to Main Menu (Keep Current GPU)                      \x1b[0m"
        } else {
            "\x1b[38;2;140;150;170m   \x1b[38;2;255;180;0;1m[Esc]\x1b[0m  Back to Main Menu (Keep Current GPU)                      "
        };
        self.buf.push_str(pad);
        self.buf.push_str(&format_line(back_str));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;140;150;170m[↑/↓] Navigate   [Enter] Select Adapter   [0-9] Quick Pick   [Esc] Cancel\x1b[0m",
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
            "\x1b[38;2;0;255;200;1m● LIVE SENSORS POLLING (150ms)\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        let mut items = vec![
            format!("\x1b[38;2;140;150;170mGPU Device         : \x1b[38;2;255;255;255;1m{}\x1b[0m", adapter_info.name),
            format!("\x1b[38;2;140;150;170mDriver & Backend   : \x1b[38;2;0;255;200m{} ({:?})\x1b[0m", adapter_info.driver, adapter_info.backend),
            format!("\x1b[38;2;140;150;170mAdapter Type       : \x1b[38;2;255;180;0m{:?}\x1b[0m", adapter_info.device_type),
            format!("\x1b[38;2;140;150;170mGPU Utilization    : \x1b[38;2;0;255;200;1m{:.1} %\x1b[38;2;140;150;170m (Peak: {:.1} %)\x1b[0m", stats.gpu_util_pct, stats.peak_gpu_util_pct),
            format!("\x1b[38;2;140;150;170mGPU Temperature    : \x1b[38;2;0;255;200;1m{:.1} °C\x1b[38;2;140;150;170m (Standard Package Temp)\x1b[0m", stats.gpu_temp_c),
        ];
        if stats.gpu_hotspot_c > 0.0 {
            items.push(format!("\x1b[38;2;140;150;170mGPU Hotspot (Jct)  : \x1b[38;2;255;180;0;1m{:.1} °C\x1b[38;2;140;150;170m (Internal Junction Peak)\x1b[0m", stats.gpu_hotspot_c));
        }
        if stats.gpu_mem_temp_c > 0.0 {
            items.push(format!("\x1b[38;2;140;150;170mGPU Memory Temp    : \x1b[38;2;0;255;200;1m{:.1} °C\x1b[38;2;140;150;170m (VRAM IC Sensors)\x1b[0m", stats.gpu_mem_temp_c));
        }
        items.push(format!("\x1b[38;2;140;150;170mGPU Power Draw     : \x1b[38;2;255;180;0;1m{:.1} W\x1b[38;2;140;150;170m (Board PPT)\x1b[0m", stats.gpu_power_w));
        items.push(format!("\x1b[38;2;140;150;170mGPU Core Clock     : \x1b[38;2;0;220;255;1m{} MHz\x1b[38;2;140;150;170m (Peak: {} MHz)\x1b[0m", stats.gpu_clock_mhz, stats.peak_gpu_clock_mhz));
        items.push(format!("\x1b[38;2;140;150;170mVRAM Allocation    : \x1b[38;2;0;255;200m{} MB / {} MB\x1b[0m", stats.vram_used_mb, stats.vram_total_mb));
        items.push(format!("\x1b[38;2;140;150;170mCPU Processor      : \x1b[38;2;255;255;255;1m{}\x1b[0m", stats.cpu_name));
        items.push(format!("\x1b[38;2;140;150;170mCPU Parallelism    : \x1b[38;2;0;255;200m{} Hardware Threads\x1b[0m", stats.cpu_threads));
        items.push(format!("\x1b[38;2;140;150;170mCPU Utilization    : \x1b[38;2;255;180;0m{:.1} %\x1b[0m", stats.cpu_usage_pct));
        items.push(format!("\x1b[38;2;140;150;170mActive Resolution  : \x1b[38;2;255;255;255;1m{}\x1b[0m", config.resolution_name));

        for item in items {
            self.buf.push_str(pad);
            self.buf.push_str(&format_line(&item));
            self.buf.push_str("\r\n");
        }

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255m├──────────────────────────────────────────────────────────────────────────┤\x1b[0m\r\n");

        self.buf.push_str(pad);
        self.buf.push_str(&format_centered(
            "\x1b[38;2;0;255;200;1mPress [Enter], [Esc], or [Backspace] to return to Main Menu...\x1b[0m",
        ));
        self.buf.push_str("\r\n");

        self.buf.push_str(pad);
        self.buf.push_str("\x1b[38;2;0;220;255;1m└──────────────────────────────────────────────────────────────────────────┘\x1b[0m\r\n");
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
    }
}
