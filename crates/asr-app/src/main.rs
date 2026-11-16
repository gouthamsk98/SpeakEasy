// SPDX-License-Identifier: GPL-3.0-or-later
//! Desktop app: live captions + calibration (voice baseline) sessions.
//!
//! Threads:
//!  * audio callback (cpal)  -> lock-free ring buffer
//!  * engine worker          -> pops audio every 2 ms, runs features + model, sends events;
//!    records "takes" (raw audio + text) on command for calibration
//!  * UI (egui, GPU via wgpu) -> repaints immediately when an event arrives

#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod calibrate;

use anyhow::Result;
use asr_engine::{load_recognizer, Device, EndpointConfig, Event, Pipeline};
use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::egui;
use std::time::{Duration, Instant};

pub struct Stats {
    name: String,
    ema_ms: f64,
    max_ms: f64,
    rtf: f64,
    chunk_ms: f64,
}

pub enum Msg {
    Ready(String),
    Event(Event),
    /// A calibration take finished: 16 kHz mono audio + everything recognised in it.
    TakeDone {
        audio: Vec<f32>,
        text: String,
    },
    Stats(Stats),
    Error(String),
}

pub enum Cmd {
    StartTake,
    StopTake,
    /// Stop the engine (model switch): releases the microphone and the model.
    Shutdown,
}

/// Longest take we keep (protects memory if Stop is never pressed).
const MAX_TAKE_SECS: usize = 60;

fn spawn_engine(model: String, device: Device, ctx: egui::Context) -> (Receiver<Msg>, Sender<Cmd>) {
    let (tx, rx) = unbounded();
    let (cmd_tx, cmd_rx) = unbounded::<Cmd>();
    std::thread::Builder::new()
        .name("asr-engine".into())
        .spawn(move || {
            let run = || -> Result<()> {
                let mut p =
                    Pipeline::new(load_recognizer(&model, device)?, EndpointConfig::default());
                let mut cap = asr_audio::start_capture(None)?;
                let _ = tx.send(Msg::Ready(p.recognizer_name()));
                ctx.request_repaint();
                let mut buf = Vec::with_capacity(4096);
                let mut ev = Vec::new();
                let mut take: Option<(Vec<f32>, Vec<String>)> = None;
                let mut last_stats = Instant::now();
                loop {
                    while let Ok(c) = cmd_rx.try_recv() {
                        match c {
                            Cmd::StartTake => {
                                // Start clean: drop pending audio and any half utterance.
                                while cap.consumer.pop().is_ok() {}
                                let mut junk = Vec::new();
                                p.flush(&mut junk)?;
                                p.set_endpointing(false); // whole take = one utterance
                                take = Some((Vec::with_capacity(16_000 * 10), Vec::new()));
                            }
                            Cmd::Shutdown => return Ok(()),
                            Cmd::StopTake => {
                                if let Some((audio, mut finals)) = take.take() {
                                    p.flush(&mut ev)?;
                                    p.set_endpointing(true);
                                    for e in ev.drain(..) {
                                        if let Event::Final(t) = e {
                                            finals.push(t);
                                        }
                                    }
                                    let text = finals.join(" ");
                                    let _ = tx.send(Msg::TakeDone { audio, text });
                                    ctx.request_repaint();
                                }
                            }
                        }
                    }

                    buf.clear();
                    while let Ok(s) = cap.consumer.pop() {
                        buf.push(s);
                    }
                    if !buf.is_empty() {
                        if let Some((audio, _)) = &mut take {
                            let room =
                                MAX_TAKE_SECS * 16_000 - audio.len().min(MAX_TAKE_SECS * 16_000);
                            audio.extend_from_slice(&buf[..buf.len().min(room)]);
                        }
                        p.push_audio(&buf, &mut ev)?;
                        if !ev.is_empty() {
                            for e in ev.drain(..) {
                                if let (Some((_, finals)), Event::Final(t)) = (&mut take, &e) {
                                    finals.push(t.clone());
                                }
                                let _ = tx.send(Msg::Event(e));
                            }
                            ctx.request_repaint();
                        }
                    }
                    if last_stats.elapsed() > Duration::from_millis(250) {
                        let m = &p.metrics;
                        let _ = tx.send(Msg::Stats(Stats {
                            name: p.recognizer_name(),
                            ema_ms: m.ema_compute_ms,
                            max_ms: m.max_compute.as_secs_f64() * 1e3,
                            rtf: m.rtf(),
                            chunk_ms: m.chunk_audio.as_secs_f64() * 1e3,
                        }));
                        ctx.request_repaint(); // 4 Hz: keeps stats fresh, negligible cost
                        last_stats = Instant::now();
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
            };
            if let Err(e) = run() {
                log::error!("{e:#}");
                let _ = tx.send(Msg::Error(format!("{e:#}")));
                ctx.request_repaint();
            }
        })
        .expect("spawn engine thread");
    (rx, cmd_tx)
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Tab {
    Live,
    Calibrate,
}

struct App {
    rx: Receiver<Msg>,
    cmd: Sender<Cmd>,
    ctx: egui::Context,
    device: Device,
    /// The model the engine was started with ("auto", a path, or "dummy").
    model: String,
    models: Vec<std::path::PathBuf>,
    notice: Option<(egui::Color32, String)>,
    tab: Tab,
    model_name: Option<String>,
    finals: Vec<String>,
    partial: String,
    stats: Option<Stats>,
    error: Option<String>,
    font_size: f32,
    show_debug: bool,
    calib: calibrate::Calibration,
}

/// The model chosen last (dropdown or import) is used again on the next start.
fn last_model_file() -> std::path::PathBuf {
    asr_engine::app_home().join("last_model.txt")
}

fn remember_model(model: &str) {
    let f = last_model_file();
    let _ = std::fs::create_dir_all(f.parent().unwrap());
    let abs = std::fs::canonicalize(model).unwrap_or_else(|_| model.into());
    let _ = std::fs::write(f, abs.to_string_lossy().as_bytes());
}

fn remembered_model() -> Option<String> {
    let m = std::fs::read_to_string(last_model_file()).ok()?;
    let m = m.trim().to_string();
    std::path::Path::new(&m)
        .join("model.json")
        .is_file()
        .then_some(m)
}

impl App {
    /// Restart the engine with another model.
    fn switch_model(&mut self, model: String) {
        remember_model(&model);
        let _ = self.cmd.send(Cmd::Shutdown);
        self.calib.cancel_take();
        self.model_name = None;
        self.stats = None;
        self.error = None;
        self.partial.clear();
        let (rx, cmd) = spawn_engine(model.clone(), self.device, self.ctx.clone());
        (self.rx, self.cmd, self.model) = (rx, cmd, model);
    }

    fn import_model(&mut self, zip: &std::path::Path) {
        let dir = asr_engine::install_models_dir();
        let res = std::fs::create_dir_all(&dir)
            .map_err(anyhow::Error::from)
            .and_then(|_| asr_calibration::package::import_model(zip, &dir));
        match res {
            Ok(d) => {
                self.notice = Some((
                    egui::Color32::from_rgb(60, 180, 90),
                    format!(
                        "Installed model {}; now using it.",
                        d.file_name().unwrap().to_string_lossy()
                    ),
                ));
                self.models = asr_engine::list_models();
                self.switch_model(d.to_string_lossy().into_owned());
            }
            Err(e) => {
                self.notice = Some((
                    egui::Color32::LIGHT_RED,
                    format!("Model import failed: {e:#}"),
                ))
            }
        }
    }

    fn model_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Model");
            let current = self.model_name.clone().unwrap_or_else(|| "loading…".into());
            let mut pick: Option<String> = None;
            egui::ComboBox::from_id_salt("model")
                .selected_text(current)
                .width(380.0)
                .show_ui(ui, |ui| {
                    for m in &self.models {
                        let name = m.file_name().unwrap_or_default().to_string_lossy().to_string();
                        if ui.selectable_label(false, name).clicked() {
                            pick = Some(m.to_string_lossy().into_owned());
                        }
                    }
                });
            if let Some(m) = pick {
                self.switch_model(m);
            }
            if ui.button("Import model…").on_hover_text("Install a model package (.zip) from your trainer. You can also drop the file onto this window.").clicked() {
                if let Some(f) = rfd::FileDialog::new()
                    .add_filter("Model package", &["zip"])
                    .set_title("Import model package")
                    .pick_file()
                {
                    self.import_model(&f);
                }
            }
        });
        if let Some((c, n)) = &self.notice {
            ui.colored_label(*c, n);
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let dropped: Vec<std::path::PathBuf> = ui.ctx().input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf())
                .collect()
        });
        for f in dropped
            .iter()
            .filter(|f| f.extension().is_some_and(|e| e == "zip"))
        {
            self.import_model(f);
        }

        while let Ok(m) = self.rx.try_recv() {
            match m {
                Msg::Ready(name) => self.model_name = Some(name),
                Msg::Event(e) if self.calib.is_recording() => self.calib.on_event(e),
                Msg::Event(Event::Partial(t)) => self.partial = t,
                Msg::Event(Event::Final(t)) => {
                    self.partial.clear();
                    self.finals.push(t);
                    if self.finals.len() > 200 {
                        self.finals.remove(0);
                    }
                }
                Msg::TakeDone { audio, text } => self.calib.on_take_done(audio, text),
                Msg::Stats(s) => self.stats = Some(s),
                Msg::Error(e) => self.error = Some(e),
            }
        }

        egui::Panel::top("controls").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Live, "Live captions");
                ui.selectable_value(&mut self.tab, Tab::Calibrate, "Calibration");
                ui.separator();
                ui.label("Text size");
                ui.add(egui::Slider::new(&mut self.font_size, 18.0..=96.0));
                if self.tab == Tab::Live && ui.button("Clear").clicked() {
                    self.finals.clear();
                    self.partial.clear();
                }
                ui.checkbox(&mut self.show_debug, "Latency stats");
            });
            self.model_bar(ui);
            if self.show_debug {
                match &self.stats {
                    Some(s) => ui.monospace(format!(
                        "{} | chunk {:.0} ms | compute {:.2} ms (max {:.2}) | RTF {:.4}",
                        s.name, s.chunk_ms, s.ema_ms, s.max_ms, s.rtf
                    )),
                    None if self.model_name.is_some() => ui.monospace("model ready"),
                    None => ui.monospace("loading model…"),
                };
            }
            if let Some(e) = &self.error {
                ui.colored_label(egui::Color32::LIGHT_RED, e);
            }
        });

        egui::CentralPanel::default().show(ui, |ui| match self.tab {
            Tab::Live => {
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .auto_shrink(false)
                    .show(ui, |ui| {
                        let font = egui::FontId::proportional(self.font_size);
                        for line in &self.finals {
                            ui.label(egui::RichText::new(line).font(font.clone()));
                        }
                        if !self.partial.is_empty() {
                            ui.label(
                                egui::RichText::new(&self.partial)
                                    .font(font)
                                    .color(ui.visuals().weak_text_color()),
                            );
                        }
                    });
            }
            Tab::Calibrate => {
                self.calib
                    .ui(ui, &self.cmd, self.model_name.as_deref(), self.font_size)
            }
        });
    }
}

fn main() -> eframe::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    // Usage: speak-easy [MODEL_DIR|auto|dummy] [auto|cpu|cuda|tensorrt|directml]
    //        speak-easy cuda          (device only; model = first installed)
    let mut model = "auto".to_string();
    let mut device = Device::Auto;
    for a in std::env::args().skip(1) {
        match a.parse::<Device>() {
            Ok(d) => device = d,
            Err(_) => model = a,
        }
    }
    if model == "auto" {
        if let Some(m) = remembered_model() {
            model = m;
        }
    }
    let data_root = asr_engine::data_root();
    log::info!("data folder: {}", data_root.display());
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 640.0])
            .with_title("SpeakEasy"),
        ..Default::default()
    };
    eframe::run_native(
        "SpeakEasy",
        opts,
        Box::new(move |cc| {
            let (rx, cmd) = spawn_engine(model.clone(), device, cc.egui_ctx.clone());
            Ok(Box::new(App {
                rx,
                cmd,
                ctx: cc.egui_ctx.clone(),
                device,
                model,
                models: asr_engine::list_models(),
                notice: None,
                tab: Tab::Live,
                model_name: None,
                finals: Vec::new(),
                partial: String::new(),
                stats: None,
                error: None,
                font_size: 40.0,
                show_debug: true,
                calib: calibrate::Calibration::new(data_root),
            }))
        }),
    )
}
