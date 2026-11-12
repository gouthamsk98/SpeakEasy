// SPDX-License-Identifier: GPL-3.0-or-later
//! Desktop captioning app.
//!
//! Threads:
//!  * audio callback (cpal)  -> lock-free ring buffer
//!  * engine worker          -> pops audio every 2 ms, runs features + model, sends events
//!  * UI (egui, GPU via wgpu) -> repaints immediately when an event arrives
//!
//! The worker calls `ctx.request_repaint()` on every event, so text hits the screen on
//! the next vsync instead of waiting for input or a timer.

#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use anyhow::Result;
use asr_engine::{load_recognizer, Device, EndpointConfig, Event, Pipeline};
use crossbeam_channel::{unbounded, Receiver};
use eframe::egui;
use std::time::Duration;

struct Stats {
    name: String,
    ema_ms: f64,
    max_ms: f64,
    rtf: f64,
    chunk_ms: f64,
}

enum Msg {
    Event(Event),
    Stats(Stats),
    Error(String),
}

fn spawn_engine(model: String, device: Device, ctx: egui::Context) -> Receiver<Msg> {
    let (tx, rx) = unbounded();
    std::thread::Builder::new()
        .name("asr-engine".into())
        .spawn(move || {
            let run = || -> Result<()> {
                let mut p =
                    Pipeline::new(load_recognizer(&model, device)?, EndpointConfig::default());
                let mut cap = asr_audio::start_capture(None)?;
                let mut buf = Vec::with_capacity(4096);
                let mut ev = Vec::new();
                let mut last_stats = std::time::Instant::now();
                loop {
                    buf.clear();
                    while let Ok(s) = cap.consumer.pop() {
                        buf.push(s);
                    }
                    if !buf.is_empty() {
                        p.push_audio(&buf, &mut ev)?;
                        if !ev.is_empty() {
                            for e in ev.drain(..) {
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
                        last_stats = std::time::Instant::now();
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
    rx
}

struct App {
    rx: Receiver<Msg>,
    finals: Vec<String>,
    partial: String,
    stats: Option<Stats>,
    error: Option<String>,
    font_size: f32,
    show_debug: bool,
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        while let Ok(m) = self.rx.try_recv() {
            match m {
                Msg::Event(Event::Partial(t)) => self.partial = t,
                Msg::Event(Event::Final(t)) => {
                    self.partial.clear();
                    self.finals.push(t);
                    if self.finals.len() > 200 {
                        self.finals.remove(0);
                    }
                }
                Msg::Stats(s) => self.stats = Some(s),
                Msg::Error(e) => self.error = Some(e),
            }
        }

        egui::Panel::top("controls").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Text size");
                ui.add(egui::Slider::new(&mut self.font_size, 18.0..=96.0));
                if ui.button("Clear").clicked() {
                    self.finals.clear();
                    self.partial.clear();
                }
                ui.checkbox(&mut self.show_debug, "Latency stats");
            });
            if self.show_debug {
                if let Some(s) = &self.stats {
                    ui.monospace(format!(
                        "{} | chunk {:.0} ms | compute {:.2} ms (max {:.2}) | RTF {:.4}",
                        s.name, s.chunk_ms, s.ema_ms, s.max_ms, s.rtf
                    ));
                }
            }
            if let Some(e) = &self.error {
                ui.colored_label(egui::Color32::LIGHT_RED, e);
            }
        });

        egui::CentralPanel::default().show(ui, |ui| {
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
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 500.0])
            .with_title("SpeakEasy"),
        ..Default::default()
    };
    eframe::run_native(
        "SpeakEasy",
        opts,
        Box::new(move |cc| {
            let rx = spawn_engine(model, device, cc.egui_ctx.clone());
            Ok(Box::new(App {
                rx,
                finals: Vec::new(),
                partial: String::new(),
                stats: None,
                error: None,
                font_size: 40.0,
                show_debug: true,
            }))
        }),
    )
}
