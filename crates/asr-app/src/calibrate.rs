// SPDX-License-Identifier: GPL-3.0-or-later
//! Calibration / recording tab.
//!
//! The user reads items shown on screen (words, word lists, phrases, sentences, tongue
//! twisters, reading passages, story parts), some under a recording condition (far from
//! the microphone, a long silence first, quietly, ...). Each take is scored against the text
//! (so they see how well the current model understands them) and saved as
//! audio + correct text: the training data for their personal model.
//!
//! Flow: Setup (speaker + consent) -> one item at a time (record / result / next)
//!       -> Summary. "Export dataset" packs every recording into a zip for the trainer.

use crate::Cmd;
use asr_calibration::{
    all_prompts, package, score,
    store::{self, valid_speaker_id, CONSENT_STATEMENT},
    Condition, Consent, Kind, Op, Prompt, Score, SessionWriter,
};
use asr_engine::Event;
use crossbeam_channel::Sender;
use eframe::egui::{self, Color32, RichText};
use std::path::PathBuf;
use std::time::Instant;

const GREEN: Color32 = Color32::from_rgb(60, 180, 90);
const RED: Color32 = Color32::from_rgb(225, 80, 80);
const ORANGE: Color32 = Color32::from_rgb(230, 150, 40);
const BLUE: Color32 = Color32::from_rgb(70, 140, 230);

/// Minutes of recordings to aim for: about where personal fine-tuning starts to pay off
/// reliably.
const GOAL_MINUTES: f32 = 30.0;
/// Bump when the prompt set changes (sessions are compared within a set).
const PROMPT_SET: &str = "en-v3";

enum Stage {
    Setup,
    Prompt,
    Summary,
}

struct Take {
    audio: Vec<f32>,
    text: String,
    score: Score,
}

pub struct Calibration {
    data_root: PathBuf,
    stage: Stage,
    speaker: String,
    n_prompts: usize,
    consent_ticked: bool,
    consent: Option<Consent>,
    prompts: Vec<Prompt>,
    idx: usize,
    recording: Option<Instant>,
    /// For Condition::PauseBefore: the screen says "stay quiet" until this moment.
    quiet_until: Option<Instant>,
    waiting_result: bool,
    live_finals: Vec<String>,
    live_partial: String,
    take: Option<Take>,
    writer: Option<SessionWriter>,
    status: Option<(Color32, String)>,
    last_export: Option<PathBuf>,
}

impl Calibration {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            stage: Stage::Setup,
            speaker: "user01".into(),
            n_prompts: 20,
            consent_ticked: false,
            consent: None,
            prompts: Vec::new(),
            idx: 0,
            recording: None,
            quiet_until: None,
            waiting_result: false,
            live_finals: Vec::new(),
            live_partial: String::new(),
            take: None,
            writer: None,
            status: None,
            last_export: None,
        }
    }

    pub fn is_recording(&self) -> bool {
        self.recording.is_some()
    }

    /// The engine is being restarted (model switch): abandon an in-flight take.
    pub fn cancel_take(&mut self) {
        self.recording = None;
        self.waiting_result = false;
    }

    pub fn on_event(&mut self, e: Event) {
        match e {
            Event::Partial(t) => self.live_partial = t,
            Event::Final(t) => {
                self.live_partial.clear();
                self.live_finals.push(t);
            }
        }
    }

    pub fn on_take_done(&mut self, audio: Vec<f32>, text: String) {
        self.waiting_result = false;
        let Some(prompt) = self.prompts.get(self.idx) else {
            return;
        };
        let score = score(&prompt.text, &text);
        self.take = Some(Take { audio, text, score });
    }

    fn start_recording(&mut self, cmd: &Sender<Cmd>) {
        self.take = None;
        self.live_finals.clear();
        self.live_partial.clear();
        let now = Instant::now();
        self.recording = Some(now);
        let quiet = self
            .prompts
            .get(self.idx)
            .map_or(0.0, |p| p.condition.quiet_before_secs());
        self.quiet_until = (quiet > 0.0).then(|| now + std::time::Duration::from_secs_f32(quiet));
        let _ = cmd.send(Cmd::StartTake);
    }

    fn stop_recording(&mut self, cmd: &Sender<Cmd>) {
        self.recording = None;
        self.quiet_until = None;
        self.waiting_result = true;
        let _ = cmd.send(Cmd::StopTake);
    }

    fn accept_take(&mut self, model: &str) {
        let Some(take) = self.take.take() else { return };
        let res = (|| -> anyhow::Result<()> {
            if self.writer.is_none() {
                let consent = self.consent.as_ref().expect("consent before session");
                self.writer = Some(SessionWriter::new(
                    &self.data_root,
                    consent,
                    model,
                    PROMPT_SET,
                )?);
            }
            let w = self.writer.as_mut().unwrap();
            let p = &self.prompts[self.idx];
            w.add_take(
                self.idx,
                &p.text,
                &take.audio,
                &take.text,
                take.score,
                p.condition,
            )
        })();
        if let Err(e) = res {
            self.status = Some((RED, format!("could not save: {e:#}")));
        }
        self.next_prompt();
    }

    fn next_prompt(&mut self) {
        self.take = None;
        self.live_finals.clear();
        self.live_partial.clear();
        self.idx += 1;
        if self.idx >= self.prompts.len() {
            self.stage = Stage::Summary;
        }
    }

    fn start_session(&mut self) {
        self.status = None;
        let consent = match Consent::load(&self.data_root, &self.speaker) {
            Some(c) => c,
            None => match Consent::record(&self.data_root, &self.speaker) {
                Ok(c) => c,
                Err(e) => {
                    self.status = Some((RED, format!("could not store consent: {e:#}")));
                    return;
                }
            },
        };
        self.consent = Some(consent);
        self.prompts = store::next_session(&self.data_root, &self.speaker, self.n_prompts);
        self.idx = 0;
        self.writer = None;
        self.take = None;
        self.stage = Stage::Prompt;
    }

    fn export(&mut self) {
        let out = dirs::download_dir()
            .unwrap_or_else(asr_engine::app_home)
            .join(package::dataset_file_name(&self.speaker));
        match package::export_dataset(&self.data_root, &self.speaker, &out) {
            Ok(r) => {
                self.status = Some((
                    GREEN,
                    format!(
                        "Exported {} recordings ({:.1} min) to {}. Send this file to your trainer.",
                        r.utterances,
                        r.total_seconds / 60.0,
                        r.path.display()
                    ),
                ));
                self.last_export = Some(r.path);
            }
            Err(e) => self.status = Some((RED, format!("export failed: {e:#}"))),
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, cmd: &Sender<Cmd>, model: Option<&str>, font: f32) {
        match self.stage {
            Stage::Setup => self.ui_setup(ui, model),
            Stage::Prompt => self.ui_prompt(ui, cmd, model.unwrap_or("unknown"), font),
            Stage::Summary => self.ui_summary(ui),
        }
        ui.add_space(8.0);
        if let Some((col, s)) = &self.status {
            ui.colored_label(*col, s);
        }
        if let Some(p) = &self.last_export {
            if ui.button("Open folder").clicked() {
                open_folder(p.parent().unwrap_or(p));
            }
        }
    }

    fn export_button(&mut self, ui: &mut egui::Ui) {
        let has_data = !store::sessions(&self.data_root, &self.speaker).is_empty();
        if ui
            .add_enabled(
                has_data,
                egui::Button::new(RichText::new("Export dataset").size(18.0)),
            )
            .on_hover_text("Pack all your recordings into one zip file to send for training")
            .clicked()
        {
            self.export();
        }
    }

    fn ui_setup(&mut self, ui: &mut egui::Ui, model: Option<&str>) {
        ui.heading("Record your voice");
        ui.label(
            "Read what is shown on screen: words, short phrases, sentences, tongue twisters \
             and short stories. Some items ask you to say them in a special way (far from \
             the microphone, after a long silence, quietly...), because in real life you \
             don't always sit still in a quiet room. You'll see what the app understood, word \
             by word. Every recording is saved on this computer with the correct text, so a \
             personal model can be trained on your voice. The more you record, the better it \
             gets.",
        );
        ui.add_space(8.0);
        egui::Grid::new("setup")
            .num_columns(2)
            .spacing([12.0, 8.0])
            .show(ui, |ui| {
                ui.label("Speaker ID");
                ui.text_edit_singleline(&mut self.speaker);
                ui.end_row();
                ui.label("Items this session");
                ui.add(egui::Slider::new(&mut self.n_prompts, 5..=50))
                    .on_hover_text("Each session mixes single words, word lists, short phrases, sentences, tongue twisters, a reading passage and a part of a story. About 20 items take 10 minutes.");
                ui.end_row();
            });
        let valid = valid_speaker_id(&self.speaker);
        if !valid {
            ui.colored_label(RED, "Speaker ID: letters, digits, - and _ only (max 32).");
        }
        ui.add_space(8.0);

        let existing = valid
            .then(|| Consent::load(&self.data_root, &self.speaker))
            .flatten();
        match &existing {
            Some(c) => {
                ui.label(format!("Consent on file since {}.", &c.accepted_at[..10]));
            }
            None => {
                ui.group(|ui| {
                    ui.label(RichText::new(CONSENT_STATEMENT).italics());
                    ui.checkbox(&mut self.consent_ticked, "I agree");
                });
            }
        }
        if valid {
            let sessions = store::sessions(&self.data_root, &self.speaker);
            let takes: usize = sessions.iter().map(|(_, s)| s.takes.len()).sum();
            let secs: f32 = sessions
                .iter()
                .flat_map(|(_, s)| s.takes.iter().map(|t| t.seconds))
                .sum::<f32>()
                + 0.0; // an empty f32 sum is -0.0
            ui.label(format!(
                "Recorded so far: {takes} recordings, {:.1} minutes, in {} sessions.",
                secs / 60.0,
                sessions.len(),
            ));
            goal_bar(ui, secs);
            // Coverage per kind: how much of each list this speaker has read at least once.
            let counts = store::recorded_counts(&self.data_root, &self.speaker);
            let all = all_prompts();
            let parts: Vec<String> = Kind::ALL
                .iter()
                .map(|k| {
                    let of_kind: Vec<_> = all.iter().filter(|p| p.kind == *k).collect();
                    let done = of_kind
                        .iter()
                        .filter(|p| counts.contains_key(&p.text))
                        .count();
                    format!("{} {done}/{}", plural(*k), of_kind.len())
                })
                .collect();
            ui.label(RichText::new(format!("Covered: {}", parts.join("  ·  "))).weak());
            let cc = store::condition_counts(&self.data_root, &self.speaker);
            let conds: Vec<String> = Condition::SPECIAL
                .iter()
                .map(|c| format!("{} {}", c.short(), cc.get(c).copied().unwrap_or(0)))
                .collect();
            ui.label(RichText::new(format!("Special recordings: {}", conds.join("  ·  "))).weak());
        }
        ui.add_space(12.0);
        let ready = valid && model.is_some() && (existing.is_some() || self.consent_ticked);
        if model.is_none() {
            ui.label("Waiting for the speech model to load…");
        }
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    ready,
                    egui::Button::new(RichText::new("Start recording").size(20.0)),
                )
                .clicked()
            {
                self.start_session();
            }
            if valid {
                self.export_button(ui);
            }
        });
        if valid {
            history(ui, &self.data_root, &self.speaker);
        }
    }

    fn ui_prompt(&mut self, ui: &mut egui::Ui, cmd: &Sender<Cmd>, model: &str, font: f32) {
        let n = self.prompts.len();
        ui.horizontal(|ui| {
            ui.label(format!("Item {} of {n}", self.idx + 1));
            ui.add(egui::ProgressBar::new(self.idx as f32 / n as f32).desired_width(200.0));
            if ui.small_button("End session").clicked() {
                if self.recording.is_some() {
                    self.stop_recording(cmd);
                }
                self.stage = Stage::Summary;
            }
        });
        ui.add_space(12.0);
        let prompt = self.prompts[self.idx].clone();
        let heading = match &prompt.context {
            Some(c) => format!("{}: {c}. Please read aloud:", prompt.kind.label()),
            None => format!("{}. Please read aloud:", prompt.kind.label()),
        };
        ui.label(RichText::new(heading).weak());
        if let Some(h) = prompt.kind.hint() {
            ui.label(RichText::new(h).weak().italics());
        }
        if prompt.condition != Condition::Normal {
            condition_banner(ui, prompt.condition);
        }
        // Short items big, long passages smaller so they fit on screen.
        let size = match prompt.kind {
            Kind::Word => font * 1.4,
            Kind::Phrase => font * 1.2,
            Kind::Sentence => font,
            Kind::WordList => font * 0.9,
            Kind::TongueTwister => font * 0.8,
            Kind::Paragraph | Kind::Story => (font * 0.6).max(18.0),
        };
        if prompt.condition == Condition::PauseInside {
            let (a, b) = prompt.pause_split();
            ui.label(RichText::new(a).size(size).strong());
            ui.label(
                RichText::new("( pause here: count to three silently )")
                    .size((size * 0.6).max(14.0))
                    .color(BLUE)
                    .italics(),
            );
            ui.label(RichText::new(b).size(size).strong());
        } else {
            ui.label(RichText::new(&prompt.text).size(size).strong());
        }
        ui.add_space(16.0);

        let (space, enter) = ui.ctx().input(|i| {
            (
                i.key_pressed(egui::Key::Space),
                i.key_pressed(egui::Key::Enter),
            )
        });

        if let Some(t0) = self.recording {
            if let Some(until) = self.quiet_until {
                let left = until
                    .saturating_duration_since(Instant::now())
                    .as_secs_f32();
                let (txt, col) = if left > 0.0 {
                    (format!("Stay quiet...  {}", left.ceil() as u32), ORANGE)
                } else {
                    ("NOW SPEAK".to_string(), GREEN)
                };
                ui.label(RichText::new(txt).size(font * 1.2).color(col).strong());
            }
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("Recording {:.1}s", t0.elapsed().as_secs_f32()))
                        .color(RED)
                        .size(20.0),
                );
                if ui
                    .button(RichText::new("Stop  (Space)").size(20.0))
                    .clicked()
                    || space
                {
                    self.stop_recording(cmd);
                }
            });
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(100));
            let mut heard = self.live_finals.join(" ");
            if !self.live_partial.is_empty() {
                if !heard.is_empty() {
                    heard.push(' ');
                }
                heard.push_str(&self.live_partial);
            }
            ui.label(
                RichText::new(heard)
                    .size(font * 0.8)
                    .color(ui.visuals().weak_text_color()),
            );
        } else if self.waiting_result {
            ui.spinner();
        } else if let Some(take) = &self.take {
            ui.label(RichText::new("What the app heard:").weak());
            word_diff(ui, &take.score, font * 0.8);
            ui.add_space(6.0);
            ui.label(format!(
                "Word errors: {}/{} ({:.0}%)   ·   Character error rate: {:.0}%   ·   {:.1}s",
                take.score.word_errors,
                take.score.ref_words,
                take.score.wer() * 100.0,
                take.score.cer() * 100.0,
                take.audio.len() as f32 / 16_000.0
            ));
            legend(ui);
            ui.label(
                RichText::new(
                    "Mistakes are fine: the recording is saved with the correct text, \
                     and that is exactly what the personal model learns from.",
                )
                .weak(),
            );
            ui.add_space(10.0);
            let too_short = take.audio.len() < 16_000 / 4;
            ui.horizontal(|ui| {
                if ui
                    .button(RichText::new("Record again").size(18.0))
                    .clicked()
                    || space
                {
                    self.start_recording(cmd);
                }
                let next = ui.add_enabled(
                    !too_short,
                    egui::Button::new(RichText::new("Save & next  (Enter)").size(18.0)),
                );
                if next.clicked() || (enter && !too_short) {
                    self.accept_take(model);
                }
            });
            if too_short {
                ui.colored_label(
                    ORANGE,
                    "That recording was very short; please record again.",
                );
            }
        } else {
            ui.horizontal(|ui| {
                if ui
                    .button(RichText::new("Record  (Space)").size(22.0))
                    .clicked()
                    || space
                {
                    self.start_recording(cmd);
                }
                if ui.button("Skip").clicked() {
                    self.next_prompt();
                }
            });
            ui.label(
                RichText::new(
                    "Press Record, read it at your own pace, then press Stop. \
                     If you said something different, use Record again.",
                )
                .weak(),
            );
        }
    }

    fn ui_summary(&mut self, ui: &mut egui::Ui) {
        ui.heading("Session results");
        match &self.writer {
            None => {
                ui.label("Nothing was saved in this session.");
            }
            Some(w) => {
                let s = w.summary();
                ui.label(
                    RichText::new(format!(
                        "This session: word error {:.0}%  ·  character error {:.0}%  ({} items, model {})",
                        s.wer() * 100.0,
                        s.cer() * 100.0,
                        s.takes.len(),
                        s.model
                    ))
                    .size(18.0),
                );
                ui.add_space(6.0);
                egui::ScrollArea::vertical()
                    .max_height(260.0)
                    .show(ui, |ui| {
                        egui::Grid::new("summary")
                            .striped(true)
                            .num_columns(3)
                            .spacing([16.0, 6.0])
                            .show(ui, |ui| {
                                ui.strong("Item");
                                ui.strong("Heard");
                                ui.strong("Word error");
                                ui.end_row();
                                for t in &s.takes {
                                    if t.condition == Condition::Normal {
                                        ui.label(&t.prompt);
                                    } else {
                                        ui.label(format!(
                                            "{}  [{}]",
                                            t.prompt,
                                            t.condition.short()
                                        ));
                                    }
                                    word_diff(ui, &t.score, 14.0);
                                    ui.label(format!("{:.0}%", t.score.wer() * 100.0));
                                    ui.end_row();
                                }
                            });
                    });
                ui.label(RichText::new(format!("Saved to {}", w.dir.display())).weak());
                let total: f32 = store::sessions(&self.data_root, &self.speaker)
                    .iter()
                    .flat_map(|(_, s)| s.takes.iter().map(|t| t.seconds))
                    .sum::<f32>()
                    + 0.0;
                goal_bar(ui, total);
            }
        }
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button(RichText::new("Record more").size(18.0)).clicked() {
                self.writer = None;
                self.start_session();
            }
            self.export_button(ui);
            if ui.button("Done").clicked() {
                self.writer = None;
                self.stage = Stage::Setup;
            }
        });
        history(ui, &self.data_root, &self.speaker);
    }
}

/// Every past session with the model used: shows whether a new model helped.
fn history(ui: &mut egui::Ui, data_root: &std::path::Path, speaker: &str) {
    let sessions = store::sessions(data_root, speaker);
    if sessions.is_empty() {
        return;
    }
    ui.add_space(10.0);
    egui::CollapsingHeader::new(format!("Past sessions ({})", sessions.len())).show(ui, |ui| {
        egui::Grid::new("history")
            .striped(true)
            .num_columns(4)
            .spacing([16.0, 4.0])
            .show(ui, |ui| {
                ui.strong("When");
                ui.strong("Model");
                ui.strong("Items");
                ui.strong("Word error");
                ui.end_row();
                for (_, s) in sessions.iter().rev() {
                    ui.label(
                        s.started_at
                            .get(..16)
                            .unwrap_or(&s.started_at)
                            .replace('T', " "),
                    );
                    ui.label(&s.model);
                    ui.label(s.takes.len().to_string());
                    ui.label(format!("{:.0}%", s.wer() * 100.0));
                    ui.end_row();
                }
            });
    });
}

fn open_folder(p: &std::path::Path) {
    let prog = if cfg!(windows) {
        "explorer"
    } else {
        "xdg-open"
    };
    let _ = std::process::Command::new(prog).arg(p).spawn();
}

fn word_diff(ui: &mut egui::Ui, s: &Score, size: f32) {
    ui.horizontal_wrapped(|ui| {
        if s.ops.is_empty() {
            ui.label(RichText::new("(nothing recognised)").size(size).color(RED));
        }
        for op in &s.ops {
            match op {
                Op::Correct(w) => {
                    ui.label(RichText::new(w).size(size).color(GREEN));
                }
                Op::Sub { expected, heard } => {
                    ui.label(RichText::new(heard).size(size).color(RED))
                        .on_hover_text(format!("expected \"{expected}\""));
                    ui.label(
                        RichText::new(format!("({expected})"))
                            .size(size * 0.55)
                            .weak(),
                    );
                }
                Op::Del(w) => {
                    ui.label(RichText::new(w).size(size).strikethrough().weak())
                        .on_hover_text("not recognised");
                }
                Op::Ins(w) => {
                    ui.label(RichText::new(w).size(size).italics().color(ORANGE))
                        .on_hover_text("extra word");
                }
            }
        }
    });
}

fn legend(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("correct").color(GREEN));
        ui.label(RichText::new("misheard (expected)").color(RED));
        ui.label(RichText::new("missed").strikethrough().weak());
        ui.label(RichText::new("extra").italics().color(ORANGE));
    });
}

/// Progress towards [`GOAL_MINUTES`] of recordings, with encouragement.
fn goal_bar(ui: &mut egui::Ui, secs: f32) {
    let min = secs / 60.0;
    let frac = (min / GOAL_MINUTES).min(1.0);
    ui.add(
        egui::ProgressBar::new(frac)
            .desired_width(320.0)
            .text(format!("{min:.1} of {GOAL_MINUTES:.0} minutes")),
    );
    let msg = if min >= GOAL_MINUTES {
        "Goal reached: great work! More recordings still help, especially special ones."
    } else if min >= GOAL_MINUTES / 2.0 {
        "Over halfway. Each session now makes your model noticeably better."
    } else {
        "Your personal model improves most with more recordings: aim for a session or two a day."
    };
    ui.label(RichText::new(msg).weak());
}

fn condition_banner(ui: &mut egui::Ui, c: Condition) {
    egui::Frame::group(ui.style())
        .fill(BLUE.gamma_multiply(0.18))
        .stroke(egui::Stroke::new(1.5, BLUE))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new(c.label()).size(20.0).strong().color(BLUE));
            ui.label(RichText::new(c.instruction()).size(17.0));
        });
    ui.add_space(6.0);
}

fn plural(k: Kind) -> &'static str {
    match k {
        Kind::Word => "words",
        Kind::WordList => "word lists",
        Kind::Paragraph => "reading passages",
        Kind::Phrase => "phrases",
        Kind::Sentence => "sentences",
        Kind::TongueTwister => "tongue twisters",
        Kind::Story => "story parts",
    }
}
