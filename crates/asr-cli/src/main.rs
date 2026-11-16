// SPDX-License-Identifier: GPL-3.0-or-later
use anyhow::Result;
use asr_engine::{load_recognizer, Device, EndpointConfig, Event, Pipeline};
use clap::{Parser, Subcommand};
use std::{
    io::Write,
    path::PathBuf,
    time::{Duration, Instant},
};

#[derive(Parser)]
#[command(
    name = "asr",
    about = "Speech recognition for glossectomy speech — headless tools"
)]
struct Cli {
    /// Model directory (model.onnx + model.json), "auto" (first in models/) or "dummy"
    #[arg(long, short, default_value = "auto", global = true)]
    model: String,
    /// auto | cpu | cuda | tensorrt | directml
    #[arg(long, short, default_value = "auto", global = true)]
    device: Device,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List microphones
    Devices,
    /// Live transcription from the microphone in the terminal
    Mic {
        #[arg(long)]
        input: Option<String>,
    },
    /// Score a model on a manifest (JSONL with "audio" and "text"), streaming exactly
    /// like the app. Prints per-utterance and overall word/character error rates.
    Eval {
        manifest: PathBuf,
        /// Feed audio in blocks of this many samples, like the live app (160 = 10 ms).
        /// 0 = whole file at once.
        #[arg(long, default_value_t = 160)]
        block: usize,
        /// Also write per-utterance results as JSON here
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Export a speaker's recordings as a dataset zip (what the app's button does)
    DatasetExport {
        #[arg(long)]
        speaker: String,
        /// Output zip (default: Downloads folder)
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Add existing recordings (a manifest JSONL with "audio" + "text") to a speaker's data,
    /// scored by the model exactly like app recordings. Needs the speaker's consent on file.
    DatasetAdd {
        #[arg(long)]
        speaker: String,
        manifest: PathBuf,
    },
    /// Install a model package zip (what the app's "Import model" does)
    ModelImport { zip: PathBuf },
    /// List installed models and where data is stored
    Models,
    /// Stream a WAV file through the pipeline as if it were live, report latency
    File {
        wav: PathBuf,
        /// Feed in 10 ms blocks at real-time pace (otherwise as fast as possible)
        #[arg(long)]
        realtime: bool,
    },
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Devices => {
            for d in asr_audio::list_input_devices()? {
                println!("{d}");
            }
        }
        Cmd::Mic { input } => {
            let mut p = Pipeline::new(
                load_recognizer(&cli.model, cli.device)?,
                EndpointConfig::default(),
            );
            let mut cap = asr_audio::start_capture(input.as_deref())?;
            eprintln!("listening with {} — Ctrl+C to stop", p.recognizer_name());
            let mut buf = Vec::with_capacity(4096);
            let mut ev = Vec::new();
            loop {
                buf.clear();
                while let Ok(s) = cap.consumer.pop() {
                    buf.push(s);
                }
                if !buf.is_empty() {
                    p.push_audio(&buf, &mut ev)?;
                    print_events(&mut ev);
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        Cmd::Eval {
            manifest,
            json,
            block,
        } => {
            let mut p = Pipeline::new(
                load_recognizer(&cli.model, cli.device)?,
                EndpointConfig::default(),
            );
            p.set_endpointing(false); // like an app take: one utterance per file
            let base = manifest.parent().unwrap_or(std::path::Path::new("."));
            let (mut we, mut wn, mut ce, mut cn) = (0, 0, 0, 0);
            let mut rows = Vec::new();
            for line in std::fs::read_to_string(&manifest)?
                .lines()
                .filter(|l| !l.trim().is_empty())
            {
                let r: serde_json::Value = serde_json::from_str(line)?;
                let (Some(audio), Some(text)) = (r["audio"].as_str(), r["text"].as_str()) else {
                    anyhow::bail!("manifest rows need \"audio\" and \"text\"");
                };
                let path = base.join(audio);
                let wav = asr_audio::read_wav_16k(&path)?;
                let mut ev = Vec::new();
                for b in wav.chunks(if block == 0 { wav.len().max(1) } else { block }) {
                    p.push_audio(b, &mut ev)?;
                }
                p.flush(&mut ev)?;
                let hyp: Vec<String> = ev
                    .into_iter()
                    .filter_map(|e| {
                        if let Event::Final(t) = e {
                            Some(t)
                        } else {
                            None
                        }
                    })
                    .collect();
                let hyp = hyp.join(" ");
                let s = asr_calibration::score(text, &hyp);
                println!("{:5.1}%  {}\n        heard: {}", s.wer() * 100.0, text, hyp);
                (we, wn, ce, cn) = (
                    we + s.word_errors,
                    wn + s.ref_words,
                    ce + s.char_errors,
                    cn + s.ref_chars,
                );
                rows.push(serde_json::json!({"audio": audio, "text": text, "heard": hyp, "wer": s.wer(), "cer": s.cer()}));
            }
            let (wer, cer) = (we as f64 / wn.max(1) as f64, ce as f64 / cn.max(1) as f64);
            println!(
                "\n{}: {} utterances | WER {:.2}% | CER {:.2}%",
                p.recognizer_name(),
                rows.len(),
                wer * 100.0,
                cer * 100.0
            );
            if let Some(j) = json {
                std::fs::write(
                    j,
                    serde_json::to_vec_pretty(&serde_json::json!({
                        "model": p.recognizer_name(), "wer": wer, "cer": cer, "utterances": rows
                    }))?,
                )?;
            }
        }
        Cmd::DatasetExport { speaker, out } => {
            let out = out.unwrap_or_else(|| {
                dirs::download_dir()
                    .unwrap_or_else(asr_engine::app_home)
                    .join(asr_calibration::package::dataset_file_name(&speaker))
            });
            let r =
                asr_calibration::package::export_dataset(&asr_engine::data_root(), &speaker, &out)?;
            println!(
                "{} utterances, {:.1} min -> {}",
                r.utterances,
                r.total_seconds / 60.0,
                r.path.display()
            );
        }
        Cmd::DatasetAdd { speaker, manifest } => {
            let root = asr_engine::data_root();
            let consent = asr_calibration::Consent::load(&root, &speaker).ok_or_else(|| {
                anyhow::anyhow!(
                    "no consent on file for {speaker:?}: record consent in the app first"
                )
            })?;
            let mut p = Pipeline::new(
                load_recognizer(&cli.model, cli.device)?,
                EndpointConfig::default(),
            );
            p.set_endpointing(false); // like an app take: one utterance per file
            let mut w = asr_calibration::SessionWriter::new(
                &root,
                &consent,
                &p.recognizer_name(),
                "imported",
            )?;
            let base = manifest.parent().unwrap_or(std::path::Path::new("."));
            let mut n = 0;
            for line in std::fs::read_to_string(&manifest)?
                .lines()
                .filter(|l| !l.trim().is_empty())
            {
                let r: serde_json::Value = serde_json::from_str(line)?;
                let (Some(audio), Some(text)) = (r["audio"].as_str(), r["text"].as_str()) else {
                    anyhow::bail!("manifest rows need \"audio\" and \"text\"");
                };
                let wav = asr_audio::read_wav_16k(&base.join(audio))?;
                let mut ev = Vec::new();
                p.push_audio(&wav, &mut ev)?;
                p.flush(&mut ev)?;
                let hyp: Vec<String> = ev
                    .into_iter()
                    .filter_map(|e| {
                        if let Event::Final(t) = e {
                            Some(t)
                        } else {
                            None
                        }
                    })
                    .collect();
                let hyp = hyp.join(" ");
                w.add_take(
                    n,
                    text,
                    &wav,
                    &hyp,
                    asr_calibration::score(text, &hyp),
                    asr_calibration::Condition::Normal,
                )?;
                n += 1;
            }
            let s = w.summary();
            println!(
                "added {n} recordings to {} (WER {:.1}%)",
                w.dir.display(),
                s.wer() * 100.0
            );
        }
        Cmd::ModelImport { zip } => {
            let dir = asr_engine::install_models_dir();
            std::fs::create_dir_all(&dir)?;
            let d = asr_calibration::package::import_model(&zip, &dir)?;
            println!("installed {}", d.display());
        }
        Cmd::Models => {
            println!("data:   {}", asr_engine::data_root().display());
            println!("models: {}", asr_engine::install_models_dir().display());
            for m in asr_engine::list_models() {
                println!("  {}", m.display());
            }
        }
        Cmd::File { wav, realtime } => {
            let audio = asr_audio::read_wav_16k(&wav)?;
            let mut p = Pipeline::new(
                load_recognizer(&cli.model, cli.device)?,
                EndpointConfig::default(),
            );
            let mut ev = Vec::new();
            let t0 = Instant::now();
            for (i, block) in audio.chunks(160).enumerate() {
                if realtime {
                    let due = Duration::from_millis(10 * i as u64);
                    if let Some(wait) = due.checked_sub(t0.elapsed()) {
                        std::thread::sleep(wait);
                    }
                }
                p.push_audio(block, &mut ev)?;
                print_events(&mut ev);
            }
            p.flush(&mut ev)?;
            print_events(&mut ev);
            let m = &p.metrics;
            eprintln!(
                "\n{}: {} chunks of {:.0} ms | compute ema {:.2} ms, max {:.2} ms | RTF {:.4} | wall {:.2}s for {:.2}s audio",
                p.recognizer_name(),
                m.chunks,
                m.chunk_audio.as_secs_f64() * 1e3,
                m.ema_compute_ms,
                m.max_compute.as_secs_f64() * 1e3,
                m.rtf(),
                t0.elapsed().as_secs_f64(),
                audio.len() as f64 / 16_000.0
            );
        }
    }
    Ok(())
}

fn print_events(ev: &mut Vec<Event>) {
    let mut out = std::io::stdout().lock();
    for e in ev.drain(..) {
        match e {
            Event::Partial(t) => write!(out, "\r\x1b[2K\x1b[90m{t}\x1b[0m").ok(),
            Event::Final(t) => writeln!(out, "\r\x1b[2K{t}").ok(),
        };
    }
    out.flush().ok();
}
