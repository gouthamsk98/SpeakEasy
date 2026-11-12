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
