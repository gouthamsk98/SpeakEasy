// SPDX-License-Identifier: GPL-3.0-or-later
//! Microphone capture.
//!
//! Design for latency:
//! * cpal (ALSA/PulseAudio/PipeWire on Linux, WASAPI on Windows) with the smallest
//!   buffer the device accepts.
//! * The audio callback does **no allocation and no locking**: it downmixes,
//!   resamples to 16 kHz and pushes into an `rtrb` SPSC ring buffer.
//! * The inference thread pops whatever is available every few milliseconds.

mod resample;

pub use resample::StreamingResampler;

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// A running capture. Drop it to stop the microphone.
pub struct Capture {
    _stream: cpal::Stream,
    pub consumer: rtrb::Consumer<f32>,
    pub device_name: String,
    pub device_sample_rate: u32,
}

/// List input device names (for a settings dropdown).
pub fn list_input_devices() -> Result<Vec<String>> {
    let host = cpal::default_host();
    Ok(host
        .input_devices()?
        .filter_map(|d| device_name(&d))
        .collect())
}

fn device_name(d: &cpal::Device) -> Option<String> {
    d.description().ok().map(|desc| desc.name().to_string())
}

/// Start capturing from `device` (or the default input). Samples arrive in
/// `Capture::consumer` as 16 kHz mono f32.
pub fn start_capture(device: Option<&str>) -> Result<Capture> {
    let host = cpal::default_host();
    let dev = match device {
        Some(want) => host
            .input_devices()?
            .find(|d| device_name(d).as_deref() == Some(want))
            .ok_or_else(|| anyhow!("input device {want:?} not found"))?,
        None => host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device"))?,
    };
    let name = device_name(&dev).unwrap_or_else(|| "<unknown>".into());

    let supported = dev.default_input_config().context("query input config")?;
    let channels = supported.channels() as usize;
    let sr: u32 = supported.sample_rate();
    let mut config: cpal::StreamConfig = supported.config();
    // Ask for a small hardware buffer: ~5 ms. Falls back to default if refused.
    config.buffer_size = cpal::BufferSize::Fixed((sr / 200).max(64));

    let format = supported.sample_format();
    let (stream, consumer) = match build_any(&dev, &config, format, sr, channels) {
        Ok(v) => v,
        Err(e) => {
            log::warn!("small fixed buffer refused ({e}); using device default buffer");
            config.buffer_size = cpal::BufferSize::Default;
            build_any(&dev, &config, format, sr, channels)?
        }
    };
    stream.play()?;
    log::info!("capturing from {name:?} @ {sr} Hz x{channels} -> 16 kHz mono");
    Ok(Capture {
        _stream: stream,
        consumer,
        device_name: name,
        device_sample_rate: sr,
    })
}

type Built = (cpal::Stream, rtrb::Consumer<f32>);

fn build_any(
    dev: &cpal::Device,
    config: &cpal::StreamConfig,
    format: cpal::SampleFormat,
    sr: u32,
    channels: usize,
) -> Result<Built> {
    use cpal::SampleFormat as F;
    match format {
        F::F32 => build_stream::<f32>(dev, config, sr, channels),
        F::I16 => build_stream::<i16>(dev, config, sr, channels),
        F::I32 => build_stream::<i32>(dev, config, sr, channels),
        F::U16 => build_stream::<u16>(dev, config, sr, channels),
        other => Err(anyhow!("unsupported device sample format {other:?}")),
    }
}

fn build_stream<T>(
    dev: &cpal::Device,
    config: &cpal::StreamConfig,
    sr: u32,
    channels: usize,
) -> Result<Built>
where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    use cpal::Sample;
    // 2 s of headroom; if inference stalls longer than that we drop audio, never block.
    let (mut producer, consumer) = rtrb::RingBuffer::<f32>::new(TARGET_SAMPLE_RATE as usize * 2);
    let mut resampler = StreamingResampler::new(sr, TARGET_SAMPLE_RATE);
    let mut mono = Vec::<f32>::with_capacity(8192);
    let mut out = Vec::<f32>::with_capacity(8192);
    let stream = dev.build_input_stream(
        *config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            mono.clear();
            mono.extend(
                data.chunks(channels)
                    .map(|c| c.iter().map(|&x| f32::from_sample(x)).sum::<f32>() / channels as f32),
            );
            out.clear();
            resampler.process(&mono, &mut out);
            for &s in &out {
                if producer.push(s).is_err() {
                    break; // consumer is behind: drop audio rather than block this thread
                }
            }
        },
        |e| log::error!("audio stream error: {e}"),
        None,
    )?;
    Ok((stream, consumer))
}

/// Load a WAV file as 16 kHz mono f32 (for the CLI / benchmarks / tests).
pub fn read_wav_16k(path: &std::path::Path) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("open {path:?}"))?;
    let spec = r.spec();
    let ch = spec.channels as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    let mono: Vec<f32> = interleaved
        .chunks(ch)
        .map(|c| c.iter().sum::<f32>() / ch as f32)
        .collect();
    let mut out = Vec::with_capacity(mono.len());
    StreamingResampler::new(spec.sample_rate, TARGET_SAMPLE_RATE).process(&mono, &mut out);
    Ok(out)
}
