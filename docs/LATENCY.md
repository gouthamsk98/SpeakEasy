# Latency budget

"Zero latency" is physically impossible: the model has to *hear* a sound before it can
recognise it, so some delay is unavoidable. The aim is to make each part of the delay as
small as possible, and to measure every part.

End-to-end latency, from the moment a sound is made to the moment its text is visible:

| Stage | Today | Target | Lever |
|---|---|---|---|
| Hardware/OS audio buffer | ~5–20 ms | ≤ 10 ms | small fixed cpal buffer (we request 5 ms), PipeWire quantum, WASAPI exclusive |
| Ring buffer poll | ≤ 2 ms | ≤ 1 ms | condvar/eventfd wakeup instead of sleep |
| Feature window | 25 ms window, 10 ms hop | same | (already causal) |
| **Chunk accumulation** (algorithmic) | 80 ms (`chunk_frames=8`) | **40 ms** | smaller chunks; model trained for them |
| **Model lookahead** (algorithmic) | 0 ms (causal LSTM) | ≤ 80 ms if needed for accuracy | trade-off: lookahead helps accuracy |
| Compute per chunk | toy: 0.08 ms CUDA | < 5 ms for a 30–100M param model | FP16/INT8, TensorRT, IoBinding (state stays on GPU), CUDA graphs |
| Decode (greedy CTC) | µs | µs | |
| UI: event → pixels | ≤ 1 frame (16.7 ms @60 Hz) | ≤ 7 ms | high-refresh display, low-latency present mode |

**Rough target: under 150 ms end to end for partial text; stretch goal under 100 ms.**
People perceive under about 100 ms as instant.

The two big items are algorithmic: chunk size and lookahead. GPU speed only matters once
compute stops being negligible. Most research effort should therefore go into **models
that stay accurate with small chunks and little lookahead.**

## Measuring

- `asr --model M --device D file x.wav`: per-chunk compute (EMA and max) and RTF.
- The app's "Latency stats" toggle shows the same numbers live.
- TODO (roadmap): a loopback rig that plays a WAV through the speaker into the mic, or
  through a virtual device, and timestamps the waveform onset against the frame in which
  the text appears. That gives true end-to-end numbers, not just compute.

Known measurement: the first CUDA call costs about 100 ms, so every model is warmed up
when it loads.
