# Roadmap

## Phase 0: foundation ✅
- [x] Rust workspace: features, audio, engine, app, CLI
- [x] ONNX Runtime backend with CUDA / TensorRT / DirectML / CPU, warm-up, strict device choice
- [x] End-to-end verified with a toy contract model on CPU and CUDA
- [x] Python: manifests with consent IDs, streaming CTC baseline, training, streaming eval, verified export
- [x] PyO3 bindings so training uses the Rust features

## Phase 0.5: real-time demo on normal speech ✅
- [x] Pretrained NeMo streaming FastConformer CTC runs in our runtime (CUDA + CPU)
- [x] Contract extended: overlapping windows, mel-major layout, int64 states, Slaney mel
- [x] Live mic → GPU → GUI path verified

## Phase 1: data (the real bottleneck)
- [ ] Consent process and forms with a clinical partner (see DATA_ETHICS.md)
- [ ] **In-app enrollment recorder:** shows a prompt sentence, records, saves wav + txt + consent ID
- [ ] Prompt set: phonetically balanced sentences, plus the user's own everyday phrases
      (names, needs, medical words)
- [ ] Gather public atypical-speech corpora for pre-adaptation, checking each licence:
      UASpeech, TORGO (dysarthria). Glossectomy-specific corpora are rare, so search the
      literature, contact the research groups, and record our own.

## Phase 2: first useful model
- [ ] Start from a **pretrained streaming English model** (an unimpaired-speech base), e.g.
      a cache-aware streaming FastConformer (NeMo) or streaming Zipformer (k2/icefall),
      exported to the contract
- [ ] Two-stage fine-tune: atypical-speech corpora, then **per-speaker** personalisation
- [ ] Parameter-efficient personalisation (LoRA / adapters) so each user's model is small
      and needs only minutes of speech
- [ ] Glossectomy-specific augmentation: simulate articulatory loss (formant shifts,
      consonant smearing) to stretch scarce data
- [ ] Vocabulary biasing / personal phrase list (contextual biasing in the decoder)

## Phase 3: latency
- [ ] True end-to-end latency rig (loopback timing)
- [ ] Smaller chunks (40 ms) with limited lookahead; measure the WER/latency trade-off curve
- [ ] ORT IoBinding: state stays on the GPU between chunks (FastConformer copies ~2.5 MB/call)
- [ ] CUDA graphs for the FastConformer step (10 ms now; profile first)
- [ ] Our fbank uses `snip_edges=true`; Kaldi/NeMo use `false` (a half-frame offset). Measure the WER impact
- [ ] TensorRT FP16 / INT8 engines and CUDA graphs
- [ ] Wake the engine with a condvar instead of a 2 ms sleep
- [ ] Polyphase resampler (measure the WER impact first)
- [ ] Neural VAD / model-based endpointing (blank runs) instead of the energy threshold

## Phase 4: product
- [ ] Settings UI: microphone, model, device, text size, endpoint silence
      (glossectomy speech may need longer pauses)
- [ ] Always-on-top caption overlay mode; text-to-speech of the recognised text (optional)
- [ ] Indic and other scripts: bundle fonts (egui's default fonts are Latin-centric)
- [ ] Multilingual models (which languages? decide with users)
- [ ] Packaging: Linux AppImage / .deb, Windows MSI with DirectML ORT bundled
- [ ] Code signing for Windows

## Research questions
- How much speech per user is needed for a useful personal model?
- Does the resection site (partial vs. total, base vs. oral tongue) predict which
  phonemes fail? Can the model exploit that?
- What is the best latency/accuracy trade-off: CTC vs. transducer, chunk size, lookahead?
