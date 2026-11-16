# Personal model workflow: user ⇄ trainer

The person who speaks (the **user**) runs the app. The person with a GPU and this repo (the
**trainer**) turns their recordings into a personal model and sends it back. No audio ever
travels except the dataset file the user chooses to send.

```
USER (app)                                     TRAINER (this repo, NVIDIA GPU)
1. Calibration tab: read sentences on screen
   (each take = audio + the correct sentence)
2. "Export dataset" → Downloads/
   speakeasy-dataset-<speaker>-<date>.zip  ──►  3. asr-dataset-import <zip>
                                                4. asr-finetune ../data/datasets/<speaker> --out runs/<speaker>-v1
                                                5. asr-export-nemo runs/<speaker>-v1/model.nemo --name <speaker>-v1
6. "Import model…" (or drop the zip   ◄──  speakeasy-model-<speaker>-v1.zip
   onto the window) → app switches to it
7. Record more → the history shows the error rate per model
```

## User side (app)

- **Calibration** tab → enter a Speaker ID, tick consent once, **Start recording**.
- Each session (default 20 items, up to 50) mixes single words, word lists (short words
  with a pause after each), short phrases and minimal pairs, sentences, tongue twisters,
  a reading passage and a part of a story, taking the items recorded least so far.
- About 2 in 5 short items come with a **recording condition**:
  - close to the microphone, or about 1 m away;
  - a long silence first (the screen counts down 4 s), or a pause in the middle;
  - slowly or quickly;
  - quietly or loudly;
  - head turned away.

  The condition is saved with each take and in the dataset manifest (`condition`).
- The setup screen shows coverage per kind and per condition, and a progress bar towards
  30 minutes of recordings: roughly where fine-tuning starts paying off reliably.
- Space = record/stop, Enter = save & next, **Record again** if you said something different.
  **Wrong recognitions are fine and useful:** the take is saved with the *correct* sentence.
- **Export dataset** writes one zip to Downloads with all recordings, a manifest, the consent
  record and a summary.
- **Import model…** in the top bar (or drag and drop) installs a model package and switches
  to it. The model dropdown switches between installed models.
- Data lives in `~/.local/share/SpeakEasy` (Linux) / `%APPDATA%\SpeakEasy` (Windows); in a
  development checkout, the repo's `data/` and `models/` are used.

## Trainer side

```bash
cd research && source .venv/bin/activate     # torch + nemo_toolkit installed (see README)
asr-dataset-import ~/Downloads/speakeasy-dataset-user01-20261112-1830.zip
asr-finetune ../data/datasets/user01 --out runs/user01-v1            # ~minutes on an RTX 4060
asr-export-nemo runs/user01-v1/model.nemo --name user01-v1
# -> research/runs/packages/speakeasy-model-user01-v1.zip  (send this back)
```

- **Honest evaluation:** `asr-dataset-import` splits by *sentence* with a stable hash. About
  15% of sentences are never trained on. `asr-export-nemo` scores the base model and the
  new model on those sentences **with the app's own streaming runtime** (`asr eval`) and
  writes the result into `model-card.json` inside the package.
- **Next rounds:** import the new dataset zip (it merges with earlier ones), then continue from
  the last personal model: `asr-finetune … --base runs/user01-v1/model.nemo --out runs/user01-v2`.
- Tiny datasets overfit: watch the held-out WER. If it gets worse, use fewer `--epochs`, a
  lower `--lr`, or `--freeze-encoder-layers 8`.
- Clinic recordings that already exist (wav + text) can be added for a speaker with
  `asr dataset-add --speaker user01 manifest.jsonl` (consent must be on file).

## What was verified (synthetic test speaker)

See the end-to-end test results in the README's status section. The test speaker was Piper
TTS with a consistent muffling distortion. That proves the loop works, not how much a real
glossectomy speaker will gain.
