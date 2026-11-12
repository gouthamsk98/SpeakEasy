# data/ (local only, never committed)

Patient speech is health data. Read ../DATA_ETHICS.md first.

```
data/
  raw/<speaker>/*.wav + *.txt     recordings + transcripts (pseudonymous speaker IDs)
  manifests/train.jsonl valid.jsonl tokens.txt    built by `asr-prepare`
```
Keep this folder on an encrypted disk.
