# Data ethics & consent

Recordings of people after glossectomy are **health data**. They reveal a medical
condition, and a voice can identify a person. Treat every recording as you would a
medical record.

## Hard rules

1. **No consent, no data.** Every utterance in a manifest carries a `consent_id` that
   points to a signed consent record kept *outside* this repository. The training code
   refuses rows without one.
2. **Never commit audio, transcripts of patients, or weights trained on them.**
   `.gitignore` blocks the usual formats. Do not work around it.
3. **Local by default.** The app never sends audio anywhere. Do not add network calls,
   analytics or crash reporters that could carry audio or text.
4. **Consent has to be specific.** It must state what is recorded, who can access it,
   where it is stored, how long it is kept, whether it may be used to train a *shared*
   model or only the person's *own* model, and how to withdraw.
5. **Withdrawal is real.** A person who withdraws has their audio deleted, and every model
   trained on it is either retrained without them or deleted.
6. **Ethics approval.** Data collection in a clinical or research setting needs
   institutional ethics (IRB/IEC) approval and must follow local law, e.g. India's DPDP
   Act 2023, the EU GDPR (special-category data) or US HIPAA, depending on where it happens.

## Model weights

Neural networks can memorise training data. Therefore:
- **Personal models** (trained on one user) belong to that user and stay on their device.
- **Shared/base models** trained on several users' speech may be published only if every
  contributor consented to that explicitly, and only after a privacy review.
- Base models trained purely on public datasets may be published under each dataset's
  licence terms.

## Storage

- Keep raw data on an encrypted disk. The `data/` folder is gitignored.
- Use pseudonymous speaker IDs (`spk01`). Keep the mapping to real identities offline and
  with the clinical partner.

*This document is project policy, not legal advice. Talk to your institution's ethics
board and a lawyer before collecting data.*
