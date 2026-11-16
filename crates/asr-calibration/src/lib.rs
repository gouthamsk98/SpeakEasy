// SPDX-License-Identifier: GPL-3.0-or-later
//! Calibration sessions: the user reads fixed sentences, we compare what the model heard
//! with what was meant, and keep the recordings as the user's **voice baseline**.
//!
//! * The same fixed sentence set is used every session, so results are comparable over
//!   time and across models: the first session is the baseline.
//! * Each take is saved as `NN.wav` + `NN.txt` (the prompt = ground-truth transcript),
//!   directly usable by `research/` (`asr-prepare`) for personalised fine-tuning.
//! * Nothing is saved without a local consent record (DATA_ETHICS.md).

pub mod align;
pub mod package;
pub mod prompts;
pub mod store;

pub use align::{score, Op, Score};
pub use prompts::{all_prompts, kind_of, Condition, Kind, Prompt};
pub use store::{Consent, SessionSummary, SessionWriter, TakeResult};
