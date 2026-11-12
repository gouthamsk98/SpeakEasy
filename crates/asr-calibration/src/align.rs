// SPDX-License-Identifier: GPL-3.0-or-later
//! Word alignment (Levenshtein with backtrace) between the prompt and what was recognised.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Op {
    Correct(String),
    /// Expected `expected`, recognised `heard`.
    Sub {
        expected: String,
        heard: String,
    },
    /// Expected word that was not recognised.
    Del(String),
    /// Recognised word that was not in the prompt.
    Ins(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    pub ops: Vec<Op>,
    pub ref_words: usize,
    pub word_errors: usize,
    pub ref_chars: usize,
    pub char_errors: usize,
}

impl Score {
    pub fn wer(&self) -> f64 {
        self.word_errors as f64 / self.ref_words.max(1) as f64
    }
    pub fn cer(&self) -> f64 {
        self.char_errors as f64 / self.ref_chars.max(1) as f64
    }
}

/// Lowercase, keep letters/digits/apostrophes, everything else becomes a word break.
pub fn normalize(s: &str) -> String {
    let cleaned: String = s
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '\'' {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn levenshtein<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let sub = prev[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Score `hypothesis` against `reference` (both normalised first).
pub fn score(reference: &str, hypothesis: &str) -> Score {
    let (r, h) = (normalize(reference), normalize(hypothesis));
    let rw: Vec<&str> = r.split_whitespace().collect();
    let hw: Vec<&str> = h.split_whitespace().collect();
    let (n, m) = (rw.len(), hw.len());

    // Full DP table for backtrace (sentences are short).
    let mut d = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in d[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let sub = d[i - 1][j - 1] + usize::from(rw[i - 1] != hw[j - 1]);
            d[i][j] = sub.min(d[i - 1][j] + 1).min(d[i][j - 1] + 1);
        }
    }
    let mut ops = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && d[i][j] == d[i - 1][j - 1] + usize::from(rw[i - 1] != hw[j - 1]) {
            ops.push(if rw[i - 1] == hw[j - 1] {
                Op::Correct(rw[i - 1].into())
            } else {
                Op::Sub {
                    expected: rw[i - 1].into(),
                    heard: hw[j - 1].into(),
                }
            });
            i -= 1;
            j -= 1;
        } else if i > 0 && d[i][j] == d[i - 1][j] + 1 {
            ops.push(Op::Del(rw[i - 1].into()));
            i -= 1;
        } else {
            ops.push(Op::Ins(hw[j - 1].into()));
            j -= 1;
        }
    }
    ops.reverse();

    let rc: Vec<char> = r.chars().collect();
    let hc: Vec<char> = h.chars().collect();
    Score {
        ops,
        ref_words: n,
        word_errors: d[n][m],
        ref_chars: rc.len(),
        char_errors: levenshtein(&rc, &hc),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect() {
        let s = score(
            "I need a glass of cold water.",
            "i need a glass of cold water",
        );
        assert_eq!(s.word_errors, 0);
        assert_eq!(s.cer(), 0.0);
        assert!(s.ops.iter().all(|o| matches!(o, Op::Correct(_))));
    }

    #[test]
    fn sub_del_ins() {
        let s = score(
            "please call my daughter today",
            "please tall my daughter to day",
        );
        // call->tall (S), today->to (S) + day (I)  => 3 errors / 5 words
        assert_eq!(s.word_errors, 3);
        assert_eq!(s.ref_words, 5);
        assert!(s.ops.contains(&Op::Sub {
            expected: "call".into(),
            heard: "tall".into()
        }));
        assert!(s.ops.iter().any(|o| matches!(o, Op::Ins(_))));

        let s = score("turn the lights on", "turn lights on");
        assert_eq!(
            s.ops,
            vec![
                Op::Correct("turn".into()),
                Op::Del("the".into()),
                Op::Correct("lights".into()),
                Op::Correct("on".into()),
            ]
        );
    }

    #[test]
    fn empty_hypothesis() {
        let s = score("thank you", "");
        assert_eq!(s.word_errors, 2);
        assert_eq!(s.wer(), 1.0);
    }

    #[test]
    fn normalize_punctuation() {
        assert_eq!(normalize("Don't — STOP, now!"), "don't stop now");
    }
}
