// SPDX-License-Identifier: GPL-3.0-or-later
//! `import asr_native` — the exact Rust fbank used by the app, callable from training code.

use asr_features::{fbank as rs_fbank, FbankConfig};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

/// fbank(samples: np.ndarray[float32], n_mels=80, sample_rate=16000) -> np.ndarray[T, n_mels]
#[pyfunction]
#[pyo3(signature = (samples, n_mels=80, sample_rate=16000))]
fn fbank<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray1<'py, f32>,
    n_mels: usize,
    sample_rate: u32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let cfg = FbankConfig {
        n_mels,
        sample_rate,
        ..FbankConfig::default()
    };
    let s = samples.as_slice()?;
    let (feats, n) = py.detach(|| rs_fbank(s, &cfg));
    let arr = numpy::PyArray1::from_vec(py, feats);
    arr.reshape([n, n_mels])
}

/// The default feature config as a dict — written into model.json at export time.
#[pyfunction]
fn default_feature_config(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let c = FbankConfig::default();
    let d = pyo3::types::PyDict::new(py);
    d.set_item("sample_rate", c.sample_rate)?;
    d.set_item("frame_length", c.frame_length)?;
    d.set_item("frame_shift", c.frame_shift)?;
    d.set_item("n_fft", c.n_fft)?;
    d.set_item("n_mels", c.n_mels)?;
    d.set_item("f_min", c.f_min)?;
    d.set_item("f_max", c.f_max)?;
    d.set_item("preemphasis", c.preemphasis)?;
    d.set_item("remove_dc", c.remove_dc)?;
    Ok(d)
}

#[pymodule]
fn asr_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fbank, m)?)?;
    m.add_function(wrap_pyfunction!(default_feature_config, m)?)?;
    Ok(())
}
