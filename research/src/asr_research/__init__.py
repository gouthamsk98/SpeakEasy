# SPDX-License-Identifier: GPL-3.0-or-later
"""Research / training side of the glossectomy ASR project.

Rule of the house: anything that the runtime also does (features, decoding) is either
imported from the Rust code via `asr_native`, or verified against it by a test.
"""
