# -*- coding: utf-8 -*-
# Copyright (c) 2026 ESA.
#
# This file is part of hyperspectral2chime.
# See https://github.com/senbox-org/sen2like/hyperspectral2chime for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the generalized spectral aggregation core.

Only numpy is required (no GDAL/h5py/pyrsr), so this runs anywhere.

Run with:  PYTHONPATH=h2c python -m pytest hyperspectral2chime/tests
       or:  PYTHONPATH=hyperspectral2chime/chime python hyperspectral2chime/tests/test_spectral_aggregation.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "h2c"))

from band_definitions import BandSet, chime_band_set, sentinel2a_band_set  # noqa: E402
from spectral_aggregation import (  # noqa: E402
    aggregate_cube,
    compute_aggregation_matrix,
    gaussian_response,
    radiance_to_reflectance,
    uncovered_target_bands,
)


def _bandset(name, cw, fwhm):
    return BandSet(
        names=tuple(f"S{i}" for i in range(len(cw))),
        central_wavelengths=np.asarray(cw, dtype=float),
        fwhm=np.asarray(fwhm, dtype=float),
        name=name,
    )


def test_gaussian_unit_area():
    """The FWHM-parameterised Gaussian integrates to ~1 over a fine grid."""
    grid = np.arange(400.0, 600.0, 0.01)
    g = gaussian_response(500.0, 10.0, grid)
    area = np.trapezoid(g, grid) if hasattr(np, "trapezoid") else np.trapz(g, grid)
    assert abs(area - 1.0) < 1e-3, area


def test_gaussian_fwhm_half_max():
    """Response at cw +/- FWHM/2 is half of the peak response."""
    grid = np.array([500.0, 505.0, 495.0])
    g = gaussian_response(500.0, 10.0, grid)
    assert abs(g[1] / g[0] - 0.5) < 1e-6
    assert abs(g[2] / g[0] - 0.5) < 1e-6


def test_columns_are_partition_of_unity():
    """Every covered target band's source weights sum to 1."""
    source = _bandset("hs", np.arange(450.0, 850.0, 5.0), np.full(80, 6.0))
    target = _bandset("multi", [490.0, 560.0, 665.0, 705.0], [60.0, 35.0, 30.0, 15.0])
    agg = compute_aggregation_matrix(source, target)
    col_sums = agg.sum(axis=0)
    assert np.allclose(col_sums, 1.0, atol=1e-9), col_sums
    assert uncovered_target_bands(target, agg) == []


def test_identity_when_source_equals_target():
    """Aggregating a band set onto itself is (close to) the identity."""
    bs = _bandset("same", [490.0, 560.0, 665.0, 842.0], [60.0, 35.0, 30.0, 106.0])
    agg = compute_aggregation_matrix(bs, bs)
    # diagonal dominates; off-diagonal leakage is small for well-separated bands
    assert np.all(np.diag(agg) > 0.78), np.diag(agg)
    assert np.argmax(agg, axis=0).tolist() == [0, 1, 2, 3]


def test_constant_spectrum_preserved():
    """A spectrally flat scene maps to the same constant on covered target bands."""
    source = _bandset("hs", np.arange(450.0, 850.0, 5.0), np.full(80, 6.0))
    target = _bandset("multi", [490.0, 560.0, 665.0, 705.0], [60.0, 35.0, 30.0, 15.0])
    agg = compute_aggregation_matrix(source, target)
    cube = np.full((4, 7, source.n_bands), 0.123, dtype=float)
    out = aggregate_cube(cube, agg)
    assert out.shape == (4, 7, target.n_bands)
    assert np.allclose(out, 0.123, atol=1e-9), out.min()


def test_two_sources_weighted_average():
    """Two narrow source bands inside one broad target band give a weighted average."""
    source = _bandset("hs", [495.0, 505.0], [4.0, 4.0])
    target = _bandset("one", [500.0], [40.0])
    agg = compute_aggregation_matrix(source, target)
    # symmetric placement -> equal weights
    assert np.allclose(agg[:, 0], [0.5, 0.5], atol=1e-3), agg[:, 0]
    cube = np.zeros((1, 1, 2))
    cube[0, 0] = [10.0, 20.0]
    out = aggregate_cube(cube, agg)
    assert abs(out[0, 0, 0] - 15.0) < 1e-2, out[0, 0, 0]


def test_uncovered_target_band_is_zero_column():
    """A target band outside the source spectral range gets an all-zero column."""
    source = _bandset("vnir", np.arange(450.0, 850.0, 5.0), np.full(80, 6.0))
    target = _bandset("with_swir", [560.0, 2200.0], [35.0, 175.0])
    agg = compute_aggregation_matrix(source, target)
    assert uncovered_target_bands(target, agg) == ["S1"]
    assert np.all(agg[:, 1] == 0.0)


def test_per_column_aggregation_smile():
    """A per-column (smile-aware) aggregation matrix is accepted and applied."""
    source = _bandset("hs", np.arange(450.0, 850.0, 5.0), np.full(80, 6.0))
    target = _bandset("multi", [490.0, 560.0, 665.0], [60.0, 35.0, 30.0])
    agg = compute_aggregation_matrix(source, target)
    n_cols = 5
    per_col = np.repeat(agg[np.newaxis, :, :], n_cols, axis=0)  # (cols, n_source, n_target)
    cube = np.random.default_rng(0).random((3, n_cols, source.n_bands))
    out_uniform = aggregate_cube(cube, agg)
    out_percol = aggregate_cube(cube, per_col)
    assert np.allclose(out_uniform, out_percol)


def test_radiance_to_reflectance_roundtrip():
    """Reflectance conversion matches the closed-form relation."""
    rad = np.array([100.0])
    refl = radiance_to_reflectance(rad, esun=1500.0, sza=30.0, sun_earth_distance=1.0)
    expected = np.pi * 100.0 / (1500.0 * np.cos(np.radians(30.0)))
    assert np.allclose(refl, expected)


def test_aux_band_sets_load():
    """Shipped Sentinel-2A and CHIME band-set CSVs load and have expected sizes."""
    s2a = sentinel2a_band_set()
    assert s2a.n_bands == 13
    assert s2a.names[0] == "B01"
    chime = chime_band_set()
    assert chime.n_bands > 100  # placeholder ~211 narrow bands
    assert chime.central_wavelengths.min() >= 350.0
    assert chime.central_wavelengths.max() <= 2550.0


def test_s2a_target_from_hyperspectral_source():
    """End-to-end: a synthetic 10 nm hyperspectral source aggregates onto the S2A set."""
    s2a = sentinel2a_band_set()
    # synthetic CHIME-like source spanning the S2 range at 10 nm / 10 nm FWHM
    cw = np.arange(420.0, 2300.0, 10.0)
    source = _bandset("hs10nm", cw, np.full(cw.size, 10.0))
    agg = compute_aggregation_matrix(source, s2a)
    covered = [b for b in s2a.names if b not in uncovered_target_bands(s2a, agg)]
    # all S2A bands within 420-2300 nm should be covered (B12@2202 included)
    assert set(covered) == set(s2a.names), uncovered_target_bands(s2a, agg)
    cube = np.ones((2, 2, source.n_bands))
    out = aggregate_cube(cube, agg)
    assert out.shape == (2, 2, s2a.n_bands)
    assert np.allclose(out, 1.0, atol=1e-6)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {exc!r}")
    print(f"\n{'OK' if failures == 0 else 'FAILED'} ({failures} failure(s))")
    sys.exit(1 if failures else 0)
