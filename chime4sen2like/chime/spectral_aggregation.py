# -*- coding: utf-8 -*-
# Copyright (c) 2026 ESA.
#
# This file is part of chime4sen2like.
# See https://github.com/senbox-org/sen2like/chime4sen2like for further info.
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
"""Generalized spectral aggregation (spectral harmonisation).

This is a band-set-agnostic generalization of
``prisma4sen2like/prisma/spectral_aggregation_functions.py``.

The PRISMA implementation was hard-wired to:
  * a target of exactly 13 Sentinel-2 bands (``n_bands_s2 = 13``);
  * Sentinel-2A spectral responses loaded through the ``pyrsr`` package;
  * a hand-split VNIR (b < 10) / SWIR (b >= 10) processing path.

Here the operation is expressed purely between two :class:`BandSet` objects, each
described by central wavelength + FWHM, which is exactly what the CHIME Fusion
Roadmap (section 5.6.5) specifies:

    "a spectral regridding via convolution using the narrowband response
     functions (central wavelengths + FWHM) from both CHIME and the other
     Hyperspectral mission".

Algorithm (identical in spirit to PRISMA, generalized):
  1. Model each source and target band as a normalized Gaussian spectral
     response from its (central wavelength, FWHM).
  2. For every (source band s, target band t) compute the overlap integral
     W[s, t] = sum_lambda g_source[s](lambda) * g_target[t](lambda).
  3. Normalize per target band so the source weights sum to one:
     P[s, t] = W[s, t] / sum_s' W[s', t].
  4. Each output (target) band is the weighted sum of source bands:
     out[t] = sum_s P[s, t] * source[s].

Only numpy is required, so this module is unit-testable without GDAL/h5py.

NOTE on smile/keystone (Roadmap section 5.6.6): when the source spectral
response varies across the field of view, pass a per-column aggregation matrix
of shape (n_columns, n_source, n_target) to :func:`aggregate_cube`. Building such
a matrix only requires calling :func:`compute_aggregation_matrix` per detector
column with that column's BandSet.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from band_definitions import BandSet

logger = logging.getLogger(__name__)

# 4 * ln(2), the constant in the FWHM-parameterised Gaussian
_FOUR_LN2 = 4.0 * np.log(2.0)


def gaussian_response(central_wavelength: float, fwhm: float, grid: NDArray) -> NDArray:
    """Normalized Gaussian spectral response sampled on ``grid``.

    Unit-area Gaussian parameterised by FWHM rather than sigma, with the
    half-maximum reached exactly at ``cw +/- FWHM/2``:

        g(l) = 2*sqrt(ln2)/(sqrt(pi)*FWHM) * exp(-4 ln2 ((l - cw)/FWHM)^2)

    (prisma4sen2like wrote the exponent as ``(4 ln2 (l-cw)/FWHM)^2`` which makes
    the response narrower than the stated FWHM; the form here is the standard
    FWHM Gaussian.)

    Args:
        central_wavelength: band central wavelength (nm).
        fwhm: band Full Width at Half Maximum (nm).
        grid: wavelengths at which to sample (nm).

    Returns:
        Spectral response values on ``grid``. All zeros if ``fwhm <= 0``.
    """
    if fwhm <= 0:
        return np.zeros_like(grid)
    norm = 2.0 * np.sqrt(np.log(2.0)) / (np.sqrt(np.pi) * fwhm)
    return norm * np.exp(-_FOUR_LN2 * np.power((grid - central_wavelength) / fwhm, 2))


def build_wavelength_grid(*band_sets: BandSet, step: float = 0.1, n_fwhm: float = 3.0) -> NDArray:
    """Build a common high-resolution wavelength grid spanning the given band sets.

    Args:
        band_sets: one or more BandSets to cover.
        step: grid spacing in nm (default 0.1 nm, as in prisma4sen2like).
        n_fwhm: how many FWHM beyond the extreme central wavelengths to extend
            the grid, so band tails are captured.

    Returns:
        1-D array of wavelengths (nm).
    """
    lows = []
    highs = []
    for band_set in band_sets:
        lows.append(float(np.min(band_set.central_wavelengths - n_fwhm * band_set.fwhm)))
        highs.append(float(np.max(band_set.central_wavelengths + n_fwhm * band_set.fwhm)))
    start = min(lows)
    stop = max(highs)
    return np.arange(start, stop + step, step)


def _responses_on_grid(band_set: BandSet, grid: NDArray) -> NDArray:
    """Matrix of each band's Gaussian response on ``grid``; shape (n_bands, len(grid))."""
    responses = np.zeros((band_set.n_bands, grid.size), dtype=np.float64)
    for i in range(band_set.n_bands):
        responses[i, :] = gaussian_response(
            float(band_set.central_wavelengths[i]), float(band_set.fwhm[i]), grid
        )
    return responses


def compute_aggregation_matrix(
    source: BandSet,
    target: BandSet,
    step: float = 0.1,
    n_fwhm: float = 3.0,
    coverage_tol: float = 1e-6,
) -> NDArray:
    """Compute the normalized spectral aggregation matrix from ``source`` to ``target``.

    Args:
        source: input (e.g. hyperspectral) band set.
        target: output (e.g. CHIME) band set.
        step: integration grid spacing in nm.
        n_fwhm: grid extension in FWHM units.
        coverage_tol: a target band is considered uncovered when its summed
            overlap weight is below ``coverage_tol`` times the largest target
            column sum. This rejects negligible Gaussian-tail overlap (e.g. a
            VNIR-only source "leaking" ~1e-70 into a SWIR target band).

    Returns:
        P, shape (source.n_bands, target.n_bands). Column ``t`` holds the weights
        of each source band contributing to target band ``t`` and sums to 1 for
        every target band that is covered by the source range. A target band with
        no spectral overlap (outside the source range) yields an all-zero column;
        :func:`uncovered_target_bands` reports those.
    """
    grid = build_wavelength_grid(source, target, step=step, n_fwhm=n_fwhm)

    source_resp = _responses_on_grid(source, grid)  # (n_source, n_grid)
    target_resp = _responses_on_grid(target, grid)  # (n_target, n_grid)

    # overlap integral W[s, t] = sum_grid g_source[s] * g_target[t]
    weights = source_resp @ target_resp.T  # (n_source, n_target)

    # normalize per target band so contributing source weights sum to 1.
    # A target band is "covered" only if its overlap is non-negligible relative
    # to the best-covered band (guards against denormal Gaussian-tail overlap).
    col_sums = weights.sum(axis=0)  # (n_target,)
    threshold = coverage_tol * col_sums.max() if col_sums.size and col_sums.max() > 0 else 0.0
    covered = col_sums > threshold
    aggregation = np.zeros_like(weights)
    aggregation[:, covered] = weights[:, covered] / col_sums[covered]

    n_uncovered = int(np.count_nonzero(~covered))
    if n_uncovered:
        logger.warning(
            "%d/%d target band(s) of '%s' have no spectral overlap with source '%s'",
            n_uncovered,
            target.n_bands,
            target.name,
            source.name,
        )

    return aggregation


def uncovered_target_bands(target: BandSet, aggregation: NDArray) -> list[str]:
    """Return the names of target bands not covered by the source (all-zero columns)."""
    col_sums = aggregation.sum(axis=0)
    return [target.names[t] for t in range(target.n_bands) if col_sums[t] == 0]


def aggregate_cube(cube: NDArray, aggregation: NDArray) -> NDArray:
    """Apply an aggregation matrix to a (rows, cols, n_source) radiance/reflectance cube.

    Args:
        cube: input cube, shape (rows, cols, n_source).
        aggregation: either
            * (n_source, n_target): one spectral response for the whole image, or
            * (cols, n_source, n_target): a per-column matrix (handles smile, i.e.
              an across-track varying spectral response).

    Returns:
        Output cube, shape (rows, cols, n_target).
    """
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3-D (rows, cols, n_source), got shape {cube.shape}")

    if aggregation.ndim == 2:
        if aggregation.shape[0] != cube.shape[2]:
            raise ValueError(
                f"aggregation source dim {aggregation.shape[0]} != cube band dim {cube.shape[2]}"
            )
        return np.einsum("rcs,st->rct", cube, aggregation, optimize=True)

    if aggregation.ndim == 3:
        if aggregation.shape[0] != cube.shape[1]:
            raise ValueError(
                f"per-column aggregation has {aggregation.shape[0]} columns, cube has {cube.shape[1]}"
            )
        if aggregation.shape[1] != cube.shape[2]:
            raise ValueError(
                f"aggregation source dim {aggregation.shape[1]} != cube band dim {cube.shape[2]}"
            )
        return np.einsum("rcs,cst->rct", cube, aggregation, optimize=True)

    raise ValueError(f"aggregation must be 2-D or 3-D, got {aggregation.ndim}-D")


def radiance_to_reflectance(
    radiance: NDArray, esun: float, sza: float, sun_earth_distance: float
) -> NDArray:
    """Convert TOA radiance (W.m-2.sr-1.um-1) to TOA reflectance (unitless).

    Same relation as prisma4sen2like.

    Args:
        radiance: TOA radiance.
        esun: band solar irradiance (W.m-2.um-1); from the mission's solar
            irradiance model (Roadmap Table 4-1 recommends TSIS).
        sza: solar zenith angle (degrees).
        sun_earth_distance: Sun-Earth distance (AU).
    """
    return (1.0 / (esun * np.cos(np.radians(sza)))) * np.pi * radiance * sun_earth_distance**2
