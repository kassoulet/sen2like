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
"""Aggregate a hyperspectral product onto a target (CHIME) band set.

This is the chime4sen2like counterpart of
``prisma4sen2like/prisma/prisma_s2_spectral_aggregation.py`` -- but instead of
being hard-wired to 13 Sentinel-2 bands and Sentinel-2A spectral responses, it
aggregates any :class:`HyperspectralProduct` onto any target :class:`BandSet`
(the CHIME band set by default) via :mod:`spectral_aggregation`.

It writes a multi-band GeoTIFF of TOA reflectance (and optionally radiance) in
the target band configuration. Re-projection to the Sentinel-2 MGRS grid and
SAFE packaging are handled downstream (see module docstring of
:mod:`product_builder` / README).
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

from band_definitions import BandSet
from hs_product import HyperspectralProduct
from spectral_aggregation import (
    aggregate_cube,
    compute_aggregation_matrix,
    radiance_to_reflectance,
    uncovered_target_bands,
)

logger = logging.getLogger(__name__)


class ChimeAggregation:
    """Spectrally aggregate a hyperspectral product onto a target band set."""

    def __init__(self, product: HyperspectralProduct, target_band_set: BandSet, work_dir: str):
        self._product = product
        self._target = target_band_set
        self._work_dir = work_dir

    def process(self) -> tuple[str, str]:
        """Run the aggregation and write radiance + reflectance GeoTIFFs.

        Returns:
            (radiance_path, reflectance_path)
        """
        start = time.time()
        source = self._product.band_set
        logger.info(
            "Spectral harmonisation: %s (%d bands) -> %s (%d bands)",
            source.name,
            source.n_bands,
            self._target.name,
            self._target.n_bands,
        )

        # 1. spectral aggregation matrix (source -> target), section 5.6.5
        aggregation = compute_aggregation_matrix(source, self._target)
        uncovered = uncovered_target_bands(self._target, aggregation)
        if uncovered:
            logger.warning(
                "%d target band(s) not covered by the source spectral range: %s%s",
                len(uncovered),
                ", ".join(uncovered[:10]),
                " ..." if len(uncovered) > 10 else "",
            )

        # 2. read source radiance cube and aggregate
        radiance_cube = self._product.read_radiance_cube()  # (rows, cols, n_source)
        logger.info("Read source cube %s", radiance_cube.shape)
        target_radiance = aggregate_cube(radiance_cube, aggregation)  # (rows, cols, n_target)

        # 3. radiance -> TOA reflectance per target band
        esun = _resolve_target_esun(self._product, source, self._target, aggregation)
        sza = self._product.sun_zenith_angle
        d = self._product.sun_earth_distance
        target_reflectance = np.empty_like(target_radiance)
        for b in range(self._target.n_bands):
            target_reflectance[:, :, b] = radiance_to_reflectance(
                target_radiance[:, :, b], esun[b], sza, d
            )

        # 4. write outputs
        rad_path = os.path.join(self._work_dir, f"{self._target.name}_toa_radiance.tif")
        ref_path = os.path.join(self._work_dir, f"{self._target.name}_toa_reflectance.tif")
        _write_geotiff(rad_path, target_radiance, reference_path=self._reference_raster())
        _write_geotiff(ref_path, target_reflectance, reference_path=self._reference_raster())

        logger.info("Aggregation done in %.2fs", time.time() - start)
        logger.info("Radiance:    %s", rad_path)
        logger.info("Reflectance: %s", ref_path)
        return rad_path, ref_path

    def _reference_raster(self) -> str | None:
        # If the source is a plain GeoTIFF, copy its geo-transform/projection.
        return getattr(self._product, "_cube_path", None)


def _resolve_target_esun(product, source, target, aggregation) -> np.ndarray:
    """Per-target-band solar irradiance.

    If the source provides per-band E_sun, propagate it through the same spectral
    weights used for the radiance (so reflectance stays consistent). Otherwise
    fall back to ones (radiance pass-through).
    """
    source_esun = np.asarray(product.solar_irradiance, dtype=float)
    if source_esun.shape[0] != source.n_bands or np.allclose(source_esun, 1.0):
        return np.ones(target.n_bands)
    # weighted combination of source E_sun using the (normalized) aggregation cols
    target_esun = source_esun @ aggregation  # (n_target,)
    target_esun[target_esun == 0] = 1.0
    return target_esun


def _write_geotiff(path: str, cube: np.ndarray, reference_path: str | None = None) -> None:
    """Write a (rows, cols, bands) cube to a multi-band Float32 GeoTIFF."""
    from osgeo import gdal

    rows, cols, bands = cube.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, cols, rows, bands, gdal.GDT_Float32, options=["COMPRESS=LZW"])
    if reference_path:
        ref = gdal.Open(reference_path)
        if ref is not None:
            dataset.SetGeoTransform(ref.GetGeoTransform())
            dataset.SetProjection(ref.GetProjection())
    for b in range(bands):
        dataset.GetRasterBand(b + 1).WriteArray(cube[:, :, b])
    dataset = None
