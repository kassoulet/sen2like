# -*- coding: utf-8 -*-
# Copyright (c) 2026 ESA.
#
# This file is part of hyperspectral2chime.
# See https://github.com/senbox-org/sen2like for further info.
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
"""Band-set-agnostic CHIME processing blocks.

These mirror the CHIME Fusion Roadmap (chapter 5) processing steps, but each one
loops over the CHIME band set (any N) rather than the fixed 13 Sentinel-2 bands.
They reuse the sen2like algorithms as components where applicable.

This module ships the foundation blocks. Coefficient/aux-data-dependent blocks
(atmospheric correction with CHIME LUTs, real inter-calibration / BRDF / fusion)
plug in here as they are implemented; the placeholders below are clearly marked.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from chime_product import ChimeProduct, write_cube

logger = logging.getLogger(__name__)


class ChimeBlock(ABC):
    """A processing block transforming a ChimeProduct in place / to a new raster."""

    name = "block"

    @abstractmethod
    def process(self, product: ChimeProduct, work_dir: str) -> ChimeProduct:
        """Run the block, returning the (possibly new) product."""

    def _transform_cube(self, product: ChimeProduct, work_dir: str, func, out_name: str) -> ChimeProduct:
        """Helper: apply ``func(cube)->cube`` and write a new raster, keeping the grid."""
        cube = product.read_cube()
        gt, proj, _, _ = product.grid()
        out = func(cube)
        out_path = os.path.join(work_dir, out_name)
        write_cube(out_path, out.astype(np.float32), gt, proj, nodata=0.0)
        product.raster_path = out_path
        product.provenance.append(self.name)
        return product


class ToaBlock(ChimeBlock):
    """Top-of-atmosphere reflectance (roadmap-equivalent of S2L_Toa).

    The ingested cube already holds TOA reflectance, so this is currently a
    pass-through that records the step. A radiance->reflectance conversion would
    plug in here when the source provides radiance + solar model.
    """

    name = "TOA"

    def process(self, product: ChimeProduct, work_dir: str) -> ChimeProduct:
        logger.info("%s (pass-through: cube already TOA reflectance)", self.name)
        product.provenance.append(self.name)
        return product


class InterCalibrationBlock(ChimeBlock):
    """Radiometric inter-calibration (roadmap 5.2): per-band ``out = in*slope + offset``.

    Coefficients harmonise the input mission to CHIME radiometry. Until CHIME-vs-HS
    inter-calibration coefficients exist, the default is identity (slope 1, offset 0)
    for every band.
    """

    name = "InterCalibration"

    def __init__(self, coefficients: dict[str, tuple[float, float]] | None = None):
        # band name -> (slope, offset); missing bands default to identity
        self._coef = coefficients or {}

    def process(self, product: ChimeProduct, work_dir: str) -> ChimeProduct:
        names = product.band_set.names
        slope = np.array([self._coef.get(n, (1.0, 0.0))[0] for n in names], dtype=np.float32)
        offset = np.array([self._coef.get(n, (1.0, 0.0))[1] for n in names], dtype=np.float32)
        applied = int(np.count_nonzero((slope != 1.0) | (offset != 0.0)))
        logger.info("%s: %d/%d bands with non-identity coefficients", self.name, applied, len(names))
        if applied == 0:
            product.provenance.append(f"{self.name} (identity)")
            return product
        return self._transform_cube(
            product, work_dir, lambda c: c * slope[np.newaxis, np.newaxis, :] + offset[np.newaxis, np.newaxis, :],
            "chime_intercal.tif",
        )
