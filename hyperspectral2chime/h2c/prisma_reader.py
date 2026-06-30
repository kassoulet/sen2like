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
"""PRISMA L1 (HCO) reader for hyperspectral2chime.

Reads a native PRISMA L1 ``.he5`` product into the form the CHIME chain needs: a
TOA-radiance band cube plus a source :class:`BandSet` (central wavelength + FWHM),
acquisition metadata, and the per-pixel geolocation grids used later for
ortho-rectification onto the CHIME grid.

The radiance conversion and cube orientation mirror
``prisma4sen2like/prisma/spectral_aggregation_functions.read_cube_to_radiance``.
VNIR (≈63 usable bands, ~407–977 nm) and SWIR (≈171 bands, ~943–2497 nm) are
concatenated into a single ~234-band hyperspectral source; bands with a zero
central wavelength (unused detector rows) are dropped. The cube stays in PRISMA
swath geometry — ``geolocation_grids`` provides the lat/lon needed to project it.

Requires ``h5py`` (e.g. the ``sen2like`` conda env after ``pip install h5py``).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
from numpy.typing import NDArray

from band_definitions import BandSet

logger = logging.getLogger(__name__)

_HCO = "HDFEOS/SWATHS/PRS_L1_HCO/"
_CENTER = 499  # 1000x1000 swath -> centre pixel index


class PrismaReader:
    """Reader for a PRISMA L1 STD (HCO) product."""

    def __init__(self, product_path: str):
        import h5py

        self._path = product_path
        self._f = h5py.File(product_path, "r")
        self._a = self._f.attrs
        self._band_set: BandSet | None = None
        self._keep: dict[str, NDArray] = {}  # spectrometer -> kept band index mask

    # --- helpers ----------------------------------------------------------------
    def _attr(self, key: str):
        value = self._a.get(key)
        return value.decode("utf-8") if isinstance(value, bytes) else value

    def _spectrometer_bands(self, spectro: str) -> tuple[NDArray, NDArray, NDArray]:
        """Return (kept_mask, central_wavelengths, fwhm) for 'Vnir' or 'Swir'."""
        cw = np.asarray(self._a.get(f"List_Cw_{spectro}"), dtype=np.float64)
        fwhm = np.asarray(self._a.get(f"List_Fwhm_{spectro}"), dtype=np.float64)
        keep = cw > 0
        return keep, cw[keep], fwhm[keep]

    def _cube_radiance(self, spectro: str, dataset: str) -> NDArray:
        """Read a spectrometer cube as (rows, cols, kept_bands) TOA radiance."""
        scale = float(self._a.get(f"ScaleFactor_{spectro}"))
        offset = float(self._a.get(f"Offset_{spectro}"))
        cube = np.asarray(self._f[_HCO + f"Data Fields/{dataset}_Cube"])  # (rows, bands, cols)
        cube = cube.swapaxes(1, 2)  # (rows, cols, bands)
        cube = np.rot90(cube, k=-1)  # align with lat/lon grids (as prisma4sen2like)
        radiance = cube / scale + offset
        keep, _, _ = self._spectrometer_bands(spectro)
        return radiance[:, :, keep].astype(np.float32)

    # --- spectral configuration -------------------------------------------------
    @property
    def band_set(self) -> BandSet:
        """Concatenated VNIR+SWIR source band set (central wavelength + FWHM, nm)."""
        if self._band_set is not None:
            return self._band_set
        _, cw_v, fw_v = self._spectrometer_bands("Vnir")
        _, cw_s, fw_s = self._spectrometer_bands("Swir")
        cw = np.concatenate([cw_v, cw_s])
        fwhm = np.concatenate([fw_v, fw_s])
        names = tuple(
            [f"VNIR_{i:03d}" for i in range(cw_v.size)] + [f"SWIR_{i:03d}" for i in range(cw_s.size)]
        )
        self._band_set = BandSet(names=names, central_wavelengths=cw, fwhm=fwhm, name="PRISMA")
        return self._band_set

    def read_radiance_cube(self) -> NDArray:
        """TOA radiance cube (rows, cols, n_source) — VNIR bands then SWIR bands."""
        vnir = self._cube_radiance("Vnir", "VNIR")
        swir = self._cube_radiance("Swir", "SWIR")
        logger.info("PRISMA cubes: VNIR %s + SWIR %s", vnir.shape, swir.shape)
        return np.concatenate([vnir, swir], axis=2)

    # --- metadata ---------------------------------------------------------------
    @property
    def acquisition_datetime(self) -> datetime:
        raw = self._attr("Product_StartTime")
        return datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)

    @property
    def sun_zenith_angle(self) -> float:
        return float(self._a.get("Sun_zenith_angle"))

    @property
    def sun_azimuth_angle(self) -> float:
        return float(self._a.get("Sun_azimuth_angle"))

    @property
    def cloudy_pixel_percentage(self) -> float:
        value = self._a.get("L1_Quality_CCPerc")
        return float(value) if value is not None else 0.0

    @property
    def scene_center(self) -> tuple[float, float]:
        lat = float(self._f[_HCO + "Geolocation Fields/Latitude_VNIR"][_CENTER, _CENTER])
        lon = float(self._f[_HCO + "Geolocation Fields/Longitude_VNIR"][_CENTER, _CENTER])
        return lat, lon

    # --- geometry / masks (for ortho-rectification, next step) ------------------
    def geolocation_grids(self) -> tuple[NDArray, NDArray, NDArray]:
        """Per-pixel (latitude, longitude, altitude) grids, oriented like the cube."""
        lat = np.rot90(np.asarray(self._f[_HCO + "Geolocation Fields/Latitude_VNIR"]), k=-1)
        lon = np.rot90(np.asarray(self._f[_HCO + "Geolocation Fields/Longitude_VNIR"]), k=-1)
        alt = np.zeros_like(lat)
        return lat, lon, alt

    def cloud_mask(self) -> NDArray | None:
        path = _HCO + "Data Fields/Cloud_Mask"
        if path not in self._f:
            return None
        return np.rot90(np.asarray(self._f[path]), k=-1)

    def close(self):
        self._f.close()
