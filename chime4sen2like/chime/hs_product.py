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
"""Hyperspectral input-product reader interface.

The CHIME Fusion Roadmap (section 4, Table 4-1) states that "specific readers for
each mission need to be created or adapted". :class:`HyperspectralProduct` is the
contract such a reader must satisfy so the rest of chime4sen2like (spectral
aggregation, ortho-rectification, packaging) stays mission-agnostic.

Two kinds of reader are expected:

* Mission-specific readers (CHIME L1C, SBG L1B, EnMAP, PRISMA, EMIT, DESIS ...)
  parse the native format and metadata. These are TODO and require the relevant
  product-format specifications ([RD02]/[RD03] for CHIME).

* :class:`GeoTiffHyperspectralProduct` is a generic, immediately usable reader:
  it reads a multi-band GeoTIFF data cube plus a sidecar band-definition CSV
  (central wavelength + FWHM). It lets the spectral-harmonisation pipeline be
  exercised end-to-end on any already-orthorectified hyperspectral cube, before a
  mission-specific reader exists.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from band_definitions import BandSet

logger = logging.getLogger(__name__)


class HyperspectralProduct(ABC):
    """Abstract hyperspectral L1 product.

    Concrete subclasses adapt a specific mission/format. The property set mirrors
    the information the CHIME Fusion Roadmap requires from other missions
    (Table 4-1): geometry, per-pixel/scene angles, spectral configuration,
    radiometry and atmosphere/quality metadata.
    """

    # --- spectral configuration -------------------------------------------------
    @property
    @abstractmethod
    def band_set(self) -> BandSet:
        """Source spectral configuration (central wavelengths + FWHM)."""

    @abstractmethod
    def read_radiance_cube(self) -> NDArray:
        """Return the TOA radiance cube, shape (rows, cols, n_source_bands).

        Units: W.m-2.sr-1.um-1 (consistent with :func:`radiance_to_reflectance`).
        Readers that natively provide reflectance should override
        :meth:`read_reflectance_cube` instead and may raise here.
        """

    def read_reflectance_cube(self) -> NDArray:
        """Return a TOA reflectance cube if the product provides one natively.

        Default implementation signals "not available"; the orchestrator then
        converts radiance to reflectance using the solar model.
        """
        raise NotImplementedError("product does not expose a native reflectance cube")

    # --- radiometry -------------------------------------------------------------
    @property
    @abstractmethod
    def solar_irradiance(self) -> NDArray:
        """Per-band solar irradiance E_sun (W.m-2.um-1). Roadmap recommends TSIS."""

    @property
    @abstractmethod
    def sun_earth_distance(self) -> float:
        """Sun-Earth distance at acquisition (AU)."""

    # --- geometry / angles ------------------------------------------------------
    @property
    @abstractmethod
    def sun_zenith_angle(self) -> float:
        """Scene mean solar zenith angle (degrees)."""

    @property
    @abstractmethod
    def sun_azimuth_angle(self) -> float:
        """Scene mean solar azimuth angle (degrees)."""

    @property
    @abstractmethod
    def scene_center(self) -> tuple[float, float]:
        """Scene centre as (latitude, longitude) in degrees (for MGRS tiling)."""

    # Optional per-pixel geolocation, required for the L1B ortho flow (5.1.6).
    @property
    def geolocation_grids(self) -> tuple[NDArray, NDArray, NDArray] | None:
        """(latitude, longitude, altitude) grids, or None for orthorectified L1C."""
        return None

    # --- masks ------------------------------------------------------------------
    @property
    def cloud_mask(self) -> NDArray | None:
        """Cloud mask grid (1 = cloud), or None if not provided by the mission."""
        return None

    # --- metadata ---------------------------------------------------------------
    @property
    @abstractmethod
    def acquisition_datetime(self) -> datetime:
        """Scene-centre acquisition UTC datetime."""

    @property
    def cloudy_pixel_percentage(self) -> float:
        return 0.0

    @property
    def spacecraft(self) -> str:
        return "CHIME-like"


class GeoTiffHyperspectralProduct(HyperspectralProduct):
    """Generic reader: a multi-band GeoTIFF cube + a band-definition CSV.

    This reader works on any already-orthorectified hyperspectral cube and is the
    quickest way to drive the spectral-harmonisation pipeline without a
    mission-specific reader. Geometry/angles/atmosphere metadata are supplied via
    the constructor (sensible defaults provided) since a plain GeoTIFF carries
    none of it.

    Args:
        cube_path: path to a multi-band GeoTIFF (band order matches the CSV).
        band_csv_path: CSV of band, central_wavelength_nm, fwhm_nm.
        sun_zenith: scene solar zenith angle (deg).
        sun_azimuth: scene solar azimuth angle (deg).
        acquisition: acquisition UTC datetime.
        scene_center: (lat, lon) degrees.
        solar_irradiance: per-band E_sun; defaults to ones (radiance pass-through).
        sun_earth_distance: AU; defaults to 1.0.
        is_reflectance: if True, the cube already holds reflectance.
    """

    def __init__(
        self,
        cube_path: str,
        band_csv_path: str,
        sun_zenith: float = 30.0,
        sun_azimuth: float = 160.0,
        acquisition: datetime | None = None,
        scene_center: tuple[float, float] = (0.0, 0.0),
        solar_irradiance: NDArray | None = None,
        sun_earth_distance: float = 1.0,
        is_reflectance: bool = False,
    ):
        self._cube_path = cube_path
        self._band_set = BandSet.from_csv(band_csv_path, name="source")
        self._sun_zenith = sun_zenith
        self._sun_azimuth = sun_azimuth
        self._acquisition = acquisition or datetime(2026, 1, 1)
        self._scene_center = scene_center
        self._solar_irradiance = (
            np.asarray(solar_irradiance, dtype=float)
            if solar_irradiance is not None
            else np.ones(self._band_set.n_bands)
        )
        self._sun_earth_distance = sun_earth_distance
        self._is_reflectance = is_reflectance

    def _read_cube(self) -> NDArray:
        # Local import: GDAL is only needed at runtime, not to import this module
        # or run the unit tests.
        from osgeo import gdal

        dataset = gdal.Open(self._cube_path)
        if dataset is None:
            raise FileNotFoundError(f"cannot open raster {self._cube_path}")
        if dataset.RasterCount != self._band_set.n_bands:
            raise ValueError(
                f"{self._cube_path} has {dataset.RasterCount} bands but band CSV "
                f"declares {self._band_set.n_bands}"
            )
        # GDAL ReadAsArray -> (bands, rows, cols); we want (rows, cols, bands)
        arr = dataset.ReadAsArray()
        return np.moveaxis(np.asarray(arr, dtype=np.float32), 0, -1)

    @property
    def band_set(self) -> BandSet:
        return self._band_set

    def read_radiance_cube(self) -> NDArray:
        if self._is_reflectance:
            raise NotImplementedError("cube holds reflectance; use read_reflectance_cube()")
        return self._read_cube()

    def read_reflectance_cube(self) -> NDArray:
        if not self._is_reflectance:
            raise NotImplementedError("cube holds radiance; use read_radiance_cube()")
        return self._read_cube()

    @property
    def solar_irradiance(self) -> NDArray:
        return self._solar_irradiance

    @property
    def sun_earth_distance(self) -> float:
        return self._sun_earth_distance

    @property
    def sun_zenith_angle(self) -> float:
        return self._sun_zenith

    @property
    def sun_azimuth_angle(self) -> float:
        return self._sun_azimuth

    @property
    def scene_center(self) -> tuple[float, float]:
        return self._scene_center

    @property
    def acquisition_datetime(self) -> datetime:
        return self._acquisition
