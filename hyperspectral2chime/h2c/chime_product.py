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
"""Internal CHIME product model.

A CHIME product is a band-cube on the CHIME grid (Sentinel-2 MGRS, 30 m GSD) plus
metadata. The internal on-disk format is deliberately band-count-agnostic — a
single multi-band Float32 GeoTIFF of (reflectance) values, named ``*_<level>.tif``,
beside a ``*_<level>.json`` metadata sidecar. This avoids the fixed 13-band
Sentinel-2 SAFE structure and scales to CHIME's ~210 narrow bands.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from band_definitions import BandSet
from mgrs_tiling import TileInfo

CHIME_GSD = 30.0  # CHIME native ground sampling distance (m)


def read_cube(raster_path: str) -> NDArray:
    """Read a multi-band raster as a (rows, cols, bands) float32 array."""
    from osgeo import gdal

    dataset = gdal.Open(raster_path)
    if dataset is None:
        raise FileNotFoundError(f"cannot open raster {raster_path}")
    arr = np.asarray(dataset.ReadAsArray(), dtype=np.float32)
    if arr.ndim == 2:  # single band
        arr = arr[np.newaxis, :, :]
    return np.moveaxis(arr, 0, -1)


def write_cube(out_path: str, cube: NDArray, geotransform, projection, nodata: float | None = None) -> str:
    """Write a (rows, cols, bands) array to a multi-band Float32 GeoTIFF."""
    from osgeo import gdal

    rows, cols, bands = cube.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(out_path, cols, rows, bands, gdal.GDT_Float32, options=["COMPRESS=LZW", "BIGTIFF=IF_SAFER"])
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)
    for b in range(bands):
        band = dataset.GetRasterBand(b + 1)
        band.WriteArray(cube[:, :, b])
        if nodata is not None:
            band.SetNoDataValue(nodata)
    dataset = None
    return out_path


@dataclass
class ChimeProduct:
    """A CHIME band-cube on the CHIME grid plus acquisition/processing metadata."""

    raster_path: str
    band_set: BandSet
    tile: TileInfo
    acquisition_datetime: datetime
    sun_zenith: float
    sun_azimuth: float
    processing_level: str = "L1C"  # L1C -> L2H -> L2F
    radiometric_unit: str = "TOA_reflectance"  # or "TOA_radiance" (W.m-2.sr-1.um-1)
    provenance: list[str] = field(default_factory=list)

    def read_cube(self) -> NDArray:
        return read_cube(self.raster_path)

    def grid(self):
        """Return (geotransform, projection, xsize, ysize) of the band cube."""
        from osgeo import gdal

        dataset = gdal.Open(self.raster_path)
        return dataset.GetGeoTransform(), dataset.GetProjection(), dataset.RasterXSize, dataset.RasterYSize

    def metadata(self) -> dict:
        return {
            "mission": "CHIME-like",
            "reference_mission": "CHIME",
            "processing_level": self.processing_level,
            "radiometric_unit": self.radiometric_unit,
            "tile": self.tile.tile_id,
            "epsg": self.tile.epsg,
            "gsd_m": CHIME_GSD,
            "acquisition_datetime": self.acquisition_datetime.isoformat(),
            "sun_zenith_angle": self.sun_zenith,
            "sun_azimuth_angle": self.sun_azimuth,
            "n_bands": self.band_set.n_bands,
            "bands": [
                {
                    "name": self.band_set.names[i],
                    "central_wavelength_nm": float(self.band_set.central_wavelengths[i]),
                    "fwhm_nm": float(self.band_set.fwhm[i]),
                }
                for i in range(self.band_set.n_bands)
            ],
            "provenance": self.provenance,
        }

    def write_metadata(self, json_path: str) -> str:
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(self.metadata(), handle, indent=2)
        return json_path

    @property
    def basename(self) -> str:
        date = self.acquisition_datetime.strftime("%Y%m%dT%H%M%S")
        return f"CHIME_{self.processing_level}_{date}_T{self.tile.tile_id}"
