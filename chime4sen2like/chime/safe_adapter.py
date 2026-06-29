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
"""Adapter exposing an aggregated cube as a Sentinel-2-like SAFE source.

Adapted from ``prisma4sen2like/prisma/adapter.py``. Where the PRISMA adapter
orthorectifies swath data from per-pixel geolocation grids, here the input is the
already-projected, spectrally-aggregated cube produced by
:class:`chime_aggregation.ChimeAggregation` (target = the Sentinel-2 13-band set),
so band extraction is a simple ``gdal.Warp`` onto the MGRS tile.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hs_product import HyperspectralProduct
from mgrs_tiling import TileInfo

logger = logging.getLogger(__name__)

# Sentinel-2 13-band order (matches aux_data/sentinel2a_bands.csv) and L1C native
# resolutions. The aggregated cube has one band per entry, in this order.
BAND_LIST = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")
_BAND_INDEX = {band: i for i, band in enumerate(BAND_LIST)}  # 0-based cube band index
IMAGES_RES = {
    "B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20, "B07": 20,
    "B08": 10, "B8A": 20, "B09": 60, "B10": 60, "B11": 20, "B12": 20,
}

# Sentinel-2 L1C DN encoding (must match MTD_MSIL1C.xml): the reflectance stored
# as DN is recovered by physical = (DN + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE.
QUANTIFICATION_VALUE = 10000
RADIO_ADD_OFFSET = -1000  # so DN = reflectance * QUANTIFICATION_VALUE - RADIO_ADD_OFFSET
_WARP_NODATA = -9999.0  # float fill outside the source footprint before DN encoding


@dataclass
class MeanAngle:
    """Mean Sun/Viewing angle (degrees)."""

    zenith_angle: float
    azimuth_angle: float


@dataclass
class AngleGrid:
    """Sun/Viewing angle grid (zenith + azimuth 2-D arrays, deg)."""

    zenith_angle: NDArray
    azimuth_angle: NDArray


@dataclass(unsafe_hash=True)
class MaskFileDef:
    """SAFE QI mask file definition (mirrors prisma4sen2like)."""

    type_attr: str
    band_id_attr: str | None
    value: str


class ProjectedCubeAdapter:
    """Adapt an aggregated, projected band cube to the SAFE product interface."""

    def __init__(
        self,
        product: HyperspectralProduct,
        cube_path: str,
        tile: TileInfo,
        work_dir: str,
        spacecraft: str = "CHIME-like",
        platform: str = "S2H",
    ):
        self._product = product
        self._cube_path = cube_path
        self._tile = tile
        self._wd = work_dir
        self._spacecraft = spacecraft
        self._platform = platform
        self._band_files: dict[str, str] = {}
        self._classi_file: str | None = None

    # --- identity / metadata ----------------------------------------------------
    @property
    def spacecraft(self) -> str:
        return self._spacecraft

    @property
    def platform(self) -> str:
        return self._platform

    @property
    def shot_level(self) -> str:
        return "L1C"

    @property
    def processing_level(self) -> str:
        return "Level-1C"

    @property
    def processing_center(self) -> str:
        return "C4SL"

    @property
    def archiving_center(self) -> str:
        return "C4SL"

    @property
    def reception_station(self) -> str:
        return "XXXX"

    @property
    def station(self) -> str:
        return "XXXX"

    @property
    def sensing_orbit_number(self) -> int:
        return 0

    @property
    def absolute_orbit_number(self) -> int:
        return 0

    @property
    def sensing_orbit_direction(self) -> str:
        return "DESCENDING"

    @property
    def sun_earth_correction(self) -> float:
        d = self._product.sun_earth_distance
        return 1.0 / (d * d) if d else 1.0

    # --- dates ------------------------------------------------------------------
    @property
    def acquisition_datetime(self):
        return self._product.acquisition_datetime

    product_start_time = acquisition_datetime
    product_stop_time = acquisition_datetime
    datatake_sensing_start = acquisition_datetime
    datastrip_sensing_start = acquisition_datetime
    datastrip_sensing_stop = acquisition_datetime
    granule_sensing_start = acquisition_datetime
    tile_sensing_time = acquisition_datetime

    @property
    def processing_time(self):
        return self._product.acquisition_datetime

    # --- geometry ---------------------------------------------------------------
    @property
    def tile_number(self) -> str:
        return self._tile.tile_id

    @property
    def tile(self) -> TileInfo:
        return self._tile

    @property
    def cloudy_pixel_percentage(self) -> float:
        return self._product.cloudy_pixel_percentage

    @property
    def snow_pixel_percentage(self) -> float:
        return 0.0

    # --- angles -----------------------------------------------------------------
    @property
    def mean_sun_angle(self) -> MeanAngle:
        return MeanAngle(self._product.sun_zenith_angle, self._product.sun_azimuth_angle)

    @property
    def mean_viewing_angle(self) -> MeanAngle:
        # nadir placeholder; a real reader supplies per-pixel viewing angles
        return MeanAngle(0.0, 0.0)

    @property
    def sun_angle_grid(self) -> AngleGrid:
        return AngleGrid(
            np.full((23, 23), self._product.sun_zenith_angle),
            np.full((23, 23), self._product.sun_azimuth_angle),
        )

    @property
    def viewing_angle_grid(self) -> AngleGrid:
        return AngleGrid(np.full((23, 23), 0.0), np.full((23, 23), 0.0))

    # --- images -----------------------------------------------------------------
    def get_band_file(self, band_name: str) -> str:
        """Reframe one cube band onto the MGRS tile and return the GeoTIFF path."""
        if band_name in self._band_files:
            return self._band_files[band_name]
        if band_name not in _BAND_INDEX:
            raise ValueError(f"unknown band {band_name}")

        from osgeo import gdal

        res = IMAGES_RES[band_name]
        band_index = _BAND_INDEX[band_name] + 1  # GDAL is 1-based

        # 1. extract the single (reflectance) band
        single = os.path.join(self._wd, f"{band_name}_src.tif")
        gdal.Translate(single, self._cube_path, bandList=[band_index])

        # 2. reframe to the MGRS tile (CRS / extent / resolution), keeping float
        #    reflectance with an explicit nodata fill outside the footprint
        warped = os.path.join(self._wd, f"{band_name}_{res}m_refl.tif")
        gdal.Warp(
            warped,
            single,
            options=gdal.WarpOptions(
                outputType=gdal.GDT_Float32,
                dstSRS=f"EPSG:{self._tile.epsg}",
                outputBounds=self._tile.bounds,
                xRes=res,
                yRes=res,
                resampleAlg="bilinear",
                dstNodata=_WARP_NODATA,
            ),
        )
        os.remove(single)

        # 3. encode reflectance to Sentinel-2 L1C DN (uint16, 0 = nodata) so the
        #    raster is consistent with the MTD quantification/offset
        out_file = os.path.join(self._wd, f"{band_name}_{res}m.tif")
        self._encode_dn(warped, out_file)
        os.remove(warped)

        self._band_files[band_name] = out_file
        logger.info("Reframed + DN-encoded %s -> %s (%dm)", band_name, os.path.basename(out_file), res)
        return out_file

    @staticmethod
    def _encode_dn(reflectance_path: str, out_path: str) -> None:
        """Convert a float reflectance raster to Sentinel-2 L1C uint16 DN (0 = nodata)."""
        from osgeo import gdal

        source = gdal.Open(reflectance_path)
        band = source.GetRasterBand(1)
        array = band.ReadAsArray()
        nodata = band.GetNoDataValue()

        invalid = ~np.isfinite(array)
        if nodata is not None:
            invalid |= array == nodata
        dn = np.rint(array * QUANTIFICATION_VALUE - RADIO_ADD_OFFSET)
        dn = np.clip(dn, 0, 65535)
        dn[invalid] = 0

        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            out_path, source.RasterXSize, source.RasterYSize, 1, gdal.GDT_UInt16, options=["COMPRESS=LZW"]
        )
        dataset.SetGeoTransform(source.GetGeoTransform())
        dataset.SetProjection(source.GetProjection())
        out_band = dataset.GetRasterBand(1)
        out_band.WriteArray(dn.astype(np.uint16))
        out_band.SetNoDataValue(0)
        dataset = None

    # --- masks ------------------------------------------------------------------
    def get_mask_file(self, mask_file_def: MaskFileDef) -> str | None:
        """Only MSK_CLASSI is produced (3-band, all-zero = no cloud/snow/cirrus)."""
        if mask_file_def.type_attr != "MSK_CLASSI":
            return None
        if self._classi_file:
            return self._classi_file

        from osgeo import gdal

        # use B01 (60 m) as the geometry reference
        ref = gdal.Open(self.get_band_file("B01"))
        cols, rows = ref.RasterXSize, ref.RasterYSize
        path = os.path.join(self._wd, mask_file_def.value)
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, cols, rows, 3, gdal.GDT_Byte, options=["COMPRESS=LZW"])
        dataset.SetGeoTransform(ref.GetGeoTransform())
        dataset.SetProjection(ref.GetProjection())
        zeros = np.zeros((rows, cols), dtype=np.uint8)
        for b in range(1, 4):
            dataset.GetRasterBand(b).WriteArray(zeros)
        dataset = None
        self._classi_file = path
        logger.info("Created %s (no-cloud placeholder)", mask_file_def.value)
        return path
