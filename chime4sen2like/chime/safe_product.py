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
"""Sentinel-2-like SAFE product model.

Adapted from ``prisma4sen2like/prisma/sen2like_product.py`` -- the naming/identifier
scheme and the attribute surface consumed by the SAFE metadata templates. It
delegates data access to :class:`safe_adapter.ProjectedCubeAdapter`.
"""
from __future__ import annotations

from datetime import datetime, timezone

from osgeo import osr

from safe_adapter import AngleGrid, MaskFileDef, MeanAngle, ProjectedCubeAdapter

YYYYMMDDTHHMMSS = "%Y%m%dT%H%M%S"

# QI mask files the builder will try to create (only MSK_CLASSI is produced).
MASK_TABLE = [
    MaskFileDef("MSK_CLASSI", None, "MSK_CLASSI_B00.tif"),
]


class SafeProduct:
    """SAFE L1C product representation backed by a ProjectedCubeAdapter."""

    _safe_name_tpl = "{}_MSIL1C_{}_N{}_R{}_T{}_{}.SAFE"
    _datastrip_identifier_tpl = "{}_OPER_MSI_{}_DS_{}_{}_S{}_N{}"
    _datatake_identifier_tpl = "G{}_{}_{}_N{}"
    _long_granule_identifier_tpl = "{}_OPER_MSI_{}_TL_{}_{}_A{}_T{}_N{}"
    _short_granule_identifier_tpl = "{}_T{}_A{}_{}"
    _image_filename_tpl = "T{}_{}_{}"

    _processing_baseline = "0000"
    _processing_baseline_dotted = "00.00"

    def __init__(self, adapter: ProjectedCubeAdapter):
        self._adapter = adapter
        self._product_date = datetime.now(timezone.utc)

    # --- identifiers ------------------------------------------------------------
    @property
    def datatake_identifier(self) -> str:
        return self._datatake_identifier_tpl.format(
            self._adapter.platform,
            self._adapter.datatake_sensing_start.strftime(YYYYMMDDTHHMMSS),
            f"{self._adapter.absolute_orbit_number:06}",
            self._processing_baseline_dotted,
        )

    @property
    def product_name(self) -> str:
        return self._safe_name_tpl.format(
            self._adapter.platform,
            self._adapter.datatake_sensing_start.strftime(YYYYMMDDTHHMMSS),
            self._processing_baseline,
            f"{self._adapter.sensing_orbit_number:03}",
            self._adapter.tile_number,
            self._product_date.strftime(YYYYMMDDTHHMMSS),
        )

    @property
    def datastrip_identifier(self) -> str:
        return self._datastrip_identifier_tpl.format(
            self._adapter.platform,
            self._adapter.shot_level,
            self._adapter.processing_center,
            self._product_date.strftime(YYYYMMDDTHHMMSS),
            self.product_start_time.strftime(YYYYMMDDTHHMMSS),
            self._processing_baseline_dotted,
        )

    @property
    def long_granule_identifier(self) -> str:
        return self._long_granule_identifier_tpl.format(
            self._adapter.platform,
            self._adapter.shot_level,
            self._adapter.station,
            self._product_date.strftime(YYYYMMDDTHHMMSS),
            f"{self._adapter.absolute_orbit_number:06}",
            self._adapter.tile_number,
            self._processing_baseline_dotted,
        )

    @property
    def short_granule_identifier(self) -> str:
        return self._short_granule_identifier_tpl.format(
            self._adapter.shot_level,
            self._adapter.tile_number,
            f"{self._adapter.absolute_orbit_number:06}",
            self._adapter.granule_sensing_start.strftime(YYYYMMDDTHHMMSS),
        )

    def get_image_filename(self, band: str) -> str:
        return self._image_filename_tpl.format(
            self._adapter.tile_number,
            self._adapter.granule_sensing_start.strftime(YYYYMMDDTHHMMSS),
            band,
        )

    @property
    def image_filename_list(self):
        from safe_adapter import BAND_LIST

        for band in BAND_LIST:
            yield self.get_image_filename(band)

    @property
    def pvi_filename(self) -> str:
        return self.get_image_filename("PVI") + ".tif"

    # --- passthrough metadata ---------------------------------------------------
    @property
    def baseline(self) -> str:
        return self._processing_baseline_dotted

    @property
    def spacecraft(self) -> str:
        return self._adapter.spacecraft

    @property
    def processing_level(self) -> str:
        return self._adapter.processing_level

    @property
    def archiving_center(self) -> str:
        return self._adapter.archiving_center

    @property
    def reception_station(self) -> str:
        return self._adapter.reception_station

    @property
    def processing_center(self) -> str:
        return self._adapter.processing_center

    @property
    def processing_time(self) -> datetime:
        return self._adapter.processing_time

    @property
    def product_date(self) -> datetime:
        return self._product_date

    @property
    def product_start_time(self) -> datetime:
        return self._adapter.product_start_time

    @property
    def product_stop_time(self) -> datetime:
        return self._adapter.product_stop_time

    @property
    def datatake_sensing_start(self) -> datetime:
        return self._adapter.datatake_sensing_start

    @property
    def datastrip_sensing_start(self) -> datetime:
        return self._adapter.datastrip_sensing_start

    @property
    def datastrip_sensing_stop(self) -> datetime:
        return self._adapter.datastrip_sensing_stop

    @property
    def sensing_orbit_number(self) -> int:
        return self._adapter.sensing_orbit_number

    @property
    def sensing_orbit_direction(self) -> str:
        return self._adapter.sensing_orbit_direction

    @property
    def tile_sensing_time(self) -> datetime:
        return self._adapter.tile_sensing_time

    @property
    def sun_earth_correction(self) -> float:
        return self._adapter.sun_earth_correction

    @property
    def cloudy_pixel_percentage(self) -> float:
        return self._adapter.cloudy_pixel_percentage

    @property
    def snow_pixel_percentage(self) -> float:
        return self._adapter.snow_pixel_percentage

    @property
    def mean_sun_angle(self) -> MeanAngle:
        return self._adapter.mean_sun_angle

    @property
    def mean_viewing_angle(self) -> MeanAngle:
        return self._adapter.mean_viewing_angle

    @property
    def sun_angle_grid(self) -> AngleGrid:
        return self._adapter.sun_angle_grid

    @property
    def viewing_angle_grid(self) -> AngleGrid:
        return self._adapter.viewing_angle_grid

    # --- geocoding --------------------------------------------------------------
    @property
    def ulx(self) -> int:
        return int(self._adapter.tile.ulx)

    @property
    def uly(self) -> int:
        return int(self._adapter.tile.uly)

    @property
    def epsg_code(self) -> str:
        return self._adapter.tile.epsg

    @property
    def epsg_name(self) -> str:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(self.epsg_code))
        return srs.GetAttrValue("projcs")

    def get_band_file(self, band_name: str) -> str:
        return self._adapter.get_band_file(band_name)

    def get_mask_file(self, mask_file_def: MaskFileDef) -> str | None:
        return self._adapter.get_mask_file(mask_file_def)
