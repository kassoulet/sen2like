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
"""Build a Sentinel-2-like SAFE L1C product.

Adapted from ``prisma4sen2like/prisma/product_builder.py``: it creates the SAFE
folder tree, renders the (reused) MTD templates with jinja2, writes the reframed
band images and the QI masks. Re-projection/reframing happens lazily in the
adapter's ``get_band_file``.
"""
from __future__ import annotations

import logging
import os
import shutil

from jinja2 import Environment, FileSystemLoader, select_autoescape

from safe_adapter import BAND_LIST, MaskFileDef
from safe_product import MASK_TABLE, SafeProduct
from utils import utc_format

logger = logging.getLogger(__name__)


class SafeBuilder:
    """Build/package a Sentinel-2-like SAFE L1C product."""

    _TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "product_template")

    def __init__(self, product: SafeProduct, work_dir: str, dest_dir: str):
        self._product = product
        self._product_dir = os.path.join(work_dir, product.product_name)
        self._dest_dir = dest_dir
        self._env = Environment(loader=FileSystemLoader(self._TEMPLATE_DIR), autoescape=select_autoescape())
        self._created_masks: list[MaskFileDef] = []

    def build(self) -> str:
        """Build the product in the working dir, move it to the destination, return its path."""
        self._create_product_structure()
        self._create_mask_files()
        self._create_band_images_file()
        self._render_product_mtd()
        self._render_datastrip_mtd()
        self._render_tile_mtd()

        dest = os.path.join(self._dest_dir, self._product.product_name)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.move(self._product_dir, dest)
        logger.info("SAFE product available in %s", dest)
        return dest

    # --- path helpers -----------------------------------------------------------
    @property
    def _datastrip_path(self):
        return os.path.join(self._product_dir, "DATASTRIP", self._product.datastrip_identifier)

    @property
    def _granule_path(self):
        return os.path.join(self._product_dir, "GRANULE", self._product.short_granule_identifier)

    @property
    def _granule_img_data_path(self):
        return os.path.join(self._granule_path, "IMG_DATA")

    @property
    def _granule_qi_data_path(self):
        return os.path.join(self._granule_path, "QI_DATA")

    @property
    def _relative_granule_qi_data_path(self):
        return "/".join(["GRANULE", self._product.short_granule_identifier, "QI_DATA"])

    # --- structure --------------------------------------------------------------
    def _create_product_structure(self):
        logger.info("Create SAFE folder tree in %s", self._product_dir)
        os.makedirs(self._product_dir)
        os.makedirs(os.path.join(self._product_dir, "AUX_DATA"))
        os.makedirs(os.path.join(self._product_dir, "HTML"))
        rep_info = os.path.join(self._product_dir, "rep_info")
        os.makedirs(rep_info)
        os.makedirs(self._datastrip_path)
        os.makedirs(os.path.join(self._datastrip_path, "QI_DATA"))
        os.makedirs(self._granule_path)
        os.makedirs(os.path.join(self._granule_path, "AUX_DATA"))
        os.makedirs(self._granule_img_data_path)
        os.makedirs(self._granule_qi_data_path)
        shutil.copy(
            os.path.join(self._TEMPLATE_DIR, "rep_info", "S2_User_Product_Level-1C_Metadata.xsd"), rep_info
        )

    def _create_band_images_file(self):
        for band in BAND_LIST:
            band_file = self._product.get_band_file(band)
            dest = os.path.join(self._granule_img_data_path, self._product.get_image_filename(band) + ".TIF")
            shutil.copyfile(band_file, dest)
        logger.info("Wrote %d band images", len(BAND_LIST))

    def _create_mask_files(self):
        for mask_def in MASK_TABLE:
            mask_path = self._product.get_mask_file(mask_def)
            if mask_path:
                shutil.copyfile(mask_path, os.path.join(self._granule_qi_data_path, mask_def.value))
                self._created_masks.append(mask_def)
            else:
                logger.warning("Mask %s not produced", mask_def.value)

    # --- metadata rendering -----------------------------------------------------
    def _render(self, template_name: str, out_path: str, **context):
        template = self._env.get_template(template_name)
        with open(out_path, "w", encoding="UTF-8") as handle:
            handle.write(template.render(**context))

    def _render_product_mtd(self):
        logger.info("Render MTD_MSIL1C.xml")
        self._render(
            "MTD_MSIL1C.xml",
            os.path.join(self._product_dir, "MTD_MSIL1C.xml"),
            product=self._product,
            product_start=utc_format(self._product.product_start_time),
            product_stop=utc_format(self._product.product_stop_time),
            datatake_sensing_start=utc_format(self._product.datatake_sensing_start),
            generation_time=self._product.product_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        )

    def _render_datastrip_mtd(self):
        logger.info("Render MTD_DS.xml")
        self._render(
            "DATASTRIP/DS_ID/MTD_DS.xml",
            os.path.join(self._datastrip_path, "MTD_DS.xml"),
            product=self._product,
            datatake_sensing_start=utc_format(self._product.datatake_sensing_start),
            datastrip_sensing_start=utc_format(self._product.datastrip_sensing_start),
            datastrip_sensing_stop=utc_format(self._product.datastrip_sensing_stop),
            processing_time=self._product.processing_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def _render_tile_mtd(self):
        logger.info("Render MTD_TL.xml")
        self._render(
            "GRANULE/TL_ID/MTD_TL.xml",
            os.path.join(self._granule_path, "MTD_TL.xml"),
            product=self._product,
            sensing_time=self._product.tile_sensing_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            granule_qi_path=self._relative_granule_qi_data_path,
            mask_files=self._created_masks,
        )
