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
"""CHIME L2H / L2F product packager (internal format).

Writes a band-count-agnostic CHIME product: a single multi-band Float32 GeoTIFF of
the CHIME-band cube plus a JSON metadata sidecar, under a product directory. This
intentionally avoids the fixed Sentinel-2 SAFE structure so it scales to CHIME's
~210 narrow bands.
"""
from __future__ import annotations

import logging
import os
import shutil

from blocks import ChimeBlock
from chime_product import ChimeProduct

logger = logging.getLogger(__name__)


class PackagerL2H(ChimeBlock):
    """Finalise the product as a CHIME L2H internal product."""

    name = "PackagerL2H"
    level = "L2H"

    def __init__(self, dest_dir: str):
        self._dest_dir = dest_dir

    def process(self, product: ChimeProduct, work_dir: str) -> ChimeProduct:
        product.processing_level = self.level
        product.provenance.append(self.name)

        product_dir = os.path.join(self._dest_dir, product.basename)
        img_dir = os.path.join(product_dir, "IMG_DATA")
        os.makedirs(img_dir, exist_ok=True)

        cube_dest = os.path.join(img_dir, product.basename + ".tif")
        shutil.copyfile(product.raster_path, cube_dest)
        product.raster_path = cube_dest

        product.write_metadata(os.path.join(product_dir, product.basename + ".json"))

        logger.info("CHIME %s product written: %s", self.level, product_dir)
        product.provenance.append(f"packaged {product_dir}")
        # stash the product dir for the caller
        self.product_dir = product_dir
        return product
