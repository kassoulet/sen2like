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
"""CHIME processing pipeline: run an ordered list of band-agnostic blocks."""
from __future__ import annotations

import logging

from blocks import ChimeBlock
from chime_product import ChimeProduct

logger = logging.getLogger(__name__)


class ChimePipeline:
    """Run a sequence of :class:`blocks.ChimeBlock` over a ChimeProduct."""

    def __init__(self, blocks: list[ChimeBlock]):
        self._blocks = blocks

    def run(self, product: ChimeProduct, work_dir: str) -> ChimeProduct:
        for block in self._blocks:
            logger.info("----- %s -----", block.name)
            product = block.process(product, work_dir)
        return product
