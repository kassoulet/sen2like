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
"""prisma2chime entry point.

Harmonise a native PRISMA L1 product onto CHIME and produce a CHIME L2H product:
read PRISMA -> aggregate onto the CHIME band set -> ortho-rectify to the CHIME grid
(S2-MGRS @ 30 m) -> band-agnostic L2 pipeline -> CHIME L2H.

Requires h5py + gdal (e.g. the sen2like conda env after `pip install h5py`).
"""
import logging
import os
import sys
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, ArgumentTypeError

from band_definitions import BandSet, chime_band_set
from blocks import InterCalibrationBlock, ToaBlock
from log import configure_logging
from mgrs_tiling import resolve_tile
from packager import PackagerL2H
from pipeline import ChimePipeline
from prisma_ingest import ingest_prisma
from prisma_reader import PrismaReader
from version import __version__ as version

logger = logging.getLogger(__name__)


def _file(path):
    if not os.path.isfile(path):
        raise ArgumentTypeError(f"{path} is not an existing file")
    return path


def _folder(path):
    if not os.path.isdir(path):
        raise ArgumentTypeError(f"{path} is not an existing directory")
    return path


_parser = ArgumentParser(
    prog="prisma2chime",
    formatter_class=ArgumentDefaultsHelpFormatter,
    description="Harmonise a PRISMA L1 product onto CHIME and produce a CHIME L2H product.",
)
_parser.add_argument("prisma_file", type=_file, help="PRISMA L1 .he5 product", metavar="PRISMA_L1")
_parser.add_argument("working_dir", type=_folder, help="Working / output directory", metavar="WORKING_DIR")
_parser.add_argument("--tile", required=True, help="CHIME/MGRS tile code, e.g. 31TFJ")
_parser.add_argument("--target-band-csv", default=None, help="CHIME band CSV (defaults to the shipped placeholder)")
_parser.add_argument("--debug", action="store_true", help="Verbose logging")


def main(argv: list[str]) -> int:
    args = _parser.parse_args(argv)
    configure_logging(args.debug, False)
    logger.info("Start prisma2chime %s with Python %s", version, sys.version.split()[0])

    work_dir = os.path.join(args.working_dir, str(round(time.time() * 1000)))
    os.mkdir(work_dir)

    target = BandSet.from_csv(args.target_band_csv, name="CHIME") if args.target_band_csv else chime_band_set()
    tile = resolve_tile(args.tile)

    reader = PrismaReader(args.prisma_file)
    logger.info("PRISMA scene centre (lat,lon): %s | sun zen/azi %.2f/%.2f",
                tuple(round(x, 4) for x in reader.scene_center), reader.sun_zenith_angle, reader.sun_azimuth_angle)

    product = ingest_prisma(reader, target, tile, work_dir)
    pipeline = ChimePipeline([ToaBlock(), InterCalibrationBlock(), PackagerL2H(args.working_dir)])
    product = pipeline.run(product, work_dir)

    logger.info("Done. CHIME %s (%s) product at: %s", product.processing_level, product.radiometric_unit, product.raster_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
