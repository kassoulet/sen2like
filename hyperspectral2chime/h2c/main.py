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
"""hyperspectral2chime entry point.

Harmonise a hyperspectral product onto CHIME and produce a CHIME L2H product.

Current input path: a projected multi-band hyperspectral GeoTIFF cube + a band
CSV (the generic path; a native PRISMA reader is the next step). The chain is:
ingest (spectral harmonisation -> CHIME bands; reframe to the CHIME grid @ 30 m)
then the band-agnostic L2 pipeline (TOA, inter-calibration, L2H packaging).
"""
import logging
import os
import sys
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, ArgumentTypeError
from datetime import datetime, timezone

from band_definitions import BandSet, chime_band_set
from blocks import InterCalibrationBlock, ToaBlock
from ingest import ingest_cube
from log import configure_logging
from mgrs_tiling import resolve_tile
from packager import PackagerL2H
from pipeline import ChimePipeline
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
    prog="hyperspectral2chime",
    formatter_class=ArgumentDefaultsHelpFormatter,
    description="Harmonise a hyperspectral cube onto CHIME and produce a CHIME L2H product.",
)
_parser.add_argument("cube_file", type=_file, help="Projected multi-band hyperspectral GeoTIFF cube", metavar="HS_CUBE")
_parser.add_argument("band_csv", type=_file, help="Source band CSV (band, central_wavelength_nm, fwhm_nm)", metavar="BAND_CSV")
_parser.add_argument("working_dir", type=_folder, help="Working / output directory", metavar="WORKING_DIR")
_parser.add_argument("--tile", required=True, help="CHIME/MGRS tile code, e.g. 31TFJ")
_parser.add_argument("--target-band-csv", default=None, help="CHIME band CSV (defaults to the shipped placeholder)")
_parser.add_argument("--sun-zenith", type=float, default=30.0, help="Scene solar zenith angle (deg)")
_parser.add_argument("--sun-azimuth", type=float, default=160.0, help="Scene solar azimuth angle (deg)")
_parser.add_argument("--debug", action="store_true", help="Verbose logging")


def main(argv: list[str]) -> int:
    args = _parser.parse_args(argv)
    configure_logging(args.debug, False)
    logger.info("Start hyperspectral2chime %s with Python %s", version, sys.version.split()[0])

    work_dir = os.path.join(args.working_dir, str(round(time.time() * 1000)))
    os.mkdir(work_dir)

    source_band_set = BandSet.from_csv(args.band_csv, name="source")
    target_band_set = BandSet.from_csv(args.target_band_csv, name="CHIME") if args.target_band_csv else chime_band_set()
    tile = resolve_tile(args.tile)

    product = ingest_cube(
        source_cube_path=args.cube_file,
        source_band_set=source_band_set,
        target_band_set=target_band_set,
        tile=tile,
        work_dir=work_dir,
        acquisition_datetime=datetime.now(timezone.utc),
        sun_zenith=args.sun_zenith,
        sun_azimuth=args.sun_azimuth,
    )

    pipeline = ChimePipeline([ToaBlock(), InterCalibrationBlock(), PackagerL2H(args.working_dir)])
    product = pipeline.run(product, work_dir)

    logger.info("Done. CHIME %s product at: %s", product.processing_level, product.raster_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
