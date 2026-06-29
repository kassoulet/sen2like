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
"""chime4sen2like preprocessor entry point.

Reads a hyperspectral L1 product and spectrally harmonises it onto the CHIME band
set, producing CHIME-band TOA radiance/reflectance rasters that downstream steps
(MGRS re-projection + SAFE packaging) turn into a sen2like-ingestable L1C product.

Current status:
  * spectral harmonisation (section 5.6) -- implemented and unit-tested;
  * generic GeoTIFF-cube reader -- implemented (works on orthorectified cubes);
  * mission-specific readers, L1B ortho-rectification and SAFE packaging -- TODO
    (see README "Roadmap").
"""
import logging
import os
import sys
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, ArgumentTypeError

from band_definitions import chime_band_set
from chime_aggregation import ChimeAggregation
from hs_product import GeoTiffHyperspectralProduct
from log import configure_logging
from version import __version__ as version

logger = logging.getLogger(__name__)


def _validate_file(file_path):
    if not os.path.isfile(file_path):
        raise ArgumentTypeError(f"{file_path} is not an existing file")
    return file_path


def _validate_folder(folder_path):
    if not os.path.isdir(folder_path):
        raise ArgumentTypeError(f"{folder_path} is not an existing directory")
    return folder_path


_arg_parser = ArgumentParser(
    prog="chime4sen2like",
    formatter_class=ArgumentDefaultsHelpFormatter,
    description="Spectrally harmonise a hyperspectral cube onto the CHIME band set.",
)
_arg_parser.add_argument(
    dest="cube_file", type=_validate_file, help="Multi-band hyperspectral GeoTIFF cube", metavar="HS_CUBE"
)
_arg_parser.add_argument(
    dest="band_csv",
    type=_validate_file,
    help="Source band-definition CSV (band, central_wavelength_nm, fwhm_nm)",
    metavar="BAND_CSV",
)
_arg_parser.add_argument(
    dest="working_dir", type=_validate_folder, help="Working / output directory", metavar="WORKING_DIR"
)
_arg_parser.add_argument(
    "--target-band-csv",
    default=None,
    help="Target band CSV (defaults to the shipped CHIME placeholder band set)",
)
_arg_parser.add_argument("--sun-zenith", type=float, default=30.0, help="Scene solar zenith angle (deg)")
_arg_parser.add_argument("--debug", action="store_true", help="Verbose logging")


def main(argv: list[str]) -> int:
    args = _arg_parser.parse_args(argv)
    configure_logging(args.debug, False)

    logger.info("Start chime4sen2like %s with Python %s", version, sys.version.split()[0])

    work_dir = os.path.join(args.working_dir, str(round(time.time() * 1000)))
    os.mkdir(work_dir)
    logger.info("Working dir: %s", work_dir)

    if args.target_band_csv:
        from band_definitions import BandSet

        target = BandSet.from_csv(args.target_band_csv, name="CHIME")
    else:
        target = chime_band_set()

    product = GeoTiffHyperspectralProduct(
        cube_path=args.cube_file,
        band_csv_path=args.band_csv,
        sun_zenith=args.sun_zenith,
    )

    aggregation = ChimeAggregation(product, target, work_dir)
    aggregation.process()

    # TODO (see README "Roadmap"):
    #   * MGRS re-projection / reframing to the Sentinel-2 grid
    #     (reuse sen2like grids/mgrs_framing.py, as prisma4sen2like does);
    #   * SAFE packaging into a CHIME-internal L1C product
    #     (reuse the prisma4sen2like product_builder + adapter).
    logger.info("Spectral harmonisation finished. SAFE packaging is not yet wired (see README).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
