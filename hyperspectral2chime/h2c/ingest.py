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
"""Ingest a hyperspectral source into a CHIME L1C product.

Two steps, both reusing existing components:

1. Spectral harmonisation (roadmap section 5.6): aggregate the source bands onto
   the CHIME band set via :mod:`spectral_aggregation`.
2. Geometry refinement (roadmap section 5.1): reframe onto the CHIME grid
   (Sentinel-2 MGRS, 30 m GSD) via ``gdal.Warp``.

The result is a :class:`chime_product.ChimeProduct` at L1C, ready for the L2
processing chain.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

from band_definitions import BandSet
from chime_product import CHIME_GSD, ChimeProduct, read_cube, write_cube
from mgrs_tiling import TileInfo
from spectral_aggregation import aggregate_cube, compute_aggregation_matrix, uncovered_target_bands

logger = logging.getLogger(__name__)


def ingest_cube(
    source_cube_path: str,
    source_band_set: BandSet,
    target_band_set: BandSet,
    tile: TileInfo,
    work_dir: str,
    acquisition_datetime: datetime,
    sun_zenith: float,
    sun_azimuth: float,
) -> ChimeProduct:
    """Aggregate a projected hyperspectral cube onto CHIME bands and reframe to the tile.

    Args:
        source_cube_path: projected multi-band hyperspectral GeoTIFF (reflectance).
        source_band_set: source spectral configuration.
        target_band_set: CHIME band set.
        tile: target MGRS tile (CHIME uses the S2 MGRS grid).
        work_dir: working directory.
        acquisition_datetime, sun_zenith, sun_azimuth: acquisition metadata.

    Returns:
        ChimeProduct at L1C on the CHIME grid (30 m).
    """
    from osgeo import gdal

    # 1. spectral harmonisation: source -> CHIME bands
    logger.info(
        "Spectral harmonisation %s (%d bands) -> %s (%d bands)",
        source_band_set.name, source_band_set.n_bands, target_band_set.name, target_band_set.n_bands,
    )
    aggregation = compute_aggregation_matrix(source_band_set, target_band_set)
    uncovered = uncovered_target_bands(target_band_set, aggregation)
    if uncovered:
        logger.warning("%d CHIME band(s) outside source spectral range (zero-filled)", len(uncovered))

    source_cube = read_cube(source_cube_path)
    chime_cube = aggregate_cube(source_cube, aggregation)  # (rows, cols, n_chime)

    # write the aggregated cube preserving the source grid, then reframe
    src_gt, src_proj, _, _ = _grid_of(source_cube_path)
    aggregated_path = os.path.join(work_dir, "chime_bands_src_grid.tif")
    write_cube(aggregated_path, chime_cube, src_gt, src_proj, nodata=0.0)

    # 2. geometry refinement: reframe onto the CHIME grid (MGRS tile @ 30 m)
    reframed_path = os.path.join(work_dir, "chime_L1C.tif")
    logger.info("Reframe to tile %s (EPSG:%s) @ %.0f m", tile.tile_id, tile.epsg, CHIME_GSD)
    gdal.Warp(
        reframed_path,
        aggregated_path,
        options=gdal.WarpOptions(
            outputType=gdal.GDT_Float32,
            dstSRS=f"EPSG:{tile.epsg}",
            outputBounds=tile.bounds,
            xRes=CHIME_GSD,
            yRes=CHIME_GSD,
            resampleAlg="bilinear",
            dstNodata=0.0,
            creationOptions=["COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
        ),
    )

    return ChimeProduct(
        raster_path=reframed_path,
        band_set=target_band_set,
        tile=tile,
        acquisition_datetime=acquisition_datetime,
        sun_zenith=sun_zenith,
        sun_azimuth=sun_azimuth,
        processing_level="L1C",
        provenance=[
            f"spectral harmonisation {source_band_set.name}->{target_band_set.name}",
            f"reframed to {tile.tile_id} @ {CHIME_GSD:.0f} m",
        ],
    )


def _grid_of(raster_path: str):
    from osgeo import gdal

    dataset = gdal.Open(raster_path)
    return dataset.GetGeoTransform(), dataset.GetProjection(), dataset.RasterXSize, dataset.RasterYSize
