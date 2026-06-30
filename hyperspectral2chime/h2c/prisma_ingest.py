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
"""Ingest a PRISMA swath product into a CHIME L1C product.

PRISMA L1 data is in swath (sensor) geometry with per-pixel lat/lon grids, so the
ingest differs from the projected-GeoTIFF path (:mod:`ingest`):

1. Spectral harmonisation (roadmap 5.6): aggregate the swath radiance cube onto the
   CHIME band set via :mod:`spectral_aggregation`.
2. Geometry refinement / ortho-rectification (roadmap 5.1): project the aggregated
   swath cube onto the CHIME grid (Sentinel-2 MGRS, 30 m) using the geolocation
   grids, via a GDAL GEOLOCATION-array VRT + ``gdal.Warp`` — the same technique as
   ``prisma4sen2like/prisma/geometry.py``.

The output is a :class:`chime_product.ChimeProduct` at L1C, carrying TOA *radiance*
(reflectance conversion needs a CHIME solar-irradiance model, e.g. TSIS — TODO).
"""
from __future__ import annotations

import logging
import os

from band_definitions import BandSet
from chime_product import CHIME_GSD, ChimeProduct
from mgrs_tiling import TileInfo
from prisma_reader import PrismaReader
from spectral_aggregation import aggregate_cube, compute_aggregation_matrix, uncovered_target_bands

logger = logging.getLogger(__name__)


def _write_grid(path, array):
    from osgeo import gdal
    import numpy as np

    array = np.ascontiguousarray(array, dtype=np.float32)  # rot90 views are non-contiguous
    rows, cols = array.shape
    ds = gdal.GetDriverByName("GTiff").Create(path, cols, rows, 1, gdal.GDT_Float32)
    ds.GetRasterBand(1).WriteArray(array)
    ds = None
    return path


def _write_swath_cube(path, cube):
    """Write a (rows, cols, bands) cube as a plain (un-georeferenced) GeoTIFF."""
    from osgeo import gdal
    import numpy as np

    rows, cols, bands = cube.shape
    ds = gdal.GetDriverByName("GTiff").Create(path, cols, rows, bands, gdal.GDT_Float32, options=["BIGTIFF=IF_SAFER"])
    for b in range(bands):
        ds.GetRasterBand(b + 1).WriteArray(np.ascontiguousarray(cube[:, :, b], dtype=np.float32))
    ds = None
    return path


def ingest_prisma(
    reader: PrismaReader,
    target_band_set: BandSet,
    tile: TileInfo,
    work_dir: str,
    gsd: float = CHIME_GSD,
) -> ChimeProduct:
    """Aggregate a PRISMA product onto CHIME bands and ortho-rectify to the CHIME grid."""
    from osgeo import gdal

    source = reader.band_set
    logger.info(
        "Spectral harmonisation %s (%d bands) -> %s (%d bands)",
        source.name, source.n_bands, target_band_set.name, target_band_set.n_bands,
    )
    aggregation = compute_aggregation_matrix(source, target_band_set)
    uncovered = uncovered_target_bands(target_band_set, aggregation)
    if uncovered:
        logger.warning("%d CHIME band(s) outside the PRISMA spectral range (zero-filled)", len(uncovered))

    radiance = reader.read_radiance_cube()             # (rows, cols, n_source)
    chime_swath = aggregate_cube(radiance, aggregation)  # (rows, cols, n_target)
    logger.info("Aggregated swath cube: %s", chime_swath.shape)

    swath_path = _write_swath_cube(os.path.join(work_dir, "prisma_chime_swath.tif"), chime_swath)
    lat, lon, _ = reader.geolocation_grids()
    lat_path = _write_grid(os.path.join(work_dir, "lat.tif"), lat)
    lon_path = _write_grid(os.path.join(work_dir, "lon.tif"), lon)

    # Attach a GEOLOCATION array (per-pixel lon/lat in EPSG:4326) to a VRT of the
    # swath cube, then warp with METHOD=GEOLOC_ARRAY. This is the robust GDAL-API
    # equivalent of the prisma4sen2like geolocation VRT.
    out_path = os.path.join(work_dir, "chime_L1C.tif")
    logger.info("Ortho-rectify to tile %s (EPSG:%s) @ %.0f m via geolocation grids", tile.tile_id, tile.epsg, gsd)
    from osgeo import osr

    srs4326 = osr.SpatialReference()
    srs4326.ImportFromEPSG(4326)
    wkt4326 = srs4326.ExportToWkt()

    vrt_path = os.path.join(work_dir, "prisma_chime_geoloc.vrt")
    vrt = gdal.Translate(vrt_path, swath_path, format="VRT")
    vrt.SetMetadata(
        {
            "SRS": wkt4326,
            "X_DATASET": lon_path, "X_BAND": "1",
            "Y_DATASET": lat_path, "Y_BAND": "1",
            "PIXEL_OFFSET": "0", "LINE_OFFSET": "0",
            "PIXEL_STEP": "1", "LINE_STEP": "1",
            "GEOREFERENCING_CONVENTION": "PIXEL_CENTER",
        },
        "GEOLOCATION",
    )
    vrt.FlushCache()
    vrt = None  # close so the GEOLOCATION metadata is persisted to the .vrt on disk
    # NB: GDAL logs "Too many points failed to transform / unable to compute output
    # bounds" here — this is benign. The swath only covers part of the tile, so the
    # inverse geoloc transform fails for tile corners outside the swath; because we
    # pass explicit outputBounds, the warp proceeds and fills the covered footprint.
    gdal.Warp(
        out_path,
        vrt_path,
        options=gdal.WarpOptions(
            dstSRS=f"EPSG:{tile.epsg}",
            outputBounds=tile.bounds,
            xRes=gsd,
            yRes=gsd,
            resampleAlg="bilinear",
            dstNodata=0.0,
            transformerOptions=["METHOD=GEOLOC_ARRAY"],
            creationOptions=["COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
        ),
    )

    return ChimeProduct(
        raster_path=out_path,
        band_set=target_band_set,
        tile=tile,
        acquisition_datetime=reader.acquisition_datetime,
        sun_zenith=reader.sun_zenith_angle,
        sun_azimuth=reader.sun_azimuth_angle,
        processing_level="L1C",
        radiometric_unit="TOA_radiance",
        provenance=[
            f"PRISMA {source.n_bands} bands -> {target_band_set.name} {target_band_set.n_bands} bands",
            f"ortho-rectified to {tile.tile_id} @ {gsd:.0f} m (geolocation array)",
        ],
    )
