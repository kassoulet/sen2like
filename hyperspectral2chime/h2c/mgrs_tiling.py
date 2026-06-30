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
"""MGRS tiling helper (dependency-light).

prisma4sen2like resolves MGRS tile geometry through its vendored
``sen2like/grids.py``, which needs ``pandas``, ``shapely`` and ``mgrs`` plus a
40 MB ``s2tiles.db`` copy. Here we reuse the *same* Sentinel-2 tiling database
already present in the repository, but query it with the standard-library
``sqlite3`` and parse the tile geometry with ``osgeo.ogr`` — no pandas/shapely/
mgrs and no duplicated database.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass

from osgeo import ogr

logger = logging.getLogger(__name__)

# Known locations of the Sentinel-2 tiling DB shipped in the repository.
_DB_CANDIDATES = (
    os.path.join("sen2like", "sen2like", "core", "product_archive", "data", "s2tiles.db"),
    os.path.join("prisma4sen2like", "prisma", "sen2like", "s2tiles.db"),
)


@dataclass(frozen=True)
class TileInfo:
    """MGRS tile geo information."""

    tile_id: str
    epsg: str
    bounds: tuple[float, float, float, float]  # (minx, miny, maxx, maxy) in tile UTM

    @property
    def ulx(self) -> float:
        return self.bounds[0]

    @property
    def uly(self) -> float:
        return self.bounds[3]


def find_s2tiles_db(explicit_path: str | None = None) -> str:
    """Locate an existing s2tiles.db.

    Args:
        explicit_path: if given, used directly.

    Returns:
        Path to a readable s2tiles.db.

    Raises:
        FileNotFoundError: if no database can be found.
    """
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(f"s2tiles.db not found at {explicit_path}")
        return explicit_path

    # search relative to the repo root (two levels above this package)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for candidate in _DB_CANDIDATES:
        path = os.path.join(repo_root, candidate)
        if os.path.isfile(path):
            return path
    # also try current working directory layout
    for candidate in _DB_CANDIDATES:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        "Could not locate s2tiles.db. Pass an explicit path (looked for: "
        + "; ".join(_DB_CANDIDATES)
        + ")"
    )


def resolve_tile(tile_code: str, db_path: str | None = None) -> TileInfo:
    """Resolve an MGRS tile code to its EPSG and UTM bounds.

    Args:
        tile_code: 5-char MGRS tile id, optionally prefixed with 'T' (e.g. '31TFJ').
        db_path: optional explicit s2tiles.db path.

    Returns:
        TileInfo with EPSG and (minx, miny, maxx, maxy) UTM bounds.
    """
    code = tile_code[1:] if tile_code.upper().startswith("T") else tile_code
    code = code.upper()

    database = find_s2tiles_db(db_path)
    connection = sqlite3.connect(database)
    try:
        row = connection.execute(
            "SELECT TILE_ID, EPSG, UTM_WKT FROM s2tiles WHERE TILE_ID=?", (code,)
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        raise ValueError(f"MGRS tile '{code}' not found in {database}")

    tile_id, epsg, utm_wkt = row
    geometry = ogr.CreateGeometryFromWkt(utm_wkt)
    if geometry is None:
        raise ValueError(f"could not parse UTM geometry for tile '{code}'")
    minx, maxx, miny, maxy = geometry.GetEnvelope()  # ogr returns (minX, maxX, minY, maxY)

    logger.info("Tile %s -> EPSG:%s bounds=(%.1f, %.1f, %.1f, %.1f)", tile_id, epsg, minx, miny, maxx, maxy)
    return TileInfo(tile_id=str(tile_id), epsg=str(epsg), bounds=(minx, miny, maxx, maxy))
