# Copyright (c) 2023 ESA.
#
# This file is part of sen2like.
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

"""
Stitch two adjacent Landsat WRS-2 L1TP scenes into a single L1TP-like product
directory, so sen2cor can run once on a scene that fully covers the target
MGRS tile instead of running independently on two partial scenes whose
resulting L2A outputs (each with its own scene-level atmospheric correction)
cannot be safely stitched together afterward.
"""
import logging
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import numpy as np
from core.metadata_extraction import NOT_FOUND, reg_exp
from osgeo import gdal, osr

logger = logging.getLogger("Sen2Like")

_SCENE_BASE_NAME_RE = re.compile(r"(L[COTEM]0[1-9]_L\w+_\d+_\d+_\d+_\d+_T\d)")

_MERGE_CREATION_OPTIONS = [
    "COMPRESS=LZW",
    "TILED=YES",
    "BIGTIFF=IF_NEEDED",
    "BLOCKXSIZE=256",
    "BLOCKYSIZE=256",
    "NUM_THREADS=ALL_CPUS",
]


class LandsatStitchError(Exception):
    pass


def stitch_l1tp_products(scene1_dir: str, scene2_dir: str, output_dir: str) -> str:
    """Merge all raster files common to both scenes, plus MTL metadata,
    into a single L1TP-like product directory.

    The merged product keeps scene1's identifiers/base name.

    Args:
        scene1_dir: path of the primary scene
        scene2_dir: path of the adjacent scene to merge in
        output_dir: directory to write the merged product into

    Returns:
        output_dir
    """
    common_rasters = _find_common_rasters(scene1_dir, scene2_dir)
    if not common_rasters:
        raise LandsatStitchError(f"No common raster files found between {scene1_dir} and {scene2_dir}")

    os.makedirs(output_dir, exist_ok=True)
    try:
        base_name = _scene_base_name(scene1_dir)
        primary_suffix = "B1" if "B1" in common_rasters else sorted(common_rasters)[0]

        logger.info(
            "Stitching %d rasters from %s and %s into %s", len(common_rasters), scene1_dir, scene2_dir, output_dir
        )

        extent_info = None
        with ThreadPoolExecutor(max_workers=min(len(common_rasters), os.cpu_count() or 1)) as executor:
            futures = {
                executor.submit(
                    _merge_raster,
                    path1,
                    path2,
                    os.path.join(output_dir, f"{base_name}_{suffix}.TIF"),
                    suffix,
                    suffix == primary_suffix,
                ): suffix
                for suffix, (path1, path2) in common_rasters.items()
            }
            for future, suffix in futures.items():
                band_extent = future.result()
                if band_extent is not None:
                    extent_info = band_extent
                logger.debug("Merged raster %s", suffix)

        if extent_info is None:
            raise LandsatStitchError(f"Failed to compute merged scene extent from band {primary_suffix}")

        _write_merged_mtl(scene1_dir, scene2_dir, output_dir, extent_info)
        _copy_primary_ang(scene1_dir, scene2_dir, output_dir)

        return output_dir
    except Exception:
        logger.error("Stitching failed, cleaning up partial output %s", output_dir, exc_info=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def _scene_base_name(product_dir: str) -> str:
    name = os.path.basename(os.path.normpath(product_dir))
    match = _SCENE_BASE_NAME_RE.match(name)
    if not match:
        raise LandsatStitchError(f"Not a recognized Landsat L1TP product directory name: {name}")
    return match.group(1)


def _find_common_rasters(scene1_dir: str, scene2_dir: str) -> dict:
    """Find raster files present, with the same band/suffix, in both scenes.

    Returns:
        dict[str, tuple[str, str]]: suffix -> (scene1_file, scene2_file)
    """

    def suffixes(scene_dir: str, base_name: str) -> dict:
        found = {}
        for filepath in glob(os.path.join(scene_dir, f"{base_name}_*.TIF")) + glob(
            os.path.join(scene_dir, f"{base_name}_*.tif")
        ):
            filename = os.path.basename(filepath)
            ext = os.path.splitext(filename)[1]
            found[filename[len(base_name) + 1 : -len(ext)]] = filepath
        return found

    files1 = suffixes(scene1_dir, _scene_base_name(scene1_dir))
    files2 = suffixes(scene2_dir, _scene_base_name(scene2_dir))

    return {suffix: (files1[suffix], files2[suffix]) for suffix in files1.keys() & files2.keys()}


def _raster_corners(dataset) -> np.ndarray:
    gt = dataset.GetGeoTransform()
    cols, rows = dataset.RasterXSize, dataset.RasterYSize
    corners = []
    for px in (0, cols):
        for py in (0, rows):
            corners.append((gt[0] + px * gt[1] + py * gt[2], gt[3] + px * gt[4] + py * gt[5]))
    return np.array(corners)


def _combined_geometry(ds1, ds2) -> tuple:
    """Compute the pixel-aligned geotransform and size of the union extent of ds1 and ds2."""
    corners1 = _raster_corners(ds1)
    corners2 = _raster_corners(ds2)

    min_x = min(corners1[:, 0].min(), corners2[:, 0].min())
    max_x = max(corners1[:, 0].max(), corners2[:, 0].max())
    min_y = min(corners1[:, 1].min(), corners2[:, 1].min())
    max_y = max(corners1[:, 1].max(), corners2[:, 1].max())

    gt1 = ds1.GetGeoTransform()
    pixel_width = abs(gt1[1])
    pixel_height = abs(gt1[5])

    raster_width = int(round((max_x - min_x) / pixel_width))
    raster_height = int(round((max_y - min_y) / pixel_height))
    geotransform = (min_x, pixel_width, 0, max_y, 0, -pixel_height)

    return geotransform, raster_width, raster_height


def _rect(dataset, merged_geotransform: tuple) -> dict:
    """Source and destination rectangles for `dataset` within the merged extent."""
    ds_gt = dataset.GetGeoTransform()
    ds_min_x = ds_gt[0]
    ds_max_y = ds_gt[3]
    ds_max_x = ds_min_x + ds_gt[1] * dataset.RasterXSize
    ds_min_y = ds_max_y + ds_gt[5] * dataset.RasterYSize

    merged_min_x, _, _, merged_max_y, _, merged_res_y = merged_geotransform
    merged_res_x = merged_geotransform[1]

    return {
        "src": {"x": 0, "y": 0, "width": dataset.RasterXSize, "height": dataset.RasterYSize},
        "dst": {
            "x": int(round((ds_min_x - merged_min_x) / merged_res_x)),
            "y": int(round((merged_max_y - ds_max_y) / (-merged_res_y))),
            "width": int(round((ds_max_x - ds_min_x) / merged_res_x)),
            "height": int(round((ds_max_y - ds_min_y) / (-merged_res_y))),
        },
    }


def _write_merge_vrt(ds1, ds2, geotransform, width, height, nodata_value, vrt_path):
    rect1 = _rect(ds1, geotransform)
    rect2 = _rect(ds2, geotransform)
    datatype = gdal.GetDataTypeName(ds1.GetRasterBand(1).DataType)

    # scene2 is drawn on top of scene1; NODATA on scene2 lets scene1 show through there
    vrt_xml = f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
    <SRS>{ds1.GetProjection()}</SRS>
    <GeoTransform>{geotransform[0]}, {geotransform[1]}, 0, {geotransform[3]}, 0, {geotransform[5]}</GeoTransform>
    <VRTRasterBand dataType="{datatype}" band="1">
        <ComplexSource>
            <SourceFilename relativeToVRT="0">{ds1.GetDescription()}</SourceFilename>
            <SourceBand>1</SourceBand>
            <SrcRect xOff="{rect1['src']['x']}" yOff="{rect1['src']['y']}"
                    xSize="{rect1['src']['width']}" ySize="{rect1['src']['height']}" />
            <DstRect xOff="{rect1['dst']['x']}" yOff="{rect1['dst']['y']}"
                    xSize="{rect1['dst']['width']}" ySize="{rect1['dst']['height']}" />
        </ComplexSource>
        <ComplexSource>
            <SourceFilename relativeToVRT="0">{ds2.GetDescription()}</SourceFilename>
            <SourceBand>1</SourceBand>
            <SrcRect xOff="{rect2['src']['x']}" yOff="{rect2['src']['y']}"
                    xSize="{rect2['src']['width']}" ySize="{rect2['src']['height']}" />
            <DstRect xOff="{rect2['dst']['x']}" yOff="{rect2['dst']['y']}"
                    xSize="{rect2['dst']['width']}" ySize="{rect2['dst']['height']}" />
            <NODATA>{nodata_value}</NODATA>
        </ComplexSource>
    </VRTRasterBand>
</VRTDataset>"""

    with open(vrt_path, "w", encoding="utf-8") as f:
        f.write(vrt_xml)


def _scene_extent(vrt_dataset) -> dict:
    gt = vrt_dataset.GetGeoTransform()
    cols, rows = vrt_dataset.RasterXSize, vrt_dataset.RasterYSize
    x_min, y_max = gt[0], gt[3]
    x_max = x_min + gt[1] * cols
    y_min = y_max + gt[5] * rows

    src_srs = osr.SpatialReference(vrt_dataset.GetProjection())
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    transform = osr.CoordinateTransformation(src_srs, wgs84_srs)

    ul_lat, ul_lon, _ = transform.TransformPoint(x_min, y_max)
    ur_lat, ur_lon, _ = transform.TransformPoint(x_max, y_max)
    ll_lat, ll_lon, _ = transform.TransformPoint(x_min, y_min)
    lr_lat, lr_lon, _ = transform.TransformPoint(x_max, y_min)

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "ul_lat": ul_lat,
        "ul_lon": ul_lon,
        "ur_lat": ur_lat,
        "ur_lon": ur_lon,
        "ll_lat": ll_lat,
        "ll_lon": ll_lon,
        "lr_lat": lr_lat,
        "lr_lon": lr_lon,
        "cols": cols,
        "rows": rows,
    }


def _merge_raster(path1: str, path2: str, output_path: str, suffix: str, want_extent: bool):
    ds1 = gdal.Open(path1)
    ds2 = gdal.Open(path2)
    if ds1 is None or ds2 is None:
        raise LandsatStitchError(f"Failed to open input rasters for band {suffix}: {path1}, {path2}")

    geotransform, width, height = _combined_geometry(ds1, ds2)
    # USGS Collection 2 QA_PIXEL fill value is 1, every other band/angle grid uses 0
    nodata_value = 1 if "QA_PIXEL" in suffix.upper() else 0

    vrt_path = output_path + ".merge.vrt"
    _write_merge_vrt(ds1, ds2, geotransform, width, height, nodata_value, vrt_path)

    try:
        merged_vrt = gdal.Open(vrt_path)

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(
            output_path, width, height, 1, ds1.GetRasterBand(1).DataType, _MERGE_CREATION_OPTIONS
        )
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(ds1.GetProjection())

        band_nodata = ds1.GetRasterBand(1).GetNoDataValue()
        if band_nodata is not None:
            out_ds.GetRasterBand(1).SetNoDataValue(band_nodata)

        gdal.ReprojectImage(merged_vrt, out_ds, None, None, gdal.GRA_NearestNeighbour)

        return _scene_extent(merged_vrt) if want_extent else None
    finally:
        gdal.Unlink(vrt_path)


def _write_merged_mtl(scene1_dir: str, scene2_dir: str, output_dir: str, extent_info: dict) -> None:
    base_name = _scene_base_name(scene1_dir)
    scene2_base_name = _scene_base_name(scene2_dir)
    scene1_mtl_path = os.path.join(scene1_dir, f"{base_name}_MTL.txt")
    scene2_mtl_path = os.path.join(scene2_dir, f"{scene2_base_name}_MTL.txt")

    if not os.path.exists(scene1_mtl_path) or not os.path.exists(scene2_mtl_path):
        raise LandsatStitchError(f"MTL file missing: {scene1_mtl_path} or {scene2_mtl_path}")

    with open(scene1_mtl_path, "r", encoding="utf-8") as f:
        mtl_text = f.read()
    with open(scene2_mtl_path, "r", encoding="utf-8") as f:
        scene2_mtl_text = f.read()

    cloud_covers = [
        float(v) for v in (reg_exp(mtl_text, "CLOUD_COVER =.*"), reg_exp(scene2_mtl_text, "CLOUD_COVER =.*"))
        if v != NOT_FOUND
    ]
    cloud_cover = max(cloud_covers) if cloud_covers else None

    replacements = {
        r"CORNER_UL_PROJECTION_X_PRODUCT = .*": f'CORNER_UL_PROJECTION_X_PRODUCT = {extent_info["x_min"]}',
        r"CORNER_UL_PROJECTION_Y_PRODUCT = .*": f'CORNER_UL_PROJECTION_Y_PRODUCT = {extent_info["y_max"]}',
        r"CORNER_UR_PROJECTION_X_PRODUCT = .*": f'CORNER_UR_PROJECTION_X_PRODUCT = {extent_info["x_max"]}',
        r"CORNER_UR_PROJECTION_Y_PRODUCT = .*": f'CORNER_UR_PROJECTION_Y_PRODUCT = {extent_info["y_max"]}',
        r"CORNER_LL_PROJECTION_X_PRODUCT = .*": f'CORNER_LL_PROJECTION_X_PRODUCT = {extent_info["x_min"]}',
        r"CORNER_LL_PROJECTION_Y_PRODUCT = .*": f'CORNER_LL_PROJECTION_Y_PRODUCT = {extent_info["y_min"]}',
        r"CORNER_LR_PROJECTION_X_PRODUCT = .*": f'CORNER_LR_PROJECTION_X_PRODUCT = {extent_info["x_max"]}',
        r"CORNER_LR_PROJECTION_Y_PRODUCT = .*": f'CORNER_LR_PROJECTION_Y_PRODUCT = {extent_info["y_min"]}',
        r"CORNER_UL_LAT_PRODUCT = .*": f'CORNER_UL_LAT_PRODUCT = {extent_info["ul_lat"]}',
        r"CORNER_UL_LON_PRODUCT = .*": f'CORNER_UL_LON_PRODUCT = {extent_info["ul_lon"]}',
        r"CORNER_UR_LAT_PRODUCT = .*": f'CORNER_UR_LAT_PRODUCT = {extent_info["ur_lat"]}',
        r"CORNER_UR_LON_PRODUCT = .*": f'CORNER_UR_LON_PRODUCT = {extent_info["ur_lon"]}',
        r"CORNER_LL_LAT_PRODUCT = .*": f'CORNER_LL_LAT_PRODUCT = {extent_info["ll_lat"]}',
        r"CORNER_LL_LON_PRODUCT = .*": f'CORNER_LL_LON_PRODUCT = {extent_info["ll_lon"]}',
        r"CORNER_LR_LAT_PRODUCT = .*": f'CORNER_LR_LAT_PRODUCT = {extent_info["lr_lat"]}',
        r"CORNER_LR_LON_PRODUCT = .*": f'CORNER_LR_LON_PRODUCT = {extent_info["lr_lon"]}',
        r"REFLECTIVE_LINES = .*": f'REFLECTIVE_LINES = {extent_info["rows"]}',
        r"REFLECTIVE_SAMPLES = .*": f'REFLECTIVE_SAMPLES = {extent_info["cols"]}',
        r"THERMAL_LINES = .*": f'THERMAL_LINES = {extent_info["rows"]}',
        r"THERMAL_SAMPLES = .*": f'THERMAL_SAMPLES = {extent_info["cols"]}',
    }
    if cloud_cover is not None:
        replacements[r"CLOUD_COVER = .*"] = f"CLOUD_COVER = {cloud_cover}"

    for pattern, replacement in replacements.items():
        mtl_text = re.sub(pattern, replacement, mtl_text)

    mtl_text += "\n\nGROUP = L1TP_STITCHING\n"
    mtl_text += f"  SOURCE_SCENE_1 = {base_name}\n"
    mtl_text += f"  SOURCE_SCENE_2 = {scene2_base_name}\n"
    mtl_text += "END_GROUP = L1TP_STITCHING\n"

    with open(os.path.join(output_dir, f"{base_name}_MTL.txt"), "w", encoding="utf-8") as f:
        f.write(mtl_text)


def _copy_primary_ang(scene1_dir: str, scene2_dir: str, output_dir: str) -> None:
    """Copy scene1's ANG file into the merged product directory.

    The resulting ANG file only reflects scene1's coverage, not the merged extent.
    This is acceptable because downstream angle computation uses the MTL metadata
    (which is correctly merged) and the band image extent, not this file directly.
    """
    base_name = _scene_base_name(scene1_dir)
    scene1_ang_path = os.path.join(scene1_dir, f"{base_name}_ANG.txt")

    if not os.path.exists(scene1_ang_path):
        logger.warning("ANG file missing for %s, no ANG file written to merged product", scene1_dir)
        return

    logger.info(
        "ANG file in merged product is copied from scene1 (%s) only and does not cover the merged extent. "
        "Downstream angle extraction should derive angles from the merged MTL metadata and band image extent.",
        base_name,
    )

    shutil.copyfile(scene1_ang_path, os.path.join(output_dir, f"{base_name}_ANG.txt"))
