import os
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from osgeo import gdal, osr

from core.landsat_l1tp_stitcher import (
    LandsatStitchError,
    _combined_geometry,
    _find_common_rasters,
    _raster_corners,
    _rect,
    _scene_base_name,
    _scene_extent,
    _copy_primary_ang,
    _write_merged_mtl,
    stitch_l1tp_products,
)

UTM31N_WKT = osr.SpatialReference()
UTM31N_WKT.ImportFromEPSG(32631)
UTM31N_WKT = UTM31N_WKT.ExportToWkt()

SCENE1_NAME = "LC09_L1TP_198024_20230614_20230614_02_T1"
SCENE2_NAME = "LC09_L1TP_198025_20230614_20230614_02_T1"

PIXEL_SIZE = 3000.0
RASTER_SIZE = 10
UL_X = 300000.0
UL_Y = 5730000.0

MTL_TEMPLATE = """GROUP = LANDSAT_METADATA_FILE
  COLLECTION_NUMBER = 02
  WRS_PATH = 198
  WRS_ROW = {row}
  CLOUD_COVER = {cloud_cover}
  CORNER_UL_PROJECTION_X_PRODUCT = {ul_x}
  CORNER_UL_PROJECTION_Y_PRODUCT = {ul_y}
  CORNER_UR_PROJECTION_X_PRODUCT = {ur_x}
  CORNER_UR_PROJECTION_Y_PRODUCT = {ul_y}
  CORNER_LL_PROJECTION_X_PRODUCT = {ul_x}
  CORNER_LL_PROJECTION_Y_PRODUCT = {ll_y}
  CORNER_LR_PROJECTION_X_PRODUCT = {ur_x}
  CORNER_LR_PROJECTION_Y_PRODUCT = {ll_y}
  REFLECTIVE_LINES = {size}
  REFLECTIVE_SAMPLES = {size}
END_GROUP = LANDSAT_METADATA_FILE
"""


def _write_band(path: str, y_origin: float, size: int, pixel_value: int, nodata: int) -> None:
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, size, size, 1, gdal.GDT_UInt16)
    ds.SetGeoTransform((UL_X, PIXEL_SIZE, 0, y_origin, 0, -PIXEL_SIZE))
    ds.SetProjection(UTM31N_WKT)
    ds.GetRasterBand(1).WriteArray(np.full((size, size), pixel_value, dtype=np.uint16))
    ds.GetRasterBand(1).SetNoDataValue(nodata)
    ds = None


def _make_scene(
    base_dir: str,
    name: str,
    row: str,
    y_origin: float,
    cloud_cover: str,
    pixel_value: int,
    bands=("B1", "QA_PIXEL"),
    with_ang: bool = False,
    with_mtl: bool = True,
) -> str:
    scene_dir = os.path.join(base_dir, name)
    os.makedirs(scene_dir, exist_ok=True)

    for suffix in bands:
        nodata = 1 if suffix == "QA_PIXEL" else 0
        _write_band(os.path.join(scene_dir, f"{name}_{suffix}.TIF"), y_origin, RASTER_SIZE, pixel_value, nodata)

    if with_mtl:
        with open(os.path.join(scene_dir, f"{name}_MTL.txt"), "w", encoding="utf-8") as f:
            f.write(
                MTL_TEMPLATE.format(
                    row=row,
                    cloud_cover=cloud_cover,
                    ul_x=UL_X,
                    ur_x=UL_X + RASTER_SIZE * PIXEL_SIZE,
                    ul_y=y_origin,
                    ll_y=y_origin - RASTER_SIZE * PIXEL_SIZE,
                    size=RASTER_SIZE,
                )
            )

    if with_ang:
        with open(os.path.join(scene_dir, f"{name}_ANG.txt"), "w", encoding="utf-8") as f:
            f.write(f"ANG content for {name}\n")

    return scene_dir


class TestSceneBaseName(TestCase):

    def test_valid_name(self):
        self.assertEqual(_scene_base_name(f"/data/{SCENE1_NAME}"), SCENE1_NAME)

    def test_valid_name_trailing_slash(self):
        self.assertEqual(_scene_base_name(f"/data/{SCENE1_NAME}/"), SCENE1_NAME)

    def test_invalid_name_raises(self):
        with self.assertRaises(LandsatStitchError):
            _scene_base_name("/data/not_a_landsat_scene")


class TestFindCommonRasters(TestCase):

    def test_returns_intersection_of_suffixes(self):
        with TemporaryDirectory() as tmp:
            scene1 = os.path.join(tmp, SCENE1_NAME)
            scene2 = os.path.join(tmp, SCENE2_NAME)
            os.makedirs(scene1)
            os.makedirs(scene2)

            for suffix in ("B1", "B2", "QA_PIXEL"):
                open(os.path.join(scene1, f"{SCENE1_NAME}_{suffix}.TIF"), "w", encoding="utf-8").close()
            for suffix in ("B1", "QA_PIXEL", "SAA"):
                open(os.path.join(scene2, f"{SCENE2_NAME}_{suffix}.TIF"), "w", encoding="utf-8").close()
            # not a raster band, must be ignored
            open(os.path.join(scene1, f"{SCENE1_NAME}_MTL.txt"), "w", encoding="utf-8").close()

            common = _find_common_rasters(scene1, scene2)

            self.assertEqual(set(common.keys()), {"B1", "QA_PIXEL"})
            self.assertEqual(common["B1"], (os.path.join(scene1, f"{SCENE1_NAME}_B1.TIF"), os.path.join(scene2, f"{SCENE2_NAME}_B1.TIF")))

    def test_no_common_suffix_returns_empty(self):
        with TemporaryDirectory() as tmp:
            scene1 = os.path.join(tmp, SCENE1_NAME)
            scene2 = os.path.join(tmp, SCENE2_NAME)
            os.makedirs(scene1)
            os.makedirs(scene2)
            open(os.path.join(scene1, f"{SCENE1_NAME}_B1.TIF"), "w", encoding="utf-8").close()
            open(os.path.join(scene2, f"{SCENE2_NAME}_B2.TIF"), "w", encoding="utf-8").close()

            self.assertEqual(_find_common_rasters(scene1, scene2), {})


class TestGeometryHelpers(TestCase):
    """Test the pure geometry helpers with in-memory (MEM driver) datasets."""

    @staticmethod
    def _mem_dataset(y_origin: float, size: int = RASTER_SIZE):
        ds = gdal.GetDriverByName("MEM").Create("", size, size, 1)
        ds.SetGeoTransform((UL_X, PIXEL_SIZE, 0, y_origin, 0, -PIXEL_SIZE))
        ds.SetProjection(UTM31N_WKT)
        return ds

    def test_raster_corners(self):
        ds = self._mem_dataset(UL_Y)
        corners = _raster_corners(ds)

        expected = np.array(
            [
                [UL_X, UL_Y],
                [UL_X, UL_Y - RASTER_SIZE * PIXEL_SIZE],
                [UL_X + RASTER_SIZE * PIXEL_SIZE, UL_Y],
                [UL_X + RASTER_SIZE * PIXEL_SIZE, UL_Y - RASTER_SIZE * PIXEL_SIZE],
            ]
        )
        np.testing.assert_array_equal(corners, expected)

    def test_combined_geometry_union_extent(self):
        # ds2 is shifted 9 rows south of ds1, overlapping by 1 row
        ds1 = self._mem_dataset(UL_Y)
        ds2 = self._mem_dataset(UL_Y - 9 * PIXEL_SIZE)

        geotransform, width, height = _combined_geometry(ds1, ds2)

        self.assertEqual(geotransform, (UL_X, PIXEL_SIZE, 0, UL_Y, 0, -PIXEL_SIZE))
        self.assertEqual(width, RASTER_SIZE)
        self.assertEqual(height, RASTER_SIZE + 9)

    def test_rect_for_each_source_within_merged_extent(self):
        ds1 = self._mem_dataset(UL_Y)
        ds2 = self._mem_dataset(UL_Y - 9 * PIXEL_SIZE)
        merged_geotransform, _, _ = _combined_geometry(ds1, ds2)

        rect1 = _rect(ds1, merged_geotransform)
        rect2 = _rect(ds2, merged_geotransform)

        self.assertEqual(rect1["dst"], {"x": 0, "y": 0, "width": RASTER_SIZE, "height": RASTER_SIZE})
        self.assertEqual(rect2["dst"], {"x": 0, "y": 9, "width": RASTER_SIZE, "height": RASTER_SIZE})

    def test_scene_extent(self):
        merged_height = RASTER_SIZE + 9
        ds = gdal.GetDriverByName("MEM").Create("", RASTER_SIZE, merged_height, 1)
        ds.SetGeoTransform((UL_X, PIXEL_SIZE, 0, UL_Y, 0, -PIXEL_SIZE))
        ds.SetProjection(UTM31N_WKT)

        extent = _scene_extent(ds)

        self.assertEqual(extent["cols"], RASTER_SIZE)
        self.assertEqual(extent["rows"], merged_height)
        self.assertAlmostEqual(extent["x_min"], UL_X)
        self.assertAlmostEqual(extent["x_max"], UL_X + RASTER_SIZE * PIXEL_SIZE)
        self.assertAlmostEqual(extent["y_max"], UL_Y)
        self.assertAlmostEqual(extent["y_min"], UL_Y - merged_height * PIXEL_SIZE)
        for lat in (extent["ul_lat"], extent["ur_lat"], extent["ll_lat"], extent["lr_lat"]):
            self.assertTrue(-90 <= lat <= 90)
        for lon in (extent["ul_lon"], extent["ur_lon"], extent["ll_lon"], extent["lr_lon"]):
            self.assertTrue(-180 <= lon <= 180)


class TestWriteMergedMtl(TestCase):

    def test_corner_and_dimension_fields_are_patched(self):
        with TemporaryDirectory() as tmp:
            scene1 = _make_scene(tmp, SCENE1_NAME, "024", UL_Y, "10.5", 100)
            scene2 = _make_scene(tmp, SCENE2_NAME, "025", UL_Y - 9 * PIXEL_SIZE, "25.0", 200)
            output_dir = os.path.join(tmp, "OUT")
            os.makedirs(output_dir)

            extent_info = {
                "x_min": UL_X,
                "x_max": UL_X + RASTER_SIZE * PIXEL_SIZE,
                "y_min": UL_Y - 19 * PIXEL_SIZE,
                "y_max": UL_Y,
                "ul_lat": 1.0,
                "ul_lon": 2.0,
                "ur_lat": 3.0,
                "ur_lon": 4.0,
                "ll_lat": 5.0,
                "ll_lon": 6.0,
                "lr_lat": 7.0,
                "lr_lon": 8.0,
                "cols": RASTER_SIZE,
                "rows": 19,
            }

            _write_merged_mtl(scene1, scene2, output_dir, extent_info)

            with open(os.path.join(output_dir, f"{SCENE1_NAME}_MTL.txt"), encoding="utf-8") as f:
                mtl_text = f.read()

            # regression test: LR corner X must be x_max, not x_min (prototype bug)
            self.assertIn(f"CORNER_LR_PROJECTION_X_PRODUCT = {extent_info['x_max']}", mtl_text)
            self.assertIn(f"CORNER_LR_PROJECTION_Y_PRODUCT = {extent_info['y_min']}", mtl_text)
            self.assertIn(f"CORNER_UL_PROJECTION_X_PRODUCT = {extent_info['x_min']}", mtl_text)
            self.assertIn(f"CORNER_UR_PROJECTION_X_PRODUCT = {extent_info['x_max']}", mtl_text)
            self.assertIn("REFLECTIVE_LINES = 19", mtl_text)
            self.assertIn(f"REFLECTIVE_SAMPLES = {RASTER_SIZE}", mtl_text)
            # cloud cover merged conservatively (max of both scenes)
            self.assertIn("CLOUD_COVER = 25.0", mtl_text)
            self.assertIn("SOURCE_SCENE_1 = " + SCENE1_NAME, mtl_text)
            self.assertIn("SOURCE_SCENE_2 = " + SCENE2_NAME, mtl_text)

    def test_missing_cloud_cover_is_left_untouched(self):
        with TemporaryDirectory() as tmp:
            scene1 = os.path.join(tmp, SCENE1_NAME)
            scene2 = os.path.join(tmp, SCENE2_NAME)
            os.makedirs(scene1)
            os.makedirs(scene2)
            with open(os.path.join(scene1, f"{SCENE1_NAME}_MTL.txt"), "w", encoding="utf-8") as f:
                f.write("GROUP = LANDSAT_METADATA_FILE\nEND_GROUP = LANDSAT_METADATA_FILE\n")
            with open(os.path.join(scene2, f"{SCENE2_NAME}_MTL.txt"), "w", encoding="utf-8") as f:
                f.write("GROUP = LANDSAT_METADATA_FILE\nEND_GROUP = LANDSAT_METADATA_FILE\n")

            output_dir = os.path.join(tmp, "OUT")
            os.makedirs(output_dir)
            extent_info = {
                "x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1,
                "ul_lat": 0, "ul_lon": 0, "ur_lat": 0, "ur_lon": 0,
                "ll_lat": 0, "ll_lon": 0, "lr_lat": 0, "lr_lon": 0,
                "cols": 1, "rows": 1,
            }

            _write_merged_mtl(scene1, scene2, output_dir, extent_info)

            with open(os.path.join(output_dir, f"{SCENE1_NAME}_MTL.txt"), encoding="utf-8") as f:
                mtl_text = f.read()
            self.assertNotIn("CLOUD_COVER", mtl_text)

    def test_missing_mtl_raises(self):
        with TemporaryDirectory() as tmp:
            scene1 = os.path.join(tmp, SCENE1_NAME)
            scene2 = os.path.join(tmp, SCENE2_NAME)
            os.makedirs(scene1)
            os.makedirs(scene2)
            with self.assertRaises(LandsatStitchError):
                _write_merged_mtl(scene1, scene2, tmp, {})


class TestWriteMergedAng(TestCase):

    def test_copies_scene1_ang_file_only(self):
        with TemporaryDirectory() as tmp:
            scene1 = _make_scene(tmp, SCENE1_NAME, "024", UL_Y, "10.5", 100, with_ang=True)
            scene2 = _make_scene(tmp, SCENE2_NAME, "025", UL_Y - 9 * PIXEL_SIZE, "25.0", 200, with_ang=True)
            output_dir = os.path.join(tmp, "OUT")
            os.makedirs(output_dir)

            _copy_primary_ang(scene1, scene2, output_dir)

            with open(os.path.join(output_dir, f"{SCENE1_NAME}_ANG.txt"), encoding="utf-8") as f:
                ang_text = f.read()
            # Only scene1's ANG content is present; scene2's is intentionally excluded
            # because the ANG file is copied as-is from scene1 (not merged).
            self.assertIn(f"ANG content for {SCENE1_NAME}", ang_text)
            self.assertNotIn(f"ANG content for {SCENE2_NAME}", ang_text)

    def test_copies_even_when_scene2_ang_missing(self):
        with TemporaryDirectory() as tmp:
            scene1 = _make_scene(tmp, SCENE1_NAME, "024", UL_Y, "10.5", 100, with_ang=True)
            scene2 = _make_scene(tmp, SCENE2_NAME, "025", UL_Y - 9 * PIXEL_SIZE, "25.0", 200, with_ang=False)
            output_dir = os.path.join(tmp, "OUT")
            os.makedirs(output_dir)

            _copy_primary_ang(scene1, scene2, output_dir)

            with open(os.path.join(output_dir, f"{SCENE1_NAME}_ANG.txt"), encoding="utf-8") as f:
                ang_text = f.read()
            self.assertIn(f"ANG content for {SCENE1_NAME}", ang_text)

    def test_skips_when_scene1_ang_missing(self):
        with TemporaryDirectory() as tmp:
            scene1 = _make_scene(tmp, SCENE1_NAME, "024", UL_Y, "10.5", 100, with_ang=False)
            scene2 = _make_scene(tmp, SCENE2_NAME, "025", UL_Y - 9 * PIXEL_SIZE, "25.0", 200, with_ang=True)
            output_dir = os.path.join(tmp, "OUT")
            os.makedirs(output_dir)

            _copy_primary_ang(scene1, scene2, output_dir)  # must not raise

            self.assertFalse(os.path.exists(os.path.join(output_dir, f"{SCENE1_NAME}_ANG.txt")))


class TestStitchL1tpProducts(TestCase):

    def test_cleans_up_partial_output_on_merge_failure(self):
        """When a band merge fails after others have already been written, the
        partially populated output directory must be removed so a retry starts
        from a clean state."""
        with TemporaryDirectory() as tmp:
            # scene1: both bands valid
            scene1 = _make_scene(tmp, SCENE1_NAME, "024", UL_Y, "10.5", 100)

            # scene2: valid B1, but QA_PIXEL is a corrupt file (not a GeoTIFF)
            scene2_dir = os.path.join(tmp, SCENE2_NAME)
            os.makedirs(scene2_dir)
            _write_band(
                os.path.join(scene2_dir, f"{SCENE2_NAME}_B1.TIF"),
                UL_Y - 9 * PIXEL_SIZE,
                RASTER_SIZE,
                200,
                0,
            )
            with open(os.path.join(scene2_dir, f"{SCENE2_NAME}_QA_PIXEL.TIF"), "w", encoding="utf-8") as f:
                f.write("not a valid GeoTIFF")
            with open(os.path.join(scene2_dir, f"{SCENE2_NAME}_MTL.txt"), "w", encoding="utf-8") as f:
                f.write(MTL_TEMPLATE.format(
                    row="025",
                    cloud_cover="25.0",
                    ul_x=UL_X,
                    ur_x=UL_X + RASTER_SIZE * PIXEL_SIZE,
                    ul_y=UL_Y - 9 * PIXEL_SIZE,
                    ll_y=UL_Y - 19 * PIXEL_SIZE,
                    size=RASTER_SIZE,
                ))

            output_dir = os.path.join(tmp, "STITCHED")

            with self.assertRaises(Exception):
                stitch_l1tp_products(scene1, scene2_dir, output_dir)

            self.assertFalse(os.path.exists(output_dir))

    def test_merges_bands_and_writes_metadata(self):
        with TemporaryDirectory() as tmp:
            scene1 = _make_scene(tmp, SCENE1_NAME, "024", UL_Y, "10.5", 100)
            scene2 = _make_scene(tmp, SCENE2_NAME, "025", UL_Y - 9 * PIXEL_SIZE, "25.0", 200)
            output_dir = os.path.join(tmp, "STITCHED")

            result = stitch_l1tp_products(scene1, scene2, output_dir)

            self.assertEqual(result, output_dir)
            self.assertEqual(
                set(os.listdir(output_dir)),
                {f"{SCENE1_NAME}_B1.TIF", f"{SCENE1_NAME}_QA_PIXEL.TIF", f"{SCENE1_NAME}_MTL.txt"},
            )

            ds = gdal.Open(os.path.join(output_dir, f"{SCENE1_NAME}_B1.TIF"))
            self.assertEqual((ds.RasterXSize, ds.RasterYSize), (RASTER_SIZE, RASTER_SIZE + 9))
            self.assertEqual(ds.GetGeoTransform(), (UL_X, PIXEL_SIZE, 0, UL_Y, 0, -PIXEL_SIZE))

            merged_values = set(np.unique(ds.GetRasterBand(1).ReadAsArray()).tolist())
            self.assertEqual(merged_values, {100, 200})

            with open(os.path.join(output_dir, f"{SCENE1_NAME}_MTL.txt"), encoding="utf-8") as f:
                mtl_text = f.read()
            self.assertIn(f"REFLECTIVE_LINES = {RASTER_SIZE + 9}", mtl_text)
            self.assertIn("CLOUD_COVER = 25.0", mtl_text)

    def test_raises_when_no_common_rasters(self):
        with TemporaryDirectory() as tmp:
            scene1 = _make_scene(tmp, SCENE1_NAME, "024", UL_Y, "10.5", 100, bands=("B1",))
            scene2 = _make_scene(tmp, SCENE2_NAME, "025", UL_Y - 9 * PIXEL_SIZE, "25.0", 200, bands=("B2",))

            with self.assertRaises(LandsatStitchError):
                stitch_l1tp_products(scene1, scene2, os.path.join(tmp, "STITCHED"))
