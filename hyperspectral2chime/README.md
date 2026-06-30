# hyperspectral2chime (H2C)

![License: Apache2](https://img.shields.io/badge/license-Apache%202-blue.svg?&logo=apache)

**hyperspectral2chime** harmonises and fuses hyperspectral missions **onto CHIME**,
following the *CHIME Fusion Roadmap* (CHIME-L2-FUSION). **CHIME is the reference
mission**: input missions (PRISMA first, then SBG, EnMAP, EMIT, DESIS, …) are
spectrally aggregated onto the CHIME band set, reframed to the CHIME grid
(Sentinel-2 MGRS, 30 m GSD — roadmap §5.1.2), and processed through a
**band-set-agnostic** chain that reuses the sen2like algorithms to produce
CHIME-like **L2H/L2F** Analysis Ready Data.

> Direction matters: this is *hyperspectral → CHIME* (CHIME is the output/target).
> It is **not** `prisma4sen2like` / `chime4sen2like`, which target Sentinel-2.
> The first concrete input path is **PRISMA** (`prisma2chime`).

## Why a new pipeline instead of retargeting sen2like

sen2like's pipeline is hard-wired to the 13 Sentinel-2 bands. CHIME has ~210
narrow bands, so instead of retrofitting sen2like we run a lean, band-count-agnostic
pipeline that **reuses the sen2like algorithms as components** (spectral
aggregation, MGRS framing, KLT matching, Li-Sparse BRDF kernel, high/low-frequency
fusion). The internal CHIME product is a multi-band GeoTIFF + JSON sidecar, which
scales to any band count.

## Status

**Prototype.**

| Capability | Status |
|------------|--------|
| Spectral harmonisation onto the CHIME band set (Gaussian-SRF, §5.6) | **Implemented + unit-tested** |
| Reframe to the CHIME grid (S2-MGRS @ 30 m, §5.1) | **Implemented** |
| Band-agnostic L2 pipeline → internal **CHIME L2H** product | **Implemented + run end-to-end** |
| TOA, inter-calibration blocks | **Implemented** (inter-cal identity until CHIME-vs-HS coefficients exist) |
| NBAR / BRDF, geometry matching, fusion (L2F) blocks | TODO (reuse sen2like kernels; need CHIME aux data) |
| Native PRISMA reader (`prisma2chime`) | TODO (reuse `prisma4sen2like` reader) |
| Atmospheric correction (CHIME monochromatic LUTs) | TODO |
| Official CHIME band set / product format | TODO (placeholder band set + internal format for now) |

## Usage

```console
PYTHONPATH=h2c python h2c/main.py HS_CUBE BAND_CSV WORKING_DIR --tile 31TFJ
```

* `HS_CUBE` — projected multi-band hyperspectral GeoTIFF (reflectance).
* `BAND_CSV` — source band definitions: `band,central_wavelength_nm,fwhm_nm`.
* `--tile` — CHIME/MGRS tile code; `--target-band-csv` overrides the CHIME band set
  (defaults to the shipped placeholder `h2c/aux_data/chime_bands.csv`).

Output: a CHIME L2H product directory `CHIME_L2H_<date>_T<tile>/` with a multi-band
cube (`IMG_DATA/*.tif`, CHIME bands @ 30 m on the MGRS grid) and a `*.json`
metadata sidecar (band set, tile, angles, provenance).

> **Data volume note:** a full ~210-band CHIME tile at 30 m is ~11 GB uncompressed
> (3660² × 210 × float32). Use a band subset for quick tests; production output is
> LZW-compressed BigTIFF.

## Tests

```console
PYTHONPATH=h2c python tests/test_spectral_aggregation.py
```

## Layout

```
hyperspectral2chime/
├── h2c/
│   ├── band_definitions.py     # BandSet (cw + FWHM); CHIME band set
│   ├── spectral_aggregation.py # Gaussian-SRF aggregation source -> CHIME (§5.6)
│   ├── mgrs_tiling.py          # MGRS tile lookup (reuses repo s2tiles.db)
│   ├── chime_product.py        # internal CHIME product (N-band GeoTIFF + JSON)
│   ├── ingest.py               # aggregate to CHIME + reframe to grid @ 30 m
│   ├── blocks.py               # band-agnostic blocks (TOA, inter-calibration, ...)
│   ├── packager.py             # CHIME L2H/L2F internal-format packager
│   ├── pipeline.py             # run an ordered list of blocks
│   ├── main.py                 # CLI entry point
│   └── aux_data/chime_bands.csv (placeholder), sentinel2a_bands.csv
└── tests/test_spectral_aggregation.py
```

## Roadmap (next steps)

1. Add the NBAR/BRDF, geometry-matching and fusion (L2F) blocks, reusing the
   sen2like kernels per-component (band-agnostic).
2. Native **PRISMA** reader (reuse `prisma4sen2like`) → `prisma2chime` end to end.
3. Replace the placeholder CHIME band set with the official CHIME SRF; conform the
   internal product to the CHIME format once specified.
4. Atmospheric correction with CHIME monochromatic-LUT radiative transfer.

## License

Apache 2.0.
