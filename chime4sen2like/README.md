# C4S2L (chime4sen2like)

![License: Apache2](https://img.shields.io/badge/license-Apache%202-blue.svg?&logo=apache)

**chime4sen2like** turns a hyperspectral L1 product into a **CHIME-like L1C
product** that the [sen2like](../sen2like) processor can ingest, following the
*CHIME Fusion Roadmap* (CHIME-L2-FUSION). It is the CHIME counterpart of
[`prisma4sen2like`](../prisma4sen2like): a thin preprocessor that
**orthorectifies** (when needed) and **spectrally harmonises** any hyperspectral
mission (CHIME, SBG, EnMAP, PRISMA, EMIT, DESIS, ...) onto a common band set, so
the existing sen2like processing blocks (geometry, inter-calibration, atmospheric
correction, BRDF, fusion, packaging) can be reused unchanged.

See [`docs/sen2like-reuse-for-chime.md`](../../chimelike/docs/sen2like-reuse-for-chime.md)
for the full reuse assessment.

## Project status

**Prototype.**

| Capability | Status |
|------------|--------|
| Spectral harmonisation — Gaussian-SRF convolution onto an arbitrary band set (Roadmap §5.6.5) | **Implemented + unit-tested** |
| Band-set model (replaces the hard-coded 13-band assumption) | **Implemented** |
| Generic GeoTIFF-cube reader (orthorectified input) | **Implemented + run end-to-end** |
| Radiance → TOA reflectance conversion | **Implemented** |
| Mission-specific readers (CHIME L1C, SBG L1B, EnMAP, ...) | TODO |
| L1B → orthorectification (alternative flow, Roadmap §5.1.6) | TODO |
| MGRS re-projection + SAFE packaging | TODO (reuse `prisma4sen2like` builder) |

## What is different from prisma4sen2like

`prisma4sen2like` hard-wires the target to **13 Sentinel-2 bands**, loads
Sentinel-2A spectral responses through `pyrsr`, and hand-splits VNIR/SWIR. Here
the spectral aggregation is expressed between two `BandSet` objects, each defined
only by **central wavelength + FWHM** — exactly the inputs the Roadmap §5.6.5
calls for — so the same code targets the CHIME band set (211 placeholder bands)
or, for validation, the Sentinel-2A set. No `pyrsr` dependency, no fixed band
count, and optional **per-detector** aggregation matrices to handle smile/keystone
(Roadmap §5.6.6).

## Installation

```console
cd sen2like/chime4sen2like
conda env create -f environment.yml
conda activate chime4sen2like
```

The spectral-aggregation core and its tests need only `numpy`/`scipy`; `gdal` is
required to read/write rasters at runtime.

## Usage

```console
PYTHONPATH=chime python chime/main.py HS_CUBE BAND_CSV WORKING_DIR
```

* `HS_CUBE` — a multi-band (orthorectified) hyperspectral GeoTIFF.
* `BAND_CSV` — source band definitions: `band,central_wavelength_nm,fwhm_nm`.
* `WORKING_DIR` — output directory.

By default the target is the shipped CHIME placeholder band set
(`chime/aux_data/chime_bands.csv`); override with `--target-band-csv`. The output
is a multi-band CHIME-band TOA radiance/reflectance GeoTIFF.

## Tests

```console
python chime4sen2like/tests/test_spectral_aggregation.py
# or, with pytest:
PYTHONPATH=chime4sen2like/chime python -m pytest chime4sen2like/tests
```

## Layout

```
chime4sen2like/
├── chime/
│   ├── band_definitions.py     # BandSet (cw + FWHM); CSV loader; S2A & CHIME sets
│   ├── spectral_aggregation.py # generalized Gaussian-SRF convolution (§5.6.5)
│   ├── hs_product.py           # HyperspectralProduct interface + GeoTIFF reader
│   ├── chime_aggregation.py    # orchestrator: product -> CHIME-band rasters
│   ├── main.py                 # CLI entry point
│   ├── aux_data/
│   │   ├── chime_bands.csv      # CHIME target band set (PLACEHOLDER)
│   │   └── sentinel2a_bands.csv # S2A band set (validation/parity)
│   └── log.py, utils.py, version.py
└── tests/test_spectral_aggregation.py
```

## Roadmap (next steps)

1. **Mission readers** — implement `HyperspectralProduct` subclasses for CHIME
   L1C ([RD02]/[RD03]) and other missions (metadata, angles, SRF, cube access).
2. **CHIME band set** — replace the placeholder `chime_bands.csv` with the
   official CHIME spectral response definition ([RD01]/[AD01]).
3. **Geometry** — for the L1B flow (§5.1.6), add orthorectification from per-pixel
   geolocation; reuse sen2like `grids/mgrs_framing.py` for reframing.
4. **Packaging** — reuse the `prisma4sen2like` `product_builder` + `adapter` to
   emit a SAFE L1C product, then register the CHIME product class in sen2like
   `core/products` + `PROC_BLOCKS`.

## License

Apache 2.0.
