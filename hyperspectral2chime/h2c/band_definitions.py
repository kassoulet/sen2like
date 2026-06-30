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
"""Band-set definitions.

This module replaces the hard-coded ``n_bands_s2 = 13`` assumption found in the
``prisma4sen2like`` spectral aggregation. A :class:`BandSet` describes a sensor's
spectral configuration purely as a list of (name, central wavelength, FWHM)
triplets, so the very same spectral-aggregation code can target the Sentinel-2
13-band set (for validation against prisma4sen2like) *or* the CHIME band set
(many narrow VNIR/SWIR bands), or any other mission.

Per the CHIME Fusion Roadmap (CHIME-L2-FUSION, section 5.6.5), spectral
harmonisation is "a spectral regridding via convolution using the narrowband
response functions (central wavelengths + FWHM) from both CHIME and the other
Hyperspectral mission". A BandSet therefore carries exactly that information.

Wavelengths and FWHM are expressed in nanometres (nm).
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_AUX_DIR = os.path.join(os.path.dirname(__file__), "aux_data")


@dataclass(frozen=True)
class BandSet:
    """A sensor spectral configuration: ordered bands with central wavelength + FWHM.

    Attributes:
        names: ordered band names (e.g. ["B01", "B02", ...] or CHIME band ids).
        central_wavelengths: central wavelength of each band, in nm.
        fwhm: Full Width at Half Maximum of each band, in nm.
        name: an identifier for the band set (e.g. "Sentinel-2A", "CHIME").
    """

    names: tuple[str, ...]
    central_wavelengths: NDArray  # shape (n_bands,), nm
    fwhm: NDArray  # shape (n_bands,), nm
    name: str = "unnamed"

    def __post_init__(self):
        n = len(self.names)
        if len(self.central_wavelengths) != n or len(self.fwhm) != n:
            raise ValueError(
                f"BandSet '{self.name}': names ({n}), central_wavelengths "
                f"({len(self.central_wavelengths)}) and fwhm ({len(self.fwhm)}) must have equal length"
            )

    @property
    def n_bands(self) -> int:
        return len(self.names)

    def __len__(self) -> int:
        return len(self.names)

    @classmethod
    def from_csv(cls, csv_path: str, name: str | None = None) -> "BandSet":
        """Load a band set from a CSV file.

        Expected columns (header row required, order-independent, case-insensitive):
            band, central_wavelength_nm, fwhm_nm

        Args:
            csv_path: path to the CSV file.
            name: optional band-set name; defaults to the file base name.

        Returns:
            BandSet: the loaded band set, ordered as in the file.
        """
        names: list[str] = []
        cws: list[float] = []
        fwhms: list[float] = []
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(_strip_comments(handle))
            field_map = {key.strip().lower(): key for key in (reader.fieldnames or [])}
            for required in ("band", "central_wavelength_nm", "fwhm_nm"):
                if required not in field_map:
                    raise ValueError(f"{csv_path}: missing required column '{required}'")
            for row in reader:
                names.append(row[field_map["band"]].strip())
                cws.append(float(row[field_map["central_wavelength_nm"]]))
                fwhms.append(float(row[field_map["fwhm_nm"]]))

        return cls(
            names=tuple(names),
            central_wavelengths=np.asarray(cws, dtype=np.float64),
            fwhm=np.asarray(fwhms, dtype=np.float64),
            name=name or os.path.splitext(os.path.basename(csv_path))[0],
        )


def _strip_comments(lines):
    """Yield CSV lines, skipping blank lines and ``#`` comment lines."""
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            yield line


def sentinel2a_band_set() -> BandSet:
    """Sentinel-2A MSI 13-band set.

    Provided so hyperspectral2chime can reproduce the prisma4sen2like target (13 S2
    bands) for validation/parity. Values come from ``aux_data/sentinel2a_bands.csv``.
    """
    return BandSet.from_csv(os.path.join(_AUX_DIR, "sentinel2a_bands.csv"), name="Sentinel-2A")


def chime_band_set() -> BandSet:
    """CHIME target band set.

    Loaded from ``aux_data/chime_bands.csv``. The shipped CSV is a PLACEHOLDER
    derived from the CHIME Mission Requirements Document envelope (VNIR-SWIR,
    spectral sampling ~10 nm, FWHM <= 12 nm). Replace it with the official CHIME
    spectral response definition ([RD01]/[AD01]) once available.
    """
    return BandSet.from_csv(os.path.join(_AUX_DIR, "chime_bands.csv"), name="CHIME")
