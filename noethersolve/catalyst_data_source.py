"""catalyst_data_source.py -- Fetch catalyst data from Materials Project.

Provides surface energies, d-band centers, and adsorption energies from
the Materials Project database for electrochemical catalyst screening.

Usage:
    from noethersolve.catalyst_data_source import CatalystDataSource

    ds = CatalystDataSource(api_key="your_mp_api_key")
    data = ds.get_surface_data("Pt", miller_index=(1,1,1))

Data source:
    Materials Project (https://next-gen.materialsproject.org/)
    - Requires free API key from materialsproject.org
    - Rate limited: 30 requests/minute on free tier

⚠️  NOTE: Materials Project API key required for full functionality.
    Without API key, falls back to built-in reference data.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests

# Built-in reference data (no API needed)
REFERENCE_D_BAND_CENTERS: Dict[str, float] = {
    # d-band center in eV relative to Fermi level (Hammer-Nørskov)
    "Pt": -2.25,
    "Pd": -1.83,
    "Ni": -1.29,
    "Cu": -2.67,
    "Au": -3.56,
    "Ag": -4.30,
    "Ru": -1.41,
    "Rh": -1.73,
    "Ir": -2.11,
    "Co": -1.17,
    "Fe": -0.92,
    "Mo": -0.35,
    "W": -0.60,
}

REFERENCE_HER_ADSORPTION: Dict[str, float] = {
    # Hydrogen adsorption free energy ΔG_H* in eV (DFT, Nørskov et al.)
    "Pt": -0.09,
    "Pd": -0.35,
    "Ni": -0.30,
    "Cu": 0.15,
    "Au": 0.40,
    "Ag": 0.60,
    "Ru": -0.15,
    "Rh": -0.20,
    "Ir": -0.10,
    "Co": -0.25,
    "Fe": -0.20,
    "Mo": -0.05,
    "W": -0.10,
    "MoS2": 0.08,  # edge sites
    "Pt-Ni": -0.05,  # alloy
}


@dataclass
class SurfaceData:
    """Surface properties for a metal catalyst."""
    element: str
    miller_index: Tuple[int, int, int]
    surface_energy: Optional[float]  # J/m²
    d_band_center: Optional[float]   # eV
    h_adsorption: Optional[float]    # eV (ΔG_H*)
    o_adsorption: Optional[float]    # eV (ΔG_O*)
    oh_adsorption: Optional[float]   # eV (ΔG_OH*)
    data_source: str


class CatalystDataSource:
    """Fetch catalyst data from Materials Project or use built-in reference."""

    MP_API_URL = "https://api.materialsproject.org/v3"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with optional Materials Project API key.

        Args:
            api_key: Materials Project API key. If None, checks MP_API_KEY env var.
                    Falls back to reference data if no key available.
        """
        self.api_key = api_key or os.environ.get("MP_API_KEY")
        self._cache: Dict[str, SurfaceData] = {}

    def get_surface_data(
        self,
        element: str,
        miller_index: Tuple[int, int, int] = (1, 1, 1),
    ) -> SurfaceData:
        """Get surface properties for a metal catalyst.

        Args:
            element: Element symbol (e.g., "Pt", "Ni")
            miller_index: Surface Miller indices, default (1,1,1)

        Returns:
            SurfaceData with available properties
        """
        cache_key = f"{element}_{miller_index}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try Materials Project API if available
        if self.api_key:
            mp_data = self._fetch_from_mp(element, miller_index)
            if mp_data:
                self._cache[cache_key] = mp_data
                return mp_data

        # Fall back to reference data
        ref_data = SurfaceData(
            element=element,
            miller_index=miller_index,
            surface_energy=None,
            d_band_center=REFERENCE_D_BAND_CENTERS.get(element),
            h_adsorption=REFERENCE_HER_ADSORPTION.get(element),
            o_adsorption=None,
            oh_adsorption=None,
            data_source="reference",
        )
        self._cache[cache_key] = ref_data
        return ref_data

    def _fetch_from_mp(
        self,
        element: str,
        miller_index: Tuple[int, int, int],
    ) -> Optional[SurfaceData]:
        """Fetch data from Materials Project API."""
        try:
            # Query for elemental material
            headers = {"X-API-KEY": self.api_key}
            params = {
                "elements": element,
                "num_elements": 1,
                "fields": "material_id,formula_pretty,surface_energy_anisotropy",
            }

            resp = requests.get(
                f"{self.MP_API_URL}/materials/summary/",
                headers=headers,
                params=params,
                timeout=10,
            )

            if resp.status_code != 200:
                return None

            data = resp.json()
            if not data.get("data"):
                return None

            # Extract surface energy if available
            material = data["data"][0]
            surface_energy = material.get("surface_energy_anisotropy")

            return SurfaceData(
                element=element,
                miller_index=miller_index,
                surface_energy=surface_energy,
                d_band_center=REFERENCE_D_BAND_CENTERS.get(element),
                h_adsorption=REFERENCE_HER_ADSORPTION.get(element),
                o_adsorption=None,
                oh_adsorption=None,
                data_source="materials_project",
            )

        except Exception:
            return None

    def get_volcano_optimal(self, reaction: str = "HER") -> List[str]:
        """Get elements near the volcano apex for a reaction.

        Args:
            reaction: "HER", "OER", or "ORR"

        Returns:
            List of element symbols ranked by predicted activity
        """
        if reaction == "HER":
            # Sort by |ΔG_H*| closest to 0
            ranked = sorted(
                REFERENCE_HER_ADSORPTION.items(),
                key=lambda x: abs(x[1])
            )
            return [elem for elem, _ in ranked]
        else:
            # For OER/ORR, would need more adsorption data
            return list(REFERENCE_D_BAND_CENTERS.keys())

    def search_alloys(self, base: str, dopants: List[str]) -> List[str]:
        """Search for alloy combinations.

        Args:
            base: Base metal (e.g., "Pt")
            dopants: List of dopant elements

        Returns:
            List of alloy formula strings to investigate
        """
        alloys = []
        for dopant in dopants:
            alloys.append(f"{base}-{dopant}")
            alloys.append(f"{base}3{dopant}")
            alloys.append(f"{base}{dopant}3")
        return alloys


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Catalyst data source CLI")
    parser.add_argument("element", nargs="?", default="Pt", help="Element symbol")
    parser.add_argument("--ranking", "-r", action="store_true",
                        help="Show volcano ranking for HER")
    args = parser.parse_args()

    ds = CatalystDataSource()

    if args.ranking:
        print("\nHER volcano ranking (|ΔG_H*| closest to 0 is best):")
        for i, elem in enumerate(ds.get_volcano_optimal("HER"), 1):
            dg = REFERENCE_HER_ADSORPTION.get(elem, 0)
            print(f"  {i:2d}. {elem:4s}  ΔG_H* = {dg:+.2f} eV")
    else:
        data = ds.get_surface_data(args.element)
        print(f"\n{data.element} ({data.miller_index}):")
        print(f"  d-band center: {data.d_band_center} eV")
        print(f"  ΔG_H*: {data.h_adsorption} eV")
        print(f"  Source: {data.data_source}")


if __name__ == "__main__":
    main()
