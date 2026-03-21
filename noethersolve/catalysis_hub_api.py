"""catalysis_hub_api.py -- Fetch DFT adsorption energies from Catalysis-Hub.

Connects to the Catalysis-Hub GraphQL API (https://api.catalysis-hub.org/graphql)
to retrieve DFT-calculated adsorption energies for catalyst screening.

Usage:
    from noethersolve.catalysis_hub_api import (
        fetch_adsorption_energy,
        fetch_volcano_data,
        search_catalysts,
    )

    # Single surface lookup
    dg = fetch_adsorption_energy("Pt", adsorbate="H")
    # -> -0.09 eV (or None if not found)

    # Volcano plot data for multiple surfaces
    data = fetch_volcano_data("H", surfaces=["Pt", "Pd", "Ni", "Cu", "Au"])
    # -> {"Pt": -0.09, "Pd": -0.45, ...}

    # Search by reaction type
    records = search_catalysts(reaction="HER", max_results=10)

Data source:
    Catalysis-Hub (https://www.catalysis-hub.org/)
    - Free, no API key required
    - GraphQL endpoint: https://api.catalysis-hub.org/graphql
    - DFT adsorption energies from published computational catalysis studies

Fallback:
    When the API is unreachable, returns reference values from:
    Norskov et al., J. Electrochem. Soc. 152, J23 (2005)
    -- H adsorption free energies on close-packed metal surfaces.

Zero external dependencies beyond stdlib (uses urllib.request).
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Dict, List, Optional

# ---- Constants ----------------------------------------------------------------

GRAPHQL_URL = "https://api.catalysis-hub.org/graphql"
REQUEST_TIMEOUT_S = 10

# ---- Fallback cache: Norskov 2005 reference H adsorption energies (eV) ------
# delta_G_H* on close-packed surfaces, from:
# Norskov, Bligaard, Logadottir, Kitchin, Chen, Pandelov, Stimming,
# "Trends in the Exchange Current for Hydrogen Evolution",
# J. Electrochem. Soc. 152, J23 (2005), Table 1.
#
# Values include ZPE and entropy corrections: delta_G = delta_E + 0.24 eV

NORSKOV_2005_H_ADSORPTION: Dict[str, float] = {
    "Pt(111)":   -0.09,
    "Pd(111)":   -0.45,
    "Ni(111)":   -0.56,
    "Cu(111)":    0.04,
    "Au(111)":    0.38,
    "Ru(0001)":  -0.42,
    "Ir(111)":   -0.26,
    "Rh(111)":   -0.33,
}

# Map short names to full facet names for convenience
_SURFACE_ALIASES: Dict[str, str] = {
    "Pt":   "Pt(111)",
    "Pd":   "Pd(111)",
    "Ni":   "Ni(111)",
    "Cu":   "Cu(111)",
    "Au":   "Au(111)",
    "Ru":   "Ru(0001)",
    "Ir":   "Ir(111)",
    "Rh":   "Rh(111)",
}

# Reaction-type to adsorbate product mapping for search_catalysts
_REACTION_ADSORBATES: Dict[str, str] = {
    "HER": "H",
    "OER": "O",
    "ORR": "OH",
    "CO2RR": "CO",
    "NRR": "N",
    "N2RR": "N",
}


# ---- Data classes -------------------------------------------------------------

@dataclass
class CatalystRecord:
    """A single catalyst adsorption energy record."""

    surface: str
    adsorbate: str
    delta_g_ev: float
    source: str  # "api" or "cache"
    pub_id: str = ""

    def __str__(self) -> str:
        src_tag = f" [{self.source}]" if self.source else ""
        pub_tag = f" (pub: {self.pub_id})" if self.pub_id else ""
        return (
            f"{self.surface} + {self.adsorbate}*: "
            f"delta_G = {self.delta_g_ev:+.3f} eV{src_tag}{pub_tag}"
        )


# ---- Internal helpers ---------------------------------------------------------

def _graphql_query(query: str, timeout: int = REQUEST_TIMEOUT_S) -> Optional[dict]:
    """Send a GraphQL query to Catalysis-Hub. Returns parsed JSON or None."""
    payload = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return None


def _normalize_surface(surface: str) -> str:
    """Normalize a surface name: 'Pt' -> 'Pt', 'Pt(111)' -> 'Pt'."""
    # Strip facet for API queries (API uses surfaceComposition without facet)
    s = surface.strip()
    if "(" in s:
        s = s[: s.index("(")]
    return s


def _resolve_cache_key(surface: str) -> Optional[str]:
    """Find the matching key in the Norskov cache."""
    s = surface.strip()
    if s in NORSKOV_2005_H_ADSORPTION:
        return s
    if s in _SURFACE_ALIASES:
        return _SURFACE_ALIASES[s]
    # Try case-insensitive match
    s_lower = s.lower()
    for key in NORSKOV_2005_H_ADSORPTION:
        if key.lower() == s_lower or key.lower().startswith(s_lower + "("):
            return key
    for alias, full in _SURFACE_ALIASES.items():
        if alias.lower() == s_lower:
            return full
    return None


def _extract_energy_from_edges(edges: list) -> Optional[float]:
    """Extract the median reaction energy from a list of GraphQL result edges."""
    energies = []
    for edge in edges:
        node = edge.get("node", {})
        energy = node.get("reactionEnergy")
        if energy is not None:
            try:
                energies.append(float(energy))
            except (TypeError, ValueError):
                continue
    if not energies:
        return None
    # Return median to be robust against outliers
    energies.sort()
    n = len(energies)
    if n % 2 == 1:
        return energies[n // 2]
    return (energies[n // 2 - 1] + energies[n // 2]) / 2.0


def _extract_records_from_edges(
    edges: list, adsorbate: str, source: str = "api"
) -> List[CatalystRecord]:
    """Convert GraphQL result edges into CatalystRecord objects."""
    records = []
    for edge in edges:
        node = edge.get("node", {})
        energy = node.get("reactionEnergy")
        if energy is None:
            continue
        try:
            energy_f = float(energy)
        except (TypeError, ValueError):
            continue
        records.append(
            CatalystRecord(
                surface=node.get("surfaceComposition", "unknown"),
                adsorbate=adsorbate,
                delta_g_ev=energy_f,
                source=source,
                pub_id=node.get("pubId", ""),
            )
        )
    return records


# ---- Public API ---------------------------------------------------------------

def fetch_adsorption_energy(
    surface: str, adsorbate: str = "H"
) -> Optional[float]:
    """Query the Catalysis-Hub GraphQL API for a surface+adsorbate adsorption energy.

    Args:
        surface: Surface composition, e.g. "Pt", "Pt(111)", "Pd", "Cu(111)".
        adsorbate: Adsorbate species, default "H".

    Returns:
        Median delta_G in eV from available DFT data, or None if not found.
        Falls back to Norskov 2005 reference cache when the API is unreachable.

    Example:
        >>> dg = fetch_adsorption_energy("Pt", "H")
        >>> dg is not None
        True
    """
    comp = _normalize_surface(surface)
    product = f"{adsorbate}*" if not adsorbate.endswith("*") else adsorbate

    query = (
        '{ reactions(first: 50, surfaceComposition: "%s", products: "%s") '
        "{ edges { node { reactionEnergy surfaceComposition Equation pubId } } } }"
        % (comp, product)
    )

    result = _graphql_query(query)
    if result is not None:
        data = result.get("data", {})
        reactions = data.get("reactions", {})
        edges = reactions.get("edges", [])
        energy = _extract_energy_from_edges(edges)
        if energy is not None:
            return energy

    # Fallback to cache for H adsorbate
    if adsorbate.replace("*", "") == "H":
        cache_key = _resolve_cache_key(surface)
        if cache_key is not None:
            return NORSKOV_2005_H_ADSORPTION[cache_key]

    return None


def fetch_volcano_data(
    adsorbate: str = "H",
    surfaces: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Fetch adsorption energies for multiple surfaces.

    Args:
        adsorbate: Adsorbate species, default "H".
        surfaces: List of surface names. If None, uses all Norskov 2005 reference
                  surfaces: Pt(111), Pd(111), Ni(111), Cu(111), Au(111),
                  Ru(0001), Ir(111), Rh(111).

    Returns:
        Dict mapping surface name to delta_G in eV.
        Only includes surfaces for which data was found.

    Example:
        >>> data = fetch_volcano_data("H", ["Pt", "Pd", "Au"])
        >>> "Pt" in data or "Pt(111)" in data
        True
    """
    if surfaces is None:
        surfaces = list(NORSKOV_2005_H_ADSORPTION.keys())

    result: Dict[str, float] = {}
    for surf in surfaces:
        energy = fetch_adsorption_energy(surf, adsorbate)
        if energy is not None:
            result[surf] = energy

    return result


def search_catalysts(
    reaction: str = "HER",
    max_results: int = 20,
) -> List[CatalystRecord]:
    """Search for catalysts by reaction type.

    Args:
        reaction: Reaction type. Supported: "HER", "OER", "ORR", "CO2RR", "NRR".
                  Also accepts a raw adsorbate string (e.g. "H", "OH").
        max_results: Maximum number of records to return.

    Returns:
        List of CatalystRecord objects sorted by |delta_G| (closest to zero first,
        i.e. nearest to the volcano peak for HER).

    Example:
        >>> records = search_catalysts("HER", max_results=5)
        >>> all(isinstance(r, CatalystRecord) for r in records)
        True
    """
    # Resolve reaction name to adsorbate product
    reaction_upper = reaction.upper().strip()
    if reaction_upper in _REACTION_ADSORBATES:
        adsorbate = _REACTION_ADSORBATES[reaction_upper]
    else:
        # Treat as raw adsorbate
        adsorbate = reaction.strip()

    product = f"{adsorbate}*" if not adsorbate.endswith("*") else adsorbate

    query = (
        '{ reactions(first: %d, products: "%s") '
        "{ edges { node { reactionEnergy surfaceComposition Equation pubId } } } }"
        % (max_results * 3, product)  # over-fetch to have room after dedup
    )

    result = _graphql_query(query)
    records: List[CatalystRecord] = []

    if result is not None:
        data = result.get("data", {})
        reactions = data.get("reactions", {})
        edges = reactions.get("edges", [])
        records = _extract_records_from_edges(edges, adsorbate, source="api")

    # If API returned nothing or was unreachable, fall back to cache for H
    if not records and adsorbate.replace("*", "") == "H":
        for surf, dg in NORSKOV_2005_H_ADSORPTION.items():
            records.append(
                CatalystRecord(
                    surface=surf,
                    adsorbate="H",
                    delta_g_ev=dg,
                    source="cache",
                    pub_id="Norskov2005",
                )
            )

    # Deduplicate by surface (keep median energy per surface)
    surface_energies: Dict[str, List[float]] = {}
    surface_pubs: Dict[str, str] = {}
    surface_sources: Dict[str, str] = {}
    for r in records:
        surface_energies.setdefault(r.surface, []).append(r.delta_g_ev)
        if not surface_pubs.get(r.surface):
            surface_pubs[r.surface] = r.pub_id
        surface_sources[r.surface] = r.source

    deduped: List[CatalystRecord] = []
    for surf, energies in surface_energies.items():
        energies.sort()
        n = len(energies)
        if n % 2 == 1:
            median = energies[n // 2]
        else:
            median = (energies[n // 2 - 1] + energies[n // 2]) / 2.0
        deduped.append(
            CatalystRecord(
                surface=surf,
                adsorbate=adsorbate,
                delta_g_ev=median,
                source=surface_sources[surf],
                pub_id=surface_pubs.get(surf, ""),
            )
        )

    # Sort by |delta_G| (closest to thermoneutral = best for HER volcano peak)
    deduped.sort(key=lambda r: abs(r.delta_g_ev))

    return deduped[:max_results]
