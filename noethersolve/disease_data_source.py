"""disease_data_source.py -- Fetch disease epidemiological parameters.

Provides R0, generation time, CFR, and vaccine efficacy from literature
and WHO/CDC databases for epidemic modeling.

Usage:
    from noethersolve.disease_data_source import DiseaseDataSource

    ds = DiseaseDataSource()
    params = ds.get_disease("COVID-19")
    print(params.R0, params.generation_time_days)

Data sources:
    - WHO Global Health Observatory (https://www.who.int/data/gho)
    - CDC MMWR reports
    - Peer-reviewed literature (curated)

⚠️  NOTE: Epidemiological parameters vary by population, time, and
    measurement method. Values are central estimates with uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class DiseaseParameters:
    """Epidemiological parameters for a disease."""
    name: str
    display_name: str

    # Transmission
    R0: float                         # Basic reproduction number
    R0_range: Tuple[float, float]     # Uncertainty range
    generation_time_days: float       # Serial interval
    incubation_days: float            # Time to symptom onset
    infectious_days: float            # Duration of infectiousness

    # Severity
    cfr: float                        # Case fatality ratio (0-1)
    ifr: float                        # Infection fatality ratio (0-1)
    hospitalization_rate: float       # Fraction requiring hospitalization

    # Vaccine
    vaccine_available: bool
    vaccine_efficacy: Optional[float]  # Against infection (0-1)
    vaccine_coverage: Optional[float]  # Current global coverage

    # Metadata
    data_source: str
    last_updated: str
    notes: str = ""


# Curated disease database from WHO/CDC/literature
DISEASE_DATABASE: Dict[str, DiseaseParameters] = {
    "covid19_omicron": DiseaseParameters(
        name="covid19_omicron",
        display_name="COVID-19 (Omicron BA.5+)",
        R0=11.5,
        R0_range=(8.0, 15.0),
        generation_time_days=3.0,
        incubation_days=3.0,
        infectious_days=5.0,
        cfr=0.001,
        ifr=0.0005,
        hospitalization_rate=0.02,
        vaccine_available=True,
        vaccine_efficacy=0.50,  # Against infection (waned)
        vaccine_coverage=0.65,
        data_source="CDC/WHO 2023",
        last_updated="2024-01",
        notes="High transmissibility, partial vaccine escape",
    ),
    "covid19_original": DiseaseParameters(
        name="covid19_original",
        display_name="COVID-19 (Wuhan)",
        R0=3.0,
        R0_range=(2.5, 3.5),
        generation_time_days=5.0,
        incubation_days=5.5,
        infectious_days=7.0,
        cfr=0.02,
        ifr=0.006,
        hospitalization_rate=0.15,
        vaccine_available=True,
        vaccine_efficacy=0.95,
        vaccine_coverage=0.70,
        data_source="Lancet/WHO 2020",
        last_updated="2020-12",
        notes="Original strain, high mRNA vaccine efficacy",
    ),
    "measles": DiseaseParameters(
        name="measles",
        display_name="Measles",
        R0=15.0,
        R0_range=(12.0, 18.0),
        generation_time_days=12.0,
        incubation_days=10.0,
        infectious_days=8.0,
        cfr=0.002,
        ifr=0.001,
        hospitalization_rate=0.25,
        vaccine_available=True,
        vaccine_efficacy=0.97,
        vaccine_coverage=0.85,
        data_source="WHO/CDC",
        last_updated="2023-01",
        notes="Highest R0 of common diseases, excellent vaccine",
    ),
    "influenza_seasonal": DiseaseParameters(
        name="influenza_seasonal",
        display_name="Seasonal Influenza",
        R0=1.5,
        R0_range=(1.2, 1.8),
        generation_time_days=3.0,
        incubation_days=2.0,
        infectious_days=5.0,
        cfr=0.001,
        ifr=0.0001,
        hospitalization_rate=0.02,
        vaccine_available=True,
        vaccine_efficacy=0.45,  # Varies by season
        vaccine_coverage=0.50,
        data_source="CDC FluView",
        last_updated="2023-12",
        notes="Low R0 but poor vaccine match; annual drift",
    ),
    "influenza_1918": DiseaseParameters(
        name="influenza_1918",
        display_name="1918 Influenza Pandemic",
        R0=2.5,
        R0_range=(2.0, 3.0),
        generation_time_days=3.0,
        incubation_days=2.0,
        infectious_days=5.0,
        cfr=0.025,
        ifr=0.025,
        hospitalization_rate=0.20,
        vaccine_available=False,
        vaccine_efficacy=None,
        vaccine_coverage=None,
        data_source="Historical records, Taubenberger 2006",
        last_updated="2006-01",
        notes="W-shaped mortality curve; cytokine storm in young adults",
    ),
    "ebola": DiseaseParameters(
        name="ebola",
        display_name="Ebola (Zaire)",
        R0=2.0,
        R0_range=(1.5, 2.5),
        generation_time_days=15.0,
        incubation_days=10.0,
        infectious_days=10.0,
        cfr=0.50,
        ifr=0.50,
        hospitalization_rate=1.0,
        vaccine_available=True,
        vaccine_efficacy=0.975,  # rVSV-ZEBOV
        vaccine_coverage=0.40,  # Ring vaccination
        data_source="WHO/Henao-Restrepo 2017",
        last_updated="2019-12",
        notes="Low R0, high CFR, ring vaccination strategy",
    ),
    "smallpox": DiseaseParameters(
        name="smallpox",
        display_name="Smallpox",
        R0=5.0,
        R0_range=(3.5, 6.0),
        generation_time_days=17.0,
        incubation_days=12.0,
        infectious_days=14.0,
        cfr=0.30,
        ifr=0.30,
        hospitalization_rate=1.0,
        vaccine_available=True,
        vaccine_efficacy=0.95,
        vaccine_coverage=None,  # Eradicated
        data_source="WHO historical",
        last_updated="1980-05",
        notes="Eradicated 1980; only disease eliminated by vaccination",
    ),
    "polio": DiseaseParameters(
        name="polio",
        display_name="Poliomyelitis",
        R0=6.0,
        R0_range=(5.0, 7.0),
        generation_time_days=14.0,
        incubation_days=10.0,
        infectious_days=21.0,
        cfr=0.005,  # Among paralytic cases ~15%
        ifr=0.002,
        hospitalization_rate=0.01,
        vaccine_available=True,
        vaccine_efficacy=0.99,  # IPV
        vaccine_coverage=0.90,
        data_source="WHO GPEI",
        last_updated="2024-01",
        notes="Near eradication; oral vs injectable vaccines",
    ),
    "pertussis": DiseaseParameters(
        name="pertussis",
        display_name="Pertussis (Whooping Cough)",
        R0=15.0,
        R0_range=(12.0, 17.0),
        generation_time_days=21.0,
        incubation_days=10.0,
        infectious_days=21.0,
        cfr=0.001,
        ifr=0.001,
        hospitalization_rate=0.05,
        vaccine_available=True,
        vaccine_efficacy=0.85,  # DTaP
        vaccine_coverage=0.85,
        data_source="CDC/WHO",
        last_updated="2023-06",
        notes="Waning immunity requires boosters; cocooning strategy",
    ),
    "hiv": DiseaseParameters(
        name="hiv",
        display_name="HIV/AIDS",
        R0=4.0,
        R0_range=(2.0, 6.0),
        generation_time_days=1460.0,  # ~4 years
        incubation_days=2920.0,  # ~8 years to AIDS without treatment
        infectious_days=3650.0,  # ~10 years chronic
        cfr=1.0,  # Without treatment
        ifr=0.02,  # With ART
        hospitalization_rate=0.10,
        vaccine_available=False,
        vaccine_efficacy=None,
        vaccine_coverage=None,
        data_source="UNAIDS/WHO",
        last_updated="2024-01",
        notes="Chronic infection; ART transforms to manageable condition",
    ),
}


class DiseaseDataSource:
    """Fetch disease parameters from curated database or WHO API."""

    WHO_GHO_URL = "https://ghoapi.azureedge.net/api"

    def __init__(self):
        self._cache = DISEASE_DATABASE.copy()

    def get_disease(self, name: str) -> Optional[DiseaseParameters]:
        """Get parameters for a disease by name.

        Args:
            name: Disease name (case-insensitive, underscores or spaces)

        Returns:
            DiseaseParameters or None if not found
        """
        key = name.lower().replace(" ", "_").replace("-", "_")

        # Check aliases
        aliases = {
            "covid": "covid19_omicron",
            "corona": "covid19_omicron",
            "coronavirus": "covid19_omicron",
            "flu": "influenza_seasonal",
            "influenza": "influenza_seasonal",
            "spanish_flu": "influenza_1918",
            "whooping_cough": "pertussis",
        }
        key = aliases.get(key, key)

        return self._cache.get(key)

    def list_diseases(self) -> List[str]:
        """List all available diseases."""
        return list(self._cache.keys())

    def get_by_r0_range(
        self,
        min_r0: float = 0,
        max_r0: float = 100,
    ) -> List[DiseaseParameters]:
        """Get diseases within an R0 range."""
        return [
            d for d in self._cache.values()
            if min_r0 <= d.R0 <= max_r0
        ]

    def get_vaccine_preventable(self) -> List[DiseaseParameters]:
        """Get diseases with effective vaccines."""
        return [
            d for d in self._cache.values()
            if d.vaccine_available and d.vaccine_efficacy and d.vaccine_efficacy > 0.8
        ]


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Disease data source CLI")
    parser.add_argument("disease", nargs="?", help="Disease name")
    parser.add_argument("--list", "-l", action="store_true", help="List all")
    parser.add_argument("--high-r0", action="store_true", help="Show R0 > 5")
    args = parser.parse_args()

    ds = DiseaseDataSource()

    if args.list:
        print("\nAvailable diseases:")
        for name in ds.list_diseases():
            d = ds.get_disease(name)
            print(f"  {d.display_name:30s} R0={d.R0:.1f}")
    elif args.high_r0:
        print("\nHigh-R0 diseases (R0 > 5):")
        for d in ds.get_by_r0_range(5, 100):
            print(f"  {d.display_name:30s} R0={d.R0:.1f} ({d.R0_range[0]}-{d.R0_range[1]})")
    elif args.disease:
        d = ds.get_disease(args.disease)
        if d:
            print(f"\n{d.display_name}")
            print(f"  R0: {d.R0} ({d.R0_range[0]}-{d.R0_range[1]})")
            print(f"  Generation time: {d.generation_time_days} days")
            print(f"  CFR: {d.cfr*100:.2f}%")
            print(f"  IFR: {d.ifr*100:.3f}%")
            if d.vaccine_available:
                print(f"  Vaccine efficacy: {d.vaccine_efficacy*100:.0f}%")
                print(f"  Vaccine coverage: {d.vaccine_coverage*100:.0f}%")
            print(f"  Source: {d.data_source}")
            print(f"  Notes: {d.notes}")
        else:
            print(f"Disease '{args.disease}' not found")
            print("Use --list to see available diseases")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
