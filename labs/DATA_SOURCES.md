# NoetherSolve Data Sources Registry

Public APIs and databases integrated with NoetherSolve labs. When building or extending a lab, **always search for a public data source first**.

---

## Currently Integrated

### Drug & Pharmacology

| Source | Module | API Type | Auth | Rate Limit |
|--------|--------|----------|------|------------|
| **ChEMBL** | `drug_data_source.py` | REST | None | 30/min |
| **PubChem** | `drug_data_source.py` | REST | None | 5/sec |
| **DrugBank** | ❌ Not integrated | REST | API key ($) | - |

**ChEMBL** (https://www.ebi.ac.uk/chembl/)
- ADMET properties, bioactivity, PK data
- `pip install chembl-webresource-client`
- Free, no authentication

**PubChem** (https://pubchem.ncbi.nlm.nih.gov/)
- Molecular properties, synonyms, assay data
- REST API, no auth needed
- Rate: 5 requests/second

---

### Epidemiology

| Source | Module | API Type | Auth | Rate Limit |
|--------|--------|----------|------|------------|
| **Built-in DB** | `disease_data_source.py` | Local | None | N/A |
| **WHO GHO** | ❌ Planned | REST | None | Unknown |
| **CDC API** | ❌ Planned | REST | None | Unknown |

**WHO Global Health Observatory** (https://www.who.int/data/gho)
- Disease surveillance, health indicators
- REST API: `https://ghoapi.azureedge.net/api`
- No auth, free access

**Our World in Data** (https://ourworldindata.org/)
- COVID-19, vaccinations, global health
- GitHub CSV downloads
- No API, but machine-readable

---

### Materials Science

| Source | Module | API Type | Auth | Rate Limit |
|--------|--------|----------|------|------------|
| **Built-in DB** | `catalyst_data_source.py` | Local | None | N/A |
| **Materials Project** | `catalyst_data_source.py` | REST | API key (free) | 30/min |

**Materials Project** (https://next-gen.materialsproject.org/)
- DFT-computed properties, band structures, surfaces
- Free API key required (register)
- `pip install mp-api`

**AFLOW** (http://aflowlib.org/)
- Crystal structures, thermodynamics
- REST API, no auth
- Alternative to Materials Project

**NOMAD** (https://nomad-lab.eu/)
- Ab initio calculation repository
- REST API, free registration

---

### Genetics & Genomics

| Source | Module | Status | API Type | Auth |
|--------|--------|--------|----------|------|
| **Ensembl** | ❌ Planned | REST | None | - |
| **NCBI** | ❌ Planned | REST | API key (free) | - |
| **UniProt** | ❌ Planned | REST | None | - |

**Ensembl** (https://rest.ensembl.org/)
- Gene sequences, variants, annotations
- REST API, no auth
- Rate: 15 requests/second (with email header)

**NCBI E-utilities** (https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- PubMed, GenBank, dbSNP
- API key recommended (free)
- Rate: 3/sec without key, 10/sec with key

**UniProt** (https://www.uniprot.org/)
- Protein sequences, function, structure
- REST API, no auth

---

### Climate & Environment

| Source | Module | Status | API Type | Auth |
|--------|--------|--------|----------|------|
| **Built-in** | `climate_lab.py` | Local | None | N/A |
| **NOAA** | ❌ Planned | REST | API key (free) | - |
| **Copernicus** | ❌ Planned | REST | Registration | - |

**NOAA Climate Data Online** (https://www.ncdc.noaa.gov/cdo-web/webservices/)
- Historical weather, climate indices
- Free API key required

**Copernicus Climate Data Store** (https://cds.climate.copernicus.eu/)
- ERA5 reanalysis, CMIP6 projections
- Free registration required
- Python API: `cdsapi`

---

### Finance & Economics

| Source | Module | Status | API Type | Auth |
|--------|--------|--------|----------|------|
| **Built-in** | `behavioral_economics_lab.py` | Local | N/A | - |
| **FRED** | ❌ Planned | REST | API key (free) | - |
| **Yahoo Finance** | ❌ Planned | REST | None | - |

**FRED** (https://fred.stlouisfed.org/)
- Federal Reserve economic data
- Free API key required
- `pip install fredapi`

---

## Discovery Process for New Labs

When creating a new lab, follow this process to find data sources:

### 1. Search for Public APIs

```bash
# Use these search patterns:
"[domain] public API"
"[domain] REST API free"
"[domain] database download"
"[domain] open data"
```

**Go-to repositories:**
- **APIs.guru** (https://apis.guru/) — OpenAPI directory
- **Public APIs** (https://github.com/public-apis/public-apis) — Curated list
- **RapidAPI** (https://rapidapi.com/) — API marketplace (some free)
- **Kaggle Datasets** (https://www.kaggle.com/datasets) — Static downloads

### 2. Evaluate Data Source Quality

| Criterion | Good | Bad |
|-----------|------|-----|
| **Auth** | None or free API key | Paid subscription |
| **Rate limit** | >10 req/min | <1 req/min |
| **Data format** | JSON, CSV | PDF, HTML scraping |
| **Update frequency** | Weekly or better | Yearly |
| **Documentation** | OpenAPI spec | None |

### 3. Implement Data Source Module

Create `noethersolve/{domain}_data_source.py`:

```python
"""domain_data_source.py -- Fetch domain data from [SOURCE].

Data sources:
    - [Source Name] (url) -- description
"""

from dataclasses import dataclass
from typing import Optional
import requests

# Built-in fallback data (always include this)
REFERENCE_DATA = {
    "item1": {...},
}

@dataclass
class DomainRecord:
    name: str
    # fields...
    data_source: str

class DomainDataSource:
    API_URL = "https://api.example.com"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DOMAIN_API_KEY")
        self._cache = REFERENCE_DATA.copy()

    def get(self, name: str) -> Optional[DomainRecord]:
        # Try cache first
        if name in self._cache:
            return self._cache[name]
        # Try API if key available
        if self.api_key:
            return self._fetch_from_api(name)
        return None
```

### 4. Add to Lab

```python
from noethersolve.domain_data_source import DomainDataSource

class MyLab:
    def __init__(self):
        self.data_source = DomainDataSource()

    def run(self):
        data = self.data_source.get("item_name")
        # Use real data instead of hardcoded values
```

### 5. Document in DATA_SOURCES.md

Add entry to the appropriate section with:
- Source name and URL
- Module location
- Auth requirements
- Rate limits
- What data it provides

---

## APIs Wanted (High Priority)

These would significantly improve existing labs:

| Lab | Needed Data | Potential Sources |
|-----|-------------|-------------------|
| **genetic_therapeutics** | Gene sequences | Ensembl, NCBI |
| **epidemic** | Real-time surveillance | WHO GHO, CDC API |
| **catalyst** | More surface energies | AFLOW, NOMAD |
| **supply_chain** | Shipping costs | Freightos, shipping APIs |
| **climate** | Historical CO2 | NOAA, Mauna Loa |

---

## Environment Variables

For APIs requiring keys, use these standard env var names:

```bash
export MP_API_KEY="your_materials_project_key"
export NCBI_API_KEY="your_ncbi_key"
export NOAA_API_KEY="your_noaa_key"
export FRED_API_KEY="your_fred_key"
```

All data source modules should check env vars as fallback when no key is passed to constructor.
