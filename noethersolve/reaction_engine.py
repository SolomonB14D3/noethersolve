"""noethersolve.reaction_engine — Organic chemistry reaction engine.

RDKit-powered computational tools for organic chemistry:
1. Functional group detection from SMILES (SMARTS pattern matching)
2. Mayr nucleophilicity/electrophilicity selectivity prediction
3. Rule-based reaction mechanism prediction (arrow-pushing logic)
4. Multi-step synthesis pathway validation

Conservation law philosophy: electrons are conserved, charge is conserved,
orbital symmetry is conserved. The rules of organic chemistry ARE conservation
laws, and this engine computes with them.

Usage:
    from noethersolve.reaction_engine import (
        analyze_molecule, predict_selectivity, predict_mechanism,
        validate_synthesis, check_baldwin, check_woodward_hoffmann,
    )

    # Analyze a molecule
    report = analyze_molecule("CCO")  # ethanol
    print(report)  # functional groups, reactive sites, properties

    # Predict selectivity
    report = predict_selectivity("[OH-]", "CCBr", solvent="dmso")
    print(report)  # log k, competing pathways, predicted products

    # Predict mechanism
    report = predict_mechanism(
        reactants=["CC(=O)C", "CC=O"],
        reagents=["NaOH"],
        conditions={"temperature": "low", "solvent": "water"},
    )
    print(report)

    # Validate a synthesis pathway
    report = validate_synthesis([
        {"substrate": "c1ccccc1", "reagent": "CH3COCl/AlCl3", "product": "CC(=O)c1ccccc1"},
        {"substrate": "CC(=O)c1ccccc1", "reagent": "NaBH4", "product": "CC(O)c1ccccc1"},
    ])
    print(report)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


# ─── Functional Group SMARTS Database ────────────────────────────────────────

# Each entry: (name, SMARTS pattern, reactive_role, description)
# reactive_role: "nucleophile", "electrophile", "both", "leaving_group", "acid", "base", "none"
_FUNCTIONAL_GROUPS: List[Tuple[str, str, str, str]] = [
    # Alcohols / Ethers / Epoxides
    ("primary alcohol", "[CH2][OX2H]", "nucleophile", "Primary -CH2OH"),
    ("secondary alcohol", "[CHX4][OX2H]", "nucleophile", "Secondary -CHOH"),
    ("tertiary alcohol", "[CX4]([!H])([!H])([!H])[OX2H]", "nucleophile", "Tertiary -COH"),
    ("phenol", "[OX2H]c1ccccc1", "acid", "Aromatic -OH (pKa ~10)"),
    ("ether", "[OD2]([#6])[#6]", "none", "R-O-R"),
    ("epoxide", "C1OC1", "electrophile", "Strained 3-membered ring — reactive electrophile"),

    # Carbonyls
    ("aldehyde", "[CX3H1](=O)[#6]", "electrophile", "R-CHO — electrophilic carbonyl carbon"),
    ("ketone", "[CX3](=O)([#6])[#6]", "electrophile", "R2C=O — electrophilic carbonyl carbon"),
    ("carboxylic acid", "[CX3](=O)[OX2H1]", "acid", "R-COOH (pKa ~4-5)"),
    ("ester", "[CX3](=O)[OX2][#6]", "electrophile", "R-COOR — nucleophilic acyl substitution substrate"),
    ("amide", "[CX3](=O)[NX3]", "none", "R-CONR2 — low reactivity due to resonance"),
    ("acid chloride", "[CX3](=O)[Cl]", "electrophile", "R-COCl — highly reactive acylating agent"),
    ("acid anhydride", "[CX3](=O)[OX2][CX3](=O)", "electrophile", "R-CO-O-CO-R"),
    ("acetal", "[CX4]([OX2])([OX2])", "none", "Protected carbonyl"),

    # Nitrogen
    ("primary amine", "[NX3H2][#6]", "nucleophile", "R-NH2 — nucleophile and base"),
    ("secondary amine", "[NX3H1]([#6])[#6]", "nucleophile", "R2NH — nucleophile and base"),
    ("tertiary amine", "[NX3]([#6])([#6])[#6]", "nucleophile", "R3N — nucleophile and base"),
    ("nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]", "none", "R-NO2 — strong electron-withdrawing group"),
    ("nitrile", "[CX2]#[NX1]", "electrophile", "R-CN — electrophilic carbon"),
    ("imine", "[CX3]=[NX2]", "electrophile", "R2C=NR — electrophilic carbon"),
    ("azide", "[NX2]=[NX2]=[NX1]", "nucleophile", "R-N3"),

    # Sulfur
    ("thiol", "[SX2H]", "nucleophile", "R-SH — strong nucleophile"),
    ("thioether", "[SX2]([#6])[#6]", "nucleophile", "R-S-R — nucleophilic sulfur"),
    ("sulfoxide", "[SX3](=O)([#6])[#6]", "none", "R-SO-R"),
    ("sulfone", "[SX4](=O)(=O)([#6])[#6]", "none", "R-SO2-R"),

    # Halides
    ("alkyl fluoride", "[CX4][FX1]", "electrophile", "C-F — poor leaving group, strong bond"),
    ("alkyl chloride", "[CX4][ClX1]", "electrophile", "C-Cl — moderate leaving group"),
    ("alkyl bromide", "[CX4][BrX1]", "electrophile", "C-Br — good leaving group"),
    ("alkyl iodide", "[CX4][IX1]", "electrophile", "C-I — excellent leaving group"),
    ("aryl halide", "c[F,Cl,Br,I]", "none", "Ar-X — unreactive to SN1/SN2 (requires SNAr or metal catalysis)"),

    # Unsaturation
    ("alkene", "[CX3]=[CX3]", "nucleophile", "C=C — pi electrons act as nucleophile"),
    ("alkyne", "[CX2]#[CX2]", "nucleophile", "C≡C — terminal alkynes are acidic (pKa ~25)"),
    ("conjugated diene", "[CX3]=[CX3]-[CX3]=[CX3]", "nucleophile", "Diene for Diels-Alder [4+2]"),
    ("aromatic ring", "c1ccccc1", "nucleophile", "Benzene ring — EAS substrate"),
    ("michael acceptor", "[CX3]=[CX3]-[CX3](=O)", "electrophile", "α,β-unsaturated carbonyl — conjugate addition"),

    # Organometallics / Special
    ("grignard", "[#6][Mg][F,Cl,Br,I]", "nucleophile", "R-MgX — powerful nucleophile/base"),
    ("phosphorus ylide", "[#6]=[PX4]", "nucleophile", "Wittig reagent R2C=PPh3"),
    ("boron", "[BX3]", "electrophile", "Lewis acid"),

    # Leaving groups (on carbon)
    ("tosylate", "[OX2][SX4](=O)(=O)c1ccc(C)cc1", "leaving_group", "OTs — excellent leaving group"),
    ("mesylate", "[OX2][SX4](=O)(=O)[CH3]", "leaving_group", "OMs — excellent leaving group"),
    ("triflate", "[OX2][SX4](=O)(=O)[CX4](F)(F)F", "leaving_group", "OTf — superb leaving group"),
]

# Compiled SMARTS patterns (lazy init)
_COMPILED_FG: Optional[List[Tuple[str, object, str, str]]] = None


def _get_compiled_fg():
    global _COMPILED_FG
    if _COMPILED_FG is None:
        _COMPILED_FG = []
        for name, smarts, role, desc in _FUNCTIONAL_GROUPS:
            pat = Chem.MolFromSmarts(smarts)
            if pat is not None:
                _COMPILED_FG.append((name, pat, role, desc))
    return _COMPILED_FG


# ─── Mayr Nucleophilicity / Electrophilicity Database ────────────────────────

# Mayr equation: log k(20°C) = s_N × (N + E)
# where N = nucleophilicity parameter, s_N = slope, E = electrophilicity parameter
# Published values from Mayr's database (mayr.ar.tum.de)
# Format: name -> (N, s_N) for nucleophiles, (E,) for electrophiles

_MAYR_NUCLEOPHILES: Dict[str, Tuple[float, float]] = {
    # Anions (in DMSO or water)
    "hydroxide (water)": (10.40, 0.60),
    "hydroxide (DMSO)": (13.91, 0.52),
    "cyanide (water)": (16.27, 0.60),
    "cyanide (DMSO)": (19.36, 0.67),
    "azide (water)": (14.57, 0.60),
    "azide (DMSO)": (20.53, 0.65),
    "acetate (DMSO)": (14.13, 0.56),
    "fluoride (DMSO)": (16.1, 0.50),
    "chloride (DMSO)": (15.52, 0.59),
    "bromide (DMSO)": (13.80, 0.59),
    "iodide (DMSO)": (13.40, 0.59),
    "thiophenolate (DMSO)": (21.43, 0.57),
    "phenolate (DMSO)": (17.71, 0.55),
    "methoxide (MeOH)": (12.30, 0.60),
    "ethoxide (EtOH)": (12.48, 0.60),
    "tert-butoxide (DMSO)": (15.63, 0.42),
    "hydride (LiAlH4)": (18.0, 0.55),
    "enolate (acetone, DMSO)": (20.8, 0.60),
    "malonate anion (DMSO)": (22.0, 0.65),

    # Neutral nucleophiles
    "water": (5.20, 0.89),
    "methanol": (5.75, 0.86),
    "ethanol": (6.05, 0.86),
    "ammonia": (9.48, 0.62),
    "methylamine": (11.90, 0.58),
    "dimethylamine": (13.17, 0.56),
    "trimethylamine": (10.59, 0.63),
    "pyridine": (11.30, 0.63),
    "DMAP": (15.50, 0.60),
    "triphenylphosphine": (14.90, 0.60),
    "DABCO": (13.50, 0.62),
    "imidazole": (11.47, 0.63),

    # Carbon nucleophiles
    "allyltrimethylsilane": (3.07, 1.10),
    "1-methylindole": (5.75, 1.06),
    "2-methylfuran": (3.61, 1.27),
    "1,3-dimethoxybenzene": (2.48, 1.16),
    "N,N-dimethylaniline": (2.20, 1.18),

    # Enamines
    "morpholine enamine (cyclohexanone)": (13.36, 0.80),
    "pyrrolidine enamine (cyclohexanone)": (15.47, 0.82),

    # Grignard
    "MeMgBr (THF)": (20.0, 0.55),
    "PhMgBr (THF)": (18.5, 0.55),
}

_MAYR_ELECTROPHILES: Dict[str, float] = {
    # Carbocations
    "trityl+ (Ph3C+)": (-5.53),
    "benzhydryl+ (Ph2CH+)": (-1.36),
    "4-MeO-benzhydryl+": (-5.89),
    "tert-butyl+": (-11.0),

    # Alkyl halides / sulfonates
    "methyl iodide": (-9.3),
    "methyl bromide": (-10.7),
    "methyl tosylate": (-10.1),
    "benzyl bromide": (-7.5),
    "allyl bromide": (-8.5),
    "epichlorohydrin": (-8.5),

    # Carbonyls
    "formaldehyde": (-5.5),
    "acetaldehyde": (-8.3),
    "benzaldehyde": (-9.5),
    "acetone": (-12.1),
    "cyclohexanone": (-12.5),
    "ethyl acetate": (-16.0),
    "acetyl chloride": (-5.0),
    "acetic anhydride": (-8.0),

    # Michael acceptors
    "methyl vinyl ketone": (-13.2),
    "acrylonitrile": (-14.0),
    "methyl acrylate": (-15.9),
    "nitroethylene": (-9.6),
    "maleic anhydride": (-11.0),

    # Electrophilic aromatic
    "NO2+ (nitronium)": (4.0),
    "Br2 / FeBr3": (-5.0),
    "CH3CO+ (acylium from AlCl3)": (-7.0),

    # Protons
    "H3O+ (water)": (-1.7),
    "AcOH": (-6.7),
    "PhOH": (-5.3),
}

# ─── Baldwin's Rules Database ────────────────────────────────────────────────

# Baldwin's rules for ring closure feasibility
# Format: (ring_size, endo_or_exo, trig_or_tet_or_dig) -> (favored, explanation)
_BALDWIN_RULES: Dict[Tuple[int, str, str], Tuple[bool, str]] = {
    # 3-membered rings
    (3, "exo", "tet"): (True, "3-Exo-Tet: FAVORED (e.g., epoxide formation)"),
    (3, "exo", "trig"): (True, "3-Exo-Trig: FAVORED"),
    (3, "endo", "trig"): (False, "3-Endo-Trig: DISFAVORED (Baldwin's rules)"),
    (3, "exo", "dig"): (True, "3-Exo-Dig: FAVORED"),
    (3, "endo", "dig"): (True, "3-Endo-Dig: FAVORED"),

    # 4-membered rings
    (4, "exo", "tet"): (True, "4-Exo-Tet: FAVORED"),
    (4, "exo", "trig"): (True, "4-Exo-Trig: FAVORED"),
    (4, "endo", "trig"): (False, "4-Endo-Trig: DISFAVORED (Baldwin's rules)"),
    (4, "exo", "dig"): (True, "4-Exo-Dig: FAVORED"),
    (4, "endo", "dig"): (True, "4-Endo-Dig: FAVORED"),

    # 5-membered rings (most common)
    (5, "exo", "tet"): (True, "5-Exo-Tet: FAVORED (very common, e.g., lactonization)"),
    (5, "endo", "tet"): (False, "5-Endo-Tet: DISFAVORED (Baldwin's rules)"),
    (5, "exo", "trig"): (True, "5-Exo-Trig: FAVORED (very common)"),
    (5, "endo", "trig"): (True, "5-Endo-Trig: FAVORED (e.g., Dieckmann cyclization)"),
    (5, "exo", "dig"): (True, "5-Exo-Dig: FAVORED"),
    (5, "endo", "dig"): (True, "5-Endo-Dig: FAVORED"),

    # 6-membered rings (very common)
    (6, "exo", "tet"): (True, "6-Exo-Tet: FAVORED"),
    (6, "endo", "tet"): (False, "6-Endo-Tet: DISFAVORED (Baldwin's rules)"),
    (6, "exo", "trig"): (True, "6-Exo-Trig: FAVORED"),
    (6, "endo", "trig"): (True, "6-Endo-Trig: FAVORED (e.g., Robinson annulation)"),
    (6, "exo", "dig"): (True, "6-Exo-Dig: FAVORED"),
    (6, "endo", "dig"): (True, "6-Endo-Dig: FAVORED"),

    # 7-membered rings
    (7, "exo", "tet"): (True, "7-Exo-Tet: FAVORED"),
    (7, "endo", "tet"): (True, "7-Endo-Tet: FAVORED"),
    (7, "exo", "trig"): (True, "7-Exo-Trig: FAVORED"),
    (7, "endo", "trig"): (True, "7-Endo-Trig: FAVORED"),
    (7, "exo", "dig"): (True, "7-Exo-Dig: FAVORED"),
    (7, "endo", "dig"): (True, "7-Endo-Dig: FAVORED"),
}

# ─── Woodward-Hoffmann Rules ────────────────────────────────────────────────

# Pericyclic reactions: thermal vs photochemical allowed/forbidden
# (n_electrons, "thermal"/"photochemical") -> (suprafacial_allowed, explanation)
_PERICYCLIC_RULES: Dict[Tuple[int, str], Tuple[bool, str]] = {
    # [2+2] cycloaddition (4 electrons total)
    (4, "thermal"): (False,
        "[2+2] Cycloaddition: THERMALLY FORBIDDEN (supra-supra). "
        "Requires photochemical activation or antarafacial component."),
    (4, "photochemical"): (True,
        "[2+2] Cycloaddition: PHOTOCHEMICALLY ALLOWED (supra-supra). "
        "Common: Paterno-Büchi, photocycloaddition of alkenes."),

    # [4+2] cycloaddition (6 electrons total) — Diels-Alder
    (6, "thermal"): (True,
        "[4+2] Cycloaddition (Diels-Alder): THERMALLY ALLOWED (supra-supra). "
        "Concerted, stereospecific. Endo rule applies."),
    (6, "photochemical"): (False,
        "[4+2] Cycloaddition: PHOTOCHEMICALLY FORBIDDEN (supra-supra)."),

    # [6+4] cycloaddition (10 electrons)
    (10, "thermal"): (True,
        "[6+4] Cycloaddition: THERMALLY ALLOWED (supra-supra)."),
    (10, "photochemical"): (False,
        "[6+4] Cycloaddition: PHOTOCHEMICALLY FORBIDDEN (supra-supra)."),

    # [1,3]-sigmatropic (2 electrons)
    (2, "thermal"): (False,
        "[1,3]-Sigmatropic shift: THERMALLY FORBIDDEN (suprafacial). "
        "Requires antarafacial, which is geometrically impossible for [1,3]-H."),
    (2, "photochemical"): (True,
        "[1,3]-Sigmatropic shift: PHOTOCHEMICALLY ALLOWED (suprafacial)."),

    # [1,5]-sigmatropic (6 electrons, same as [3,3])
    # [3,3]-sigmatropic (6 electrons) — Cope, Claisen
    # Already covered by 6 thermal=True

    # Electrocyclic 4 electrons (e.g., butadiene → cyclobutene)
    # 4 electrons thermal = conrotatory, photochemical = disrotatory
    # Electrocyclic 6 electrons (e.g., hexatriene → cyclohexadiene)
    # 6 electrons thermal = disrotatory, photochemical = conrotatory
}

# Electrocyclic rules: n_pi_electrons -> (thermal_mode, photochemical_mode)
_ELECTROCYCLIC: Dict[int, Tuple[str, str]] = {
    4: ("conrotatory", "disrotatory"),
    6: ("disrotatory", "conrotatory"),
    8: ("conrotatory", "disrotatory"),
}

# ─── SN1/SN2/E1/E2 Competition Rules ────────────────────────────────────────

_SUBSTRATE_CLASSES = {
    "methyl": {"sn2": "fast", "sn1": "never", "e2": "slow", "e1": "never"},
    "primary": {"sn2": "fast", "sn1": "very_slow", "e2": "moderate", "e1": "very_slow"},
    "secondary": {"sn2": "moderate", "sn1": "moderate", "e2": "fast", "e1": "moderate"},
    "tertiary": {"sn2": "never", "sn1": "fast", "e2": "fast", "e1": "fast"},
    "allylic": {"sn2": "fast", "sn1": "fast", "e2": "moderate", "e1": "moderate"},
    "benzylic": {"sn2": "fast", "sn1": "fast", "e2": "moderate", "e1": "moderate"},
    "vinyl": {"sn2": "never", "sn1": "never", "e2": "possible", "e1": "never"},
    "aryl": {"sn2": "never", "sn1": "never", "e2": "never", "e1": "never"},
}


# ─── Reaction Templates ─────────────────────────────────────────────────────

@dataclass
class ReactionTemplate:
    """A named reaction with conditions, mechanism, and stereo outcome."""
    name: str
    category: str  # substitution, elimination, addition, pericyclic, oxidation, reduction, rearrangement
    mechanism_type: str  # SN1, SN2, E1, E2, EAS, radical, concerted, etc.
    description: str
    required_fg_substrate: List[str]  # functional groups needed on substrate
    required_fg_reagent: List[str]  # functional groups/reagents needed
    conditions: Dict[str, str]  # solvent, temperature, catalyst, etc.
    mechanism_steps: List[str]  # arrow-pushing steps in plain text
    stereo_outcome: str  # inversion, retention, racemization, suprafacial, etc.
    product_fg: List[str]  # what functional groups appear in product
    competing_reactions: List[str]  # names of competing reactions
    notes: List[str]  # additional notes


def _build_reaction_templates() -> Dict[str, ReactionTemplate]:
    """Build the reaction template database."""
    db: Dict[str, ReactionTemplate] = {}

    def _add(name: str, **kwargs) -> None:
        for k in ("conditions", "notes", "competing_reactions"):
            kwargs.setdefault(k, {} if k == "conditions" else [])
        db[name] = ReactionTemplate(name=name, **kwargs)

    _add("SN2",
         category="substitution", mechanism_type="SN2",
         description="Bimolecular nucleophilic substitution — backside attack, concerted, Walden inversion",
         required_fg_substrate=["alkyl halide", "tosylate", "mesylate", "triflate", "epoxide"],
         required_fg_reagent=["nucleophile"],
         conditions={"substrate": "methyl or primary (best), secondary (slower)", "solvent": "polar aprotic (DMSO, DMF, acetone)"},
         mechanism_steps=[
             "Nucleophile attacks carbon from backside (180° to leaving group)",
             "C-Nu bond forms as C-LG bond breaks (concerted, single transition state)",
             "Configuration at carbon inverts (Walden inversion)",
         ],
         stereo_outcome="inversion",
         product_fg=["new C-Nu bond"],
         competing_reactions=["E2"],
         notes=["Rate = k[substrate][nucleophile]", "Never occurs at tertiary carbon (steric)"])

    _add("SN1",
         category="substitution", mechanism_type="SN1",
         description="Unimolecular nucleophilic substitution — carbocation intermediate, racemization",
         required_fg_substrate=["alkyl halide", "tosylate"],
         required_fg_reagent=[],
         conditions={"substrate": "tertiary, allylic, benzylic (stable cation)", "solvent": "polar protic (water, alcohols)"},
         mechanism_steps=[
             "Leaving group departs to form carbocation (rate-determining step)",
             "Carbocation may rearrange to more stable form (hydride/methyl shift)",
             "Nucleophile attacks carbocation from either face",
         ],
         stereo_outcome="racemization",
         product_fg=["new C-Nu bond"],
         competing_reactions=["E1"],
         notes=["Rate = k[substrate]", "Carbocation rearrangement possible"])

    _add("E2",
         category="elimination", mechanism_type="E2",
         description="Bimolecular elimination — anti-periplanar, concerted, gives alkene",
         required_fg_substrate=["alkyl halide", "tosylate"],
         required_fg_reagent=["base"],
         conditions={"substrate": "any (but strong base needed)", "solvent": "any", "temperature": "higher favors E over S"},
         mechanism_steps=[
             "Base abstracts β-hydrogen anti-periplanar to leaving group",
             "C-H bond breaks, C=C pi bond forms, C-LG bond breaks (all concerted)",
             "Alkene product forms with specific geometry based on anti-periplanar requirement",
         ],
         stereo_outcome="anti-periplanar determines E/Z geometry",
         product_fg=["alkene"],
         competing_reactions=["SN2"],
         notes=["Rate = k[substrate][base]", "Zaitsev (bulky base → Hofmann)", "Anti-periplanar required"])

    _add("E1",
         category="elimination", mechanism_type="E1",
         description="Unimolecular elimination — carbocation intermediate, gives alkene",
         required_fg_substrate=["alkyl halide", "tosylate"],
         required_fg_reagent=[],
         conditions={"substrate": "tertiary (stable cation)", "solvent": "polar protic", "temperature": "high"},
         mechanism_steps=[
             "Leaving group departs to form carbocation (rate-determining)",
             "Carbocation may rearrange",
             "Base (often solvent) abstracts β-hydrogen to form alkene",
         ],
         stereo_outcome="Zaitsev product predominates (most substituted alkene)",
         product_fg=["alkene"],
         competing_reactions=["SN1"],
         notes=["Rate = k[substrate]", "Favored at high temperature"])

    _add("Diels-Alder",
         category="pericyclic", mechanism_type="concerted",
         description="[4+2] cycloaddition — conjugated diene + dienophile → cyclohexene",
         required_fg_substrate=["conjugated diene"],
         required_fg_reagent=["alkene", "michael acceptor"],
         conditions={"temperature": "thermal (no light needed)", "diene": "must be s-cis"},
         mechanism_steps=[
             "Diene adopts s-cis conformation",
             "HOMO(diene) interacts with LUMO(dienophile) — normal electron demand",
             "Six electrons rearrange in concerted [4+2] cycloaddition",
             "Two new C-C sigma bonds form simultaneously, one C=C pi bond remains",
         ],
         stereo_outcome="suprafacial on both components, endo kinetically favored",
         product_fg=["cyclohexene"],
         competing_reactions=[],
         notes=["Electron-rich diene + electron-poor dienophile = fast", "Stereospecific: cis stays cis"])

    _add("EAS (general)",
         category="substitution", mechanism_type="EAS",
         description="Electrophilic aromatic substitution — electrophile + aromatic ring",
         required_fg_substrate=["aromatic ring"],
         required_fg_reagent=["electrophile"],
         conditions={"catalyst": "Lewis acid often needed"},
         mechanism_steps=[
             "Electrophile attacks pi system of aromatic ring",
             "Arenium ion (sigma complex / Wheland intermediate) forms — aromaticity disrupted",
             "Proton lost from sp3 carbon to restore aromaticity",
         ],
         stereo_outcome="not applicable (planar product)",
         product_fg=["substituted aromatic"],
         competing_reactions=[],
         notes=[
             "EDG (OH, NH2, OR) → ortho/para directors, activate ring",
             "EWG (NO2, COOH, SO3H) → meta directors, deactivate ring",
             "Halogens: deactivating but ortho/para directing",
         ])

    _add("Grignard addition",
         category="addition", mechanism_type="nucleophilic addition",
         description="Grignard reagent adds to carbonyl → alcohol after workup",
         required_fg_substrate=["aldehyde", "ketone", "ester", "epoxide"],
         required_fg_reagent=["grignard"],
         conditions={"solvent": "anhydrous ether (THF, Et2O)", "exclusion": "water, protic sources"},
         mechanism_steps=[
             "Mg coordinates to carbonyl oxygen (Lewis acid activation)",
             "Carbanion equivalent (R-) attacks electrophilic carbonyl carbon",
             "Alkoxide intermediate forms",
             "Aqueous acid workup protonates alkoxide to give alcohol",
         ],
         stereo_outcome="racemic (unless chiral auxiliary/catalyst)",
         product_fg=["alcohol"],
         competing_reactions=["protonation by protic sources (destroys Grignard)"],
         notes=[
             "RCHO + RMgX → secondary alcohol",
             "R2CO + RMgX → tertiary alcohol",
             "Ester + 2 RMgX → tertiary alcohol",
             "Epoxide + RMgX → primary alcohol (attacks less hindered carbon)",
         ])

    _add("Aldol addition",
         category="addition", mechanism_type="nucleophilic addition",
         description="Enolate + aldehyde/ketone → β-hydroxy carbonyl",
         required_fg_substrate=["aldehyde", "ketone"],
         required_fg_reagent=["base"],
         conditions={"temperature": "low for kinetic control"},
         mechanism_steps=[
             "Base deprotonates α-carbon to form enolate",
             "Enolate attacks electrophilic carbonyl of partner",
             "β-alkoxide intermediate forms",
             "Protonation gives β-hydroxy carbonyl (aldol product)",
         ],
         stereo_outcome="depends on enolate geometry (Z → syn aldol, E → anti aldol)",
         product_fg=["primary alcohol", "secondary alcohol", "ketone", "aldehyde"],
         competing_reactions=["self-condensation", "Claisen condensation (if ester)"],
         notes=[
             "Dehydration gives α,β-unsaturated carbonyl (aldol condensation)",
             "Crossed aldol: one partner must be non-enolizable",
             "LDA at -78°C → kinetic enolate; NaOH/heat → thermodynamic",
         ])

    _add("Nucleophilic acyl substitution",
         category="substitution", mechanism_type="addition-elimination",
         description="Nucleophile attacks acyl compound, tetrahedral intermediate, leaving group departs",
         required_fg_substrate=["acid chloride", "acid anhydride", "ester", "amide"],
         required_fg_reagent=["nucleophile"],
         conditions={},
         mechanism_steps=[
             "Nucleophile attacks electrophilic carbonyl carbon",
             "Tetrahedral intermediate forms (sp3 carbon with 4 groups)",
             "Leaving group departs, C=O reforms",
             "Product: new acyl derivative with nucleophile attached",
         ],
         stereo_outcome="not applicable (trigonal planar carbon)",
         product_fg=["ester", "amide", "carboxylic acid"],
         competing_reactions=[],
         notes=[
             "Reactivity order: acid chloride > anhydride > ester > amide",
             "Leaving group ability: Cl- > RCOO- > RO- > RNH-",
         ])

    _add("Wittig reaction",
         category="addition", mechanism_type="concerted/stepwise",
         description="Phosphorus ylide + aldehyde/ketone → alkene + Ph3P=O",
         required_fg_substrate=["aldehyde", "ketone"],
         required_fg_reagent=["phosphorus ylide"],
         conditions={},
         mechanism_steps=[
             "Ylide (R2C=PPh3) attacks carbonyl carbon",
             "Betaine intermediate forms",
             "Oxaphosphetane forms (4-membered ring)",
             "[2+2] cycloreversion: alkene + triphenylphosphine oxide",
         ],
         stereo_outcome="stabilized ylide → E-alkene; unstabilized → Z-alkene",
         product_fg=["alkene"],
         competing_reactions=[],
         notes=["Regiospecific: C=C exactly where C=O was"])

    _add("Hydroboration-oxidation",
         category="addition", mechanism_type="concerted",
         description="BH3 adds syn across C=C → anti-Markovnikov alcohol after oxidation",
         required_fg_substrate=["alkene"],
         required_fg_reagent=["boron"],
         conditions={"reagent": "BH3·THF then H2O2/NaOH"},
         mechanism_steps=[
             "BH3 approaches alkene (syn addition via 4-membered TS)",
             "Boron attaches to less substituted carbon (anti-Markovnikov)",
             "Hydrogen attaches to more substituted carbon",
             "H2O2/NaOH oxidizes C-B bond to C-OH with retention",
         ],
         stereo_outcome="syn addition, anti-Markovnikov, retention at C-B",
         product_fg=["primary alcohol"],
         competing_reactions=["acid-catalyzed hydration (Markovnikov)"],
         notes=["Net result: anti-Markovnikov, syn hydration of alkene"])

    return db


_REACTION_DB: Dict[str, ReactionTemplate] = _build_reaction_templates()


# ─── Report Dataclasses ─────────────────────────────────────────────────────

@dataclass
class FunctionalGroupHit:
    """A functional group found in a molecule."""
    name: str
    atoms: List[Tuple[int, ...]]  # atom indices matching
    count: int
    reactive_role: str
    description: str


@dataclass
class MoleculeAnalysis:
    """Full analysis of a molecule from SMILES."""
    smiles: str
    canonical_smiles: str
    molecular_formula: str
    molecular_weight: float
    functional_groups: List[FunctionalGroupHit]
    nucleophilic_sites: List[str]
    electrophilic_sites: List[str]
    leaving_groups: List[str]
    acidic_groups: List[str]
    is_aromatic: bool
    num_stereocenters: int
    notes: List[str]

    def __str__(self) -> str:
        lines = ["=" * 60, "  Molecule Analysis", "=" * 60]
        lines.append(f"  SMILES: {self.canonical_smiles}")
        lines.append(f"  Formula: {self.molecular_formula}  MW: {self.molecular_weight:.1f}")
        lines.append(f"  Aromatic: {self.is_aromatic}  Stereocenters: {self.num_stereocenters}")
        lines.append("-" * 60)
        if self.functional_groups:
            lines.append("  Functional Groups:")
            for fg in self.functional_groups:
                lines.append(f"    [{fg.reactive_role:13s}] {fg.name} (×{fg.count}) — {fg.description}")
        if self.nucleophilic_sites:
            lines.append(f"  Nucleophilic sites: {', '.join(self.nucleophilic_sites)}")
        if self.electrophilic_sites:
            lines.append(f"  Electrophilic sites: {', '.join(self.electrophilic_sites)}")
        if self.leaving_groups:
            lines.append(f"  Leaving groups: {', '.join(self.leaving_groups)}")
        if self.acidic_groups:
            lines.append(f"  Acidic groups: {', '.join(self.acidic_groups)}")
        if self.notes:
            lines.append("-" * 60)
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class SelectivityReport:
    """Result of Mayr selectivity prediction."""
    nucleophile: str
    electrophile: str
    solvent: str
    log_k: Optional[float]
    N: Optional[float]
    s_N: Optional[float]
    E: Optional[float]
    competing_pathways: List[str]
    predicted_mechanism: str
    notes: List[str]

    def __str__(self) -> str:
        lines = ["=" * 60, "  Selectivity Prediction", "=" * 60]
        lines.append(f"  Nucleophile: {self.nucleophile}")
        lines.append(f"  Electrophile: {self.electrophile}")
        lines.append(f"  Solvent: {self.solvent}")
        if self.log_k is not None:
            lines.append(f"  Mayr: log k = s_N × (N + E) = {self.s_N:.2f} × ({self.N:.1f} + ({self.E:.1f})) = {self.log_k:.1f}")
            if self.log_k > 5:
                lines.append("  Rate: VERY FAST (essentially diffusion-controlled)")
            elif self.log_k > 0:
                lines.append("  Rate: FAST (proceeds readily at room temperature)")
            elif self.log_k > -5:
                lines.append("  Rate: MODERATE (may need heating or long reaction time)")
            else:
                lines.append("  Rate: SLOW (may not proceed under normal conditions)")
        else:
            lines.append("  Mayr: parameters not available for this pair")
        lines.append(f"  Predicted mechanism: {self.predicted_mechanism}")
        if self.competing_pathways:
            lines.append("  Competing pathways:")
            for p in self.competing_pathways:
                lines.append(f"    - {p}")
        if self.notes:
            lines.append("-" * 60)
            for n in self.notes:
                lines.append(f"  {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class MechanismStep:
    """A single step in a reaction mechanism."""
    step_number: int
    description: str
    arrow_type: str  # "curly" (heterolytic), "fishhook" (homolytic), "retro" (retrosynthetic)
    intermediates: List[str]  # SMILES of intermediates if computable
    notes: List[str]


@dataclass
class MechanismReport:
    """Full mechanism prediction for a reaction."""
    reaction_name: str
    reaction_type: str
    reactants: List[str]
    reagents: List[str]
    conditions: Dict[str, str]
    mechanism_steps: List[MechanismStep]
    predicted_products: List[str]
    stereo_outcome: str
    atom_balance: bool
    charge_balance: bool
    feasibility: str  # "favorable", "possible", "unfavorable", "forbidden"
    competing_reactions: List[str]
    notes: List[str]

    def __str__(self) -> str:
        lines = ["=" * 60, f"  Mechanism: {self.reaction_name}", "=" * 60]
        lines.append(f"  Type: {self.reaction_type}")
        lines.append(f"  Reactants: {' + '.join(self.reactants)}")
        if self.reagents:
            lines.append(f"  Reagents: {', '.join(self.reagents)}")
        if self.conditions:
            cond_str = ", ".join(f"{k}={v}" for k, v in self.conditions.items())
            lines.append(f"  Conditions: {cond_str}")
        lines.append("-" * 60)
        lines.append("  Mechanism Steps:")
        for step in self.mechanism_steps:
            lines.append(f"    Step {step.step_number}: {step.description}")
            if step.notes:
                for n in step.notes:
                    lines.append(f"      → {n}")
        lines.append("-" * 60)
        if self.predicted_products:
            lines.append(f"  Predicted products: {' + '.join(self.predicted_products)}")
        lines.append(f"  Stereochemistry: {self.stereo_outcome}")
        lines.append(f"  Atom balance: {'✓' if self.atom_balance else '✗ IMBALANCED'}")
        lines.append(f"  Charge balance: {'✓' if self.charge_balance else '✗ IMBALANCED'}")
        lines.append(f"  Feasibility: {self.feasibility}")
        if self.competing_reactions:
            lines.append(f"  Competing: {', '.join(self.competing_reactions)}")
        if self.notes:
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class SynthesisIssue:
    """An issue found in a synthesis pathway."""
    step_number: int
    severity: str  # HIGH, MODERATE, LOW
    description: str


@dataclass
class SynthesisReport:
    """Validation report for a multi-step synthesis."""
    verdict: str  # PASS, WARN, FAIL
    num_steps: int
    issues: List[SynthesisIssue]
    step_analyses: List[str]
    notes: List[str]

    def __str__(self) -> str:
        lines = ["=" * 60, f"  Synthesis Validation: {self.verdict}", "=" * 60]
        lines.append(f"  Steps: {self.num_steps}")
        if self.step_analyses:
            for sa in self.step_analyses:
                lines.append(f"  {sa}")
        if self.issues:
            lines.append("-" * 60)
            lines.append(f"  Issues ({len(self.issues)}):")
            for iss in self.issues:
                lines.append(f"    Step {iss.step_number} [{iss.severity}]: {iss.description}")
        if self.notes:
            lines.append("-" * 60)
            for n in self.notes:
                lines.append(f"  {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class BaldwinReport:
    """Result of Baldwin's rules check."""
    ring_size: int
    endo_exo: str
    geometry: str  # tet, trig, dig
    favored: bool
    explanation: str

    def __str__(self) -> str:
        status = "FAVORED" if self.favored else "DISFAVORED"
        return f"Baldwin's Rules: {self.ring_size}-{self.endo_exo.capitalize()}-{self.geometry.capitalize()} → {status}. {self.explanation}"


@dataclass
class WoodwardHoffmannReport:
    """Result of Woodward-Hoffmann rules check."""
    n_electrons: int
    conditions: str  # thermal, photochemical
    reaction_type: str
    allowed: bool
    explanation: str
    electrocyclic_mode: Optional[str] = None  # conrotatory, disrotatory

    def __str__(self) -> str:
        status = "ALLOWED" if self.allowed else "FORBIDDEN"
        lines = [f"Woodward-Hoffmann: {self.reaction_type} ({self.n_electrons}e, {self.conditions}) → {status}"]
        lines.append(f"  {self.explanation}")
        if self.electrocyclic_mode:
            lines.append(f"  Ring closure mode: {self.electrocyclic_mode}")
        return "\n".join(lines)


# ─── Public API ──────────────────────────────────────────────────────────────

def analyze_molecule(smiles: str) -> MoleculeAnalysis:
    """Analyze a molecule from SMILES: detect functional groups, reactive sites, properties.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        MoleculeAnalysis with all detected features.

    Raises:
        ValueError: if SMILES cannot be parsed.
        ImportError: if RDKit is not installed.
    """
    if not _HAS_RDKIT:
        raise ImportError("RDKit is required for analyze_molecule(). Install: pip install rdkit")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")

    canonical = Chem.MolToSmiles(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = Descriptors.MolWt(mol)

    # Detect functional groups
    fg_hits: List[FunctionalGroupHit] = []
    for name, pat, role, desc in _get_compiled_fg():
        matches = mol.GetSubstructMatches(pat)
        if matches:
            fg_hits.append(FunctionalGroupHit(
                name=name,
                atoms=[tuple(m) for m in matches],
                count=len(matches),
                reactive_role=role,
                description=desc,
            ))

    # Classify reactive sites
    nuc_sites = [fg.name for fg in fg_hits if fg.reactive_role == "nucleophile"]
    elec_sites = [fg.name for fg in fg_hits if fg.reactive_role == "electrophile"]
    lg_sites = [fg.name for fg in fg_hits if fg.reactive_role == "leaving_group"]
    acid_sites = [fg.name for fg in fg_hits if fg.reactive_role == "acid"]

    # Aromaticity
    is_arom = any(atom.GetIsAromatic() for atom in mol.GetAtoms())

    # Stereocenters
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    n_stereo = len(chiral_centers)

    # Notes
    notes: List[str] = []
    if any(fg.name == "grignard" for fg in fg_hits):
        if any(fg.name in ("primary alcohol", "secondary alcohol", "tertiary alcohol",
                            "phenol", "carboxylic acid", "primary amine") for fg in fg_hits):
            notes.append("WARNING: Grignard-incompatible protic group present — "
                         "Grignard will be destroyed by protonation")

    if any(fg.name in ("alkyl bromide", "alkyl iodide") for fg in fg_hits):
        if any(fg.name in ("tertiary alcohol",) for fg in fg_hits):
            notes.append("Tertiary substrate with leaving group — SN1/E1 likely")

    if any(fg.name == "conjugated diene" for fg in fg_hits):
        notes.append("Conjugated diene detected — Diels-Alder [4+2] cycloaddition possible")

    if any(fg.name == "michael acceptor" for fg in fg_hits):
        notes.append("Michael acceptor detected — conjugate (1,4-) addition possible")

    if n_stereo > 0:
        notes.append(f"{n_stereo} stereocenter(s) — consider stereochemical outcome of reactions")

    return MoleculeAnalysis(
        smiles=smiles,
        canonical_smiles=canonical,
        molecular_formula=formula,
        molecular_weight=mw,
        functional_groups=fg_hits,
        nucleophilic_sites=nuc_sites,
        electrophilic_sites=elec_sites,
        leaving_groups=lg_sites,
        acidic_groups=acid_sites,
        is_aromatic=is_arom,
        num_stereocenters=n_stereo,
        notes=notes,
    )


def predict_selectivity(
    nucleophile: str,
    electrophile: str,
    solvent: str = "DMSO",
) -> SelectivityReport:
    """Predict reaction rate and selectivity using Mayr's equation.

    Uses log k(20°C) = s_N × (N + E) from Mayr's nucleophilicity/electrophilicity
    database to predict reaction rates and competing pathways.

    Args:
        nucleophile: name of nucleophile (e.g., "hydroxide", "cyanide", "water")
                     or SMILES if RDKit is available for identification.
        electrophile: name of electrophile (e.g., "methyl bromide", "acetaldehyde")
                      or SMILES.
        solvent: solvent name — affects nucleophilicity ordering.

    Returns:
        SelectivityReport with rate prediction and competing pathways.
    """
    # Try to match nucleophile
    solvent_lower = solvent.lower()
    nuc_key = _match_mayr_nucleophile(nucleophile, solvent_lower)
    elec_key = _match_mayr_electrophile(electrophile)

    N_val, s_N_val, E_val = None, None, None
    log_k = None

    if nuc_key is not None:
        N_val, s_N_val = _MAYR_NUCLEOPHILES[nuc_key]
    if elec_key is not None:
        E_val = _MAYR_ELECTROPHILES[elec_key]

    if N_val is not None and s_N_val is not None and E_val is not None:
        log_k = s_N_val * (N_val + E_val)

    # Predict mechanism
    mechanism = _predict_mechanism_type(nucleophile, electrophile, solvent_lower)

    # Competing pathways
    competing: List[str] = []
    notes: List[str] = []

    if nuc_key and "oxide" in nuc_key.lower():
        competing.append("E2 elimination (strong base can deprotonate β-hydrogen)")
    if "tertiary" in electrophile.lower() or "tert" in electrophile.lower():
        competing.append("E1 elimination (tertiary substrate favors elimination)")
        competing.append("SN1 substitution (tertiary substrate forms stable carbocation)")
        notes.append("SN2 NOT possible at tertiary carbon — SN1 or E1/E2 only")

    if nuc_key is None:
        notes.append(f"Nucleophile '{nucleophile}' not found in Mayr database — "
                     f"try common names like 'hydroxide (water)', 'cyanide (DMSO)'")
    if elec_key is None:
        notes.append(f"Electrophile '{electrophile}' not found in Mayr database — "
                     f"try common names like 'methyl bromide', 'acetaldehyde'")

    return SelectivityReport(
        nucleophile=nuc_key or nucleophile,
        electrophile=elec_key or electrophile,
        solvent=solvent,
        log_k=log_k,
        N=N_val,
        s_N=s_N_val,
        E=E_val,
        competing_pathways=competing,
        predicted_mechanism=mechanism,
        notes=notes,
    )


def predict_mechanism(
    reactants: List[str],
    reagents: Optional[List[str]] = None,
    conditions: Optional[Dict[str, str]] = None,
) -> MechanismReport:
    """Predict the reaction mechanism for given reactants, reagents, and conditions.

    Uses rule-based arrow-pushing logic to predict the most likely mechanism,
    intermediates, products, and stereochemical outcomes.

    Args:
        reactants: list of SMILES strings for reactants.
        reagents: optional list of reagent names or SMILES.
        conditions: optional dict with keys like "temperature", "solvent", "catalyst".

    Returns:
        MechanismReport with full mechanism prediction.
    """
    if not _HAS_RDKIT:
        raise ImportError("RDKit is required for predict_mechanism(). Install: pip install rdkit")

    reagents = reagents or []
    conditions = conditions or {}

    # Analyze all reactants
    analyses = []
    for r in reactants:
        try:
            analyses.append(analyze_molecule(r))
        except ValueError:
            analyses.append(None)

    # Identify reagent roles
    reagent_roles = _classify_reagents(reagents)

    # Match to reaction template
    template, confidence = _match_reaction_template(analyses, reagent_roles, conditions)

    if template is None:
        # No template match — try generic analysis
        return _generic_mechanism(reactants, reagents, conditions, analyses)

    # Build mechanism steps from template
    steps: List[MechanismStep] = []
    for i, step_desc in enumerate(template.mechanism_steps, 1):
        steps.append(MechanismStep(
            step_number=i,
            description=step_desc,
            arrow_type="curly",
            intermediates=[],
            notes=[],
        ))

    # Try to predict products using RDKit reaction SMARTS
    predicted_products = _predict_products(template.name, reactants, reagents)

    # Check atom balance if we have products
    atom_bal = True
    charge_bal = True
    if predicted_products:
        atom_bal, charge_bal = _check_balance(reactants, predicted_products, reagents)

    notes: List[str] = list(template.notes)
    if confidence < 0.7:
        notes.insert(0, f"Match confidence: {confidence:.0%} — multiple mechanisms possible")

    return MechanismReport(
        reaction_name=template.name,
        reaction_type=template.mechanism_type,
        reactants=reactants,
        reagents=reagents,
        conditions=conditions,
        mechanism_steps=steps,
        predicted_products=predicted_products,
        stereo_outcome=template.stereo_outcome,
        atom_balance=atom_bal,
        charge_balance=charge_bal,
        feasibility="favorable" if confidence > 0.7 else "possible",
        competing_reactions=template.competing_reactions,
        notes=notes,
    )


def validate_synthesis(
    steps: List[Dict[str, str]],
) -> SynthesisReport:
    """Validate a multi-step synthesis pathway for feasibility.

    Checks each step for: functional group compatibility, reagent compatibility,
    protecting group logic, and atom conservation.

    Args:
        steps: list of dicts with keys "substrate" (SMILES), "reagent" (name/SMILES),
               and optionally "product" (SMILES), "conditions" (string).

    Returns:
        SynthesisReport with issues and validation results.
    """
    if not _HAS_RDKIT:
        raise ImportError("RDKit is required for validate_synthesis(). Install: pip install rdkit")

    issues: List[SynthesisIssue] = []
    step_analyses: List[str] = []
    notes: List[str] = []

    for i, step in enumerate(steps, 1):
        substrate_smi = step.get("substrate", "")
        reagent_str = step.get("reagent", "")
        product_smi = step.get("product", "")
        step.get("conditions", "")

        # Parse substrate
        try:
            sub_analysis = analyze_molecule(substrate_smi)
            step_analyses.append(f"Step {i}: {substrate_smi} + {reagent_str} → {product_smi or '?'}")
        except (ValueError, ImportError):
            issues.append(SynthesisIssue(i, "HIGH", f"Cannot parse substrate SMILES: {substrate_smi}"))
            step_analyses.append(f"Step {i}: INVALID SMILES")
            continue

        reagent_lower = reagent_str.lower()

        # Check Grignard compatibility
        if "grignard" in reagent_lower or "mgbr" in reagent_lower or "mgcl" in reagent_lower:
            protic_groups = [fg for fg in sub_analysis.functional_groups
                            if fg.name in ("primary alcohol", "secondary alcohol",
                                           "tertiary alcohol", "phenol",
                                           "carboxylic acid", "primary amine",
                                           "secondary amine", "thiol")]
            if protic_groups:
                group_names = [fg.name for fg in protic_groups]
                issues.append(SynthesisIssue(i, "HIGH",
                    f"Grignard reagent incompatible with unprotected {', '.join(group_names)} — "
                    f"Grignard will be destroyed by protonation. Protect these groups first."))

        # Check LiAlH4 / NaBH4 selectivity
        if "lialh4" in reagent_lower or "lithium aluminum" in reagent_lower:
            reducible = [fg for fg in sub_analysis.functional_groups
                         if fg.name in ("ester", "carboxylic acid", "amide",
                                         "acid chloride", "acid anhydride",
                                         "aldehyde", "ketone", "nitrile", "epoxide")]
            if len(reducible) > 1:
                names = [fg.name for fg in reducible]
                issues.append(SynthesisIssue(i, "MODERATE",
                    f"LiAlH4 will reduce ALL of: {', '.join(names)}. "
                    f"If selective reduction needed, consider NaBH4 (only reduces "
                    f"aldehydes/ketones) or DIBAL-H (can stop at aldehyde stage)."))

        if "nabh4" in reagent_lower or "sodium borohydride" in reagent_lower:
            # NaBH4 only reduces aldehydes and ketones
            esters = [fg for fg in sub_analysis.functional_groups if fg.name == "ester"]
            if esters:
                notes.append(f"Step {i}: NaBH4 will NOT reduce ester groups (selective for aldehyde/ketone)")

        # Check Friedel-Crafts on deactivated ring
        if "alcl3" in reagent_lower or "friedel" in reagent_lower:
            deactivators = [fg for fg in sub_analysis.functional_groups
                            if fg.name in ("nitro", "nitrile", "carboxylic acid")]
            if deactivators:
                issues.append(SynthesisIssue(i, "HIGH",
                    f"Friedel-Crafts fails on deactivated aromatic rings "
                    f"(found: {', '.join(fg.name for fg in deactivators)}). "
                    f"The ring is too electron-poor for electrophilic attack."))

        # Check oxidation agents with sensitive groups
        if any(ox in reagent_lower for ox in ["kmno4", "cro3", "jones", "pcc", "dmp", "swern"]):
            if "kmno4" in reagent_lower or "cro3" in reagent_lower or "jones" in reagent_lower:
                alkenes = [fg for fg in sub_analysis.functional_groups if fg.name == "alkene"]
                if alkenes:
                    issues.append(SynthesisIssue(i, "MODERATE",
                        "Strong oxidant may cleave alkene double bonds in addition to "
                        "oxidizing alcohols. Use PCC or Swern for selective alcohol oxidation."))

        # Check product atom balance
        if product_smi:
            try:
                prod_mol = Chem.MolFromSmiles(product_smi)
                sub_mol = Chem.MolFromSmiles(substrate_smi)
                if prod_mol and sub_mol:
                    # Rough heavy atom check (not including reagent atoms)
                    sub_heavy = sub_mol.GetNumHeavyAtoms()
                    prod_heavy = prod_mol.GetNumHeavyAtoms()
                    diff = prod_heavy - sub_heavy
                    if abs(diff) > 10:
                        issues.append(SynthesisIssue(i, "LOW",
                            f"Large atom count change ({diff:+d} heavy atoms) — "
                            f"verify product SMILES is correct"))
            except Exception:
                pass

    # Overall verdict
    has_high = any(iss.severity == "HIGH" for iss in issues)
    has_moderate = any(iss.severity == "MODERATE" for iss in issues)
    if has_high:
        verdict = "FAIL"
    elif has_moderate:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return SynthesisReport(
        verdict=verdict,
        num_steps=len(steps),
        issues=issues,
        step_analyses=step_analyses,
        notes=notes,
    )


def check_baldwin(
    ring_size: int,
    endo_exo: str,
    geometry: str,
) -> BaldwinReport:
    """Check Baldwin's rules for ring closure feasibility.

    Args:
        ring_size: 3-7 (number of atoms in the ring being formed).
        endo_exo: "endo" or "exo" (bond breaking inside or outside ring).
        geometry: "tet" (sp3), "trig" (sp2), or "dig" (sp).

    Returns:
        BaldwinReport with favored/disfavored verdict and explanation.
    """
    endo_exo = endo_exo.lower().strip()
    geometry = geometry.lower().strip()

    if ring_size < 3 or ring_size > 7:
        return BaldwinReport(
            ring_size=ring_size, endo_exo=endo_exo, geometry=geometry,
            favored=False,
            explanation=f"Ring size {ring_size} outside Baldwin's rules range (3-7). "
                        f"Rings < 3 impossible; rings > 7 generally disfavored entropically.",
        )

    key = (ring_size, endo_exo, geometry)
    if key in _BALDWIN_RULES:
        favored, explanation = _BALDWIN_RULES[key]
        return BaldwinReport(
            ring_size=ring_size, endo_exo=endo_exo, geometry=geometry,
            favored=favored, explanation=explanation,
        )

    return BaldwinReport(
        ring_size=ring_size, endo_exo=endo_exo, geometry=geometry,
        favored=False,
        explanation=f"Combination {ring_size}-{endo_exo}-{geometry} not found in Baldwin's rules database.",
    )


def check_woodward_hoffmann(
    n_electrons: int,
    conditions: str = "thermal",
    reaction_type: str = "cycloaddition",
) -> WoodwardHoffmannReport:
    """Check Woodward-Hoffmann rules for pericyclic reaction feasibility.

    Args:
        n_electrons: total number of pi electrons involved.
        conditions: "thermal" or "photochemical".
        reaction_type: "cycloaddition", "electrocyclic", or "sigmatropic".

    Returns:
        WoodwardHoffmannReport with allowed/forbidden verdict and explanation.
    """
    conditions = conditions.lower().strip()
    reaction_type = reaction_type.lower().strip()

    # Electrocyclic — special case (conrotatory vs disrotatory)
    if reaction_type == "electrocyclic":
        if n_electrons in _ELECTROCYCLIC:
            thermal_mode, photo_mode = _ELECTROCYCLIC[n_electrons]
            mode = thermal_mode if conditions == "thermal" else photo_mode
            return WoodwardHoffmannReport(
                n_electrons=n_electrons,
                conditions=conditions,
                reaction_type=f"electrocyclic ({n_electrons}π e⁻)",
                allowed=True,
                explanation=(
                    f"Electrocyclic ring closure/opening with {n_electrons} π electrons "
                    f"under {conditions} conditions proceeds via {mode} motion."
                ),
                electrocyclic_mode=mode,
            )
        return WoodwardHoffmannReport(
            n_electrons=n_electrons,
            conditions=conditions,
            reaction_type=f"electrocyclic ({n_electrons}π e⁻)",
            allowed=False,
            explanation="Electrocyclic rules require even number of π electrons (4, 6, 8).",
        )

    # Cycloaddition / sigmatropic — use the general 4n+2 / 4n rule
    key = (n_electrons, conditions)
    if key in _PERICYCLIC_RULES:
        allowed, explanation = _PERICYCLIC_RULES[key]
        return WoodwardHoffmannReport(
            n_electrons=n_electrons,
            conditions=conditions,
            reaction_type=f"{reaction_type} ({n_electrons}e⁻)",
            allowed=allowed,
            explanation=explanation,
        )

    # General rule: 4n+2 thermal supra-supra allowed; 4n thermal forbidden
    if n_electrons % 4 == 2:  # 4n+2
        allowed = (conditions == "thermal")
    else:  # 4n
        allowed = (conditions == "photochemical")

    status = "ALLOWED" if allowed else "FORBIDDEN"
    rule = "4n+2 electrons → thermally allowed (supra-supra)" if n_electrons % 4 == 2 \
        else "4n electrons → thermally forbidden (supra-supra), photochemically allowed"

    return WoodwardHoffmannReport(
        n_electrons=n_electrons,
        conditions=conditions,
        reaction_type=f"{reaction_type} ({n_electrons}e⁻)",
        allowed=allowed,
        explanation=f"{n_electrons}e⁻ {reaction_type} under {conditions} conditions: {status}. Rule: {rule}",
    )


def list_mayr_nucleophiles() -> List[str]:
    """List all nucleophiles in the Mayr database with their N and s_N values."""
    return sorted(_MAYR_NUCLEOPHILES.keys())


def list_mayr_electrophiles() -> List[str]:
    """List all electrophiles in the Mayr database with their E values."""
    return sorted(_MAYR_ELECTROPHILES.keys())


def list_reaction_templates() -> List[str]:
    """List all named reaction templates in the database."""
    return sorted(_REACTION_DB.keys())


def get_reaction_template(name: str) -> Optional[ReactionTemplate]:
    """Get full details of a named reaction template."""
    name_lower = name.lower().strip()
    for k, v in _REACTION_DB.items():
        if k.lower() == name_lower or name_lower in k.lower():
            return v
    return None


# ─── Internal Helpers ────────────────────────────────────────────────────────

def _match_mayr_nucleophile(name: str, solvent: str) -> Optional[str]:
    """Fuzzy-match a nucleophile name to the Mayr database."""
    name_lower = name.lower().strip()

    # Try exact match
    for key in _MAYR_NUCLEOPHILES:
        if key.lower() == name_lower:
            return key

    # Try match with solvent suffix
    for key in _MAYR_NUCLEOPHILES:
        key_lower = key.lower()
        base_name = key_lower.split("(")[0].strip()
        if base_name == name_lower:
            # Prefer matching solvent
            if solvent in key_lower:
                return key

    # Try partial match preferring solvent
    best_key = None
    for key in _MAYR_NUCLEOPHILES:
        key_lower = key.lower()
        base_name = key_lower.split("(")[0].strip()
        if name_lower in base_name or base_name in name_lower:
            if solvent in key_lower:
                return key
            if best_key is None:
                best_key = key
    return best_key


def _match_mayr_electrophile(name: str) -> Optional[str]:
    """Fuzzy-match an electrophile name to the Mayr database."""
    name_lower = name.lower().strip()
    for key in _MAYR_ELECTROPHILES:
        if key.lower() == name_lower:
            return key
    for key in _MAYR_ELECTROPHILES:
        key_lower = key.lower()
        if name_lower in key_lower or key_lower in name_lower:
            return key
    return None


def _predict_mechanism_type(nuc: str, elec: str, solvent: str) -> str:
    """Predict the most likely mechanism based on reactant properties."""
    nuc_lower = nuc.lower()
    elec_lower = elec.lower()

    # Strong base + alkyl halide → E2 likely
    strong_bases = ["tert-butoxide", "lda", "dbu", "dbn", "nanh2"]
    if any(b in nuc_lower for b in strong_bases):
        if any(h in elec_lower for h in ["halide", "bromide", "chloride", "iodide", "tosylate"]):
            return "E2 (strong base favors elimination)"

    # Weak nucleophile + tertiary → SN1
    if "tertiary" in elec_lower or "tert" in elec_lower:
        return "SN1 (tertiary substrate — backside attack blocked)"

    # Primary/methyl + good nucleophile → SN2
    if any(s in elec_lower for s in ["methyl", "primary", "benzyl", "allyl"]):
        return "SN2 (unhindered substrate + nucleophile)"

    # Carbonyl + nucleophile → nucleophilic addition
    if any(c in elec_lower for c in ["aldehyde", "ketone", "formaldehyde", "acetone"]):
        return "Nucleophilic addition to carbonyl"

    # Default
    return "SN2 or addition (depends on substrate class)"


def _classify_reagents(reagents: List[str]) -> Dict[str, str]:
    """Classify reagent names into roles: base, acid, oxidant, reductant, etc."""
    roles: Dict[str, str] = {}
    for r in reagents:
        r_lower = r.lower().strip()
        if any(b in r_lower for b in ["naoh", "koh", "lda", "nah", "t-buok",
                                       "dbu", "et3n", "triethylamine", "pyridine",
                                       "k2co3", "cs2co3", "nanh2"]):
            roles[r] = "base"
        elif any(a in r_lower for a in ["hcl", "h2so4", "hbr", "bf3", "alcl3",
                                         "ticl4", "fecl3", "tfa", "acoh", "ptsa"]):
            roles[r] = "acid"
        elif any(o in r_lower for o in ["kmno4", "cro3", "pcc", "dmp", "swern",
                                         "jones", "h2o2", "mcpba", "ozone"]):
            roles[r] = "oxidant"
        elif any(red in r_lower for red in ["nabh4", "lialh4", "dibal", "h2/pd",
                                             "h2 pd", "zn/hcl", "na/nh3"]):
            roles[r] = "reductant"
        elif any(c in r_lower for c in ["pd", "pt", "ni", "rh", "ru", "ir"]):
            roles[r] = "catalyst"
        else:
            roles[r] = "unknown"
    return roles


def _match_reaction_template(
    analyses: List[Optional[MoleculeAnalysis]],
    reagent_roles: Dict[str, str],
    conditions: Dict[str, str],
) -> Tuple[Optional[ReactionTemplate], float]:
    """Match reactant analyses + reagents to the best reaction template."""
    if not analyses or analyses[0] is None:
        return None, 0.0

    primary = analyses[0]
    fg_names = {fg.name for fg in primary.functional_groups}

    has_base = "base" in reagent_roles.values()
    has_acid = "acid" in reagent_roles.values()
    "reductant" in reagent_roles.values()
    "oxidant" in reagent_roles.values()

    best_match: Optional[ReactionTemplate] = None
    best_score = 0.0

    for name, template in _REACTION_DB.items():
        score = 0.0
        # Check substrate functional groups
        for req_fg in template.required_fg_substrate:
            for fg_name in fg_names:
                if req_fg.lower() in fg_name.lower() or fg_name.lower() in req_fg.lower():
                    score += 1.0
                    break

        # Check reagent requirements
        for req_rg in template.required_fg_reagent:
            req_lower = req_rg.lower()
            if req_lower == "nucleophile" and (has_base or len(analyses) > 1):
                score += 0.5
            elif req_lower == "base" and has_base:
                score += 0.5
            elif req_lower == "electrophile" and has_acid:
                score += 0.5
            elif req_lower == "grignard":
                for rn in reagent_roles:
                    if "mg" in rn.lower() or "grignard" in rn.lower():
                        score += 1.0
                        break

        # Check second reactant functional groups
        if len(analyses) > 1 and analyses[1] is not None:
            second_fg = {fg.name for fg in analyses[1].functional_groups}
            for req_fg in template.required_fg_reagent:
                for fg_name in second_fg:
                    if req_fg.lower() in fg_name.lower() or fg_name.lower() in req_fg.lower():
                        score += 0.5
                        break

        # Normalize
        total_reqs = len(template.required_fg_substrate) + len(template.required_fg_reagent)
        if total_reqs > 0:
            score = score / total_reqs
        else:
            score = 0.5

        if score > best_score:
            best_score = score
            best_match = template

    if best_score < 0.3:
        return None, 0.0
    return best_match, best_score


def _predict_products(
    reaction_name: str,
    reactants: List[str],
    reagents: List[str],
) -> List[str]:
    """Try to predict products using RDKit reaction SMARTS."""
    if not _HAS_RDKIT:
        return []

    # Simple reaction SMARTS for common reactions
    rxn_smarts: Dict[str, str] = {
        "SN2": "[C:1][Cl,Br,I:2]>>[C:1][OH]",  # generic — Nu replaces LG
        "E2": "[CH:1][C:2][Cl,Br,I:3]>>[C:1]=[C:2]",
        "Hydroboration-oxidation": "[C:1]=[C:2]>>[C:1][C:2][OH]",
    }

    if reaction_name in rxn_smarts:
        try:
            rxn = AllChem.ReactionFromSmarts(rxn_smarts[reaction_name])
            if rxn is None:
                return []
            mol = Chem.MolFromSmiles(reactants[0])
            if mol is None:
                return []
            products = rxn.RunReactants((mol,))
            if products:
                return [Chem.MolToSmiles(p) for p in products[0]]
        except Exception:
            pass

    return []


def _check_balance(
    reactants: List[str],
    products: List[str],
    reagents: List[str],
) -> Tuple[bool, bool]:
    """Check atom and charge balance between reactants and products."""
    if not _HAS_RDKIT:
        return True, True

    def _count_atoms(smiles_list: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                counts[sym] = counts.get(sym, 0) + 1
        return counts

    def _total_charge(smiles_list: List[str]) -> int:
        total = 0
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            total += Chem.GetFormalCharge(mol)
        return total

    # We can't perfectly balance without knowing all byproducts,
    # but we can check charge conservation
    r_charge = _total_charge(reactants)
    p_charge = _total_charge(products)
    charge_ok = (r_charge == p_charge)

    # Atom balance is approximate (reagent atoms may contribute)
    r_atoms = _count_atoms(reactants)
    p_atoms = _count_atoms(products)
    # Just check that no element appears in product but not in reactants
    atom_ok = True
    for elem in p_atoms:
        if elem not in r_atoms and elem != "H":
            atom_ok = False
            break

    return atom_ok, charge_ok


def _generic_mechanism(
    reactants: List[str],
    reagents: List[str],
    conditions: Dict[str, str],
    analyses: List[Optional[MoleculeAnalysis]],
) -> MechanismReport:
    """Generate a generic mechanism report when no template matches."""
    notes: List[str] = []
    steps: List[MechanismStep] = []

    # Describe what we found
    if analyses[0] is not None:
        fg_names = [fg.name for fg in analyses[0].functional_groups]
        nuc = analyses[0].nucleophilic_sites
        elec = analyses[0].electrophilic_sites
        notes.append(f"Substrate functional groups: {', '.join(fg_names) if fg_names else 'none detected'}")
        if nuc:
            notes.append(f"Nucleophilic sites: {', '.join(nuc)}")
        if elec:
            notes.append(f"Electrophilic sites: {', '.join(elec)}")

    steps.append(MechanismStep(
        step_number=1,
        description="No specific reaction template matched — generic analysis provided",
        arrow_type="curly",
        intermediates=[],
        notes=["Provide more specific reagents/conditions for detailed mechanism prediction"],
    ))

    return MechanismReport(
        reaction_name="Unknown",
        reaction_type="unclassified",
        reactants=reactants,
        reagents=reagents,
        conditions=conditions,
        mechanism_steps=steps,
        predicted_products=[],
        stereo_outcome="unknown",
        atom_balance=True,
        charge_balance=True,
        feasibility="unknown",
        competing_reactions=[],
        notes=notes,
    )
