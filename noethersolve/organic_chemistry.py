"""noethersolve.organic_chemistry — Organic chemistry fact checker.

Validates claims about organic chemistry mechanisms, reagents, reactions,
stereochemistry, and synthesis strategies against a curated reference database.
Catches: confusing concerted with stepwise mechanisms, misidentifying reagent
compatibility, stereochemistry errors, and retrosynthesis misconceptions.

Covers 12 topics across 5 clusters: mechanisms, reagents, reactions, stereochem,
and synthesis.

Usage:
    from noethersolve.organic_chemistry import (
        check_organic_chemistry, list_organic_chemistry_topics,
        get_organic_chemistry_topic,
        OrganicChemistryReport, OrganicChemistryIssue, OrganicChemistryTopic,
    )

    # Check a topic and validate a claim
    report = check_organic_chemistry("E1 elimination", claim="E1 is concerted")
    print(report)
    # FAIL -- 1 issue: CLAIM_CHECK [HIGH] ...

    # Check a topic without a specific claim (returns info + common errors)
    report = check_organic_chemistry("Grignard")
    print(report)
    # PASS -- 0 issues, notes with key facts

    # List all topics in a cluster
    for name in list_organic_chemistry_topics(cluster="mechanisms"):
        print(name)

    # Get full info on a topic
    info = get_organic_chemistry_topic("Diels-Alder")
    print(info.key_facts)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─── Topic database ──────────────────────────────────────────────────────────

@dataclass
class OrganicChemistryTopic:
    """Full information about an organic chemistry topic."""
    id: str                   # e.g. "oc01_e1"
    name: str                 # human-readable name
    cluster: str              # mechanisms, reagents, reactions, stereochem, synthesis
    description: str          # one-paragraph summary
    key_facts: List[str]      # 3-5 important facts
    common_errors: List[str]  # 2-3 LLM mistakes
    references: List[str]     # textbook / review citations

    def __str__(self) -> str:
        lines = [f"{self.name} [{self.id}] (cluster: {self.cluster})"]
        lines.append(f"  {self.description}")
        if self.key_facts:
            lines.append("  Key facts:")
            for f in self.key_facts:
                lines.append(f"    - {f}")
        if self.common_errors:
            lines.append("  Common errors:")
            for err in self.common_errors:
                lines.append(f"    - {err}")
        if self.references:
            lines.append("  References:")
            for ref in self.references:
                lines.append(f"    - {ref}")
        return "\n".join(lines)


@dataclass
class OrganicChemistryIssue:
    """A single issue found when checking an organic chemistry claim."""
    check_type: str     # CLAIM_CHECK, MECHANISM_CHECK, ERROR_FLAG
    severity: str       # HIGH, MODERATE, LOW, INFO
    description: str
    details: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"  [{self.severity}] {self.check_type}: {self.description}"


@dataclass
class OrganicChemistryReport:
    """Result of check_organic_chemistry()."""
    verdict: str                                    # PASS, WARN, or FAIL
    topic: Optional[OrganicChemistryTopic]
    claim: Optional[str]
    issues: List[OrganicChemistryIssue]
    notes: List[str]

    def __str__(self) -> str:
        lines = []
        lines.append(f"{'=' * 60}")
        if self.topic:
            lines.append(f"  Organic Chemistry Check: {self.verdict}")
            lines.append(f"{'=' * 60}")
            lines.append(f"  Topic: {self.topic.name} [{self.topic.id}]")
            lines.append(f"  Cluster: {self.topic.cluster}")
        else:
            lines.append(f"  Organic Chemistry Check: {self.verdict}")
            lines.append(f"{'=' * 60}")
        if self.claim:
            lines.append(f"  Claim: {self.claim}")
        lines.append(f"{'-' * 60}")
        if self.issues:
            lines.append(f"  Issues ({len(self.issues)}):")
            for issue in self.issues:
                lines.append(str(issue))
        else:
            lines.append("  No issues found.")
        if self.notes:
            lines.append(f"{'-' * 60}")
            lines.append("  Notes:")
            for note in self.notes:
                lines.append(f"    - {note}")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


# ─── Build database ──────────────────────────────────────────────────────────

def _build_database() -> Dict[str, OrganicChemistryTopic]:
    """Build the organic chemistry topic database. Called once at module load."""
    db: Dict[str, OrganicChemistryTopic] = {}

    def _add(topic_id: str, name: str, **kwargs: object) -> None:
        kwargs.setdefault("references", [])
        topic = OrganicChemistryTopic(id=topic_id, name=name, **kwargs)  # type: ignore[arg-type]
        db[topic_id] = topic

    # ── mechanisms cluster ─────────────────────────────────────────────

    _add("oc01_e1", "E1 Elimination",
         cluster="mechanisms",
         description=(
             "Unimolecular elimination proceeding through a carbocation "
             "intermediate in two steps: ionization of the leaving group "
             "followed by deprotonation to form the alkene."
         ),
         key_facts=[
             "Two-step mechanism: (1) ionization to carbocation, (2) deprotonation",
             "Favored by tertiary substrates, weak bases, polar protic solvents, high temperature",
             "Competes with SN1 (same conditions) — elimination favored at higher temperatures",
             "Zaitsev product (more substituted alkene) typically predominates",
             "Rate = k[substrate] (unimolecular — base not in rate-determining step)",
         ],
         common_errors=[
             "Claiming E1 is concerted (it is not — concerted elimination is E2)",
             "Confusing E1 with SN1 conditions (same substrate/solvent preferences, "
             "but E1 gives alkene while SN1 gives substitution product)",
             "Saying E1 requires a strong base (strong bases favor E2, not E1)",
         ],
         references=[
             "Clayden et al., Organic Chemistry, 2nd ed., Ch. 17",
             "McMurry, Organic Chemistry, 9th ed., Ch. 11",
         ])

    _add("oc02_e2", "E2 Elimination",
         cluster="mechanisms",
         description=(
             "Bimolecular elimination requiring anti-periplanar geometry "
             "(180-degree dihedral between the departing H and leaving group). "
             "Concerted, single-step mechanism."
         ),
         key_facts=[
             "Concerted single-step mechanism: base abstracts proton while leaving group departs simultaneously",
             "Requires anti-periplanar geometry (180-degree H-C-C-LG dihedral) for orbital overlap",
             "Favored by strong bases, primary/secondary substrates, and higher temperatures",
             "Rate = k[substrate][base] (bimolecular)",
             "Bulky bases (t-BuOK) favor Hofmann product (less substituted alkene)",
         ],
         common_errors=[
             "Claiming any geometry works for E2 (anti-periplanar is required for the "
             "orbital overlap that drives concerted bond breaking)",
             "Confusing anti-periplanar (180-degree) with syn-periplanar (0-degree) — "
             "syn elimination is rare and much slower",
             "Saying E2 proceeds through a carbocation intermediate (that is E1)",
         ],
         references=[
             "Clayden et al., Organic Chemistry, 2nd ed., Ch. 17",
             "Vollhardt & Schore, Organic Chemistry, 8th ed., Ch. 7",
         ])

    _add("oc11_nucleophile", "Nucleophilicity",
         cluster="mechanisms",
         description=(
             "Nucleophilicity measures the rate at which a species attacks an "
             "electrophilic center. It increases with negative charge, "
             "polarizability, and decreasing electronegativity, but depends "
             "critically on solvent."
         ),
         key_facts=[
             "In polar aprotic solvents, nucleophilicity roughly parallels basicity "
             "(F- > Cl- > Br- > I-)",
             "In polar protic solvents, larger atoms are better nucleophiles because "
             "polarizability dominates over solvation (I- > Br- > Cl- > F-)",
             "Negative charge dramatically increases nucleophilicity (OH- >> H2O)",
             "Nucleophilicity and basicity are distinct: nucleophilicity is kinetic "
             "(rate of attack on carbon), basicity is thermodynamic (equilibrium proton affinity)",
         ],
         common_errors=[
             "Equating nucleophilicity with basicity in all solvents (the ordering "
             "reverses between protic and aprotic solvents for halides)",
             "Claiming electronegativity increases nucleophilicity (it decreases it — "
             "more electronegative atoms hold electrons more tightly)",
             "Ignoring solvent effects when comparing nucleophile strength",
         ],
         references=[
             "Clayden et al., Organic Chemistry, 2nd ed., Ch. 17",
             "Carey & Sundberg, Advanced Organic Chemistry, Part A, Ch. 4",
         ])

    # ── reagents cluster ───────────────────────────────────────────────

    _add("oc03_grignard", "Grignard Reagents",
         cluster="reagents",
         description=(
             "Organomagnesium halides (RMgX) that act as powerful nucleophiles "
             "and strong bases. React with water and protic solvents to give "
             "alkanes via protonation of the carbanion equivalent."
         ),
         key_facts=[
             "Must be prepared and used in anhydrous conditions (ether solvents: THF or Et2O)",
             "React with water/alcohols/acids to give R-H (protonation destroys the reagent)",
             "React with CO2 to give carboxylic acids (after acid workup)",
             "React with aldehydes to give secondary alcohols, ketones to give tertiary alcohols",
             "React with epoxides to give primary alcohols (nucleophilic ring opening at less substituted carbon)",
         ],
         common_errors=[
             "Claiming Grignard reagents are water-stable or can be used in protic solvents "
             "(they react immediately with any protic source)",
             "Confusing Grignard reactivity with organolithium reactivity "
             "(organolithiums are more reactive and less selective)",
             "Forgetting that substrates with acidic protons (OH, NH, COOH) "
             "must be protected before Grignard addition",
         ],
         references=[
             "Clayden et al., Organic Chemistry, 2nd ed., Ch. 9",
             "Grignard, V. C. R. Hebd. Seances Acad. Sci. 130, 1322 (1900)",
         ])

    _add("oc12_aromatic", "Aromaticity (Huckel's Rule)",
         cluster="reagents",
         description=(
             "Aromaticity requires a planar, cyclic, fully conjugated system "
             "with 4n+2 pi electrons (Huckel's rule). Anti-aromatic systems "
             "have 4n pi electrons and are destabilized."
         ),
         key_facts=[
             "Four criteria: planar + cyclic + continuous conjugation + 4n+2 pi electrons",
             "Benzene: 6 pi electrons (n=1), naphthalene: 10 (n=2), cyclopentadienyl anion: 6 (n=1)",
             "Anti-aromatic: 4n pi electrons in a planar cyclic conjugated system (e.g., cyclobutadiene, 4 pi electrons)",
             "Non-planar systems escape anti-aromaticity (cyclooctatetraene is tub-shaped, not anti-aromatic)",
             "Heteroatoms contribute lone pairs to pi system only if needed for conjugation (pyrrole N: 2e to ring; pyridine N: 0e to ring)",
         ],
         common_errors=[
             "Claiming any cyclic conjugated system is aromatic (must also satisfy 4n+2 count — "
             "cyclobutadiene is cyclic and conjugated but anti-aromatic with 4 pi electrons)",
             "Confusing 4n+2 (aromatic) with 4n (anti-aromatic) — getting the electron count wrong",
             "Forgetting planarity requirement (a system with 4n+2 electrons is not aromatic if non-planar)",
         ],
         references=[
             "Huckel, E. Z. Phys. 70, 204-286 (1931)",
             "Clayden et al., Organic Chemistry, 2nd ed., Ch. 8",
         ])

    # ── reactions cluster ──────────────────────────────────────────────

    _add("oc04_friedel", "Friedel-Crafts Alkylation",
         cluster="reactions",
         description=(
             "Electrophilic aromatic substitution using a Lewis acid catalyst "
             "(typically AlCl3) to generate a carbocation electrophile. "
             "Subject to carbocation rearrangement and polyalkylation."
         ),
         key_facts=[
             "Lewis acid catalyst (AlCl3, FeCl3, BF3) generates carbocation from alkyl halide",
             "Carbocations rearrange to more stable forms (1-degree to 2-degree to 3-degree) via hydride/methyl shifts",
             "Does NOT work on deactivated rings (nitrobenzene, benzoic acid) — "
             "the ring is too electron-poor for electrophilic attack",
             "Polyalkylation problem: product is more activated than starting material, "
             "so multiple alkylations occur",
             "Friedel-Crafts acylation avoids rearrangement (acylium cation is resonance-stabilized)",
         ],
         common_errors=[
             "Claiming no carbocation rearrangement occurs in Friedel-Crafts alkylation "
             "(rearrangement is a key limitation — e.g., n-propyl chloride + AlCl3 gives isopropylbenzene, not n-propylbenzene)",
             "Ignoring the deactivated ring limitation (Friedel-Crafts fails on "
             "rings bearing strong electron-withdrawing groups)",
             "Confusing alkylation (rearrangement possible) with acylation (no rearrangement)",
         ],
         references=[
             "Friedel, C.; Crafts, J. M. C. R. Hebd. Seances Acad. Sci. 84, 1392 (1877)",
             "Clayden et al., Organic Chemistry, 2nd ed., Ch. 22",
         ])

    _add("oc05_aldol", "Aldol Condensation",
         cluster="reactions",
         description=(
             "Two-stage reaction: aldol addition (enolate attacks aldehyde to "
             "give beta-hydroxy carbonyl) followed by dehydration to yield an "
             "alpha,beta-unsaturated carbonyl (enone or enal)."
         ),
         key_facts=[
             "Aldol addition: enolate + aldehyde/ketone -> beta-hydroxy carbonyl (aldol product)",
             "Aldol condensation: aldol product undergoes dehydration (-H2O) -> alpha,beta-unsaturated carbonyl",
             "Crossed aldol requires one non-enolizable component (e.g., benzaldehyde, formaldehyde) "
             "to avoid self-condensation mixtures",
             "LDA at -78C gives kinetic enolate; thermodynamic enolate from NaOH/heat",
             "Intramolecular aldol cyclization forms 5- and 6-membered rings preferentially",
         ],
         common_errors=[
             "Calling the initial beta-hydroxy carbonyl the 'condensation' product "
             "(condensation specifically requires the dehydration step — the initial "
             "product is the 'aldol addition' product)",
             "Forgetting that crossed aldols give mixtures unless one partner is non-enolizable",
             "Confusing Claisen condensation (ester enolate + ester) with aldol condensation "
             "(enolate + aldehyde/ketone)",
         ],
         references=[
             "Clayden et al., Organic Chemistry, 2nd ed., Ch. 27",
             "Wurtz, A. Bull. Soc. Chim. Fr. 17, 436-442 (1872)",
         ])

    _add("oc06_diels_alder", "Diels-Alder Reaction",
         cluster="reactions",
         description=(
             "Concerted [4+2] pericyclic cycloaddition between a conjugated "
             "diene (4 pi electrons) and a dienophile (2 pi electrons). "
             "Suprafacial on both components, stereospecific, and governed "
             "by the endo rule."
         ),
         key_facts=[
             "Concerted, single-step mechanism (no intermediates — pericyclic, not stepwise or radical)",
             "Diene must adopt s-cis conformation for orbital overlap with dienophile",
             "Suprafacial on both diene and dienophile — preserves cis/trans relationships",
             "Endo rule: kinetically favored product has substituents endo (under the newly forming ring) "
             "due to secondary orbital interactions",
             "Stereospecific: cis-dienophile substituents remain cis in the product",
         ],
         common_errors=[
             "Claiming the Diels-Alder reaction is stepwise or radical-mediated "
             "(it is concerted and pericyclic — single transition state, no intermediates)",
             "Confusing [4+2] cycloaddition (thermal, Diels-Alder) with [2+2] cycloaddition "
             "(photochemical, requires UV light by Woodward-Hoffmann rules)",
             "Forgetting the s-cis diene requirement (s-trans dienes cannot react)",
         ],
         references=[
             "Diels, O.; Alder, K. Justus Liebigs Ann. Chem. 460, 98-122 (1928)",
             "Clayden et al., Organic Chemistry, 2nd ed., Ch. 35",
         ])

    # ── stereochem cluster ─────────────────────────────────────────────

    _add("oc07_chirality", "Chirality",
         cluster="stereochem",
         description=(
             "A molecule is chiral if it is non-superimposable on its mirror "
             "image. Chirality does NOT require a stereocenter — axial chirality "
             "(allenes, biphenyls) and planar chirality also exist."
         ),
         key_facts=[
             "Chiral = non-superimposable mirror image; achiral = superimposable on its mirror image",
             "Does NOT require a stereocenter: allenes with different substituents, "
             "restricted-rotation biphenyls (atropisomers), and helicenes are chiral without stereocenters",
             "Enantiomers have identical physical properties (mp, bp, solubility, IR, NMR) "
             "EXCEPT optical rotation sign and interaction with other chiral molecules",
             "Meso compounds have stereocenters but are achiral due to an internal mirror plane",
         ],
         common_errors=[
             "Requiring a stereocenter (sp3 carbon with 4 different substituents) for chirality "
             "(axial and planar chirality exist without any stereocenter)",
             "Confusing enantiomers with diastereomers (enantiomers are mirror images; "
             "diastereomers are stereoisomers that are NOT mirror images)",
             "Saying enantiomers have different physical properties (only optical rotation "
             "direction and chiral interactions differ)",
         ],
         references=[
             "Clayden et al., Organic Chemistry, 2nd ed., Ch. 14",
             "Eliel & Wilen, Stereochemistry of Organic Compounds, Ch. 4",
         ])

    _add("oc08_r_s", "R/S Assignment (CIP Priority Rules)",
         cluster="stereochem",
         description=(
             "Cahn-Ingold-Prelog priority rules assign R or S configuration "
             "to stereocenters based on atomic number of substituents. "
             "R/S designation does NOT correlate with optical rotation direction."
         ),
         key_facts=[
             "Priority assigned by atomic number at the point of first difference "
             "(higher atomic number = higher priority)",
             "If tied at the first atom, move outward until a difference is found",
             "R (rectus) = clockwise arrangement of priorities 1->2->3 with #4 pointing away from viewer",
             "S (sinister) = counterclockwise arrangement with #4 pointing away",
             "R/S does NOT predict (+)/(-) optical rotation — that requires measurement or computation",
         ],
         common_errors=[
             "Equating R with (+) dextrorotatory or S with (-) levorotatory "
             "(there is no general correlation — e.g., (R)-glyceraldehyde is (-), "
             "(S)-alanine is (+))",
             "Using alphabetical order of substituent names instead of atomic number "
             "(CIP rules use atomic number, not name)",
             "Forgetting to orient the lowest-priority group away from the viewer "
             "before assigning R/S",
         ],
         references=[
             "Cahn, R.S.; Ingold, C.; Prelog, V. Angew. Chem. Int. Ed. 5, 385-415 (1966)",
             "IUPAC Recommendations 2013 on stereochemical nomenclature",
         ])

    # ── synthesis cluster ──────────────────────────────────────────────

    _add("oc09_protecting", "Protecting Groups",
         cluster="synthesis",
         description=(
             "Temporary chemical modifications that mask reactive functional "
             "groups during multi-step synthesis, allowing selective "
             "transformations on other parts of the molecule."
         ),
         key_facts=[
             "Must be selectively installable and removable under conditions that don't affect other functionality",
             "Alcohol protection: TBS/TBDMS (removed by F-, e.g., TBAF), acetyl (removed by base), benzyl (removed by hydrogenolysis)",
             "Amine protection: Boc (removed by acid, e.g., TFA), Fmoc (removed by base, e.g., piperidine), Cbz (removed by hydrogenolysis)",
             "Carbonyl protection: cyclic acetals from ethylene glycol + acid catalyst (removed by aqueous acid)",
             "Orthogonal protection: using groups removed by different conditions allows sequential deprotection",
         ],
         common_errors=[
             "Claiming protecting groups permanently modify the product "
             "(they are temporary by definition and must be cleanly removable)",
             "Confusing protecting groups with leaving groups "
             "(leaving groups depart during bond formation; protecting groups are "
             "installed then removed without forming new C-C bonds)",
             "Forgetting that protection/deprotection adds steps and reduces overall yield",
         ],
         references=[
             "Greene & Wuts, Protective Groups in Organic Synthesis, 5th ed.",
             "Clayden et al., Organic Chemistry, 2nd ed., Ch. 30",
         ])

    _add("oc10_retro", "Retrosynthetic Analysis",
         cluster="synthesis",
         description=(
             "Planning strategy introduced by E.J. Corey that works backward "
             "from the target molecule to simpler precursors via disconnections. "
             "Uses synthons (idealized fragments) and their synthetic equivalents "
             "(real reagents)."
         ),
         key_facts=[
             "Work backward from target molecule: identify key bonds to disconnect (retrosynthetic arrows =>)",
             "Synthons are idealized charged fragments from disconnection (e.g., acyl cation +C=O)",
             "Synthetic equivalents are real reagents corresponding to synthons (e.g., acyl chloride for acyl cation)",
             "Functional group interconversion (FGI) transforms groups to enable disconnections",
             "Strategic disconnections simplify molecular complexity (reduce rings, stereocenters, or heteroatom content)",
         ],
         common_errors=[
             "Confusing retrosynthetic analysis (backward from target) with forward synthesis planning "
             "(retrosynthesis is specifically the backward-reasoning step; forward synthesis is the execution plan)",
             "Claiming retrosynthetic analysis is purely computational "
             "(it is a human reasoning framework that can be computer-assisted, but the "
             "strategic choices rely on chemical intuition about feasibility and selectivity)",
             "Ignoring stereochemical considerations during disconnection planning",
         ],
         references=[
             "Corey, E.J.; Cheng, X.-M. The Logic of Chemical Synthesis, Wiley (1989)",
             "Clayden et al., Organic Chemistry, 2nd ed., Ch. 30",
         ])

    return db


def _build_aliases() -> Dict[str, str]:
    """Build alias map from common names/abbreviations to topic IDs."""
    return {
        # E1
        "e1": "oc01_e1",
        "e1 elimination": "oc01_e1",
        "e1 reaction": "oc01_e1",
        "unimolecular elimination": "oc01_e1",
        # E2
        "e2": "oc02_e2",
        "e2 elimination": "oc02_e2",
        "e2 reaction": "oc02_e2",
        "bimolecular elimination": "oc02_e2",
        "anti-periplanar": "oc02_e2",
        "antiperiplanar": "oc02_e2",
        # Nucleophilicity
        "nucleophilicity": "oc11_nucleophile",
        "nucleophile": "oc11_nucleophile",
        "nucleophilic": "oc11_nucleophile",
        # Grignard
        "grignard": "oc03_grignard",
        "grignard reagent": "oc03_grignard",
        "grignard reagents": "oc03_grignard",
        "rmgx": "oc03_grignard",
        "organomagnesium": "oc03_grignard",
        # Aromaticity
        "aromaticity": "oc12_aromatic",
        "aromatic": "oc12_aromatic",
        "huckel": "oc12_aromatic",
        "huckel's rule": "oc12_aromatic",
        "4n+2": "oc12_aromatic",
        "anti-aromatic": "oc12_aromatic",
        "antiaromatic": "oc12_aromatic",
        # Friedel-Crafts
        "friedel-crafts": "oc04_friedel",
        "friedel crafts": "oc04_friedel",
        "friedel-crafts alkylation": "oc04_friedel",
        "fc alkylation": "oc04_friedel",
        # Aldol
        "aldol": "oc05_aldol",
        "aldol condensation": "oc05_aldol",
        "aldol reaction": "oc05_aldol",
        "aldol addition": "oc05_aldol",
        # Diels-Alder
        "diels-alder": "oc06_diels_alder",
        "diels alder": "oc06_diels_alder",
        "diels-alder reaction": "oc06_diels_alder",
        "[4+2]": "oc06_diels_alder",
        "[4+2] cycloaddition": "oc06_diels_alder",
        "pericyclic": "oc06_diels_alder",
        # Chirality
        "chirality": "oc07_chirality",
        "chiral": "oc07_chirality",
        "enantiomer": "oc07_chirality",
        "enantiomers": "oc07_chirality",
        # R/S
        "r/s": "oc08_r_s",
        "r/s assignment": "oc08_r_s",
        "cip": "oc08_r_s",
        "cip rules": "oc08_r_s",
        "cahn-ingold-prelog": "oc08_r_s",
        "cahn ingold prelog": "oc08_r_s",
        "stereocenter assignment": "oc08_r_s",
        # Protecting groups
        "protecting group": "oc09_protecting",
        "protecting groups": "oc09_protecting",
        "protection": "oc09_protecting",
        "tbs": "oc09_protecting",
        "tbdms": "oc09_protecting",
        "boc": "oc09_protecting",
        "fmoc": "oc09_protecting",
        # Retrosynthesis
        "retrosynthesis": "oc10_retro",
        "retrosynthetic": "oc10_retro",
        "retrosynthetic analysis": "oc10_retro",
        "disconnection": "oc10_retro",
        "synthon": "oc10_retro",
        "corey": "oc10_retro",
    }


# ─── Module-level database (built once at import) ────────────────────────────

_DB: Dict[str, OrganicChemistryTopic] = _build_database()
_ALIASES: Dict[str, str] = _build_aliases()

# Cluster membership for fast filtering
_CLUSTERS: Dict[str, List[str]] = {}
for _tid, _topic in _DB.items():
    _CLUSTERS.setdefault(_topic.cluster, []).append(_tid)


# ─── Name resolution ─────────────────────────────────────────────────────────

def _resolve_topic(name: str) -> Optional[OrganicChemistryTopic]:
    """Resolve a topic name with fuzzy matching."""
    key = name.lower().strip()

    # Direct ID match
    if key in _DB:
        return _DB[key]

    # Alias match
    if key in _ALIASES:
        return _DB.get(_ALIASES[key])

    # Substring match on topic names
    matches = [t for t in _DB.values() if key in t.name.lower()]
    if len(matches) == 1:
        return matches[0]

    # Substring match on topic IDs
    matches = [t for tid, t in _DB.items() if key in tid]
    if len(matches) == 1:
        return matches[0]

    # Reverse substring — query contains a topic name
    matches = [t for t in _DB.values() if t.name.lower() in key]
    if len(matches) == 1:
        return matches[0]

    # Word overlap
    query_words = set(key.split())
    best_match: Optional[OrganicChemistryTopic] = None
    best_overlap = 0
    for topic in _DB.values():
        name_words = set(topic.name.lower().split())
        overlap = len(query_words & name_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = topic
    if best_overlap >= 1 and best_match is not None:
        return best_match

    return None


# ─── Claim-checking patterns ─────────────────────────────────────────────────

# Each entry: (compiled regex, target topic IDs, severity, description)
_CLAIM_PATTERNS: List[tuple] = []


def _init_claim_patterns() -> None:
    """Build claim-checking regex patterns. Called once at module load."""
    patterns = [
        # E1 errors
        (r"\be1\b.{0,30}\bconcerted\b",
         ["oc01_e1"], "HIGH",
         "E1 is NOT concerted. E1 proceeds through a carbocation intermediate "
         "in two steps. Concerted elimination is E2."),
        (r"\be1\b.{0,30}\bstrong\s+base",
         ["oc01_e1"], "MODERATE",
         "E1 does not require a strong base. Strong bases favor E2, not E1."),
        (r"\be1\b.{0,30}\bsingle\s+step\b",
         ["oc01_e1"], "HIGH",
         "E1 is NOT a single-step reaction. It proceeds in two steps through "
         "a carbocation intermediate."),

        # E2 errors
        (r"\be2\b.{0,40}\bany\s+geometry\b",
         ["oc02_e2"], "HIGH",
         "E2 requires anti-periplanar geometry (180-degree dihedral). "
         "It does not work with any geometry."),
        (r"\be2\b.{0,30}\bcarbocation\b",
         ["oc02_e2"], "HIGH",
         "E2 does NOT proceed through a carbocation intermediate. "
         "It is concerted. Carbocation intermediates are in E1."),
        (r"\be2\b.{0,30}\bsyn.periplanar\b",
         ["oc02_e2"], "MODERATE",
         "E2 requires anti-periplanar geometry, not syn-periplanar. "
         "Syn elimination is rare and much slower."),
        (r"\be2\b.{0,30}\btwo\s+step",
         ["oc02_e2"], "HIGH",
         "E2 is concerted (single step), not two steps. Two-step elimination is E1."),

        # Grignard errors
        (r"grignard.{0,30}\bwater.stable\b",
         ["oc03_grignard"], "HIGH",
         "Grignard reagents are NOT water-stable. They react immediately "
         "with any protic source."),
        (r"grignard.{0,30}\bprotic\s+solvent\b",
         ["oc03_grignard"], "HIGH",
         "Grignard reagents cannot be used in protic solvents. "
         "They require anhydrous ether solvents (THF or Et2O)."),
        (r"grignard.{0,30}\baqueous\b",
         ["oc03_grignard"], "HIGH",
         "Grignard reagents are destroyed by water. They cannot be used "
         "in aqueous conditions."),

        # Aromaticity errors
        (r"aromatic.{0,30}\b4n\b(?!\s*\+\s*2)",
         ["oc12_aromatic"], "HIGH",
         "Aromatic systems require 4n+2 pi electrons, not 4n. "
         "4n pi electrons give anti-aromatic systems."),
        (r"(?:cyclic|ring).{0,20}\bconjugat.{0,10}\b.{0,10}\baromatic\b",
         ["oc12_aromatic"], "MODERATE",
         "Not all cyclic conjugated systems are aromatic. Must also satisfy "
         "4n+2 electron count and planarity requirements."),

        # Friedel-Crafts errors
        (r"friedel.crafts.{0,30}\bno\s+rearrangement\b",
         ["oc04_friedel"], "HIGH",
         "Friedel-Crafts alkylation IS subject to carbocation rearrangement. "
         "This is a key limitation (e.g., n-propyl -> isopropyl)."),
        (r"friedel.crafts\s+alkylat.{0,30}\bno\b.{0,15}\brearrang",
         ["oc04_friedel"], "HIGH",
         "Friedel-Crafts alkylation IS subject to carbocation rearrangement."),

        # Aldol errors
        (r"aldol\s+condensation.{0,30}beta.hydroxy",
         ["oc05_aldol"], "MODERATE",
         "The beta-hydroxy carbonyl is the aldol ADDITION product. "
         "Condensation specifically refers to the dehydrated product "
         "(alpha,beta-unsaturated carbonyl)."),

        # Diels-Alder errors
        (r"diels.alder.{0,30}\b(?:stepwise|radical|two.step)\b",
         ["oc06_diels_alder"], "HIGH",
         "The Diels-Alder reaction is NOT stepwise or radical-mediated. "
         "It is a concerted pericyclic [4+2] cycloaddition."),
        (r"diels.alder.{0,30}\b\[2\+2\]\b",
         ["oc06_diels_alder"], "HIGH",
         "The Diels-Alder is a [4+2] cycloaddition, not [2+2]. "
         "[2+2] cycloadditions are photochemical."),
        (r"diels.alder.{0,30}\bs.trans\s+diene\b",
         ["oc06_diels_alder"], "MODERATE",
         "The Diels-Alder reaction requires the diene in s-cis conformation. "
         "s-trans dienes cannot react."),

        # Chirality errors
        (r"chiral.{0,20}\brequir.{0,10}\bstereocent",
         ["oc07_chirality"], "HIGH",
         "Chirality does NOT require a stereocenter. Allenes, restricted-rotation "
         "biphenyls, and helicenes are chiral without stereocenters."),
        (r"enantiomers\s+have\s+different\s+(?:melting|boiling|physical)\b",
         ["oc07_chirality"], "HIGH",
         "Enantiomers have IDENTICAL physical properties (mp, bp, solubility). "
         "They differ ONLY in optical rotation sign and interactions with "
         "other chiral molecules."),

        # R/S errors
        (r"\bR\b.{0,15}\b(?:dextro|positive|\(\+\)|clockwise\s+rotation)\b",
         ["oc08_r_s"], "HIGH",
         "R configuration does NOT correlate with (+) optical rotation. "
         "R/S is a naming convention; (+)/(-) is a measured property."),
        (r"\bS\b.{0,15}\b(?:levo|negative|\(\-\)|counterclockwise\s+rotation)\b",
         ["oc08_r_s"], "HIGH",
         "S configuration does NOT correlate with (-) optical rotation."),
        (r"(?:cip|priority).{0,20}\balphabetical\b",
         ["oc08_r_s"], "HIGH",
         "CIP priority rules use ATOMIC NUMBER, not alphabetical order "
         "of substituent names."),

        # Protecting group errors
        (r"protecting\s+group.{0,30}\bpermanent\b",
         ["oc09_protecting"], "HIGH",
         "Protecting groups are TEMPORARY by definition. "
         "They must be cleanly removable."),
        (r"protecting\s+group.{0,20}\bleaving\s+group\b",
         ["oc09_protecting"], "MODERATE",
         "Protecting groups and leaving groups are different concepts. "
         "Leaving groups depart during bond formation; protecting groups "
         "are installed then removed."),

        # Retrosynthesis errors
        (r"retrosynthes.{0,20}\bforward\b",
         ["oc10_retro"], "MODERATE",
         "Retrosynthetic analysis works BACKWARD from the target molecule. "
         "Forward synthesis is the execution plan, not the analysis."),
        (r"retrosynthes.{0,20}\bpurely\s+computational\b",
         ["oc10_retro"], "MODERATE",
         "Retrosynthetic analysis is a human reasoning framework that can "
         "be computer-assisted, but strategic choices rely on chemical "
         "intuition about feasibility and selectivity."),

        # Nucleophilicity errors
        (r"nucleophilicity.{0,20}\b(?:equals?|same\s+as|identical\s+to)\b.{0,10}\bbasicity\b",
         ["oc11_nucleophile"], "HIGH",
         "Nucleophilicity and basicity are NOT the same. Nucleophilicity is "
         "kinetic (rate of carbon attack), basicity is thermodynamic "
         "(equilibrium proton affinity). Their ordering reverses between "
         "protic and aprotic solvents for halides."),
        (r"electronegativity.{0,20}\bincreas.{0,10}\bnucleophilicity\b",
         ["oc11_nucleophile"], "HIGH",
         "Electronegativity DECREASES nucleophilicity (more electronegative "
         "atoms hold electrons more tightly, making them worse nucleophiles)."),
    ]

    for pat_str, topic_ids, severity, desc in patterns:
        _CLAIM_PATTERNS.append((re.compile(pat_str, re.IGNORECASE), topic_ids, severity, desc))


_init_claim_patterns()


# ─── Public API ───────────────────────────────────────────────────────────────

def check_organic_chemistry(
    topic: str,
    claim: Optional[str] = None,
) -> OrganicChemistryReport:
    """Look up an organic chemistry topic and optionally validate a claim.

    Args:
        topic: topic name or ID (fuzzy matched — "E1", "Grignard",
               "Diels-Alder", "oc06_diels_alder", etc.)
        claim: optional natural-language claim to validate against the
               topic's known facts and common errors.

    Returns:
        OrganicChemistryReport with verdict (PASS/WARN/FAIL), issues, and notes.
    """
    info = _resolve_topic(topic)
    if info is None:
        return OrganicChemistryReport(
            verdict="FAIL",
            topic=None,
            claim=claim,
            issues=[OrganicChemistryIssue(
                check_type="CLAIM_CHECK",
                severity="HIGH",
                description=f"Topic '{topic}' not found in database",
            )],
            notes=[f"Known topics: use list_organic_chemistry_topics() to see all {len(_DB)} entries"],
        )

    issues: List[OrganicChemistryIssue] = []
    notes: List[str] = []

    # Add key facts as notes
    for fact in info.key_facts[:3]:
        notes.append(fact)

    if claim is not None:
        claim_lower = claim.lower().strip()

        # Check claim against known error patterns
        for pattern, target_ids, severity, desc in _CLAIM_PATTERNS:
            if info.id in target_ids and pattern.search(claim_lower):
                issues.append(OrganicChemistryIssue(
                    check_type="CLAIM_CHECK",
                    severity=severity,
                    description=desc,
                    details={"claim": claim, "topic": info.id},
                ))

        # Check if claim matches any common error verbatim fragments
        for error in info.common_errors:
            error_lower = error.lower()
            # Extract key phrases from the error description
            # Check if the claim asserts something that the error warns against
            error_keywords = set(error_lower.split()) - {
                "the", "a", "an", "is", "are", "it", "that", "this", "of", "in",
                "to", "for", "and", "or", "not", "with", "from", "by", "on", "at",
            }
            claim_words = set(claim_lower.split()) - {
                "the", "a", "an", "is", "are", "it", "that", "this", "of", "in",
                "to", "for", "and", "or", "not", "with", "from", "by", "on", "at",
            }
            overlap = len(error_keywords & claim_words)
            if overlap >= 4 and not any(i.check_type == "CLAIM_CHECK" for i in issues):
                issues.append(OrganicChemistryIssue(
                    check_type="ERROR_FLAG",
                    severity="MODERATE",
                    description=f"Claim may contain a known misconception: {error}",
                    details={"claim": claim, "error": error},
                ))

    else:
        # No claim — flag common errors as informational notes
        if info.common_errors:
            notes.append("Common misconceptions to watch for:")
            for err in info.common_errors:
                notes.append(f"  {err}")

    # ── Verdict logic ─────────────────────────────────────────────────
    has_high = any(i.severity == "HIGH" for i in issues)
    has_moderate = any(i.severity == "MODERATE" for i in issues)

    if has_high:
        verdict = "FAIL"
    elif has_moderate:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return OrganicChemistryReport(
        verdict=verdict,
        topic=info,
        claim=claim,
        issues=issues,
        notes=notes,
    )


def list_organic_chemistry_topics(
    cluster: Optional[str] = None,
) -> List[str]:
    """List all topic names, optionally filtered by cluster.

    Args:
        cluster: optional filter — one of "mechanisms", "reagents",
                 "reactions", "stereochem", "synthesis". Case-insensitive.

    Returns:
        Sorted list of topic names.
    """
    filter_cluster = cluster.lower().strip() if cluster else None
    results: List[str] = []
    for topic in _DB.values():
        if filter_cluster is None or topic.cluster == filter_cluster:
            results.append(topic.name)
    return sorted(results)


def get_organic_chemistry_topic(
    topic: str,
) -> Optional[OrganicChemistryTopic]:
    """Get full information about an organic chemistry topic.

    Args:
        topic: topic name or ID (fuzzy matched).

    Returns:
        OrganicChemistryTopic if found, None otherwise.
    """
    return _resolve_topic(topic)
