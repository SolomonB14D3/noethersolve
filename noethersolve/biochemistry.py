"""noethersolve.biochemistry — Biochemistry fact checker and lookup table.

Validates claims about biochemistry topics against a curated database of
established facts.  Catches common LLM errors: confusing enzyme inhibition
types, misquoting metabolic yields, conflating signaling mechanisms.

Covers 12 topics across 5 clusters: enzymes, metabolism, molbio, proteins,
and signaling.

Usage:
    from noethersolve.biochemistry import (
        check_biochemistry, list_biochemistry_topics, get_biochemistry_topic,
        BiochemistryReport, BiochemistryIssue, BiochemistryInfo,
    )

    # Check a topic — returns report with issues and verdict
    report = check_biochemistry("michaelis")
    print(report)
    # PASS — 0 issues ...

    # Check a topic with a specific claim to validate
    report = check_biochemistry("competitive inhibition", claim="Vmax decreases")
    print(report)
    # FAIL — 1 issue: [HIGH] Matches known error ...

    # List all topics
    for name in list_biochemistry_topics():
        print(name)

    # List topics in a cluster
    for name in list_biochemistry_topics(cluster="metabolism"):
        print(name)

    # Get full info on a topic
    info = get_biochemistry_topic("krebs")
    print(info.cluster, info.key_facts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Dataclasses ──────────────────────────────────────────────────────────

@dataclass
class BiochemistryInfo:
    """Full information about a biochemistry topic."""

    id: str
    """Unique identifier (e.g., 'bc01_michaelis')."""

    name: str
    """Human-readable name."""

    cluster: str
    """Topic cluster (enzymes, metabolism, molbio, proteins, signaling)."""

    description: str
    """One-paragraph description of the topic."""

    key_facts: List[str] = field(default_factory=list)
    """3-5 important established facts."""

    common_errors: List[str] = field(default_factory=list)
    """2-3 mistakes LLMs commonly make about this topic."""

    references: List[str] = field(default_factory=list)
    """Textbook or review references."""

    def __str__(self) -> str:
        lines = [f"{self.name} [{self.id}] ({self.cluster})"]
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
class BiochemistryIssue:
    """A single issue found when checking a biochemistry claim."""

    severity: str
    """HIGH = matches known error, MODERATE = misleading, LOW = imprecise,
    INFO = correct / informational."""

    description: str
    """Why this claim is problematic (or confirmed)."""

    id: str = ""
    """Matched topic ID in the database, if any."""

    references: List[str] = field(default_factory=list)
    """Supporting references."""

    def __str__(self) -> str:
        ref_str = ""
        if self.references:
            ref_str = f" [{', '.join(self.references)}]"
        return f"  [{self.severity}] {self.description}{ref_str}"


@dataclass
class BiochemistryReport:
    """Result of check_biochemistry()."""

    verdict: str
    """PASS, WARN, or FAIL."""

    topic: Optional[BiochemistryInfo]
    """Matched topic, if found."""

    claim: Optional[str]
    """The original claim text, if provided."""

    issues: List[BiochemistryIssue] = field(default_factory=list)
    """Issues found during checking."""

    notes: List[str] = field(default_factory=list)
    """Additional notes."""

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    @property
    def n_issues(self) -> int:
        return len(self.issues)

    def __str__(self) -> str:
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append(f"  Biochemistry Check: {self.verdict}")
        lines.append("=" * 60)
        if self.topic:
            lines.append(f"  Topic: {self.topic.name} [{self.topic.id}]")
            lines.append(f"  Cluster: {self.topic.cluster}")
        else:
            lines.append("  Topic not found in database")
        if self.claim:
            lines.append(f"  Claim: {self.claim}")
        lines.append("")

        if self.issues:
            lines.append(f"  Issues ({len(self.issues)}):")
            sev_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "INFO": 3}
            for issue in sorted(
                self.issues,
                key=lambda i: sev_order.get(i.severity, 4),
            ):
                lines.append(str(issue))
            lines.append("")
        else:
            lines.append("  No issues detected.")
            lines.append("")

        if self.notes:
            lines.append("  Notes:")
            for note in self.notes:
                lines.append(f"    - {note}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ── Database ─────────────────────────────────────────────────────────────

def _build_database() -> Dict[str, BiochemistryInfo]:
    """Build the biochemistry database.  Called once at module load."""
    db: Dict[str, BiochemistryInfo] = {}

    def _add(id: str, **kwargs: object) -> None:
        kwargs.setdefault("key_facts", [])
        kwargs.setdefault("common_errors", [])
        kwargs.setdefault("references", [])
        info = BiochemistryInfo(id=id, **kwargs)  # type: ignore[arg-type]
        db[id] = info

    # ── enzymes cluster ──────────────────────────────────────────────────

    _add(
        "bc01_michaelis",
        name="Michaelis-Menten Kinetics",
        cluster="enzymes",
        description=(
            "Michaelis-Menten kinetics describes the relationship between "
            "substrate concentration [S] and reaction velocity v for "
            "enzyme-catalyzed reactions.  The Michaelis constant Km equals "
            "the substrate concentration at which the reaction rate is half "
            "of Vmax."
        ),
        key_facts=[
            "v = Vmax * [S] / (Km + [S])",
            "Km = substrate concentration at half-Vmax (units of concentration, not affinity)",
            "Vmax = maximum velocity when enzyme is fully saturated; depends on [E]total and kcat",
            "Low Km means high apparent affinity (reaches half-Vmax at low [S])",
            "Assumes steady-state: d[ES]/dt ~ 0 (Briggs-Haldane derivation)",
        ],
        common_errors=[
            "Confusing Km with the dissociation constant Kd — Km = (k-1 + kcat)/k1, "
            "which equals Kd only when kcat << k-1",
            "Saying Vmax depends on substrate concentration — Vmax depends on total "
            "enzyme concentration and kcat, NOT on [S]",
            "Treating Km as a direct measure of binding affinity — Km includes the "
            "catalytic rate constant kcat",
        ],
        references=[
            "Berg, Tymoczko, Stryer — Biochemistry, Ch. 8",
            "Michaelis & Menten (1913) Biochem Z 49:333-369",
        ],
    )

    _add(
        "bc02_allosteric",
        name="Allosteric Regulation",
        cluster="enzymes",
        description=(
            "Allosteric regulation occurs when an effector molecule binds "
            "at a site DISTINCT from the active site, causing a conformational "
            "change that alters enzyme activity.  Effectors can be positive "
            "(activators) or negative (inhibitors)."
        ),
        key_facts=[
            "Effector binds at allosteric site, NOT the active site",
            "Positive allosteric effectors increase enzyme activity (shift toward R-state)",
            "Negative allosteric effectors decrease enzyme activity (shift toward T-state)",
            "Sigmoidal kinetics typical of allosteric enzymes (Hill equation, not Michaelis-Menten)",
            "Monod-Wyman-Changeux (MWC) concerted model: all subunits switch states together",
        ],
        common_errors=[
            "Confusing allosteric inhibition with competitive inhibition — "
            "competitive inhibition IS at the active site, allosteric is NOT",
            "Saying allosteric regulators always inhibit — they can also activate",
        ],
        references=[
            "Berg, Tymoczko, Stryer — Biochemistry, Ch. 10",
            "Monod, Wyman, Changeux (1965) J Mol Biol 12:88-118",
        ],
    )

    _add(
        "bc03_competitive",
        name="Competitive Inhibition",
        cluster="enzymes",
        description=(
            "Competitive inhibition occurs when an inhibitor competes with "
            "substrate for binding at the enzyme's active site.  Apparent Km "
            "increases (lower apparent affinity) but Vmax is UNCHANGED because "
            "sufficient substrate can always outcompete the inhibitor."
        ),
        key_facts=[
            "Inhibitor binds at the active site, competing directly with substrate",
            "Apparent Km increases by factor (1 + [I]/Ki)",
            "Vmax is UNCHANGED — can always be overcome by excess substrate",
            "Lineweaver-Burk plot: lines intersect on y-axis (same 1/Vmax)",
        ],
        common_errors=[
            "Saying Vmax decreases in competitive inhibition — Vmax is unchanged; "
            "that describes noncompetitive inhibition",
            "Confusing with uncompetitive inhibition — uncompetitive decreases "
            "BOTH apparent Km AND Vmax (binds only to ES complex)",
            "Confusing with allosteric inhibition — competitive inhibition is at "
            "the active site, allosteric is at a different site",
        ],
        references=[
            "Berg, Tymoczko, Stryer — Biochemistry, Ch. 8",
            "Lineweaver & Burk (1934) J Am Chem Soc 56:658-666",
        ],
    )

    # ── metabolism cluster ────────────────────────────────────────────────

    _add(
        "bc04_krebs",
        name="Citric Acid Cycle (Krebs Cycle)",
        cluster="metabolism",
        description=(
            "The citric acid cycle (TCA / Krebs cycle) oxidizes acetyl-CoA "
            "to CO2, generating reduced electron carriers.  Per acetyl-CoA: "
            "3 NADH, 1 FADH2, 1 GTP (substrate-level phosphorylation).  "
            "Total ~10 ATP equivalents per turn."
        ),
        key_facts=[
            "Per acetyl-CoA: 3 NADH, 1 FADH2, 1 GTP",
            "NADH yields ~2.5 ATP via oxidative phosphorylation; FADH2 yields ~1.5 ATP",
            "Total ~10 ATP equivalents per acetyl-CoA (7.5 from NADH + 1.5 from FADH2 + 1 GTP)",
            "Occurs in the mitochondrial matrix",
            "Only ONE substrate-level phosphorylation step (succinyl-CoA -> succinate, producing GTP)",
        ],
        common_errors=[
            "Saying the TCA cycle produces ATP directly — it produces only 1 GTP "
            "per turn; the rest comes from NADH/FADH2 via oxidative phosphorylation",
            "Confusing TCA yield with glycolysis yield — glycolysis produces 2 ATP "
            "(net), 2 NADH, 2 pyruvate in the cytoplasm",
        ],
        references=[
            "Berg, Tymoczko, Stryer — Biochemistry, Ch. 17",
            "Krebs & Johnson (1937) Enzymologia 4:148-156",
        ],
    )

    _add(
        "bc05_oxphos",
        name="Oxidative Phosphorylation",
        cluster="metabolism",
        description=(
            "Oxidative phosphorylation couples electron transport along the "
            "inner mitochondrial membrane to ATP synthesis via the proton "
            "motive force (chemiosmotic coupling, Peter Mitchell 1961).  "
            "ATP synthase is a rotary molecular motor driven by proton "
            "flow down the gradient."
        ),
        key_facts=[
            "Electron transport chain: Complexes I-IV in inner mitochondrial membrane",
            "Proton gradient (pmf) across inner mitochondrial membrane drives ATP synthase",
            "ATP synthase (Complex V) is a rotary motor: ~3 H+ per ATP",
            "NADH enters at Complex I (~2.5 ATP); FADH2 enters at Complex II (~1.5 ATP)",
            "Chemiosmotic hypothesis: Mitchell (1961), Nobel Prize 1978",
        ],
        common_errors=[
            "Confusing oxidative phosphorylation with substrate-level phosphorylation "
            "— substrate-level produces ATP directly (e.g., glycolysis, TCA GTP step)",
            "Saying proton gradient is across the outer mitochondrial membrane "
            "— it is across the INNER mitochondrial membrane",
        ],
        references=[
            "Berg, Tymoczko, Stryer — Biochemistry, Ch. 18",
            "Mitchell (1961) Nature 191:144-148",
        ],
    )

    _add(
        "bc06_glycolysis",
        name="Glycolysis",
        cluster="metabolism",
        description=(
            "Glycolysis converts one glucose to two pyruvate in the cytoplasm.  "
            "Net yield: 2 ATP (4 produced, 2 invested in the preparatory phase), "
            "2 NADH, and 2 pyruvate."
        ),
        key_facts=[
            "Net yield per glucose: 2 ATP, 2 NADH, 2 pyruvate",
            "Gross ATP production is 4, but 2 ATP invested in preparatory phase",
            "Occurs entirely in the cytoplasm — does not require mitochondria",
            "Three irreversible regulatory steps: hexokinase, PFK-1, pyruvate kinase",
            "Anaerobic conditions: pyruvate -> lactate (regenerates NAD+)",
        ],
        common_errors=[
            "Saying glycolysis produces 4 ATP — that is gross, not net; "
            "net is 2 ATP after subtracting the 2 ATP investment",
            "Including oxidative phosphorylation yield when quoting glycolysis ATP "
            "— glycolysis itself produces only 2 ATP (net), everything else is downstream",
        ],
        references=[
            "Berg, Tymoczko, Stryer — Biochemistry, Ch. 16",
            "Embden, Meyerhof, Parnas pathway — established 1930s",
        ],
    )

    # ── molbio cluster ───────────────────────────────────────────────────

    _add(
        "bc07_dna_pol",
        name="DNA Polymerase",
        cluster="molbio",
        description=(
            "DNA polymerase synthesizes DNA by adding nucleotides to the "
            "3'-OH end of an existing primer.  It CANNOT start synthesis "
            "de novo — it requires a primer (typically RNA, laid down by "
            "primase).  Direction of synthesis is always 5' to 3'."
        ),
        key_facts=[
            "Requires a primer with a free 3'-OH group to begin synthesis",
            "Adds nucleotides in the 5' -> 3' direction ONLY",
            "Cannot initiate de novo — RNA polymerase and primase can start without a primer",
            "Proofreading: 3' -> 5' exonuclease activity in replicative polymerases",
            "Leading strand: continuous synthesis; lagging strand: Okazaki fragments",
        ],
        common_errors=[
            "Saying DNA polymerase can start synthesis de novo — it cannot; "
            "RNA polymerase can, but DNA polymerase always needs a primer",
            "Confusing DNA polymerase with reverse transcriptase — reverse "
            "transcriptase makes DNA from an RNA template (retroviral enzyme)",
        ],
        references=[
            "Alberts et al. — Molecular Biology of the Cell, Ch. 5",
            "Kornberg (1960) Science 131:1503-1508",
        ],
    )

    _add(
        "bc08_transcription",
        name="Eukaryotic mRNA Processing",
        cluster="molbio",
        description=(
            "Eukaryotic pre-mRNA undergoes three major processing steps before "
            "export from the nucleus: 5' capping (7-methylguanosine), 3' "
            "polyadenylation (~200 adenine residues), and intron splicing "
            "(by the spliceosome, a ribonucleoprotein complex)."
        ),
        key_facts=[
            "5' cap: 7-methylguanosine (m7G) added co-transcriptionally; protects from degradation",
            "3' poly-A tail: ~200 adenine residues added by poly-A polymerase; aids nuclear export and stability",
            "Intron splicing: spliceosome removes introns, joins exons; requires GU-AG splice sites",
            "All three modifications occur in the nucleus before mRNA export",
            "Alternative splicing: one gene can produce multiple mRNA variants (and proteins)",
        ],
        common_errors=[
            "Saying prokaryotes undergo the same mRNA processing — prokaryotic "
            "mRNA is NOT capped, polyadenylated, or spliced (no introns in most bacterial genes)",
            "Saying introns are kept in the mature mRNA — introns are REMOVED; "
            "exons are joined together",
        ],
        references=[
            "Alberts et al. — Molecular Biology of the Cell, Ch. 6",
            "Sharp (1994) Cell 77:805-815 (Nobel lecture on splicing)",
        ],
    )

    # ── proteins cluster ─────────────────────────────────────────────────

    _add(
        "bc09_hemoglobin",
        name="Hemoglobin Cooperative Binding",
        cluster="proteins",
        description=(
            "Hemoglobin exhibits cooperative O2 binding: the sigmoidal O2 "
            "saturation curve (Hill coefficient ~2.8) arises from the T-state "
            "(tense, low affinity) to R-state (relaxed, high affinity) "
            "transition.  Quaternary structure is an alpha2-beta2 tetramer."
        ),
        key_facts=[
            "Sigmoidal O2 binding curve (Hill coefficient ~2.8)",
            "Quaternary structure: alpha2-beta2 tetramer (four subunits)",
            "T-state (tense, deoxy): low O2 affinity; R-state (relaxed, oxy): high O2 affinity",
            "Bohr effect: lower pH and higher CO2 decrease O2 affinity (right-shift of curve)",
            "2,3-BPG (bisphosphoglycerate) stabilizes T-state, reducing O2 affinity",
        ],
        common_errors=[
            "Confusing hemoglobin's sigmoidal curve with myoglobin's hyperbolic curve "
            "— myoglobin is a monomer with NO cooperativity (Hill coefficient = 1)",
            "Saying hemoglobin has a Hill coefficient of 4 — it is ~2.8, "
            "not equal to the number of subunits because cooperativity is not perfect",
        ],
        references=[
            "Berg, Tymoczko, Stryer — Biochemistry, Ch. 7",
            "Perutz (1970) Nature 228:726-734",
        ],
    )

    _add(
        "bc12_chaperone",
        name="Molecular Chaperones",
        cluster="proteins",
        description=(
            "Molecular chaperones (e.g., GroEL/GroES in bacteria, Hsp70/Hsp90 "
            "in eukaryotes) assist protein folding by preventing aggregation of "
            "unfolded or partially folded intermediates.  They provide an "
            "isolated folding environment — they do NOT provide folding templates."
        ),
        key_facts=[
            "Prevent aggregation of unfolded/misfolded protein intermediates",
            "GroEL/GroES (bacterial): barrel-shaped cage provides isolated folding chamber",
            "Hsp70: binds exposed hydrophobic regions, prevents aggregation; ATP-dependent",
            "Chaperones do NOT dictate the final fold — the amino acid sequence determines structure (Anfinsen)",
            "ATP hydrolysis drives the chaperone cycle (binding and release of substrate)",
        ],
        common_errors=[
            "Saying chaperones provide a template or blueprint for the correct fold "
            "— they only prevent aggregation; the amino acid sequence determines the fold",
            "Saying chaperones catalyze peptide bond formation — that is the ribosome's "
            "function; chaperones assist FOLDING, not SYNTHESIS",
        ],
        references=[
            "Alberts et al. — Molecular Biology of the Cell, Ch. 6",
            "Anfinsen (1973) Science 181:223-230 (thermodynamic hypothesis of folding)",
        ],
    )

    # ── signaling cluster ────────────────────────────────────────────────

    _add(
        "bc10_gpcr",
        name="G Protein-Coupled Receptors (GPCRs)",
        cluster="signaling",
        description=(
            "GPCRs are 7-transmembrane domain receptors that activate "
            "heterotrimeric G proteins (Galpha/Gbeta-gamma) upon ligand "
            "binding.  They are the largest receptor superfamily, mediating "
            "responses to hormones, neurotransmitters, and sensory signals."
        ),
        key_facts=[
            "7-transmembrane alpha-helical domains (serpentine receptors)",
            "Activate heterotrimeric G proteins: Galpha (GTPase) + Gbeta-gamma dimer",
            "Galpha-GTP activates downstream effectors (adenylyl cyclase, PLC, ion channels)",
            "Largest receptor superfamily: >800 GPCRs in humans",
            "Desensitization: GRK phosphorylation -> arrestin binding -> internalization",
        ],
        common_errors=[
            "Confusing GPCRs with receptor tyrosine kinases (RTKs) — RTKs have "
            "intrinsic kinase activity and dimerize; GPCRs activate G proteins, not kinases",
            "Saying GPCRs are ion channels — GPCRs can modulate ion channels "
            "indirectly via G proteins, but GPCRs themselves are NOT ion channels",
        ],
        references=[
            "Alberts et al. — Molecular Biology of the Cell, Ch. 15",
            "Lefkowitz & Kobilka (2012) Nobel Prize in Chemistry for GPCR studies",
        ],
    )

    _add(
        "bc11_kinase",
        name="Protein Kinases",
        cluster="signaling",
        description=(
            "Protein kinases catalyze the transfer of a phosphate group from "
            "ATP to a specific amino acid residue (Ser, Thr, or Tyr) on a "
            "substrate protein.  This phosphorylation acts as a molecular "
            "switch.  Kinases are opposed by phosphatases, which REMOVE "
            "phosphate groups."
        ),
        key_facts=[
            "Transfer phosphate from ATP to substrate Ser/Thr/Tyr residues",
            "Opposed by phosphatases (which remove phosphate via hydrolysis)",
            "~538 protein kinases in the human kinome",
            "Key signaling cascades: MAPK/ERK, PI3K/Akt, JAK/STAT",
            "Dysregulation of kinases is common in cancer (e.g., BCR-ABL, BRAF, EGFR)",
        ],
        common_errors=[
            "Confusing kinases with phosphatases — kinases ADD phosphate, "
            "phosphatases REMOVE phosphate; they are functional opposites",
            "Saying kinases simply hydrolyze ATP — kinases transfer the "
            "gamma-phosphate to a substrate protein; ATPases hydrolyze ATP "
            "without transferring to a protein target",
        ],
        references=[
            "Berg, Tymoczko, Stryer — Biochemistry, Ch. 10",
            "Manning et al. (2002) Science 298:1912-1934 (human kinome)",
        ],
    )

    return db


# Module-level database (loaded once)
_DB: Dict[str, BiochemistryInfo] = _build_database()


# ── Aliases for fuzzy matching ───────────────────────────────────────────

_ALIASES: Dict[str, str] = {
    # enzymes
    "michaelis": "bc01_michaelis",
    "michaelis-menten": "bc01_michaelis",
    "michaelis menten": "bc01_michaelis",
    "km": "bc01_michaelis",
    "vmax": "bc01_michaelis",
    "allosteric": "bc02_allosteric",
    "allostery": "bc02_allosteric",
    "allosteric regulation": "bc02_allosteric",
    "competitive": "bc03_competitive",
    "competitive inhibition": "bc03_competitive",
    "enzyme inhibition": "bc03_competitive",
    # metabolism
    "krebs": "bc04_krebs",
    "krebs cycle": "bc04_krebs",
    "tca": "bc04_krebs",
    "tca cycle": "bc04_krebs",
    "citric acid cycle": "bc04_krebs",
    "oxphos": "bc05_oxphos",
    "oxidative phosphorylation": "bc05_oxphos",
    "electron transport": "bc05_oxphos",
    "etc": "bc05_oxphos",
    "chemiosmotic": "bc05_oxphos",
    "atp synthase": "bc05_oxphos",
    "glycolysis": "bc06_glycolysis",
    # molbio
    "dna polymerase": "bc07_dna_pol",
    "dna pol": "bc07_dna_pol",
    "dna replication": "bc07_dna_pol",
    "mrna processing": "bc08_transcription",
    "transcription": "bc08_transcription",
    "splicing": "bc08_transcription",
    "spliceosome": "bc08_transcription",
    "poly-a": "bc08_transcription",
    "polyadenylation": "bc08_transcription",
    "5 cap": "bc08_transcription",
    "5' cap": "bc08_transcription",
    # proteins
    "hemoglobin": "bc09_hemoglobin",
    "cooperative binding": "bc09_hemoglobin",
    "bohr effect": "bc09_hemoglobin",
    "chaperone": "bc12_chaperone",
    "chaperones": "bc12_chaperone",
    "groel": "bc12_chaperone",
    "hsp70": "bc12_chaperone",
    "protein folding": "bc12_chaperone",
    # signaling
    "gpcr": "bc10_gpcr",
    "gpcrs": "bc10_gpcr",
    "g protein": "bc10_gpcr",
    "g-protein coupled": "bc10_gpcr",
    "kinase": "bc11_kinase",
    "kinases": "bc11_kinase",
    "protein kinase": "bc11_kinase",
    "phosphorylation": "bc11_kinase",
    "phosphatase": "bc11_kinase",
}

# Reverse lookup: cluster -> list of topic IDs
_CLUSTERS: Dict[str, List[str]] = {}
for _tid, _info in _DB.items():
    _CLUSTERS.setdefault(_info.cluster, []).append(_tid)


# ── Resolution helpers ───────────────────────────────────────────────────

def _resolve_topic(name: str) -> Optional[BiochemistryInfo]:
    """Resolve a topic name with fuzzy matching."""
    key = name.lower().strip()

    # Direct topic ID match
    if key in _DB:
        return _DB[key]

    # Alias match
    if key in _ALIASES:
        return _DB.get(_ALIASES[key])

    # Substring match against topic IDs
    matches = [info for tid, info in _DB.items() if key in tid]
    if len(matches) == 1:
        return matches[0]

    # Substring match against names (case-insensitive)
    matches = [info for info in _DB.values() if key in info.name.lower()]
    if len(matches) == 1:
        return matches[0]

    # Word overlap match against names and descriptions
    query_words = set(key.split())
    best_match: Optional[BiochemistryInfo] = None
    best_overlap = 0
    for info in _DB.values():
        name_words = set(info.name.lower().split())
        overlap = len(query_words & name_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = info
    if best_overlap >= 2:
        return best_match

    return None


def _claim_matches_error(claim: str, errors: List[str]) -> Optional[str]:
    """Check if a claim text matches any known common error.

    Returns the matched error string, or None.
    """
    claim_lower = claim.lower()
    claim_words = set(claim_lower.split())

    for error in errors:
        error_lower = error.lower()
        # Direct substring containment
        # Extract the key misconception phrase (before the dash explanation)
        core = error_lower.split(" — ")[0] if " — " in error_lower else error_lower
        if core in claim_lower or claim_lower in core:
            return error
        # Word overlap
        error_words = set(error_lower.split())
        overlap = len(claim_words & error_words)
        if overlap >= 4 and overlap / max(len(claim_words), 1) >= 0.4:
            return error

    return None


def _claim_matches_facts(claim: str, facts: List[str]) -> bool:
    """Check if a claim text is consistent with known facts."""
    claim_lower = claim.lower()
    claim_words = set(claim_lower.split())

    for fact in facts:
        fact_lower = fact.lower()
        if fact_lower in claim_lower or claim_lower in fact_lower:
            return True
        fact_words = set(fact_lower.split())
        overlap = len(claim_words & fact_words)
        if overlap >= 4 and overlap / max(len(claim_words), 1) >= 0.4:
            return True

    return False


# ── Public API ───────────────────────────────────────────────────────────

def check_biochemistry(
    topic: str,
    claim: Optional[str] = None,
) -> BiochemistryReport:
    """Look up a biochemistry topic and optionally validate a claim.

    Parameters
    ----------
    topic : str
        Topic name or ID (fuzzy matched — "michaelis", "krebs", "bc04_krebs",
        "competitive inhibition", etc.).
    claim : str, optional
        A specific claim to validate against the topic's known facts and
        common errors.

    Returns
    -------
    BiochemistryReport
        Report with verdict (PASS/WARN/FAIL), matched topic, issues, and notes.
    """
    info = _resolve_topic(topic)

    if info is None:
        return BiochemistryReport(
            verdict="FAIL",
            topic=None,
            claim=claim,
            issues=[BiochemistryIssue(
                severity="HIGH",
                description=f"Topic '{topic}' not found in database",
            )],
            notes=[
                f"Known topics: use list_biochemistry_topics() to see all "
                f"{len(_DB)} entries",
            ],
        )

    issues: List[BiochemistryIssue] = []
    notes: List[str] = []

    # Always include key facts as notes
    for fact in info.key_facts:
        notes.append(fact)

    if claim is not None:
        # Check against common errors first
        matched_error = _claim_matches_error(claim, info.common_errors)
        if matched_error is not None:
            issues.append(BiochemistryIssue(
                severity="HIGH",
                description=f"Matches known error: {matched_error}",
                id=info.id,
                references=info.references,
            ))
        elif _claim_matches_facts(claim, info.key_facts):
            issues.append(BiochemistryIssue(
                severity="INFO",
                description=f"Claim is consistent with established facts",
                id=info.id,
                references=info.references,
            ))
        else:
            issues.append(BiochemistryIssue(
                severity="LOW",
                description=(
                    f"Claim could not be confidently matched to known facts or "
                    f"errors for {info.name}. Manual review recommended."
                ),
                id=info.id,
                references=info.references,
            ))

    # Determine verdict
    has_high = any(i.severity == "HIGH" for i in issues)
    has_moderate = any(i.severity == "MODERATE" for i in issues)
    has_low = any(i.severity == "LOW" for i in issues)

    if has_high:
        verdict = "FAIL"
    elif has_moderate or has_low:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return BiochemistryReport(
        verdict=verdict,
        topic=info,
        claim=claim,
        issues=issues,
        notes=notes,
    )


def list_biochemistry_topics(cluster: Optional[str] = None) -> List[str]:
    """List all biochemistry topic IDs, optionally filtered by cluster.

    Parameters
    ----------
    cluster : str, optional
        Filter by cluster: "enzymes", "metabolism", "molbio", "proteins",
        "signaling".

    Returns
    -------
    list of str
        Topic IDs (e.g., ["bc01_michaelis", "bc02_allosteric", ...]).
    """
    if cluster is None:
        return sorted(_DB.keys())

    cluster_lower = cluster.lower().strip()
    if cluster_lower not in _CLUSTERS:
        return []

    return sorted(_CLUSTERS[cluster_lower])


def get_biochemistry_topic(topic: str) -> Optional[BiochemistryInfo]:
    """Look up a biochemistry topic by ID or name.

    Parameters
    ----------
    topic : str
        Topic ID (e.g., "bc01_michaelis") or name (e.g., "michaelis",
        "competitive inhibition").

    Returns
    -------
    BiochemistryInfo or None
    """
    return _resolve_topic(topic)
