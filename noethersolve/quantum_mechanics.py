"""noethersolve.quantum_mechanics — Quantum mechanics fact checker.

Validates claims about quantum mechanics principles, phenomena, and systems
against a curated database of established physics.  Catches: confusing
uncertainty with measurement error, attributing FTL communication to
entanglement, wrong ground-state energies, and other common LLM mistakes.

Covers 12 topics across 3 clusters: foundations, phenomena, and systems.

Usage:
    from noethersolve.quantum_mechanics import (
        check_quantum_mechanics, list_quantum_mechanics_topics,
        get_quantum_mechanics_topic,
        QuantumMechanicsReport, QuantumMechanicsIssue, QMTopicInfo,
    )

    # Validate a claim against a topic
    report = check_quantum_mechanics("uncertainty", claim="it is caused by measurement apparatus")
    print(report)
    # FAIL -- 1 issue: CLAIM_CHECK [HIGH] ...

    # List all topics in a cluster
    for tid in list_quantum_mechanics_topics(cluster="foundations"):
        print(tid)

    # Get full topic info
    info = get_quantum_mechanics_topic("qm04_tunneling")
    print(info.name, info.cluster)
    for f in info.key_facts:
        print(f"  - {f}")

    # Check without a claim -- returns PASS with INFO summary
    report = check_quantum_mechanics("born rule")
    print(report)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Dataclasses ──────────────────────────────────────────────────────────


@dataclass
class QuantumMechanicsIssue:
    """A single issue found when checking a quantum mechanics claim."""

    severity: str
    """HIGH = factually wrong, MODERATE = misleading, LOW = imprecise,
    INFO = correct or informational."""

    tag: str
    """Issue tag (e.g. CLAIM_CHECK, TOPIC_LOOKUP)."""

    description: str
    """Why this claim is problematic (or confirmed)."""

    id: str = ""
    """Matched topic ID in the database."""

    references: List[str] = field(default_factory=list)
    """Supporting references."""

    def __str__(self) -> str:
        ref_str = ""
        if self.references:
            ref_str = f" [{', '.join(self.references)}]"
        return f"  {self.tag} [{self.severity}] {self.description}{ref_str}"


@dataclass
class QuantumMechanicsReport:
    """Result of checking a quantum mechanics topic or claim."""

    verdict: str
    """PASS = no errors found, WARN = imprecise claims, FAIL = factual errors."""

    id: str
    """The topic that was checked."""

    issues: List[QuantumMechanicsIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def n_issues(self) -> int:
        return len(self.issues)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    def __str__(self) -> str:
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append(f"  Quantum Mechanics Check: {self.verdict}")
        lines.append(f"  Topic: {self.id}")
        lines.append("=" * 60)
        n_high = sum(1 for i in self.issues if i.severity == "HIGH")
        n_mod = sum(1 for i in self.issues if i.severity == "MODERATE")
        n_low = sum(1 for i in self.issues if i.severity == "LOW")
        n_info = sum(1 for i in self.issues if i.severity == "INFO")
        lines.append(
            f"  Issues: {self.n_issues} total — "
            f"{n_high} HIGH, {n_mod} MODERATE, {n_low} LOW, {n_info} INFO"
        )
        lines.append("")
        if self.issues:
            sev_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "INFO": 3}
            for issue in sorted(
                self.issues, key=lambda i: sev_order.get(i.severity, 4)
            ):
                lines.append(str(issue))
            lines.append("")
        else:
            lines.append("  No issues detected.")
            lines.append("")
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
            lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class QMTopicInfo:
    """Full information about a quantum mechanics topic."""

    id: str
    name: str
    cluster: str
    description: str
    key_facts: List[str] = field(default_factory=list)
    common_errors: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"{self.name} ({self.id})"]
        lines.append(f"  Cluster: {self.cluster}")
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


# ── Database ─────────────────────────────────────────────────────────────


def _build_database() -> Dict[str, QMTopicInfo]:
    """Build the quantum mechanics topic database. Called once at module load."""
    db: Dict[str, QMTopicInfo] = {}

    def _add(
        id: str,
        name: str,
        cluster: str,
        description: str,
        key_facts: List[str],
        common_errors: List[str],
        references: List[str],
        keywords: List[str],
    ) -> None:
        db[id] = QMTopicInfo(
            id=id,
            name=name,
            cluster=cluster,
            description=description,
            key_facts=key_facts,
            common_errors=common_errors,
            references=references,
            keywords=keywords,
        )

    # ── Foundations cluster ────────────────────────────────────────────

    _add(
        "qm01_uncertainty",
        "Heisenberg Uncertainty Principle",
        "foundations",
        (
            "Fundamental limit on simultaneous knowledge of conjugate variables. "
            "Not a statement about measurement apparatus limitations."
        ),
        key_facts=[
            "Position-momentum: dx*dp >= hbar/2 (Robertson relation)",
            "Energy-time: dE*dt >= hbar/2 (not from commutator — dt is not an observable)",
            "Follows from wave mechanics: narrow position wavepacket requires broad momentum spectrum",
            "Applies to ALL quantum systems, not just photons or electrons",
        ],
        common_errors=[
            "Attributing uncertainty to measurement apparatus limitations (observer effect != uncertainty principle)",
            "Saying it only applies to photons or to the act of observation",
            "Confusing the Heisenberg uncertainty principle with the observer effect (photon recoil in measurement)",
        ],
        references=[
            "Heisenberg 1927 — Z. Phys. 43, 172-198",
            "Robertson 1929 — Phys. Rev. 34, 163",
            "Ozawa 2003 — Phys. Rev. A 67, 042105 (measurement-disturbance vs intrinsic uncertainty)",
        ],
        keywords=[
            "uncertainty", "heisenberg", "delta x", "delta p",
            "conjugate", "hbar", "measurement", "observer",
        ],
    )

    _add(
        "qm02_collapse",
        "Wave Function Collapse",
        "foundations",
        (
            "Upon measurement, the quantum state projects onto an eigenstate of "
            "the measured observable. Copenhagen interpretation postulate."
        ),
        key_facts=[
            "Measurement projects state onto eigenstate of the measured observable",
            "Schrodinger evolution is unitary and deterministic — collapse is NOT part of it",
            "Collapse is instantaneous and non-unitary (Copenhagen)",
            "Decoherence explains apparent collapse via entanglement with environment, but does not solve the measurement problem",
            "Alternative interpretations (Many-Worlds, Bohmian) avoid collapse entirely",
        ],
        common_errors=[
            "Saying collapse happens during Schrodinger evolution (evolution is unitary/deterministic)",
            "Confusing collapse with decoherence (decoherence is a physical process, collapse is a postulate)",
        ],
        references=[
            "von Neumann 1932 — Mathematical Foundations of Quantum Mechanics",
            "Zurek 2003 — Rev. Mod. Phys. 75, 715 (decoherence program)",
        ],
        keywords=[
            "collapse", "measurement", "wave function", "projection",
            "eigenstate", "copenhagen", "decoherence", "observation",
        ],
    )

    _add(
        "qm03_pauli",
        "Pauli Exclusion Principle",
        "foundations",
        (
            "No two identical fermions (half-integer spin) can occupy the same "
            "quantum state simultaneously. Consequence of the spin-statistics theorem."
        ),
        key_facts=[
            "Applies to ALL fermions: electrons, protons, neutrons, quarks — not just electrons",
            "Responsible for electron shell structure and the periodic table",
            "Causes degeneracy pressure in white dwarfs (electron) and neutron stars (neutron)",
            "Underlies the stability of matter — without it, all atoms would collapse",
            "Follows from the spin-statistics theorem (Pauli 1940): fermion wavefunctions are antisymmetric",
        ],
        common_errors=[
            "Saying it applies to bosons (bosons obey Bose-Einstein statistics, can share states)",
            "Limiting it to electrons only (applies to all fermions: protons, neutrons, quarks, etc.)",
            "Confusing exclusion with repulsion (it is a quantum statistical constraint, not a force)",
        ],
        references=[
            "Pauli 1925 — Z. Phys. 31, 765",
            "Pauli 1940 — Phys. Rev. 58, 716 (spin-statistics theorem)",
            "Dyson & Lenard 1967 — J. Math. Phys. 8, 423 (stability of matter)",
        ],
        keywords=[
            "pauli", "exclusion", "fermion", "half-integer", "spin-statistics",
            "degeneracy pressure", "electron shell", "antisymmetric",
        ],
    )

    _add(
        "qm06_born",
        "Born Rule",
        "foundations",
        (
            "The probability of a measurement outcome is the squared modulus of "
            "the amplitude: P = |psi|^2. Most controversial axiom — not derived "
            "from other postulates."
        ),
        key_facts=[
            "Probability = |amplitude|^2, i.e. |<phi|psi>|^2 for outcome |phi>",
            "Not derivable from the other postulates of quantum mechanics (axiomatic status)",
            "Gleason's theorem (1957) shows it is the ONLY consistent probability rule for Hilbert spaces of dimension >= 3",
            "Attempts to derive it: Everettian decision theory (Deutsch-Wallace), envariance (Zurek)",
        ],
        common_errors=[
            "Saying probability is psi directly (must be squared modulus, not the amplitude itself)",
            "Confusing Born rule probability with expectation values (<A> = <psi|A|psi>)",
        ],
        references=[
            "Born 1926 — Z. Phys. 37, 863",
            "Gleason 1957 — J. Math. Mech. 6, 885",
        ],
        keywords=[
            "born", "probability", "amplitude squared", "modulus",
            "psi squared", "measurement probability", "gleason",
        ],
    )

    _add(
        "qm09_superposition",
        "Quantum Superposition",
        "foundations",
        (
            "A quantum system exists in a linear combination of states "
            "|psi> = sum(c_i |phi_i>) until measured. The coefficients c_i "
            "are complex amplitudes."
        ),
        key_facts=[
            "State is a LINEAR combination: |psi> = sum(c_i |phi_i>) with complex c_i",
            "Probabilities are |c_i|^2, and sum to 1 (normalization)",
            "System does NOT 'rapidly switch' between states (ruled out by Bell's theorem)",
            "Coefficients can differ — not all states need equal weight",
            "Superposition is basis-dependent: every state is a superposition in SOME basis",
        ],
        common_errors=[
            "Saying system 'rapidly switches' between states (hidden variable view, ruled out by Bell)",
            "Saying all states in a superposition have equal weight (coefficients can differ arbitrarily)",
            "Treating superposition as classical ignorance (it is not — interference effects prove otherwise)",
        ],
        references=[
            "Dirac 1930 — The Principles of Quantum Mechanics",
            "Bell 1964 — Physics 1, 195 (hidden variables ruled out)",
        ],
        keywords=[
            "superposition", "linear combination", "amplitude",
            "coefficient", "basis", "interference", "cat state",
        ],
    )

    # ── Phenomena cluster ─────────────────────────────────────────────

    _add(
        "qm04_tunneling",
        "Quantum Tunneling",
        "phenomena",
        (
            "Nonzero probability of a particle penetrating a classically "
            "forbidden potential barrier. Transmission probability "
            "T ~ exp(-2*kappa*L) where kappa = sqrt(2m(V-E))/hbar."
        ),
        key_facts=[
            "Transmission probability T ~ exp(-2*kappa*L), kappa = sqrt(2m(V-E))/hbar",
            "Does NOT violate energy conservation — total energy is conserved throughout",
            "Applications: alpha decay, scanning tunneling microscope (STM), tunnel diodes, Josephson junctions",
            "Particle does not go 'over' the barrier — it penetrates through the classically forbidden region",
            "Tunneling time is still debated (Hartman effect, attoclock experiments)",
        ],
        common_errors=[
            "Saying tunneling violates energy conservation (it does not — energy is conserved exactly)",
            "Saying particles go 'over' the barrier (they penetrate through the forbidden region)",
            "Confusing tunneling probability with certainty (it is exponentially suppressed by barrier width)",
        ],
        references=[
            "Gamow 1928 — Z. Phys. 51, 204 (alpha decay)",
            "Binnig & Rohrer 1982 — Nobel Prize for STM",
            "Ramos et al. 2020 — Nature 583, 529 (attoclock tunneling time)",
        ],
        keywords=[
            "tunnel", "barrier", "penetrat", "transmission", "alpha decay",
            "stm", "classically forbidden", "evanescent",
        ],
    )

    _add(
        "qm05_spin",
        "Spin-1/2 and Spinor Rotation",
        "phenomena",
        (
            "Spin-1/2 particles require 720 degrees (4*pi) rotation to return "
            "to their original state. SU(2) is the double cover of SO(3)."
        ),
        key_facts=[
            "Spin-1/2 requires 720 degrees (4*pi) rotation for identity — 360 degrees gives a sign flip",
            "SU(2) double cover of SO(3): two SU(2) elements map to each SO(3) rotation",
            "Observable consequences: neutron interferometry (Rauch et al. 1975) confirmed 4*pi periodicity",
            "Integer-spin particles (bosons) return to original state after 360 degrees (normal rotation)",
        ],
        common_errors=[
            "Saying 360 degrees suffices for spin-1/2 (that is true only for integer-spin particles)",
            "Claiming spinor properties have no observable consequences (neutron interferometry confirms them)",
        ],
        references=[
            "Rauch et al. 1975 — Phys. Lett. A 54, 425 (neutron interferometry)",
            "Sakurai 1994 — Modern Quantum Mechanics, Ch. 3 (rotation of spin states)",
        ],
        keywords=[
            "spin", "spinor", "720", "4 pi", "su(2)", "so(3)",
            "double cover", "half-integer", "rotation",
        ],
    )

    _add(
        "qm07_degeneracy",
        "Quantum Degeneracy",
        "phenomena",
        (
            "Multiple quantum states sharing the same energy eigenvalue. "
            "Lifted by perturbations that break the underlying symmetry."
        ),
        key_facts=[
            "Degeneracy = multiple linearly independent states with the same energy",
            "Zeeman effect: magnetic field breaks rotational symmetry, lifts m_l degeneracy",
            "Stark effect: electric field breaks parity, lifts l-degeneracy",
            "Spin-orbit coupling breaks l-degeneracy in hydrogen (fine structure)",
            "Accidental degeneracy in hydrogen (l-independence) due to hidden SO(4) symmetry",
        ],
        common_errors=[
            "Saying degeneracy is always removed by temperature changes (temperature populates states, does not split them)",
            "Confusing degeneracy with decoherence or decay (completely unrelated concepts)",
        ],
        references=[
            "Griffiths & Schroeter 2018 — Introduction to Quantum Mechanics, Ch. 6",
            "Fock 1935 — Z. Phys. 98, 145 (SO(4) symmetry of hydrogen)",
        ],
        keywords=[
            "degeneracy", "degenerate", "zeeman", "stark",
            "symmetry breaking", "fine structure", "spin-orbit",
        ],
    )

    _add(
        "qm08_entanglement",
        "Quantum Entanglement",
        "phenomena",
        (
            "Bell inequality violations prove quantum correlations cannot be "
            "explained by local hidden variables. Correlations are nonlocal but "
            "no information is transferred (no-communication theorem)."
        ),
        key_facts=[
            "Bell's theorem (1964): no local hidden variable theory can reproduce all QM predictions",
            "Experimental violations of Bell/CHSH inequalities (Aspect 1982, loophole-free: Hensen et al. 2015)",
            "No-communication theorem: entanglement CANNOT transmit information faster than light",
            "EPR paradox resolved: 'spooky action at a distance' is real but useless for signaling",
            "Entanglement is a resource for quantum teleportation, QKD, and quantum computing",
        ],
        common_errors=[
            "Saying entanglement allows faster-than-light (FTL) communication (no-communication theorem forbids it)",
            "Saying classical shared randomness (hidden variables) can explain Bell violations (Bell's theorem rules this out)",
            "Confusing entanglement with cloning (no-cloning theorem is separate)",
        ],
        references=[
            "Bell 1964 — Physics 1, 195",
            "Aspect et al. 1982 — Phys. Rev. Lett. 49, 1804",
            "Hensen et al. 2015 — Nature 526, 682 (loophole-free Bell test)",
        ],
        keywords=[
            "entangle", "bell", "epr", "nonlocal", "correlation",
            "hidden variable", "chsh", "spooky", "ftl",
        ],
    )

    # ── Systems cluster ───────────────────────────────────────────────

    _add(
        "qm10_harmonic",
        "Quantum Harmonic Oscillator",
        "systems",
        (
            "Energy levels E_n = (n + 1/2)*hbar*omega. Ground state has "
            "nonzero zero-point energy hbar*omega/2. Wave functions are "
            "Hermite polynomials times a Gaussian."
        ),
        key_facts=[
            "E_n = (n + 1/2)*hbar*omega, n = 0, 1, 2, ...",
            "Ground state energy = hbar*omega/2 (zero-point energy, nonzero)",
            "Wave functions: psi_n(x) = H_n(alpha*x) * exp(-alpha^2 x^2 / 2), H_n = Hermite polynomial",
            "Equally spaced energy levels (spacing = hbar*omega)",
            "Creation/annihilation operators: a^dagger|n> = sqrt(n+1)|n+1>, a|n> = sqrt(n)|n-1>",
        ],
        common_errors=[
            "Saying the ground state energy is zero (it is hbar*omega/2, a direct consequence of the uncertainty principle)",
            "Confusing with classical oscillator (classical oscillator can have zero energy at rest)",
        ],
        references=[
            "Griffiths & Schroeter 2018 — Introduction to Quantum Mechanics, Ch. 2.3",
            "Dirac 1930 — ladder operator method",
        ],
        keywords=[
            "harmonic oscillator", "zero-point", "hbar omega",
            "ladder operator", "creation", "annihilation", "hermite",
            "ground state energy", "equally spaced",
        ],
    )

    _add(
        "qm11_hydrogen",
        "Hydrogen Atom Energy Levels",
        "systems",
        (
            "Non-relativistic energy levels E_n = -13.6 eV / n^2, depending "
            "on principal quantum number n ONLY. Accidental l-degeneracy due "
            "to SO(4) symmetry. Fine structure breaks l-degeneracy."
        ),
        key_facts=[
            "E_n = -13.6 eV / n^2 (Bohr model and exact non-relativistic QM agree)",
            "Energy depends on n ONLY (non-relativistically) — accidental degeneracy in l",
            "Accidental degeneracy explained by hidden SO(4) symmetry (Fock 1935, Bargmann 1936)",
            "Fine structure (relativistic + spin-orbit) splits l-levels: correction ~ alpha^2 * E_n",
            "Lamb shift (QED) further splits 2S_1/2 and 2P_1/2 (Lamb & Retherford 1947)",
        ],
        common_errors=[
            "Saying energy depends on l at the non-relativistic level (it does not — only n matters)",
            "Confusing principal quantum number effects with angular momentum quantum number effects",
        ],
        references=[
            "Griffiths & Schroeter 2018 — Introduction to Quantum Mechanics, Ch. 4",
            "Fock 1935 — Z. Phys. 98, 145",
            "Lamb & Retherford 1947 — Phys. Rev. 72, 241",
        ],
        keywords=[
            "hydrogen", "13.6", "bohr", "principal quantum number",
            "fine structure", "lamb shift", "degeneracy", "so(4)",
        ],
    )

    _add(
        "qm12_commutator",
        "Non-Commuting Observables",
        "systems",
        (
            "If [A, B] != 0, then DELTA_A * DELTA_B >= |<[A,B]>|/2 "
            "(generalized uncertainty relation). Cannot generally have "
            "simultaneous eigenstates."
        ),
        key_facts=[
            "Robertson uncertainty relation: DELTA_A * DELTA_B >= |<[A,B]>|/2",
            "Canonical examples: [x, p] = i*hbar, [L_x, L_y] = i*hbar*L_z",
            "Non-commuting observables cannot generally share a complete set of eigenstates",
            "Commuting observables ([A,B] = 0) CAN be simultaneously diagonalized (compatible observables)",
            "The commutator encodes the algebraic structure (Lie algebra) of the symmetry group",
        ],
        common_errors=[
            "Saying non-commuting observables can be measured simultaneously with arbitrary precision",
            "Confusing non-commutativity with mathematical undefinedness (the operators are well-defined, just non-commuting)",
        ],
        references=[
            "Robertson 1929 — Phys. Rev. 34, 163",
            "Sakurai 1994 — Modern Quantum Mechanics, Ch. 1.4",
        ],
        keywords=[
            "commutator", "commut", "non-commuting", "uncertainty relation",
            "simultaneous", "compatible", "eigenstate", "[x,p]", "lie algebra",
        ],
    )

    return db


# Build at module load time.
_DATABASE: Dict[str, QMTopicInfo] = _build_database()

# All valid cluster names.
_CLUSTERS = {"foundations", "phenomena", "systems"}


# ── Internal helpers ─────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return " ".join(text.lower().split())


def _match_topic(query: str) -> Optional[str]:
    """Find the best-matching topic ID for a query string.

    Tries, in order:
    1. Exact id match (e.g. "qm04_tunneling").
    2. Substring match on id or name.
    3. Keyword match (best overlap).

    Returns the id or None.
    """
    q = _normalize(query)

    # 1. Exact ID match.
    if q.replace(" ", "_").replace("-", "_") in _DATABASE:
        return q.replace(" ", "_").replace("-", "_")

    # 2. Substring match on id or name.
    for tid, info in _DATABASE.items():
        if q in tid or q in _normalize(info.name):
            return tid

    # 3. Keyword overlap scoring.
    best_tid: Optional[str] = None
    best_score = 0
    q_tokens = set(q.split())
    for tid, info in _DATABASE.items():
        score = 0
        for kw in info.keywords:
            kw_lower = kw.lower()
            if kw_lower in q:
                score += 2
            elif any(t in kw_lower for t in q_tokens):
                score += 1
        if score > best_score:
            best_score = score
            best_tid = tid

    return best_tid if best_score > 0 else None


def _check_claim_against_topic(
    claim: str, topic: QMTopicInfo,
) -> List[QuantumMechanicsIssue]:
    """Check a natural-language claim against a topic's common errors and key facts."""
    issues: List[QuantumMechanicsIssue] = []
    claim_lower = _normalize(claim)

    # Check against common errors.
    _error_triggers = _get_error_triggers()
    for tid, triggers in _error_triggers.items():
        if tid != topic.id:
            continue
        for trigger_words, error_idx, severity in triggers:
            if all(tw in claim_lower for tw in trigger_words):
                issues.append(QuantumMechanicsIssue(
                    severity=severity,
                    tag="CLAIM_CHECK",
                    description=topic.common_errors[error_idx],
                    id=topic.id,
                    references=topic.references,
                ))

    return issues


def _get_error_triggers() -> Dict[str, List[tuple]]:
    """Return keyword trigger patterns for each topic's common errors.

    Each entry: (trigger_words, common_error_index, severity).
    trigger_words is a list of strings that must ALL appear in the claim.
    """
    return {
        "qm01_uncertainty": [
            (["measurement", "apparatus"], 0, "HIGH"),
            (["observer", "effect"], 0, "HIGH"),
            (["measurement", "error"], 0, "HIGH"),
            (["only", "photon"], 1, "HIGH"),
            (["only", "observation"], 1, "MODERATE"),
        ],
        "qm02_collapse": [
            (["schrodinger", "evolution", "collapse"], 0, "HIGH"),
            (["unitary", "collapse"], 0, "HIGH"),
            (["collapse", "decoherence"], 1, "MODERATE"),
            (["decoherence", "same"], 1, "MODERATE"),
        ],
        "qm03_pauli": [
            (["boson"], 0, "HIGH"),
            (["only", "electron"], 1, "HIGH"),
            (["electron", "only"], 1, "HIGH"),
        ],
        "qm06_born": [
            (["probability", "psi", "not", "squared"], 0, "HIGH"),
            (["probability", "amplitude", "directly"], 0, "HIGH"),
            (["expectation"], 1, "MODERATE"),
        ],
        "qm09_superposition": [
            (["rapidly", "switch"], 0, "HIGH"),
            (["switch", "between"], 0, "MODERATE"),
            (["equal", "weight"], 1, "MODERATE"),
            (["equal", "probability"], 1, "MODERATE"),
        ],
        "qm04_tunneling": [
            (["violat", "energy", "conservation"], 0, "HIGH"),
            (["break", "energy", "conservation"], 0, "HIGH"),
            (["over", "barrier"], 1, "MODERATE"),
            (["above", "barrier"], 1, "MODERATE"),
        ],
        "qm05_spin": [
            (["360", "suffic"], 0, "HIGH"),
            (["360", "return"], 0, "HIGH"),
            (["360", "original"], 0, "HIGH"),
        ],
        "qm07_degeneracy": [
            (["temperature", "remov"], 0, "HIGH"),
            (["temperature", "lift"], 0, "HIGH"),
            (["temperature", "split"], 0, "HIGH"),
        ],
        "qm08_entanglement": [
            (["faster", "light"], 0, "HIGH"),
            (["ftl", "communicat"], 0, "HIGH"),
            (["superluminal", "communicat"], 0, "HIGH"),
            (["classical", "explain"], 1, "HIGH"),
            (["hidden", "variable", "explain"], 1, "HIGH"),
        ],
        "qm10_harmonic": [
            (["ground", "state", "zero"], 0, "HIGH"),
            (["ground", "energy", "zero"], 0, "HIGH"),
            (["lowest", "energy", "zero"], 0, "HIGH"),
        ],
        "qm11_hydrogen": [
            (["energy", "depend", "l"], 0, "HIGH"),
            (["energy", "angular", "momentum"], 0, "MODERATE"),
        ],
        "qm12_commutator": [
            (["simultaneously", "measur"], 0, "HIGH"),
            (["simultaneous", "precision"], 0, "HIGH"),
            (["undefin"], 1, "MODERATE"),
        ],
    }


# ── Public API ───────────────────────────────────────────────────────────


def check_quantum_mechanics(
    topic: str,
    claim: Optional[str] = None,
) -> QuantumMechanicsReport:
    """Check a quantum mechanics topic and optionally validate a claim.

    Parameters
    ----------
    topic : str
        Topic ID (e.g. ``"qm01_uncertainty"``) or keyword
        (e.g. ``"tunneling"``, ``"born rule"``).
    claim : str, optional
        A natural-language claim to validate against the topic.
        If *None*, returns a PASS report with INFO-level summary.

    Returns
    -------
    QuantumMechanicsReport
        Verdict is PASS, WARN, or FAIL depending on issues found.

    Examples
    --------
    >>> report = check_quantum_mechanics("uncertainty")
    >>> report.verdict
    'PASS'

    >>> report = check_quantum_mechanics(
    ...     "uncertainty",
    ...     claim="it is caused by measurement apparatus limitations",
    ... )
    >>> report.verdict
    'FAIL'
    """
    matched_id = _match_topic(topic)

    if matched_id is None:
        return QuantumMechanicsReport(
            verdict="WARN",
            id=topic,
            issues=[],
            warnings=[f"Topic '{topic}' not found in the database"],
        )

    info = _DATABASE[matched_id]
    issues: List[QuantumMechanicsIssue] = []

    if claim is not None:
        issues = _check_claim_against_topic(claim, info)
    else:
        # No claim — provide an INFO summary.
        issues.append(QuantumMechanicsIssue(
            severity="INFO",
            tag="TOPIC_LOOKUP",
            description=info.description,
            id=matched_id,
            references=info.references,
        ))

    # Determine verdict.
    if any(i.severity == "HIGH" for i in issues):
        verdict = "FAIL"
    elif any(i.severity == "MODERATE" for i in issues):
        verdict = "WARN"
    else:
        verdict = "PASS"

    return QuantumMechanicsReport(
        verdict=verdict,
        id=matched_id,
        issues=issues,
    )


def list_quantum_mechanics_topics(
    cluster: Optional[str] = None,
) -> List[str]:
    """List all quantum mechanics topic IDs, optionally filtered by cluster.

    Parameters
    ----------
    cluster : str, optional
        Filter by cluster: ``"foundations"``, ``"phenomena"``, ``"systems"``.

    Returns
    -------
    list of str
        Sorted topic IDs.

    Examples
    --------
    >>> list_quantum_mechanics_topics()
    ['qm01_uncertainty', 'qm02_collapse', ..., 'qm12_commutator']

    >>> list_quantum_mechanics_topics(cluster="systems")
    ['qm10_harmonic', 'qm11_hydrogen', 'qm12_commutator']
    """
    if cluster is None:
        return sorted(_DATABASE.keys())

    cluster_lower = cluster.lower().strip()
    if cluster_lower not in _CLUSTERS:
        raise ValueError(
            f"Unknown cluster '{cluster}'. "
            f"Valid clusters: {', '.join(sorted(_CLUSTERS))}"
        )

    return sorted(
        tid for tid, info in _DATABASE.items()
        if info.cluster == cluster_lower
    )


def get_quantum_mechanics_topic(topic: str) -> Optional[QMTopicInfo]:
    """Look up a quantum mechanics topic by ID or keyword.

    Parameters
    ----------
    topic : str
        Topic ID (e.g. ``"qm04_tunneling"``) or keyword (e.g. ``"tunneling"``).

    Returns
    -------
    QMTopicInfo or None
        Full topic information, or *None* if not found.

    Examples
    --------
    >>> info = get_quantum_mechanics_topic("tunneling")
    >>> info.name
    'Quantum Tunneling'
    >>> info.cluster
    'phenomena'
    """
    matched_id = _match_topic(topic)
    if matched_id is None:
        return None
    return _DATABASE[matched_id]
