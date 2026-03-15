"""NoetherSolve MCP Server — expose 32 verified tools to any AI agent.

The full pipeline: find gaps → flip facts → build tool → add to MCP server.
Every tool added here makes every connected agent smarter.

Usage:
    python -m noethersolve.mcp_server

    # Claude Code auto-discovers .mcp.json at the project root.
    # Or: pip install noethersolve[mcp] && noethersolve-mcp
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "NoetherSolve",
    instructions="32 computational tools for physics, math, genetics, and LLM science — "
                 "verified reference databases, not guesses.",
)


# ── Conservation Law Monitors ─────────────────────────────────────────

@mcp.tool()
def check_vortex_conservation(
    positions: list[list[float]],
    circulations: list[float],
) -> str:
    """Check conservation laws for a 2D point vortex system.

    Given vortex positions [[x1,y1], [x2,y2], ...] and circulations,
    computes conserved quantities: Hamiltonian, linear impulse, angular impulse.
    """
    from noethersolve.monitor import VortexMonitor
    import numpy as np

    pos = np.array(positions)
    circ = np.array(circulations)
    monitor = VortexMonitor(circ)
    report = monitor.check(pos)
    return str(report)


@mcp.tool()
def check_hamiltonian_system(
    system: str = "harmonic_oscillator",
    t_span: float = 100.0,
    n_eval: int = 1000,
) -> str:
    """Validate Hamiltonian system conservation (energy, Liouville volume, Poincaré).

    Systems: harmonic_oscillator, kepler_2d, henon_heiles, coupled_oscillators.
    """
    from noethersolve.hamiltonian import (
        harmonic_oscillator as _ho, kepler_2d as _k2d,
        henon_heiles as _hh, coupled_oscillators as _co,
    )

    systems = {
        "harmonic_oscillator": _ho,
        "kepler_2d": _k2d,
        "henon_heiles": _hh,
        "coupled_oscillators": _co,
    }
    if system not in systems:
        return f"Unknown system '{system}'. Available: {list(systems.keys())}"

    import numpy as np
    monitor = systems[system]()
    # Default initial conditions per system
    z0_defaults = {
        "harmonic_oscillator": np.array([1.0, 0.0]),
        "kepler_2d": np.array([1.0, 0.0, 0.0, 1.0]),
        "henon_heiles": np.array([0.3, 0.1, 0.2, 0.05]),
        "coupled_oscillators": np.array([1.0, 0.0, 0.5, 0.0]),
    }
    z0 = z0_defaults[system]
    report = monitor.validate(z0, T=t_span, n_eval=n_eval)
    return str(report)


# ── Mathematical Conjectures ──────────────────────────────────────────

@mcp.tool()
def check_conjecture(name: str) -> str:
    """Check the status of a mathematical conjecture.

    Returns: status (OPEN/PROVEN/REFUTED), solver, year, key facts,
    implications, and common errors people make about it.

    Examples: "Riemann", "P vs NP", "Goldbach", "Collatz", "Twin primes",
    "Hodge", "BSD", "Navier-Stokes", "abc conjecture"
    """
    from noethersolve.conjecture_status import check_conjecture as _check
    report = _check(name)
    return str(report)


@mcp.tool()
def list_conjectures(status: str = "") -> str:
    """List known mathematical conjectures, optionally filtered by status.

    Status: "open", "proven", "refuted", or "" for all.
    """
    from noethersolve.conjecture_status import list_conjectures as _list
    conjectures = _list(status=status if status else None)
    return "\n".join(f"  {c}" for c in conjectures) if conjectures else "No conjectures found."


# ── Complexity Theory ─────────────────────────────────────────────────

@mcp.tool()
def check_complexity_inclusion(class_a: str, class_b: str) -> str:
    """Check if complexity class A ⊆ B, A = B, or relationship is unknown.

    Examples: check_complexity_inclusion("P", "NP"),
              check_complexity_inclusion("BPP", "P/poly")
    """
    from noethersolve.complexity import check_inclusion
    result = check_inclusion(class_a, class_b)
    return str(result)


@mcp.tool()
def check_completeness(problem: str, complexity_class: str) -> str:
    """Check if a problem is complete for a complexity class.

    Examples: check_completeness("SAT", "NP"),
              check_completeness("TQBF", "PSPACE")
    """
    from noethersolve.complexity import check_completeness as _check
    result = _check(problem, complexity_class)
    return str(result)


@mcp.tool()
def audit_complexity_claims(claims: list[str]) -> str:
    """Audit a list of complexity theory claims for correctness.

    Example claims: ["P ⊆ NP", "NP = coNP", "SAT is in P"]
    """
    from noethersolve.complexity import audit_complexity
    if isinstance(claims, str):
        claims = [c.strip() for c in claims.split(",") if c.strip()]
    report = audit_complexity(claims)
    return str(report)


# ── Proof Barriers ────────────────────────────────────────────────────

@mcp.tool()
def check_proof_barriers(problem: str, technique: str = "") -> str:
    """Check which proof barriers block resolving a problem, or if a
    specific technique is blocked.

    Examples: check_proof_barriers("P vs NP"),
              check_proof_barriers("P vs NP", "diagonalization")
    """
    from noethersolve.proof_barriers import check_barriers, list_techniques
    if technique:
        report = check_barriers(technique=technique, target=problem)
        return str(report)
    else:
        techniques = list_techniques()
        lines = [f"Proof barriers for: {problem}\n"]
        for t in techniques:
            r = check_barriers(technique=t, target=problem)
            issues = r.issues if hasattr(r, 'issues') else []
            high = [i for i in issues if getattr(i, 'severity', '') == 'HIGH']
            if high:
                lines.append(f"  [{t}] BLOCKED — {high[0].description[:100]}")
        return "\n".join(lines) if len(lines) > 1 else f"No known barriers for: {problem}"


@mcp.tool()
def list_proof_barriers() -> str:
    """List all known proof technique barriers (relativization, natural proofs, etc.)."""
    from noethersolve.proof_barriers import list_barriers
    barriers = list_barriers()
    lines = []
    for b in barriers:
        lines.append(f"  {b.name} ({b.year}) — {b.summary[:80]}")
    return "\n".join(lines)


# ── Number Theory ─────────────────────────────────────────────────────

@mcp.tool()
def verify_goldbach(limit: int = 10000) -> str:
    """Verify Goldbach's conjecture up to a limit. Returns verification report."""
    from noethersolve.number_theory import verify_goldbach as _verify
    report = _verify(limit)
    return str(report)


@mcp.tool()
def verify_collatz(limit: int = 10000) -> str:
    """Verify Collatz conjecture up to a limit. Returns max steps and verification."""
    from noethersolve.number_theory import verify_collatz as _verify
    report = _verify(limit)
    return str(report)


@mcp.tool()
def check_abc_triple(a: int, b: int, c: int = 0) -> str:
    """Check if (a, b, c) is an ABC triple and compute its quality.

    If c is not provided, it defaults to a + b.
    """
    from noethersolve.number_theory import check_abc_triple as _check
    if c == 0:
        c = a + b
    report = _check(a, b, c)
    return str(report)


@mcp.tool()
def analyze_prime_gaps(limit: int = 100000) -> str:
    """Analyze prime gap distribution up to limit."""
    from noethersolve.number_theory import prime_gap_analysis
    report = prime_gap_analysis(limit)
    return str(report)


# ── Computational Reductions ──────────────────────────────────────────

@mcp.tool()
def validate_reduction_chain(problems: list[str], reduction_type: str = "karp") -> str:
    """Validate a chain of computational reductions between problems.

    problems: list of problem names, e.g. ["3-SAT", "CLIQUE", "VERTEX-COVER"]
    reduction_type: type of reduction ("karp", "cook", "turing", "levin")

    Checks each consecutive pair A →_type B.
    """
    from noethersolve.reductions import validate_chain
    # Build chain as list of (problem_a, reduction_type, problem_b) tuples
    chain = []
    for i in range(len(problems) - 1):
        chain.append((problems[i], reduction_type, problems[i + 1]))
    report = validate_chain(chain)
    return str(report)


@mcp.tool()
def check_reduction(problem_a: str, problem_b: str, reduction_type: str = "karp") -> str:
    """Check if problem A reduces to problem B.

    reduction_type: "karp", "cook", "turing", "levin"
    """
    from noethersolve.reductions import check_reduction as _check
    result = _check(problem_a, reduction_type, problem_b)
    return str(result)


# ── PDE Regularity ───────────────────────────────────────────────────

@mcp.tool()
def check_sobolev_embedding(s: float, p: float, n: int) -> str:
    """Check Sobolev embedding theorem for W^{s,p}(R^n).

    Returns target space, embedding type (continuous/compact), and critical cases.
    """
    from noethersolve.pde_regularity import check_sobolev_embedding as _check
    report = _check(s, p, n)
    return str(report)


@mcp.tool()
def check_pde_regularity(pde_name: str, dimension: int = 3) -> str:
    """Check known regularity results for a PDE.

    pde_name: "navier-stokes", "euler", "heat", "wave", "burgers", "kdv", "nls"
    dimension: spatial dimension (default: 3)
    """
    from noethersolve.pde_regularity import check_pde_regularity as _check
    report = _check(pde_name, dimension)
    return str(report) if report else f"No regularity data for '{pde_name}'"


# ── Pharmacokinetics ──────────────────────────────────────────────────

@mcp.tool()
def audit_drug_interactions(drugs: list[str]) -> str:
    """Check a list of drugs for CYP-mediated interactions.

    Returns potential interactions, severity, and affected enzymes.
    Example: audit_drug_interactions(["warfarin", "fluconazole", "omeprazole"])
    """
    from noethersolve.pharmacokinetics import audit_drug_list
    # Handle comma-separated string input (models sometimes pass strings)
    if isinstance(drugs, str):
        drugs = [d.strip() for d in drugs.split(",") if d.strip()]
    report = audit_drug_list(drugs)
    return str(report)


@mcp.tool()
def check_pharmacogenomics(enzyme: str, phenotype: str, drugs: list[str] = []) -> str:
    """Check drug impact for a CYP enzyme metabolizer phenotype.

    enzyme: e.g. "CYP2D6", "CYP2C19", "CYP2C9"
    phenotype: "poor", "intermediate", "normal", "rapid", "ultrarapid"
    drugs: optional list of drugs to check specifically

    Examples: check_pharmacogenomics("CYP2D6", "poor", ["codeine", "tramadol"])
    """
    from noethersolve.pharmacokinetics import check_phenotype
    if isinstance(drugs, str):
        drugs = [d.strip() for d in drugs.split(",") if d.strip()]
    if not drugs:
        drugs = []
    result = check_phenotype(enzyme, phenotype, drugs)
    return str(result)


# ── LLM Claims Auditor ───────────────────────────────────────────────

@mcp.tool()
def check_llm_claim(claim: str) -> str:
    """Check if an LLM-related claim is true, false, or debated.

    Covers: hallucination, reasoning, alignment, training, evaluation,
    context/memory. 35+ topics with references.

    Examples: "RLHF eliminates sycophancy",
              "scaling eliminates hallucination",
              "chain-of-thought guarantees correctness"
    """
    from noethersolve.llm_claims import check_llm_claim as _check
    result = _check(claim)
    return str(result)


@mcp.tool()
def audit_llm_claims(claims: list[str]) -> str:
    """Batch audit multiple LLM claims. Returns pass/fail with issues."""
    from noethersolve.llm_claims import audit_llm_claims as _audit
    if isinstance(claims, str):
        claims = [c.strip() for c in claims.split(",") if c.strip()]
    report = _audit(claims)
    return str(report)


@mcp.tool()
def chinchilla_scaling(
    params_B: float = 0,
    tokens_B: float = 0,
    compute_flops: float = 0,
) -> str:
    """Check if a model is Chinchilla-optimal (D_opt ≈ 20× N).

    Provide at least one of: params_B, tokens_B, compute_flops.
    """
    from noethersolve.llm_claims import chinchilla_optimal
    kwargs = {}
    if params_B > 0:
        kwargs["params_B"] = params_B
    if tokens_B > 0:
        kwargs["tokens_B"] = tokens_B
    if compute_flops > 0:
        kwargs["compute_flops"] = compute_flops
    if not kwargs:
        return "Provide at least one of: params_B, tokens_B, compute_flops"
    result = chinchilla_optimal(**kwargs)
    return "\n".join(f"  {k}: {v}" for k, v in result.items())


@mcp.tool()
def check_benchmark_score(model: str, benchmark: str, score: float) -> str:
    """Check if a claimed benchmark score is plausible.

    Example: check_benchmark_score("gpt-4", "mmlu", 87.0)
    """
    from noethersolve.llm_claims import check_benchmark_score as _check
    result = _check(model, benchmark, score)
    return str(result)


# ── Genetics / Therapeutics ───────────────────────────────────────────

@mcp.tool()
def score_crispr_guide(
    spacer: str,
    pam: str = "NGG",
) -> str:
    """Score a CRISPR guide RNA for on-target activity and off-target risk.

    spacer: 20nt guide sequence
    pam: PAM sequence (default: NGG for SpCas9)
    """
    from noethersolve.crispr import score_guide
    report = score_guide(spacer, pam=pam)
    return str(report)


@mcp.tool()
def audit_dna_sequence(sequence: str) -> str:
    """Audit a DNA/RNA sequence for therapeutic design issues.

    Checks: GC content, CpG islands, homopolymer runs, restriction sites,
    repeat elements, codon usage.
    """
    from noethersolve.audit_sequence import audit_sequence
    report = audit_sequence(sequence)
    return str(report)


@mcp.tool()
def predict_protein_aggregation(sequence: str) -> str:
    """Predict aggregation propensity of a protein sequence.

    Uses Kyte-Doolittle hydrophobicity and AGGRESCAN scales.
    """
    from noethersolve.aggregation import predict_aggregation
    report = predict_aggregation(sequence)
    return str(report)


@mcp.tool()
def score_splice_sites(sequence: str) -> str:
    """Scan a DNA sequence for splice donor/acceptor sites.

    Returns scored sites with PWM-based strength assessment.
    """
    from noethersolve.splice import scan_splice_sites
    report = scan_splice_sites(sequence)
    return str(report)


@mcp.tool()
def validate_therapy_pipeline(
    modality: str,
    target_tissue: str,
    transgene_size_kb: float = 0.0,
    vector_serotype: str = "",
    promoter: str = "",
    route: str = "",
    payload_type: str = "",
    redosing_planned: bool = False,
) -> str:
    """Validate consistency of a therapeutic design pipeline.

    modality: antisense, siRNA, mRNA, CRISPR, gene_therapy
    target_tissue: liver, CNS, muscle, eye, lung, etc.
    """
    from noethersolve.pipeline import TherapyDesign, validate_pipeline

    kwargs = {"modality": modality, "target_tissue": target_tissue}
    if transgene_size_kb > 0:
        kwargs["transgene_size_kb"] = transgene_size_kb
    if vector_serotype:
        kwargs["vector_serotype"] = vector_serotype
    if promoter:
        kwargs["promoter"] = promoter
    if route:
        kwargs["route"] = route
    if payload_type:
        kwargs["payload_type"] = payload_type
    if redosing_planned:
        kwargs["redosing_planned"] = redosing_planned

    design = TherapyDesign(**kwargs)
    report = validate_pipeline(design)
    return str(report)


# ── Chemical Kinetics ─────────────────────────────────────────────────

@mcp.tool()
def audit_chemical_network(
    species: list[str],
    stoichiometry: list[list[int]],
    rate_constants: list[float] = [],
) -> str:
    """Audit a chemical reaction network for thermodynamic consistency.

    species: list of species names, e.g. ["A", "B", "C"]
    stoichiometry: matrix (n_species × n_reactions) — each row is one species,
        each column is one reaction. E.g. for A→B, B→C with species [A,B,C]:
        [[-1, 0], [1, -1], [0, 1]]
    rate_constants: optional list of rate constants (one per reaction)
    """
    from noethersolve.audit_chem import audit_network
    import numpy as np

    S = np.array(stoichiometry)
    k = np.array(rate_constants) if rate_constants else None
    report = audit_network(species, S, rate_constants=k)
    return str(report)


# ── Knot Theory ───────────────────────────────────────────────────────

@mcp.tool()
def check_knot_invariants(knot_name: str = "trefoil") -> str:
    """Check knot invariants (crossing number, writhe, Jones polynomial).

    knot_name: "unknot", "trefoil", "figure_eight"
    """
    from noethersolve.knot import KnotMonitor, trefoil, figure_eight_knot, unknot

    knots = {
        "unknot": unknot,
        "trefoil": trefoil,
        "figure_eight": figure_eight_knot,
    }
    if knot_name not in knots:
        return f"Unknown knot '{knot_name}'. Available: {list(knots.keys())}"

    diagram = knots[knot_name]()
    monitor = KnotMonitor(diagram)
    report = monitor.validate()
    return str(report)


# ── EM Field Conservation ─────────────────────────────────────────────

@mcp.tool()
def check_em_conservation(
    E_field: list[list[list[list[float]]]],
    B_field: list[list[list[list[float]]]],
) -> str:
    """Check electromagnetic field conservation laws.

    E_field, B_field: 4D arrays [3, Nx, Ny, Nz] — vector field components.
    Returns energy density, helicity, and other conservation checks.
    """
    from noethersolve.monitor_em import EMMonitor
    import numpy as np

    E = np.array(E_field)
    B = np.array(B_field)
    N = E.shape[1] if E.ndim == 4 else E.shape[0]
    monitor = EMMonitor(N=N)
    monitor.set_initial(E, B)
    report = monitor.check(E, B)
    return str(report)


# ── Conservation Law Discovery ────────────────────────────────────────

@mcp.tool()
def discover_conservation_law(
    trajectories: list[list[list[float]]],
    weights: list[float] = [],
) -> str:
    """Attempt to discover conserved quantities from trajectories.

    trajectories: list of trajectories, each a list of position vectors.
        E.g. [[[x1,y1], [x1,y1], ...], [[x2,y2], [x2,y2], ...]]
        for two particles in 2D.
    weights: optional weight per trajectory (default: uniform).
    Uses L-BFGS-B optimization over basis functions.
    """
    from noethersolve.learner import InvariantLearner
    import numpy as np

    trajs = [np.array(t) for t in trajectories]
    if not weights:
        weights = [1.0] * len(trajs)
    learner = InvariantLearner()
    report = learner.learn_from_positions(trajs, weights)
    return str(report)


# ── Entry Point ───────────────────────────────────────────────────────

def main():
    """Run the NoetherSolve MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
