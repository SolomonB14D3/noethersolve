"""NoetherSolve MCP Server — expose 43 verified tools to any AI agent.

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
    instructions="43 computational tools for physics, math, genetics, and LLM science — "
                 "verified calculators and reference databases, not guesses.",
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


# ── Control Systems (CALCULATOR) ─────────────────────────────────

@mcp.tool()
def simulate_pid(
    Kp: float = 1.0,
    Ki: float = 0.0,
    Kd: float = 0.0,
    plant_num: list[float] = [1.0],
    plant_den: list[float] = [1.0, 3.0, 1.0],
    setpoint: float = 1.0,
    t_final: float = 20.0,
) -> str:
    """Simulate PID controller step response with a transfer-function plant.

    COMPUTES overshoot, settling time, steady-state error, and windup detection.

    Kp, Ki, Kd: PID gains
    plant_num, plant_den: transfer function coefficients (highest power first)
        Default plant: 1/(s^2 + 3s + 1) (underdamped 2nd order)
    setpoint: step input magnitude
    t_final: simulation duration (seconds)

    Example: simulate_pid(Kp=2.5, Ki=0.8, Kd=0.1)
    """
    from noethersolve.control import simulate_pid as _sim
    report = _sim(
        Kp=Kp, Ki=Ki, Kd=Kd,
        plant_num=plant_num, plant_den=plant_den,
        setpoint=setpoint, t_final=t_final,
    )
    return str(report)


@mcp.tool()
def analyze_stability(coefficients: list[float]) -> str:
    """Analyze stability of a characteristic polynomial via Routh-Hurwitz.

    COMPUTES pole locations, sign changes, and stability verdict.

    coefficients: polynomial coefficients, highest power first.
        E.g. [1, 3, 3, 1] for s^3 + 3s^2 + 3s + 1

    Example: analyze_stability([1, 6, 11, 6])  → stable, poles at -1,-2,-3
    """
    from noethersolve.control import analyze_stability as _analyze
    report = _analyze(coefficients)
    return str(report)


# ── Transaction Isolation (CALCULATOR) ───────────────────────────

@mcp.tool()
def check_isolation(
    isolation_level: str,
    anomaly: str = "",
) -> str:
    """Check which concurrency anomalies are possible under a SQL isolation level.

    COMPUTES possible anomalies, prevented anomalies, and common misconceptions.

    isolation_level: READ_UNCOMMITTED, READ_COMMITTED, REPEATABLE_READ,
                     SNAPSHOT, SERIALIZABLE
    anomaly: optional specific anomaly to check (e.g. "phantom_read",
             "write_skew", "lost_update", "dirty_read")

    Example: check_isolation("REPEATABLE_READ", "phantom_read")
    → YES, phantom reads are still possible (common misconception!)
    """
    from noethersolve.isolation import check_isolation as _check
    report = _check(isolation_level, anomaly=anomaly if anomaly else None)
    return str(report)


@mcp.tool()
def analyze_schedule(
    transactions: list[list[list[str]]],
    isolation: str = "READ_COMMITTED",
) -> str:
    """Analyze a concrete transaction schedule for conflicts and anomalies.

    COMPUTES conflict graph, possible anomalies, and serializability.

    transactions: list of transactions, each a list of [operation, item] pairs.
        Operation: "R" (read) or "W" (write). Item: data item name.
        Example: [[["R","x"],["W","x"]], [["R","x"],["W","x"]]]
    isolation: isolation level to analyze under

    Example: two transactions both read-then-write x → detects lost update risk
    """
    from noethersolve.isolation import analyze_schedule as _analyze
    # Convert list-of-lists to list-of-tuples
    txns = [[(op, item) for op, item in txn] for txn in transactions]
    report = _analyze(txns, isolation=isolation)
    return str(report)


# ── Quantum Circuit Simulation (CALCULATOR) ──────────────────────

@mcp.tool()
def simulate_quantum_circuit(
    n_qubits: int,
    gates: list[list],
) -> str:
    """Simulate a quantum circuit and compute measurement probabilities.

    COMPUTES state vector, probabilities, entanglement, and von Neumann entropy.

    n_qubits: number of qubits (max 10)
    gates: list of gates, each [name, qubits] or [name, qubits, param].
        Names: "H", "X", "Y", "Z", "S", "T", "CNOT", "CZ", "SWAP",
               "RX", "RY", "RZ", "TOFFOLI"
        qubits: list of qubit indices (0-indexed)

    Examples:
        Bell state: [["H",[0]], ["CNOT",[0,1]]]
        GHZ state:  [["H",[0]], ["CNOT",[0,1]], ["CNOT",[0,2]]]
    """
    from noethersolve.quantum_circuit import simulate_circuit as _sim
    # Convert gate lists to tuples
    gate_tuples = []
    for g in gates:
        if len(g) == 2:
            gate_tuples.append((g[0], g[1]))
        elif len(g) == 3:
            gate_tuples.append((g[0], g[1], g[2]))
        else:
            return f"Invalid gate spec: {g}. Expected [name, qubits] or [name, qubits, param]."
    report = _sim(n_qubits, gate_tuples)
    return str(report)


# ── Chemistry Calculator ──────────────────────────────────────────

@mcp.tool()
def calc_nernst(
    E_standard: float,
    n_electrons: int,
    Q: float,
    temperature: float = 298.15,
) -> str:
    """Calculate cell potential using the Nernst equation.

    COMPUTES E = E° - (RT/nF)ln(Q), spontaneity, and ΔG.

    E_standard: Standard cell potential in Volts
    n_electrons: Electrons transferred in half-reaction
    Q: Reaction quotient [products]/[reactants]
    temperature: Temperature in Kelvin (default 298.15)

    Example: calc_nernst(E_standard=1.10, n_electrons=2, Q=0.01)
    → E=1.159V, spontaneous, ΔG=-223.8 kJ/mol
    """
    from noethersolve.chemistry_calc import nernst_equation
    report = nernst_equation(E_standard, n_electrons, Q, temperature)
    return str(report)


@mcp.tool()
def calc_buffer_ph(pKa: float, acid_conc: float, base_conc: float) -> str:
    """Calculate buffer pH using Henderson-Hasselbalch equation.

    COMPUTES pH = pKa + log([A⁻]/[HA]), buffer capacity, effective range.

    pKa: Acid dissociation constant (-log10 Ka)
    acid_conc: Weak acid concentration [HA] in mol/L
    base_conc: Conjugate base concentration [A⁻] in mol/L

    Example: calc_buffer_ph(pKa=4.76, acid_conc=0.1, base_conc=0.15)
    → pH=4.94, capacity=0.057, range 3.76-5.76
    """
    from noethersolve.chemistry_calc import henderson_hasselbalch
    report = henderson_hasselbalch(pKa, acid_conc, base_conc)
    return str(report)


@mcp.tool()
def calc_crystal_field(
    d_electrons: int,
    geometry: str = "octahedral",
    strong_field: bool = False,
) -> str:
    """Calculate crystal field splitting for transition metal complexes.

    COMPUTES CFSE, spin state, unpaired electrons, d-orbital configuration.

    d_electrons: Number of d-electrons (1-10)
    geometry: "octahedral", "tetrahedral", or "square_planar"
    strong_field: True for low-spin (strong field ligands like CN⁻, CO)

    Example: calc_crystal_field(6, "octahedral", strong_field=True)
    → t2g^6 eg^0, low_spin, 0 unpaired, CFSE=-24 Dq
    """
    from noethersolve.chemistry_calc import crystal_field_splitting
    report = crystal_field_splitting(d_electrons, geometry, strong_field)
    return str(report)


# ── Cryptography Calculator ──────────────────────────────────────

@mcp.tool()
def calc_security_level(
    algorithm: str,
    key_bits: int,
    ops_per_second: float = 1e12,
) -> str:
    """Calculate effective security level for a cryptographic algorithm.

    COMPUTES classical and post-quantum security bits, brute force time.

    algorithm: "aes", "3des", "rsa", "ecc", "sha", "chacha20"
    key_bits: Key or output size in bits
    ops_per_second: Attacker throughput (default 10^12)

    Example: calc_security_level("aes", 256) → 256-bit classical, 128-bit quantum
    """
    from noethersolve.crypto_calc import security_level
    report = security_level(algorithm, key_bits, ops_per_second)
    return str(report)


@mcp.tool()
def calc_birthday_bound(output_bits: int, trials: int = 0) -> str:
    """Calculate birthday bound collision probability for a hash function.

    COMPUTES trials needed for 50% collision, probability at given trials.

    output_bits: Hash output size in bits (e.g., 256 for SHA-256)
    trials: Optional number of trials to check probability

    Example: calc_birthday_bound(128, trials=1000000) → P(collision) ≈ 2.9e-27
    """
    from noethersolve.crypto_calc import birthday_bound
    report = birthday_bound(output_bits, trials=trials if trials > 0 else None)
    return str(report)


@mcp.tool()
def calc_cipher_mode(mode: str, block_bits: int = 128) -> str:
    """Analyze security properties of a block cipher mode of operation.

    COMPUTES IV requirements, parallelizability, authentication, birthday limits.

    mode: "ECB", "CBC", "CTR", "GCM", "CCM", "OFB", "CFB"
    block_bits: Block size (default 128 for AES)

    Example: calc_cipher_mode("CBC")
    → requires random IV, decrypt parallelizable, NOT authenticated
    """
    from noethersolve.crypto_calc import cipher_mode_analysis
    report = cipher_mode_analysis(mode, block_bits)
    return str(report)


# ── Finance Calculator ───────────────────────────────────────────

@mcp.tool()
def calc_black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> str:
    """Price a European option using Black-Scholes (log-normal + constant vol).

    COMPUTES option price and all Greeks (delta, gamma, theta, vega, rho).

    S: Current stock price
    K: Strike price
    T: Time to expiry in years
    r: Risk-free rate (e.g., 0.05 for 5%)
    sigma: Volatility (e.g., 0.20 for 20%)
    option_type: "call" or "put"

    Example: calc_black_scholes(100, 105, 0.5, 0.05, 0.20)
    """
    from noethersolve.finance_calc import black_scholes
    report = black_scholes(S, K, T, r, sigma, option_type)
    return str(report)


@mcp.tool()
def calc_put_call_parity(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
) -> str:
    """Check put-call parity: C - P = S - PV(K).

    COMPUTES whether parity holds and identifies arbitrage opportunities.
    Relates call, put, stock, AND risk-free bond (common LLM error: models
    often say it only relates calls to stocks).

    Example: calc_put_call_parity(10.45, 5.58, 100, 105, 0.05, 0.5)
    """
    from noethersolve.finance_calc import put_call_parity
    report = put_call_parity(call_price, put_price, S, K, r, T)
    return str(report)


@mcp.tool()
def calc_nash_equilibrium(
    payoff_p1: list[list[float]],
    payoff_p2: list[list[float]],
) -> str:
    """Find Nash equilibria of a 2x2 game.

    COMPUTES pure and mixed equilibria, dominant strategies, game classification.

    payoff_p1: 2x2 matrix — P1's payoffs. payoff_p1[i][j] = P1 payoff when
               P1 plays row i, P2 plays column j.
    payoff_p2: 2x2 matrix — P2's payoffs.

    Example (Prisoner's Dilemma):
      calc_nash_equilibrium([[3,0],[5,1]], [[3,5],[0,1]])
      → Pure NE: (1,1) both defect, despite (0,0) being better for both
    """
    from noethersolve.finance_calc import nash_equilibrium_2x2
    report = nash_equilibrium_2x2(payoff_p1, payoff_p2)
    return str(report)


# ── Distributed Systems Calculator ──────────────────────────────

@mcp.tool()
def calc_quorum(
    total_nodes: int,
    read_quorum: int = 0,
    write_quorum: int = 0,
    strategy: str = "majority",
) -> str:
    """Calculate quorum sizes and consistency guarantees.

    COMPUTES R+W>N check, fault tolerance, strong consistency verdict.

    total_nodes: Total replica count (N)
    read_quorum: Read quorum (R), 0 to auto-derive from strategy
    write_quorum: Write quorum (W), 0 to auto-derive
    strategy: "majority", "read_heavy", "write_heavy" (if R/W not given)

    Example: calc_quorum(5) → R=3, W=3, strong consistency, tolerates 2 failures
    """
    from noethersolve.distributed_calc import quorum_calc
    r = read_quorum if read_quorum > 0 else None
    w = write_quorum if write_quorum > 0 else None
    report = quorum_calc(total_nodes, r, w, strategy)
    return str(report)


@mcp.tool()
def calc_byzantine(
    total_nodes: int = 0,
    max_faults: int = 0,
    algorithm: str = "PBFT",
) -> str:
    """Calculate Byzantine fault tolerance requirements.

    COMPUTES minimum nodes (3f+1), rounds, message complexity.
    Common LLM error: models often say 2f+1, but BFT requires 3f+1.

    total_nodes: Number of nodes (to find max faults), or
    max_faults: Desired tolerance (to find min nodes)
    algorithm: "PBFT", "synchronous", "Tendermint"

    Example: calc_byzantine(max_faults=2) → need 7 nodes minimum (3×2+1)
    """
    from noethersolve.distributed_calc import byzantine_threshold
    n = total_nodes if total_nodes > 0 else None
    f = max_faults if max_faults > 0 else None
    report = byzantine_threshold(n, f, algorithm)
    return str(report)


@mcp.tool()
def calc_vector_clock(clock_a: list[int], clock_b: list[int]) -> str:
    """Compare two vector clocks to determine causal ordering.

    COMPUTES happens-before, concurrent, or equal relationship + merge.

    clock_a: Vector clock for event A (e.g., [2, 0, 1])
    clock_b: Vector clock for event B (e.g., [1, 1, 1])

    Example: calc_vector_clock([2,0,1], [1,1,1]) → concurrent (no causal link)
    """
    from noethersolve.distributed_calc import vector_clock_compare
    report = vector_clock_compare(clock_a, clock_b)
    return str(report)


# ── Networking Calculator ────────────────────────────────────────

@mcp.tool()
def calc_bandwidth_delay(
    bandwidth_bps: float,
    rtt_seconds: float,
    window_size_bytes: int = 0,
) -> str:
    """Calculate bandwidth-delay product and TCP window requirements.

    COMPUTES BDP, minimum window for full utilization, link utilization.

    bandwidth_bps: Link bandwidth in bits/sec (e.g., 1e9 for 1 Gbps)
    rtt_seconds: Round-trip time in seconds (e.g., 0.050 for 50ms)
    window_size_bytes: Optional TCP window to check utilization

    Example: calc_bandwidth_delay(1e9, 0.050) → BDP=6.25MB, need 6.25MB window
    """
    from noethersolve.network_calc import bandwidth_delay_product
    w = window_size_bytes if window_size_bytes > 0 else None
    report = bandwidth_delay_product(bandwidth_bps, rtt_seconds, w)
    return str(report)


@mcp.tool()
def calc_subnet(ip_address: str, prefix_length: int) -> str:
    """Calculate subnet properties from IP and CIDR prefix.

    COMPUTES network/broadcast addresses, host range, masks, usable hosts.

    ip_address: IPv4 address (e.g., "192.168.1.100")
    prefix_length: CIDR prefix (0-32)

    Example: calc_subnet("192.168.1.100", 24)
    → network 192.168.1.0, broadcast 192.168.1.255, 254 usable hosts
    """
    from noethersolve.network_calc import subnet_calc
    report = subnet_calc(ip_address, prefix_length)
    return str(report)


@mcp.tool()
def calc_tcp_throughput(
    window_size_bytes: int,
    rtt_seconds: float,
    mss_bytes: int = 1460,
    loss_rate: float = 0.0,
) -> str:
    """Estimate TCP throughput from window size, RTT, and loss rate.

    COMPUTES window-limited and loss-limited (Mathis model) throughput.

    window_size_bytes: TCP window in bytes
    rtt_seconds: Round-trip time in seconds
    mss_bytes: Max segment size (default 1460)
    loss_rate: Packet loss rate 0-1 (0 = no loss)

    Example: calc_tcp_throughput(65535, 0.050, loss_rate=0.001)
    """
    from noethersolve.network_calc import tcp_throughput
    report = tcp_throughput(
        window_size_bytes, rtt_seconds, mss_bytes,
        loss_rate=loss_rate if loss_rate > 0 else None,
    )
    return str(report)


# ── Operating Systems Calculator ─────────────────────────────────

@mcp.tool()
def calc_page_table(
    virtual_bits: int,
    physical_bits: int,
    page_size_bytes: int,
    levels: int = 1,
) -> str:
    """Calculate page table dimensions from address space parameters.

    COMPUTES page count, PTE size, table size, multi-level breakdown.

    virtual_bits: Virtual address width (e.g., 48 for x86-64)
    physical_bits: Physical address width (e.g., 52)
    page_size_bytes: Page size in bytes (must be power of 2, e.g., 4096)
    levels: Page table levels (1-5, default 1)

    Example: calc_page_table(48, 52, 4096, levels=4) → x86-64 4-level page table
    """
    from noethersolve.os_calc import page_table_calc
    report = page_table_calc(virtual_bits, physical_bits, page_size_bytes, levels=levels)
    return str(report)


@mcp.tool()
def calc_scheduling(
    processes: list[list],
    algorithm: str = "FCFS",
    quantum: float = 2.0,
) -> str:
    """Simulate CPU scheduling algorithm on a process set.

    COMPUTES execution order, turnaround times, waiting times, Gantt chart.

    processes: List of [name, arrival_time, burst_time]
        Example: [["P1",0,5], ["P2",1,3], ["P3",2,8]]
    algorithm: "FCFS", "SJF", "SRTF", "RR"
    quantum: Time quantum for Round Robin

    Example: calc_scheduling([["P1",0,5],["P2",1,3],["P3",2,8]], "SJF")
    """
    from noethersolve.os_calc import schedule_fcfs, schedule_sjf, schedule_round_robin
    procs = [(p[0], float(p[1]), float(p[2])) for p in processes]
    algo = algorithm.upper()
    if algo == "FCFS":
        report = schedule_fcfs(procs)
    elif algo == "SJF":
        report = schedule_sjf(procs, preemptive=False)
    elif algo == "SRTF":
        report = schedule_sjf(procs, preemptive=True)
    elif algo == "RR":
        report = schedule_round_robin(procs, quantum=quantum)
    else:
        return f"Unknown algorithm: {algorithm}. Use FCFS, SJF, SRTF, or RR."
    return str(report)


@mcp.tool()
def calc_deadlock(
    holding: dict[str, list[str]],
    waiting: dict[str, str],
) -> str:
    """Detect deadlock using wait-for graph cycle detection.

    COMPUTES cycle in wait-for graph. Deadlock requires ALL four Coffman
    conditions: mutual exclusion, hold-wait, no preemption, circular wait.

    holding: {process: [resources held]}
    waiting: {process: resource waiting for}

    Example: calc_deadlock(
        {"P1": ["R1"], "P2": ["R2"]},
        {"P1": "R2", "P2": "R1"}
    ) → deadlock detected: P1 → P2 → P1
    """
    from noethersolve.os_calc import detect_deadlock
    report = detect_deadlock(holding, waiting)
    return str(report)


# ── Entry Point ───────────────────────────────────────────────────────

def main():
    """Run the NoetherSolve MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
