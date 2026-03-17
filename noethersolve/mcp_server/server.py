"""NoetherSolve MCP Server — expose 163 verified tools to any AI agent.

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
    instructions="163 computational tools for physics, math, genetics, chemistry, pharmacokinetics, "
                 "epidemiology, climate physics, turbulence, topological phases, ergodic theory, optimization, "
                 "numerical PDEs, MHD conservation, GR constraints, seismic waves, plasma adiabatic invariants, "
                 "intersection theory, and LLM science — verified calculators from first principles, not guesses.",
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


@mcp.tool()
def check_dimension_physics(concept: str, dimension: int = 0) -> str:
    """Check how physics formulas change with spatial dimension.

    CRITICAL: Models are systematically blind to dimension-dependent physics.
    They know 3D physics but fail to modulate for 2D contexts.

    concept: Physics concept to check. Options:
        - "greens_function" or "laplacian"
        - "coulomb" or "electrostatic"
        - "vortex" or "vortex_topology"
        - "turbulence" or "energy_cascade"
        - "enstrophy"
        - "navier_stokes" or "ns_regularity"
        - "wave" or "huygens"
        - "stream_function"

    dimension: If 0, returns all dimensions. If 1/2/3, returns that specific form.

    Example: check_dimension_physics("greens_function")
    → Shows: 1D is |x|/2, 2D is -ln(r)/(2π), 3D is 1/(4πr)

    Example: check_dimension_physics("turbulence", 2)
    → Shows: In 2D, energy cascades UPWARD (inverse cascade)
    """
    from noethersolve.dimension_physics import check_dimension_dependence, get_formula

    if dimension == 0:
        result = check_dimension_dependence(concept)
        return str(result)
    else:
        formula = get_formula(concept, dimension)
        if formula:
            return (
                f"{formula.concept} in {dimension}D:\n"
                f"  Formula: {formula.formula}\n"
                f"  LaTeX: {formula.latex}\n"
                f"  Note: {formula.notes}\n"
                f"  Common error: {formula.common_error}"
            )
        else:
            return f"No formula for '{concept}' in {dimension}D"


# ── Pharmacokinetics ──────────────────────────────────────────────────

@mcp.tool()
def calc_iv_bolus(
    dose: float,
    Vd: float,
    ke: float,
    t: float,
) -> str:
    """Compute drug concentration after IV bolus (one-compartment model).

    CALCULATOR — C(t) = (Dose/Vd) × e^(-ke×t). Reports concentration,
    half-life, clearance, AUC, and fraction remaining.

    dose: dose in mg
    Vd: volume of distribution in L
    ke: elimination rate constant in h⁻¹
    t: time in hours

    Example: calc_iv_bolus(500, 50, 0.1, 6)
    → C(6h) = 5.49 mg/L, t½ = 6.93h, CL = 5 L/h
    """
    from noethersolve.pk_model import one_compartment_iv
    return str(one_compartment_iv(dose=dose, Vd=Vd, ke=ke, t=t))


@mcp.tool()
def calc_oral_dose(
    dose: float,
    F: float,
    Vd: float,
    ka: float,
    ke: float,
    t: float,
) -> str:
    """Compute drug concentration after oral dosing (Bateman function).

    CALCULATOR — uses one-compartment model with first-order absorption.
    Reports Cmax, Tmax, AUC, and concentration at any time point.

    dose: dose in mg
    F: bioavailability (0 to 1)
    Vd: volume of distribution in L
    ka: absorption rate constant in h⁻¹
    ke: elimination rate constant in h⁻¹
    t: time in hours

    Example: calc_oral_dose(500, 0.8, 50, 1.5, 0.1, 4)
    → C(4h) = 5.12 mg/L, Cmax = 5.71 mg/L at Tmax = 1.94h
    """
    from noethersolve.pk_model import one_compartment_oral
    return str(one_compartment_oral(dose=dose, F=F, Vd=Vd, ka=ka, ke=ke, t=t))


@mcp.tool()
def calc_half_life(
    CL: float = 0,
    Vd: float = 0,
    ke: float = 0,
) -> str:
    """Compute elimination half-life from PK parameters.

    CALCULATOR — t½ = ln(2)/ke = 0.693 × Vd/CL. Provide either ke alone,
    or both CL and Vd. Reports time to 97% elimination (5 × t½).

    CL: clearance in L/h (use with Vd)
    Vd: volume of distribution in L (use with CL)
    ke: elimination rate constant in h⁻¹ (alternative to CL+Vd)

    Example: calc_half_life(CL=10, Vd=50) → t½ = 3.47h
    """
    from noethersolve.pk_model import half_life
    kw = {}
    if ke > 0:
        kw["ke"] = ke
    if CL > 0:
        kw["CL"] = CL
    if Vd > 0:
        kw["Vd"] = Vd
    return str(half_life(**kw))


@mcp.tool()
def calc_steady_state(
    dose: float,
    F: float,
    CL: float,
    tau: float,
    Vd: float = 0,
) -> str:
    """Compute steady-state concentrations for repeated dosing.

    CALCULATOR — Css_avg = F×Dose/(CL×τ). Reports average, peak, trough
    concentrations, accumulation factor, and time to steady state.

    dose: dose per administration in mg
    F: bioavailability (0 to 1)
    CL: clearance in L/h
    tau: dosing interval in hours
    Vd: volume of distribution in L (for peak/trough, optional)

    Example: calc_steady_state(500, 0.8, 10, 8, 50)
    → Css_avg = 5.0 mg/L, time to SS ≈ 17.3h
    """
    from noethersolve.pk_model import steady_state
    kw = {}
    if Vd > 0:
        kw["Vd"] = Vd
    return str(steady_state(dose=dose, F=F, CL=CL, tau=tau, **kw))


@mcp.tool()
def calc_dose_adjustment(
    original_dose: float,
    fold_change_auc: float,
    reason: str = "",
) -> str:
    """Compute adjusted dose for CYP inhibition/induction or organ impairment.

    CALCULATOR — if AUC increases N-fold, reduce dose by 1/N to maintain
    same total exposure. Reports adjusted dose and clinical notes.

    original_dose: current dose in mg
    fold_change_auc: fold-change in AUC (e.g. 5.0 for strong CYP3A4 inhibitor,
                     0.2 for strong inducer causing 80% AUC decrease)
    reason: description of interaction (optional)

    Example: calc_dose_adjustment(100, 5.0, "ketoconazole CYP3A4 inhibition")
    → Adjusted dose = 20 mg (1/5 of original)
    """
    from noethersolve.pk_model import dose_adjustment
    return str(dose_adjustment(original_dose=original_dose,
                               fold_change_auc=fold_change_auc, reason=reason))


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


# ── Science Domain Lookup Tables ──────────────────────────────────────


@mcp.tool()
def calc_michaelis_menten(
    Vmax: float,
    Km: float,
    S: float,
) -> str:
    """Compute Michaelis-Menten enzyme kinetics from first principles.

    CALCULATOR — derives initial velocity V0 = Vmax × [S] / (Km + [S]).
    Reports saturation status, regime (first-order vs zero-order), and
    reference values.

    Vmax: maximum velocity (any consistent units, e.g. µM/s)
    Km: Michaelis constant (same concentration units as S)
    S: substrate concentration

    Example: calc_michaelis_menten(100, 5, 10)
    → V0 = 66.7 µM/s (66.7% Vmax, near-saturated)
    """
    from noethersolve.enzyme_kinetics import michaelis_menten
    return str(michaelis_menten(Vmax=Vmax, Km=Km, S=S))


@mcp.tool()
def calc_enzyme_inhibition(
    Vmax: float,
    Km: float,
    S: float,
    Ki: float,
    I: float,
    mode: str = "competitive",
    Ki_prime: float = 0,
) -> str:
    """Compute enzyme kinetics with inhibitor (competitive, noncompetitive,
    uncompetitive, or mixed).

    CALCULATOR — derives apparent Km and Vmax, computes inhibited velocity,
    and reports percent inhibition with Lineweaver-Burk diagnostic.

    Vmax: maximum velocity
    Km: Michaelis constant
    S: substrate concentration
    Ki: inhibition constant
    I: inhibitor concentration
    mode: "competitive", "noncompetitive", "uncompetitive", or "mixed"
    Ki_prime: for mixed inhibition only (ES-I dissociation constant)

    Example: calc_enzyme_inhibition(100, 5, 10, 2, 4, "competitive")
    → Km_app = 15, V0 = 40.0, 40% inhibition
    """
    from noethersolve.enzyme_kinetics import inhibition
    kw = {}
    if Ki_prime > 0:
        kw["Ki_prime"] = Ki_prime
    return str(inhibition(Vmax=Vmax, Km=Km, S=S, Ki=Ki, I=I, mode=mode, **kw))


@mcp.tool()
def calc_catalytic_efficiency(
    kcat: float,
    Km: float,
) -> str:
    """Compute catalytic efficiency (specificity constant kcat/Km).

    CALCULATOR — classifies enzyme efficiency from slow to catalytically
    perfect (diffusion-limited ~10⁸ M⁻¹s⁻¹). Compares to reference enzymes
    (carbonic anhydrase, acetylcholinesterase, triosephosphate isomerase).

    kcat: turnover number in s⁻¹
    Km: Michaelis constant in M (molar)

    Example: calc_catalytic_efficiency(1000, 5e-6)
    → kcat/Km = 2.0×10⁸ M⁻¹s⁻¹ — catalytically perfect!
    """
    from noethersolve.enzyme_kinetics import catalytic_efficiency
    return str(catalytic_efficiency(kcat=kcat, Km=Km))


@mcp.tool()
def calc_cooperativity(
    Vmax: float,
    K_half: float,
    n: float,
    S: float,
) -> str:
    """Compute cooperative enzyme/binding kinetics using the Hill equation.

    CALCULATOR — V0 = Vmax × [S]ⁿ / (K₀.₅ⁿ + [S]ⁿ). Reports cooperativity
    type (positive/negative/none) and minimum binding sites implied by Hill
    coefficient.

    Vmax: maximum velocity
    K_half: substrate concentration at half-maximal velocity
    n: Hill coefficient (>1 positive cooperativity, <1 negative, =1 Michaelis-Menten)
    S: substrate concentration

    Example: calc_cooperativity(100, 26, 2.8, 26)
    → V0 = 50.0 (positive cooperativity, at least 3 binding sites)
    """
    from noethersolve.enzyme_kinetics import cooperativity
    return str(cooperativity(Vmax=Vmax, K_half=K_half, n=n, S=S))


@mcp.tool()
def calc_ph_rate_profile(
    pH: float,
    V_optimal: float,
    pKa1: float,
    pKa2: float,
) -> str:
    """Compute enzyme activity at a given pH using the bell-shaped model.

    CALCULATOR — V(pH) = Vopt / (1 + [H⁺]/Ka1 + Ka2/[H⁺]). Reports
    fraction active, optimal pH, and protonation state diagnostics.

    pH: pH value to evaluate
    V_optimal: maximum velocity at optimal pH
    pKa1: pKa of acid limb (lower pH side)
    pKa2: pKa of base limb (higher pH side)

    Example: calc_ph_rate_profile(5.0, 100, 6.0, 8.0)
    → V = 8.3 (8.3% active, below pKa1 — catalytic base protonated)
    """
    from noethersolve.enzyme_kinetics import ph_rate_profile
    return str(ph_rate_profile(pH=pH, V_optimal=V_optimal, pKa1=pKa1, pKa2=pKa2))


@mcp.tool()
def calc_particle_in_box(
    n: int,
    L: float,
    m: float = 9.109e-31,
) -> str:
    """Compute particle-in-a-box energy levels from first principles.

    CALCULATOR — E_n = n²π²ℏ²/(2mL²). Reports energy in eV, de Broglie
    wavelength, and node count.

    n: quantum number (1, 2, 3, ...)
    L: box length in meters
    m: particle mass in kg (default: electron mass 9.109e-31)

    Example: calc_particle_in_box(1, 1e-9)
    → E_1 = 0.376 eV (ground state)
    """
    from noethersolve.qm_calculator import particle_in_box
    return str(particle_in_box(n=n, L=L, m=m))


@mcp.tool()
def calc_hydrogen_energy(
    n: int,
    Z: int = 1,
) -> str:
    """Compute hydrogen atom energy levels and orbital properties.

    CALCULATOR — E_n = -13.6 × Z²/n² eV. Reports energy, Bohr radius,
    degeneracy, ionization energy, and Lyman transition wavelength.

    n: principal quantum number (1, 2, 3, ...)
    Z: nuclear charge (1 for H, 2 for He+, etc.)

    Example: calc_hydrogen_energy(2) → E_2 = -3.4 eV, r = 2.12 Å
    """
    from noethersolve.qm_calculator import hydrogen_energy
    return str(hydrogen_energy(n=n, Z=Z))


@mcp.tool()
def calc_uncertainty_check(
    delta_x: float,
    delta_p: float,
) -> str:
    """Check if position/momentum uncertainties satisfy Heisenberg's principle.

    CALCULATOR — Δx·Δp ≥ ℏ/2. Reports product, ratio to ℏ/2, and whether
    the proposed uncertainties are physically allowed.

    delta_x: position uncertainty in meters
    delta_p: momentum uncertainty in kg·m/s

    Example: calc_uncertainty_check(1e-10, 1e-24)
    → SATISFIED: Δx·Δp = 1e-34 ≥ ℏ/2
    """
    from noethersolve.qm_calculator import uncertainty_check
    return str(uncertainty_check(delta_x=delta_x, delta_p=delta_p))


@mcp.tool()
def calc_tunneling(
    E: float,
    V: float,
    L: float,
    m: float = 9.109e-31,
) -> str:
    """Compute quantum tunneling probability through a rectangular barrier.

    CALCULATOR — exact transmission coefficient using sinh formula for E < V,
    sin formula for E > V. Reports T, R, and WKB approximation.

    E: particle energy in eV
    V: barrier height in eV
    L: barrier width in meters
    m: particle mass in kg (default: electron mass)

    Example: calc_tunneling(5.0, 10.0, 1e-10)
    → T = 0.68, R = 0.32 (tunneling regime)
    """
    from noethersolve.qm_calculator import tunneling_probability
    return str(tunneling_probability(E=E, V=V, L=L, m=m))


@mcp.tool()
def calc_harmonic_oscillator_qm(
    n: int,
    omega: float,
    m: float = 9.109e-31,
) -> str:
    """Compute quantum harmonic oscillator energy levels.

    CALCULATOR — E_n = (n + ½)ℏω. Reports energy, zero-point energy,
    and classical turning point.

    n: quantum number (0, 1, 2, ...)
    omega: angular frequency in rad/s
    m: particle mass in kg (default: electron mass)

    Example: calc_harmonic_oscillator_qm(0, 1e14)
    → E_0 = ½ℏω = 0.033 eV (zero-point energy)
    """
    from noethersolve.qm_calculator import harmonic_oscillator
    return str(harmonic_oscillator(n=n, omega=omega, m=m))


@mcp.tool()
def calc_angular_momentum(
    j1: float,
    j2: float,
) -> str:
    """Compute allowed total angular momentum values from coupling j1 ⊗ j2.

    CALCULATOR — J ranges from |j1-j2| to j1+j2. Reports all allowed J values,
    state counting, and Clebsch-Gordan decomposition verification.

    j1: first angular momentum quantum number (integer or half-integer)
    j2: second angular momentum quantum number (integer or half-integer)

    Example: calc_angular_momentum(0.5, 0.5)
    → J = 0 (singlet) ⊕ J = 1 (triplet), 4 total states
    """
    from noethersolve.qm_calculator import angular_momentum_addition
    return str(angular_momentum_addition(j1=j1, j2=j2))


# ── Organic Chemistry Reaction Engine ──────────────────────────────────


@mcp.tool()
def analyze_molecule(smiles: str) -> str:
    """Analyze a molecule from SMILES — detect all functional groups, reactive sites, properties.

    COMPUTES functional groups via SMARTS pattern matching, identifies nucleophilic
    and electrophilic sites, leaving groups, acidic groups, stereocenters, and
    aromaticity. Use this FIRST when reasoning about any organic molecule's reactivity.

    smiles: SMILES string (e.g., "CCBr", "c1ccccc1", "CC(=O)OC")

    Example: analyze_molecule("OC(=O)c1ccc(C=O)cc1")
    → carboxylic acid (acid), aldehyde (electrophile), aromatic ring (nucleophile)
    """
    from noethersolve.reaction_engine import analyze_molecule as _analyze
    report = _analyze(smiles)
    return str(report)


@mcp.tool()
def predict_reaction_selectivity(
    nucleophile: str,
    electrophile: str,
    solvent: str = "DMSO",
) -> str:
    """Predict reaction rate and selectivity using Mayr's nucleophilicity/electrophilicity equation.

    COMPUTES log k = s_N × (N + E) from tabulated Mayr parameters for ~50
    nucleophiles and ~30 electrophiles. Predicts mechanism (SN1/SN2/E1/E2)
    and identifies competing pathways.

    nucleophile: name (e.g., "hydroxide", "cyanide", "water", "thiophenolate")
    electrophile: name (e.g., "methyl bromide", "acetaldehyde", "benzyl bromide")
    solvent: affects nucleophilicity ordering (e.g., "DMSO", "water", "MeOH")

    Example: predict_reaction_selectivity("cyanide", "methyl bromide", "DMSO")
    → log k = 5.7, VERY FAST, SN2 mechanism
    """
    from noethersolve.reaction_engine import predict_selectivity as _predict
    report = _predict(nucleophile, electrophile, solvent)
    return str(report)


@mcp.tool()
def predict_reaction_mechanism(
    reactants: list[str],
    reagents: list[str] = [],
    temperature: str = "",
    solvent: str = "",
    catalyst: str = "",
) -> str:
    """Predict the reaction mechanism for given reactants and conditions.

    COMPUTES the most likely mechanism using rule-based arrow-pushing logic.
    Returns step-by-step mechanism, predicted products, stereochemical outcome,
    atom/charge balance, and competing reactions.

    reactants: SMILES strings for reactants (e.g., ["C=CC=C", "C=CC(=O)C"])
    reagents: reagent names (e.g., ["NaOH"], ["LDA"], ["BH3"])
    temperature: "low", "high", "reflux", or specific temp
    solvent: e.g., "THF", "DMSO", "water", "ether"
    catalyst: e.g., "AlCl3", "Pd/C", "H2SO4"

    Example: predict_reaction_mechanism(["C=CC=C", "C=CC(=O)C"])
    → Diels-Alder [4+2], concerted, suprafacial, endo favored
    """
    from noethersolve.reaction_engine import predict_mechanism as _predict
    conditions = {}
    if temperature:
        conditions["temperature"] = temperature
    if solvent:
        conditions["solvent"] = solvent
    if catalyst:
        conditions["catalyst"] = catalyst
    report = _predict(reactants, reagents=reagents if reagents else None, conditions=conditions if conditions else None)
    return str(report)


@mcp.tool()
def validate_synthesis_pathway(
    steps: list[dict],
) -> str:
    """Validate a multi-step organic synthesis for feasibility and compatibility.

    CHECKS each step for: functional group compatibility, reagent compatibility,
    protecting group logic, oxidant selectivity, and atom conservation.
    Catches: Grignard + protic groups, Friedel-Crafts on deactivated rings,
    LiAlH4 over-reduction, strong oxidant + alkene cleavage.

    steps: list of dicts, each with:
        "substrate": SMILES of starting material
        "reagent": reagent name or SMILES
        "product": (optional) SMILES of expected product
        "conditions": (optional) reaction conditions string

    Example: validate_synthesis_pathway([
        {"substrate": "OC(=O)c1ccccc1", "reagent": "PhMgBr", "product": "?"}
    ]) → FAIL: Grignard incompatible with unprotected carboxylic acid
    """
    from noethersolve.reaction_engine import validate_synthesis as _validate
    report = _validate(steps)
    return str(report)


@mcp.tool()
def check_baldwin_rules(
    ring_size: int,
    endo_exo: str,
    geometry: str,
) -> str:
    """Check Baldwin's rules for ring closure feasibility.

    COMPUTES whether a proposed ring closure is favored or disfavored.
    Baldwin's rules predict which intramolecular ring closures succeed
    based on ring size, endo/exo, and bond hybridization.

    ring_size: 3-7 (atoms in the ring being formed)
    endo_exo: "endo" or "exo" (bond breaking inside or outside ring)
    geometry: "tet" (sp3), "trig" (sp2), or "dig" (sp)

    Example: check_baldwin_rules(5, "exo", "tet") → FAVORED (very common)
    Example: check_baldwin_rules(5, "endo", "tet") → DISFAVORED
    """
    from noethersolve.reaction_engine import check_baldwin as _check
    report = _check(ring_size, endo_exo, geometry)
    return str(report)


@mcp.tool()
def check_woodward_hoffmann(
    n_electrons: int,
    conditions: str = "thermal",
    reaction_type: str = "cycloaddition",
) -> str:
    """Check Woodward-Hoffmann rules for pericyclic reaction feasibility.

    COMPUTES whether a pericyclic reaction is symmetry-allowed or forbidden.
    Covers cycloadditions ([4+2] Diels-Alder, [2+2]), electrocyclic reactions
    (conrotatory vs disrotatory), and sigmatropic shifts.

    n_electrons: total pi electrons involved (e.g., 6 for Diels-Alder, 4 for [2+2])
    conditions: "thermal" or "photochemical"
    reaction_type: "cycloaddition", "electrocyclic", or "sigmatropic"

    Example: check_woodward_hoffmann(6, "thermal") → ALLOWED (Diels-Alder)
    Example: check_woodward_hoffmann(4, "thermal") → FORBIDDEN ([2+2] needs light)
    """
    from noethersolve.reaction_engine import check_woodward_hoffmann as _check
    report = _check(n_electrons, conditions, reaction_type)
    return str(report)


# ── Elliptic Curves ───────────────────────────────────────────────────

@mcp.tool()
def analyze_elliptic_curve(
    a: int,
    b: int,
    p: int,
) -> str:
    """Analyze an elliptic curve y² = x³ + ax + b over F_p.

    COMPUTES discriminant, j-invariant, point count, Hasse bounds verification,
    and sample points. Detects singular curves.

    a, b: curve coefficients
    p: prime field modulus

    Example: analyze_elliptic_curve(-1, 1, 23)
    → 23 points, Δ=-368, j=-300.52, Hasse satisfied
    """
    from noethersolve.elliptic_curves import analyze_curve
    report = analyze_curve(a, b, p)
    return str(report)


@mcp.tool()
def ec_point_arithmetic(
    a: int,
    b: int,
    p: int,
    P: list[int],
    Q: list[int],
) -> str:
    """Compute point arithmetic on elliptic curve y² = x³ + ax + b mod p.

    COMPUTES P + Q and 2P, verifies points are on curve.

    a, b, p: curve parameters
    P: first point [x, y] or null for point at infinity
    Q: second point [x, y] or null for point at infinity

    Example: ec_point_arithmetic(-1, 1, 23, [0, 1], [1, 1])
    → P + Q = (17, 3), 2P = (6, 2)
    """
    from noethersolve.elliptic_curves import analyze_point_arithmetic
    P_pt = tuple(P) if P else None
    Q_pt = tuple(Q) if Q else None
    report = analyze_point_arithmetic(a, b, p, P_pt, Q_pt)
    return str(report)


@mcp.tool()
def ec_scalar_mult(
    a: int,
    b: int,
    p: int,
    n: int,
    P: list[int],
) -> str:
    """Compute scalar multiplication nP on elliptic curve.

    Uses double-and-add algorithm. Returns nP and verifies result is on curve.

    a, b, p: curve parameters
    n: scalar multiplier
    P: point [x, y]

    Example: ec_scalar_mult(-1, 1, 23, 5, [0, 1])
    → 5P = (9, 13)
    """
    from noethersolve.elliptic_curves import scalar_mult, is_on_curve
    E = {"a": a, "b": b, "p": p}
    P_pt = tuple(P) if P else None
    result = scalar_mult(E, n, P_pt)
    on_curve = is_on_curve(E, result)
    if result is None:
        return f"{n}P = O (point at infinity)"
    return f"{n}P = {result}, on curve: {on_curve}"


@mcp.tool()
def ec_hasse_bounds(
    p: int,
) -> str:
    """Compute Hasse bounds for #E(F_p).

    Hasse's theorem: |#E(F_p) - (p + 1)| ≤ 2√p
    So: p + 1 - 2√p ≤ #E(F_p) ≤ p + 1 + 2√p

    p: prime modulus

    Example: ec_hasse_bounds(23) → [14, 34]
    """
    from noethersolve.elliptic_curves import hasse_bounds
    lo, hi = hasse_bounds(p)
    return f"Hasse bounds for p={p}: [{lo}, {hi}]\n  Expected count: ~{p + 1}\n  Max deviation: ~{int(2 * (p ** 0.5) + 1)}"


@mcp.tool()
def ec_discriminant(
    a: int,
    b: int,
) -> str:
    """Compute discriminant and j-invariant of y² = x³ + ax + b.

    Δ = -16(4a³ + 27b²)
    j = 1728 × 4a³ / (4a³ + 27b²)

    Curve is non-singular iff Δ ≠ 0.

    Example: ec_discriminant(-1, 1) → Δ=-368, j=-300.52
    """
    from noethersolve.elliptic_curves import discriminant, j_invariant, is_singular
    disc = discriminant(a, b)
    j_inv = j_invariant(a, b)
    sing = is_singular(a, b)
    lines = [
        f"Curve: y² = x³ + ({a})x + ({b})",
        f"Discriminant Δ = {disc}",
    ]
    if j_inv is not None:
        lines.append(f"j-invariant = {j_inv:.6f}")
    else:
        lines.append("j-invariant = undefined (singular curve)")
    lines.append(f"Singular: {sing}")
    return "\n".join(lines)


@mcp.tool()
def ec_point_order(
    a: int,
    b: int,
    p: int,
    P: list[int],
    max_order: int = 10000,
) -> str:
    """Compute the order of point P in E(F_p).

    The order is the smallest n > 0 such that nP = O (point at infinity).

    a, b, p: curve parameters
    P: point [x, y]
    max_order: search limit

    Example: ec_point_order(-1, 1, 23, [0, 1]) → order = 23
    """
    from noethersolve.elliptic_curves import point_order, is_on_curve
    E = {"a": a, "b": b, "p": p}
    P_pt = tuple(P) if P else None

    if not is_on_curve(E, P_pt):
        return f"ERROR: Point {P} is not on the curve"

    order = point_order(E, P_pt, max_order)
    if order is None:
        return f"Order of {P} exceeds {max_order}"
    return f"Order of {P} = {order}"


# ── Intersection Theory (Algebraic Geometry) ──────────────────────────

@mcp.tool()
def calc_bezout(
    degree_1: int,
    degree_2: int,
) -> str:
    """Compute intersection count by Bezout's theorem.

    Two plane curves of degrees d1 and d2 with no common component
    intersect in EXACTLY d1 × d2 points, counted with multiplicity.

    degree_1: Degree of first curve
    degree_2: Degree of second curve

    Example: calc_bezout(2, 3) → conic and cubic intersect in 6 points
    """
    from noethersolve.intersection_theory import bezout_intersection
    report = bezout_intersection(degree_1, degree_2)
    return str(report)


@mcp.tool()
def calc_genus_degree(
    degree: int,
) -> str:
    """Compute genus of smooth plane curve by genus-degree formula.

    For a smooth plane curve of degree d: g = (d-1)(d-2)/2

    degree: Degree of the smooth plane curve

    Example: calc_genus_degree(3) → genus 1 (elliptic curve)
    Example: calc_genus_degree(4) → genus 3
    """
    from noethersolve.intersection_theory import genus_degree_formula
    report = genus_degree_formula(degree)
    return str(report)


@mcp.tool()
def calc_self_intersection(
    surface: str,
    divisor: str = "line",
) -> str:
    """Compute self-intersection numbers on surfaces.

    surface: "P2" (projective plane), "blowup" (blow-up of P² at point)
    divisor: "line" (L·L=1 in P²), "exceptional" (E·E=-1 on blow-up)

    Example: calc_self_intersection("P2", "line") → L·L = 1
    Example: calc_self_intersection("blowup", "exceptional") → E·E = -1
    """
    from noethersolve.intersection_theory import (
        self_intersection_line_P2,
        self_intersection_exceptional,
    )
    if surface.lower() in ("p2", "p^2", "projective"):
        report = self_intersection_line_P2()
    elif surface.lower() in ("blowup", "blow-up", "bl"):
        report = self_intersection_exceptional()
    else:
        return f"Unknown surface: {surface}. Use 'P2' or 'blowup'."
    return str(report)


@mcp.tool()
def calc_canonical_divisor(
    surface: str,
    n_blowups: int = 0,
) -> str:
    """Compute canonical divisor on algebraic surfaces.

    surface: "P2" (K=-3H), "cubic" (K²=3), "del_pezzo" (K²=9-n)
    n_blowups: For del Pezzo, number of points blown up (0-8)

    Example: calc_canonical_divisor("P2") → K_P² = -3H, Fano
    Example: calc_canonical_divisor("cubic") → K² = 3 (del Pezzo degree 3)
    Example: calc_canonical_divisor("del_pezzo", 6) → K² = 3
    """
    from noethersolve.intersection_theory import (
        canonical_P2,
        canonical_cubic_surface,
        del_pezzo_degree,
    )
    s = surface.lower()
    if s in ("p2", "p^2", "projective"):
        report = canonical_P2()
    elif s in ("cubic", "cubic_surface"):
        report = canonical_cubic_surface()
    elif s in ("del_pezzo", "delpezzo", "dp"):
        report = del_pezzo_degree(n_blowups)
    else:
        return f"Unknown surface: {surface}. Use 'P2', 'cubic', or 'del_pezzo'."
    return str(report)


@mcp.tool()
def calc_noether_formula(
    c1_squared: int,
    c2: int,
) -> str:
    """Check the Noether formula: c₁² + c₂ = 12χ(O_S).

    For a smooth complex surface S:
    - c₁² = K² (self-intersection of canonical)
    - c₂ = e(S) (topological Euler characteristic)
    - χ(O_S) = 1 - q + p_g (holomorphic Euler characteristic)

    c1_squared: c₁² = K² for the surface
    c2: c₂ = topological Euler characteristic

    Example: calc_noether_formula(9, 3) → P²: χ = 1
    Example: calc_noether_formula(3, 9) → Cubic surface: χ = 1
    """
    from noethersolve.intersection_theory import noether_formula
    report = noether_formula(c1_squared, c2)
    return str(report)


@mcp.tool()
def calc_enumerative(
    problem: str,
    degree: int = 1,
) -> str:
    """Compute classical enumerative geometry results.

    problem: One of:
        "lines_cubic" - 27 lines on cubic surface
        "bitangents" - 28 bitangents to plane quartic
        "conics_5pts" - 1 conic through 5 general points
        "lines_4lines" - 2 lines meeting 4 general lines in P³
        "cubics_9pts" - 1 cubic through 9 general points
        "quintic" - rational curves on quintic threefold (use degree=1,2,3)
    degree: For quintic, degree of rational curves (1=2875, 2=609250, 3=317206375)

    Example: calc_enumerative("lines_cubic") → 27 lines, W(E6) symmetry
    Example: calc_enumerative("quintic", 2) → 609250 conics
    """
    from noethersolve.intersection_theory import (
        lines_on_cubic_surface,
        bitangents_to_quartic,
        conics_through_5_points,
        lines_meeting_4_general_lines_P3,
        plane_cubics_through_9_points,
        rational_curves_on_quintic_threefold,
    )
    p = problem.lower().replace("_", "").replace("-", "").replace(" ", "")
    if p in ("linescubic", "27lines", "cubliclines"):
        report = lines_on_cubic_surface()
    elif p in ("bitangents", "28bitangents", "quarticbitangents"):
        report = bitangents_to_quartic()
    elif p in ("conics5pts", "conic5points", "5pointsconic"):
        report = conics_through_5_points()
    elif p in ("lines4lines", "4lineslines", "4generallines"):
        report = lines_meeting_4_general_lines_P3()
    elif p in ("cubics9pts", "cubic9points", "9pointscubic"):
        report = plane_cubics_through_9_points()
    elif p in ("quintic", "quinticthreefold", "rationalcurves"):
        report = rational_curves_on_quintic_threefold(degree)
    else:
        return (f"Unknown problem: {problem}. Use: lines_cubic, bitangents, "
                "conics_5pts, lines_4lines, cubics_9pts, quintic")
    return str(report)


@mcp.tool()
def calc_adjunction(
    ambient_dim: int,
    divisor_degree: int,
    ambient_type: str = "Pn",
) -> str:
    """Apply the adjunction formula to compute K_D for a divisor D in X.

    CRITICAL: K_D = (K_X + D)|_D

    For smooth degree d curve in P²:
        K_C = (d-3)H|_C, genus g = (d-1)(d-2)/2

    ambient_dim: Dimension of ambient variety (e.g., 2 for P²)
    divisor_degree: Degree of the divisor
    ambient_type: "Pn" for projective space

    Example: calc_adjunction(2, 3) → cubic has genus 1
    Example: calc_adjunction(4, 5) → quintic 3-fold is Calabi-Yau (K=0)
    """
    from noethersolve.intersection_theory import adjunction_formula
    report = adjunction_formula(ambient_dim, divisor_degree, ambient_type)
    return str(report)


@mcp.tool()
def calc_adjunction_ci(
    degrees: list[int],
    ambient_dim: int,
) -> str:
    """Adjunction for complete intersection in P^n.

    For V = V(f_1, ..., f_k) ⊂ P^n:
        K_V = (Σd_i - n - 1)H|_V

    degrees: List of hypersurface degrees [d_1, ..., d_k]
    ambient_dim: n for P^n

    Example: calc_adjunction_ci([2, 2], 3) → CI(2,2) in P³ has genus 1
    Example: calc_adjunction_ci([2, 3], 4) → CI(2,3) in P⁴ is Calabi-Yau
    """
    from noethersolve.intersection_theory import adjunction_complete_intersection
    report = adjunction_complete_intersection(degrees, ambient_dim)
    return str(report)


@mcp.tool()
def calc_blowup_K_squared(
    original_K_sq: int,
    n_points: int,
) -> str:
    """Compute K² after blowing up n points.

    CRITICAL: K²_X̃ = K²_X - n (K² DECREASES by 1 per point!)

    LLMs often incorrectly say K² increases or stays the same.
    Each blow-up adds one (-1)-curve (exceptional divisor with E²=-1).

    original_K_sq: K² of original surface
    n_points: Number of points blown up

    Example: calc_blowup_K_squared(9, 1) → K² = 8 (P² blown up once)
    Example: calc_blowup_K_squared(9, 6) → K² = 3 (del Pezzo degree 3)
    """
    from noethersolve.intersection_theory import blowup_K_squared
    report = blowup_K_squared(original_K_sq, n_points)
    return str(report)


@mcp.tool()
def calc_blowup_P2(n_points: int) -> str:
    """Blow up P² at n points in general position.

    P² has K² = 9. Results:
    - n ≤ 8: del Pezzo surface of degree 9-n (Fano)
    - n = 9: rational elliptic surface (K² = 0)
    - n > 9: K² < 0, -K big but not ample

    n_points: Number of points to blow up

    Example: calc_blowup_P2(6) → del Pezzo degree 3 (cubic surface)
    Example: calc_blowup_P2(8) → del Pezzo degree 1
    Example: calc_blowup_P2(9) → K² = 0, rational elliptic
    """
    from noethersolve.intersection_theory import blowup_P2
    report = blowup_P2(n_points)
    return str(report)


@mcp.tool()
def calc_ruled_surface(
    base_genus: int,
    invariant_e: int = 0,
) -> str:
    """Compute intersection theory on ruled surface P(E) over curve C.

    CRITICAL: K² = 8(1 - g) ALWAYS, regardless of invariant e.

    Pic(S) = Zh ⊕ Zf, where h·f = 1, f² = 0, h² = -e.
    Hirzebruch surfaces F_n have g=0, e=n.

    base_genus: Genus g of base curve C
    invariant_e: The invariant e (related to bundle splitting)

    Example: calc_ruled_surface(0, 2) → F_2 with (-2)-section
    Example: calc_ruled_surface(1, 0) → K² = 0 (elliptic ruled)
    """
    from noethersolve.intersection_theory import ruled_surface
    report = ruled_surface(base_genus, invariant_e)
    return str(report)


@mcp.tool()
def calc_hirzebruch(n: int) -> str:
    """The Hirzebruch surface F_n = P(O ⊕ O(-n)) over P¹.

    F_0 = P¹ × P¹
    F_1 = Bl_p(P²) (not minimal)
    F_n (n ≥ 2) has a (-n)-curve (negative section)

    All have K² = 8, Picard rank 2.
    Fano iff n ≤ 2.

    n: The Hirzebruch index (n ≥ 0)

    Example: calc_hirzebruch(0) → P¹ × P¹
    Example: calc_hirzebruch(1) → Bl_p(P²)
    """
    from noethersolve.intersection_theory import hirzebruch_surface
    report = hirzebruch_surface(n)
    return str(report)


@mcp.tool()
def calc_toric_canonical(variety_name: str) -> str:
    """Compute canonical class for toric varieties.

    CRITICAL: For complete smooth toric variety X with fan Σ:
        K_X = -Σ_ρ D_ρ (sum over rays)
    K is ALWAYS anti-effective (never effective) on complete toric varieties.

    variety_name: One of:
        "P1", "P2", "P3", "P4" (projective space)
        "P1xP1" (product)
        "F0", "F1", "F2", ... (Hirzebruch surfaces)
        "weighted_P" (weighted projective)

    Example: calc_toric_canonical("P2") → K = -3H, Fano
    Example: calc_toric_canonical("F3") → NOT Fano (n > 2)
    """
    from noethersolve.intersection_theory import toric_canonical
    report = toric_canonical(variety_name)
    return str(report)


# ── Information Theory ─────────────────────────────────────────────────

@mcp.tool()
def calc_channel_capacity_bsc(p: float) -> str:
    """Compute Binary Symmetric Channel capacity.

    EXACT formula: C = 1 - H(p) where H is binary entropy.
    NOT C = 1 - p (common error).

    p: Crossover probability (0 to 0.5)

    Example: calc_channel_capacity_bsc(0.1) → C = 0.531 bits
    """
    from noethersolve.information_theory import capacity_bsc
    report = capacity_bsc(p)
    return str(report)


@mcp.tool()
def calc_channel_capacity_bec(epsilon: float) -> str:
    """Compute Binary Erasure Channel capacity.

    EXACT formula: C = 1 - ε (simpler than BSC!).
    BEC capacity is LINEAR in (1-ε), unlike BSC.

    epsilon: Erasure probability

    Example: calc_channel_capacity_bec(0.3) → C = 0.7 bits
    """
    from noethersolve.information_theory import capacity_bec
    report = capacity_bec(epsilon)
    return str(report)


@mcp.tool()
def calc_channel_capacity_awgn(
    snr: float,
    bandwidth: float = 1.0,
) -> str:
    """Compute AWGN channel capacity (Shannon's formula).

    EXACT formula: C = B × log₂(1 + SNR).
    NOT C = B × SNR (common error).

    snr: Signal-to-noise ratio (linear, not dB)
    bandwidth: Channel bandwidth in Hz (default 1 for normalized)

    Example: calc_channel_capacity_awgn(10) → C = 3.46 bits/use
    """
    from noethersolve.information_theory import capacity_awgn
    report = capacity_awgn(snr, bandwidth)
    return str(report)


@mcp.tool()
def calc_channel_capacity_z(p: float) -> str:
    """Compute Z-channel capacity.

    Z-channel: 0→0 always, 1→0 with probability p.
    Optimal input is NOT uniform (biased toward reliable 0).

    p: Probability of 1→0 transition

    Example: calc_channel_capacity_z(0.2) → C with P(X=1) < 0.5
    """
    from noethersolve.information_theory import capacity_z_channel
    report = capacity_z_channel(p)
    return str(report)


@mcp.tool()
def calc_rate_distortion(
    source: str,
    D: float,
    variance: float = 1.0,
) -> str:
    """Compute rate-distortion function R(D).

    source: "binary" (Hamming) or "gaussian" (MSE)
    D: Target distortion
    variance: Source variance (for Gaussian)

    Example: calc_rate_distortion("binary", 0.1) → R(0.1) ≈ 0.531 bits
    Example: calc_rate_distortion("gaussian", 0.25) → R(0.25) = 1 bit
    """
    from noethersolve.information_theory import rate_distortion_binary, rate_distortion_gaussian
    if source.lower() in ("binary", "hamming"):
        report = rate_distortion_binary(D)
    elif source.lower() in ("gaussian", "mse"):
        report = rate_distortion_gaussian(D, variance)
    else:
        return f"Unknown source: {source}. Use 'binary' or 'gaussian'."
    return str(report)


@mcp.tool()
def calc_mac_region(
    I_X1_Y: float,
    I_X2_Y: float,
    I_X1X2_Y: float,
) -> str:
    """Compute 2-user MAC capacity region.

    The region is a PENTAGON (common error: models say rectangle).
    Bounded by R₁ ≤ I(X₁;Y|X₂), R₂ ≤ I(X₂;Y|X₁), R₁+R₂ ≤ I(X₁,X₂;Y).

    I_X1_Y: I(X₁; Y | X₂) — user 1's conditional rate
    I_X2_Y: I(X₂; Y | X₁) — user 2's conditional rate
    I_X1X2_Y: I(X₁, X₂; Y) — sum rate

    Example: calc_mac_region(1.0, 1.0, 1.5) → pentagon with sum-rate 1.5
    """
    from noethersolve.information_theory import mac_capacity_region_2user
    report = mac_capacity_region_2user(I_X1_Y, I_X2_Y, I_X1X2_Y)
    return str(report)


# ── Drug Interactions ─────────────────────────────────────────────────

@mcp.tool()
def check_drug_interaction(drug_a: str, drug_b: str) -> str:
    """Check for drug-drug interaction between two medications.

    COMPUTES CYP450-mediated interactions including inhibition and induction.
    Returns mechanism, severity, AUC change, and clinical recommendations.

    drug_a: First drug name (generic, case-insensitive)
    drug_b: Second drug name (generic, case-insensitive)

    Example: check_drug_interaction("ketoconazole", "midazolam")
    → Strong CYP3A4 inhibition, AUC ↑10-15×, CONTRAINDICATED
    """
    from noethersolve.drug_interactions import check_interaction
    report = check_interaction(drug_a, drug_b)
    return str(report)


@mcp.tool()
def get_drug_cyp_profile(drug: str) -> str:
    """Get the CYP450 metabolic profile of a drug.

    RETURNS which CYP enzymes metabolize the drug, whether it's a
    sensitive substrate (narrow therapeutic index), whether it's a
    prodrug, and if it inhibits or induces any CYP enzymes.

    drug: Drug name (generic, case-insensitive)

    Example: get_drug_cyp_profile("codeine")
    → CYP2D6 substrate, PRODRUG (requires 2D6 for activation to morphine)
    """
    from noethersolve.drug_interactions import get_drug_profile
    report = get_drug_profile(drug)
    return str(report)


@mcp.tool()
def get_cyp_enzyme_info(enzyme: str) -> str:
    """Get information about a CYP enzyme including substrates and interactors.

    RETURNS all known substrates, sensitive substrates, strong/moderate
    inhibitors, and strong inducers for the specified CYP enzyme.

    enzyme: CYP enzyme name (e.g., "CYP3A4", "CYP2D6", or just "3A4")

    Example: get_cyp_enzyme_info("CYP3A4")
    → Substrates: midazolam, simvastatin, ... Inhibitors: ketoconazole, ...
    """
    from noethersolve.drug_interactions import get_cyp_info
    report = get_cyp_info(enzyme)
    return str(report)


@mcp.tool()
def check_pharmacogenomics(drug: str) -> str:
    """Check pharmacogenomic considerations for a drug.

    RETURNS relevant CYP enzyme phenotypes (PM/IM/NM/UM), their clinical
    impacts, and recommendations. Critical for prodrugs (codeine, clopidogrel)
    and narrow therapeutic index drugs (warfarin).

    drug: Drug name (generic, case-insensitive)

    Example: check_pharmacogenomics("codeine")
    → CYP2D6 PM: no analgesia; UM: TOXICITY RISK (FDA boxed warning)
    """
    from noethersolve.drug_interactions import check_pharmacogenomics as pgx
    report = pgx(drug)
    return str(report)


@mcp.tool()
def predict_ddi_auc_change(perpetrator: str, victim: str) -> str:
    """Predict AUC fold-change from a drug-drug interaction.

    COMPUTES the expected change in AUC (area under curve) when a
    perpetrator drug (inhibitor/inducer) affects a victim drug (substrate).

    perpetrator: Drug that affects metabolism (inhibitor or inducer)
    victim: Drug whose levels are affected (substrate)

    Example: predict_ddi_auc_change("rifampin", "midazolam")
    → AUC 0.05-0.2× (80-95% decrease due to strong CYP3A4 induction)
    """
    from noethersolve.drug_interactions import predict_auc_change
    import json
    result = predict_auc_change(perpetrator, victim)
    # Format nicely
    if result["auc_low"] == 1.0 and result["auc_high"] == 1.0:
        return "No known interaction: AUC unchanged (1.0×)"
    elif result["auc_low"] >= 1.0:
        return (f"AUC increase: {result['auc_low']:.1f}-{result['auc_high']:.1f}× "
                f"({result['interaction_type']}, {result['enzyme']})\n"
                f"Mechanism: {result.get('mechanism', 'N/A')}\n"
                f"Recommendation: {result.get('recommendation', 'N/A')}")
    else:
        pct_lo = (1 - result['auc_high']) * 100
        pct_hi = (1 - result['auc_low']) * 100
        return (f"AUC decrease: {pct_lo:.0f}-{pct_hi:.0f}% "
                f"({result['interaction_type']}, {result['enzyme']})\n"
                f"Mechanism: {result.get('mechanism', 'N/A')}\n"
                f"Recommendation: {result.get('recommendation', 'N/A')}")


# ── Epidemiology ─────────────────────────────────────────────────────

@mcp.tool()
def calc_herd_immunity(R0: float) -> str:
    """Calculate herd immunity threshold from basic reproduction number.

    COMPUTES HIT = 1 - 1/R₀ (the fraction that must be immune to stop
    transmission). Reports threshold percentage, susceptible at equilibrium,
    and R0-specific context.

    R0: Basic reproduction number (average secondary infections from one case
        in a fully susceptible population)

    Example: calc_herd_immunity(15.0)
    → Measles (R0=15): HIT = 93.3%, only 6.7% can remain susceptible
    """
    from noethersolve.epidemiology import herd_immunity_threshold
    report = herd_immunity_threshold(R0=R0)
    return str(report)


@mcp.tool()
def calc_reproduction_number(
    R0: float = 0.0,
    susceptible_fraction: float = 1.0,
    beta: float = 0.0,
    gamma: float = 0.0,
) -> str:
    """Calculate basic or effective reproduction number.

    COMPUTES R₀ = β/γ (from transmission/recovery rates) or
    Rt = R₀ × S (effective R given partial immunity).

    R0: Basic reproduction number (provide directly, or compute from β/γ)
    susceptible_fraction: Fraction of population still susceptible (0-1)
    beta: Transmission rate (infections per contact per time)
    gamma: Recovery rate (1/infectious period)

    Example: calc_reproduction_number(R0=3.0, susceptible_fraction=0.5)
    → Rt = 1.5, epidemic still GROWING (Rt > 1)
    """
    from noethersolve.epidemiology import reproduction_number
    report = reproduction_number(
        R0=R0 if R0 > 0 else None,
        susceptible_fraction=susceptible_fraction if susceptible_fraction < 1.0 else None,
        beta=beta if beta > 0 else None,
        gamma=gamma if gamma > 0 else None,
    )
    return str(report)


@mcp.tool()
def calc_doubling_time(
    growth_rate: float = 0.0,
    R0: float = 0.0,
    generation_time: float = 0.0,
) -> str:
    """Calculate epidemic doubling time from growth rate or R0.

    COMPUTES T_d = ln(2)/r where r is the exponential growth rate.
    Can derive r from R0 and generation time: r = ln(R0)/T_g.

    growth_rate: Exponential growth rate (per time unit)
    R0: Basic reproduction number (alternative to growth_rate)
    generation_time: Mean generation interval in same time units as growth_rate

    Example: calc_doubling_time(R0=2.0, generation_time=5.0)
    → Doubling time = 5.0 days (cases double every 5 days)
    """
    from noethersolve.epidemiology import doubling_time
    report = doubling_time(
        growth_rate=growth_rate if growth_rate != 0 else None,
        R0=R0 if R0 > 0 else None,
        generation_time=generation_time if generation_time > 0 else None,
    )
    return str(report)


@mcp.tool()
def calc_attack_rate(R0: float) -> str:
    """Calculate final attack rate (total infected) from R0.

    COMPUTES the final epidemic size by solving the transcendental final
    size equation: S_∞ = exp(-R₀ × (1 - S_∞)). This is the exact solution,
    not an approximation.

    R0: Basic reproduction number

    Example: calc_attack_rate(2.5)
    → 89.3% final attack rate (10.7% escape infection entirely)
    """
    from noethersolve.epidemiology import attack_rate
    report = attack_rate(R0=R0)
    return str(report)


@mcp.tool()
def calc_sir_model(
    beta: float,
    gamma: float,
    initial_infected: float = 0.001,
) -> str:
    """Analyze SIR model parameters and derive epidemic characteristics.

    COMPUTES R₀, generation time, herd immunity threshold, peak infected
    fraction, and epidemic duration estimates from transmission parameters.

    beta: Transmission rate (infections per contact per time)
    gamma: Recovery rate (1/infectious period)
    initial_infected: Initial fraction infected (default 0.001 = 0.1%)

    Example: calc_sir_model(beta=0.4, gamma=0.1)
    → R0 = 4.0, generation time = 10 days, HIT = 75%, peak = 29%
    """
    from noethersolve.epidemiology import sir_model
    report = sir_model(beta=beta, gamma=gamma, initial_infected_fraction=initial_infected)
    return str(report)


@mcp.tool()
def calc_vaccine_impact(
    R0: float,
    vaccine_efficacy: float,
    coverage: float,
) -> str:
    """Calculate impact of vaccination on epidemic potential.

    COMPUTES effective reproduction number after vaccination:
    Rt = R₀ × (1 - VE × coverage), and whether herd immunity is achieved.

    R0: Basic reproduction number
    vaccine_efficacy: Vaccine efficacy against infection (0-1, e.g., 0.9 = 90%)
    coverage: Fraction of population vaccinated (0-1)

    Example: calc_vaccine_impact(R0=3.0, vaccine_efficacy=0.9, coverage=0.75)
    → Rt = 0.975 < 1, herd immunity ACHIEVED (67.5% effective immunity)
    """
    from noethersolve.epidemiology import vaccine_impact
    report = vaccine_impact(R0=R0, vaccine_efficacy=vaccine_efficacy, coverage=coverage)
    return str(report)


@mcp.tool()
def calc_generation_interval(
    mean_generation: float,
    mean_serial: float = 0.0,
) -> str:
    """Analyze generation and serial intervals for an infectious disease.

    COMPUTES the relationship between generation interval (infection to
    infection) and serial interval (symptom to symptom). Serial < generation
    indicates presymptomatic transmission.

    mean_generation: Mean generation interval (days)
    mean_serial: Mean serial interval (days, optional)

    Example: calc_generation_interval(mean_generation=5.0, mean_serial=4.0)
    → Serial < generation by 1 day → significant presymptomatic transmission
    """
    from noethersolve.epidemiology import generation_interval
    report = generation_interval(
        mean_generation=mean_generation,
        mean_serial=mean_serial if mean_serial > 0 else None,
    )
    return str(report)


@mcp.tool()
def get_disease_r0(disease: str) -> str:
    """Get reference R0 range for a known infectious disease.

    RETURNS published R0 estimates for common diseases. Use these as
    inputs to other epidemiology tools.

    disease: Disease name (e.g., "measles", "covid19_omicron", "influenza_seasonal")

    Example: get_disease_r0("measles") → R0 = 12-18
    """
    from noethersolve.epidemiology import get_disease_R0, list_diseases
    r0_range = get_disease_R0(disease)
    if r0_range is None:
        diseases = list_diseases()
        return f"Unknown disease '{disease}'. Known diseases: {', '.join(sorted(diseases))}"
    return f"{disease}: R₀ = {r0_range[0]}-{r0_range[1]}"


# ── Radiative Transfer / Climate Physics ─────────────────────────────

@mcp.tool()
def calc_co2_forcing(
    co2_final: float,
    co2_initial: float = 280.0,
) -> str:
    """Calculate radiative forcing from CO2 concentration change.

    COMPUTES ΔF = 5.35 × ln(C/C₀) W/m². KEY POINT: The relationship is
    LOGARITHMIC, not linear. Each doubling adds ~3.7 W/m² regardless of
    absolute concentration. This is one of the most common errors LLMs make.

    co2_final: Final CO2 concentration in ppm
    co2_initial: Initial CO2 in ppm (default 280 = preindustrial)

    Example: calc_co2_forcing(560, 280)
    → Doubling CO2: ΔF = 3.7 W/m² (LOGARITHMIC, not linear!)
    """
    from noethersolve.radiative_transfer import radiative_forcing
    report = radiative_forcing(co2_final=co2_final, co2_initial=co2_initial)
    return str(report)


@mcp.tool()
def calc_planck_response(emission_temperature: float = 255.0) -> str:
    """Calculate the Planck (no-feedback) climate response.

    COMPUTES λ₀ = 1/(4σT³) from Stefan-Boltzmann derivative. At Earth's
    effective emission temperature (~255 K), this gives ~1.2 K warming per
    CO2 doubling WITHOUT feedbacks. This is the MINIMUM warming.

    emission_temperature: Effective emission temperature in K (default 255)

    Example: calc_planck_response(255)
    → No-feedback warming: 1.2 K per doubling (actual: 2.5-4.0 K with feedbacks)
    """
    from noethersolve.radiative_transfer import planck_response
    report = planck_response(emission_temperature=emission_temperature)
    return str(report)


@mcp.tool()
def calc_climate_sensitivity(
    ecs: float = 0.0,
    include_cloud: bool = True,
) -> str:
    """Calculate equilibrium climate sensitivity from feedback analysis.

    COMPUTES ECS = ΔF₂ₓ / (-Σfeedbacks). IPCC AR6 likely range: 2.5-4.0 K.
    Includes Planck, water vapor, lapse rate, surface albedo, and (optionally)
    cloud feedbacks. Cloud feedback is the LARGEST source of uncertainty.

    ecs: If provided, derive feedback parameter from this ECS value
    include_cloud: Whether to include cloud feedback (default True)

    Example: calc_climate_sensitivity()
    → ECS ≈ 3.0 K per CO2 doubling (with all feedbacks)
    """
    from noethersolve.radiative_transfer import climate_sensitivity
    if ecs > 0:
        report = climate_sensitivity(ecs=ecs)
    else:
        feedbacks = ["planck", "water_vapor", "lapse_rate", "surface_albedo"]
        if include_cloud:
            feedbacks.append("cloud")
        report = climate_sensitivity(include_feedbacks=feedbacks)
    return str(report)


@mcp.tool()
def calc_stefan_boltzmann(
    temperature: float,
    emissivity: float = 1.0,
) -> str:
    """Calculate blackbody radiation power density.

    COMPUTES P = εσT⁴. Exact Stefan-Boltzmann law.

    temperature: Temperature in Kelvin
    emissivity: Surface emissivity 0-1 (default 1 = perfect blackbody)

    Example: calc_stefan_boltzmann(5778) → Sun surface: 6.3×10⁷ W/m²
    Example: calc_stefan_boltzmann(255) → Earth emission: ~240 W/m²
    """
    from noethersolve.radiative_transfer import stefan_boltzmann
    report = stefan_boltzmann(temperature=temperature, emissivity=emissivity)
    return str(report)


@mcp.tool()
def calc_greenhouse_effect(
    albedo: float = 0.30,
    actual_surface_temp: float = 288.0,
) -> str:
    """Calculate effective temperature and greenhouse effect.

    COMPUTES T_eff from energy balance and compares to actual surface temp.
    Earth's greenhouse effect is ~33 K warming above the no-atmosphere case.

    albedo: Bond albedo (default 0.30 for Earth)
    actual_surface_temp: Actual average surface temperature in K (default 288)

    Example: calc_greenhouse_effect()
    → T_eff = 255 K, actual = 288 K, greenhouse effect = 33 K
    """
    from noethersolve.radiative_transfer import effective_temperature
    report = effective_temperature(albedo=albedo, actual_surface_temp=actual_surface_temp)
    return str(report)


@mcp.tool()
def analyze_climate_feedback(name: str) -> str:
    """Analyze a specific climate feedback mechanism.

    RETURNS feedback value, sign (positive=amplifying, negative=damping),
    uncertainty range, and physical mechanism.

    name: Feedback name (planck, water_vapor, lapse_rate, surface_albedo, cloud)

    Example: analyze_climate_feedback("cloud")
    → +0.5 W/(m²·K), positive, LARGEST UNCERTAINTY (±0.7)
    """
    from noethersolve.radiative_transfer import analyze_feedback
    try:
        report = analyze_feedback(name)
        return str(report)
    except ValueError as e:
        from noethersolve.radiative_transfer import list_feedbacks
        return f"{e}\nAvailable feedbacks: {', '.join(list_feedbacks())}"


# ── Turbulence Scaling ───────────────────────────────────────────────

@mcp.tool()
def calc_kolmogorov_45_law(
    separation: float,
    energy_dissipation: float,
) -> str:
    """Calculate third-order structure function using EXACT 4/5 law.

    COMPUTES S₃(r) = ⟨(Δu)³⟩ = -(4/5)εr EXACTLY. This is the ONLY exact
    scaling law in turbulence — derived rigorously from Navier-Stokes,
    NOT from dimensional analysis like the -5/3 spectrum. Common LLM error:
    confusing the exact 4/5 law with the approximate -5/3 spectrum.

    separation: Scale r in the inertial range
    energy_dissipation: Mean dissipation rate ε (m²/s³)

    Example: calc_kolmogorov_45_law(0.01, 0.1)
    → S₃ = -0.0008 (EXACT, negative = forward energy cascade)
    """
    from noethersolve.turbulence import kolmogorov_45_law
    report = kolmogorov_45_law(separation=separation, energy_dissipation=energy_dissipation)
    return str(report)


@mcp.tool()
def calc_energy_spectrum(
    wavenumber: float,
    energy_dissipation: float,
    intermittency_model: str = "",
) -> str:
    """Calculate Kolmogorov energy spectrum E(k).

    COMPUTES E(k) = C_K × ε^(2/3) × k^(-5/3). Unlike the 4/5 law, this is
    APPROXIMATE (dimensional analysis only). Intermittency corrections
    modify the exponent slightly.

    wavenumber: Wavenumber k (1/m)
    energy_dissipation: Mean dissipation rate ε (m²/s³)
    intermittency_model: "she_leveque", "k62", or empty for none

    Example: calc_energy_spectrum(100, 0.1)
    → E(k) ≈ 0.007, exponent -5/3 (APPROXIMATE, not exact!)
    """
    from noethersolve.turbulence import energy_spectrum
    model = intermittency_model if intermittency_model else None
    report = energy_spectrum(wavenumber=wavenumber, energy_dissipation=energy_dissipation,
                            intermittency_model=model)
    return str(report)


@mcp.tool()
def calc_turbulent_scales(
    integral_scale: float,
    urms: float,
    kinematic_viscosity: float,
) -> str:
    """Calculate turbulent length scales (Kolmogorov, Taylor, integral).

    COMPUTES all three fundamental scales:
    - Kolmogorov η = (ν³/ε)^(1/4) — smallest eddies, viscous dissipation
    - Taylor λ — intermediate scale, velocity gradients
    - Integral L — largest eddies, energy input

    integral_scale: Largest eddy size L (m)
    urms: RMS velocity fluctuation u' (m/s)
    kinematic_viscosity: ν (m²/s)

    Example: calc_turbulent_scales(1.0, 1.0, 1e-5)
    → η ~ 10⁻⁴ m, λ ~ 10⁻² m, L = 1 m, Re ~ 10⁵
    """
    from noethersolve.turbulence import length_scales
    report = length_scales(integral_scale=integral_scale, urms=urms,
                          kinematic_viscosity=kinematic_viscosity)
    return str(report)


@mcp.tool()
def calc_structure_exponent(
    order: int,
    model: str = "she_leveque",
) -> str:
    """Calculate structure function scaling exponent with intermittency.

    COMPUTES ζ_p for S_p(r) = ⟨|Δu|^p⟩ ~ r^ζ_p. K41 predicts ζ_p = p/3.
    Intermittency causes deviations, especially at high orders.
    KEY: ζ₃ = 1 EXACTLY (from 4/5 law), unaffected by intermittency!

    order: Moment order p (1, 2, 3, ...)
    model: "k41", "she_leveque", "k62", or "beta_model"

    Example: calc_structure_exponent(6, "she_leveque")
    → ζ₆ = 1.78 (vs K41 prediction of 2.0 — intermittency effect)
    """
    from noethersolve.turbulence import structure_function_exponent
    report = structure_function_exponent(order=order, model=model)
    return str(report)


@mcp.tool()
def analyze_intermittency(model: str = "she_leveque") -> str:
    """Analyze intermittency corrections to turbulence scaling.

    RETURNS model parameters and predicted scaling exponents. Intermittency
    refers to the burstiness of turbulent dissipation, causing deviations
    from Kolmogorov 1941 predictions.

    model: "she_leveque", "k62", or "beta_model"

    Example: analyze_intermittency("she_leveque")
    → ζ₂=0.70, ζ₃=1.00 (exact!), ζ₄=1.28, ζ₆=1.78
    """
    from noethersolve.turbulence import intermittency_analysis
    report = intermittency_analysis(model=model)
    return str(report)


# ── Topological Invariants ────────────────────────────────────────────

@mcp.tool()
def calc_chern_number(
    band_index: int = 1,
    system: str = "quantum_hall",
) -> str:
    """Calculate the Chern number for a topological band structure.

    KEY POINT: Chern numbers are EXACTLY INTEGERS — this is not an approximation!
    They are topological invariants that cannot change without closing the band gap.
    Each filled Landau level contributes C = 1 to the Hall conductance.

    band_index: Which band to analyze (default 1)
    system: "quantum_hall", "haldane", or "chern_insulator"

    Example: calc_chern_number(system="haldane")
    → C = 1 (EXACTLY integer, protected by gap)
    """
    from noethersolve.topological_invariants import chern_number
    report = chern_number(band_index=band_index, system=system)
    return str(report)


@mcp.tool()
def calc_z2_invariant(
    nu: int,
    dimension: int = 2,
) -> str:
    """Calculate the Z2 topological invariant for a time-reversal insulator.

    KEY POINT: Z2 can ONLY be 0 or 1 — exactly! It classifies whether a
    system is a trivial insulator (ν=0) or topological insulator (ν=1).
    Protected by time-reversal symmetry (T² = -1 for spin-1/2 particles).

    nu: Z2 invariant (0 = trivial, 1 = topological)
    dimension: 2 or 3 (3D has four indices: ν₀; ν₁ν₂ν₃)

    Example: calc_z2_invariant(nu=1, dimension=3)
    → Strong TI (ν₀=1): single Dirac cone surface states, robust to disorder
    """
    from noethersolve.topological_invariants import z2_invariant
    report = z2_invariant(nu=nu, dimension=dimension)
    return str(report)


@mcp.tool()
def check_bulk_boundary(
    bulk_invariant: int,
    edge_modes: int = 0,
    system_type: str = "chern_insulator",
) -> str:
    """Verify the bulk-boundary correspondence.

    A fundamental theorem: the number of protected edge modes equals the
    absolute value of the bulk topological invariant. Violation indicates
    an error or symmetry breaking.

    bulk_invariant: Chern number or Z2 invariant of the bulk
    edge_modes: Number of observed edge modes (0 = auto-calculate from bulk)
    system_type: "chern_insulator", "z2_insulator", or "quantum_hall"

    Example: check_bulk_boundary(bulk_invariant=2, edge_modes=2)
    → ✓ Correspondence SATISFIED: 2 bulk = 2 edge
    """
    from noethersolve.topological_invariants import bulk_boundary_correspondence
    em = edge_modes if edge_modes > 0 else None
    report = bulk_boundary_correspondence(bulk_invariant=bulk_invariant, edge_modes=em,
                                          system_type=system_type)
    return str(report)


@mcp.tool()
def calc_quantum_hall(
    filling_factor: float,
    is_integer: bool = True,
) -> str:
    """Calculate quantum Hall effect properties.

    KEY POINT: Hall conductance is EXACTLY quantized to σ_xy = ν × e²/h.
    This exactness is used to DEFINE the Ohm in the SI system (since 2019).
    The von Klitzing constant R_K = h/e² = 25812.80745... Ω is exact.

    filling_factor: Landau level filling ν (e.g., 1, 2, 1/3)
    is_integer: True for integer QHE, False for fractional QHE

    Example: calc_quantum_hall(filling_factor=1.0)
    → σ_xy = 1.0 e²/h (EXACTLY), R_H = 25812.81 Ω
    """
    from noethersolve.topological_invariants import quantum_hall
    report = quantum_hall(filling_factor=filling_factor, is_integer=is_integer)
    return str(report)


@mcp.tool()
def calc_berry_phase(
    phase_value: float,
    symmetry: str = "",
) -> str:
    """Calculate and analyze Berry phase.

    Berry phase can be quantized to 0 or π by certain symmetries (inversion,
    time-reversal). Without protection, it varies continuously. Physical
    observables depend on Berry curvature (gauge-invariant).

    phase_value: Berry phase in radians
    symmetry: Protecting symmetry ("inversion", "time_reversal", or empty)

    Example: calc_berry_phase(3.14159, "inversion")
    → φ_B = π (QUANTIZED by inversion symmetry, topologically protected)
    """
    from noethersolve.topological_invariants import berry_phase
    sym = symmetry if symmetry else None
    report = berry_phase(phase_value=phase_value, symmetry=sym)
    return str(report)


@mcp.tool()
def lookup_topological_class(
    symmetry_class: str,
    dimension: int,
) -> str:
    """Look up topological classification from the periodic table.

    The "periodic table" of topological phases classifies all possible
    topological insulators/superconductors based on symmetry and dimension.
    Returns Z (integer), Z2 (binary), 2Z (even integers), or 0 (trivial).

    symmetry_class: Altland-Zirnbauer class (A, AIII, AI, BDI, D, DIII, AII, CII, C, CI)
    dimension: Spatial dimension (1, 2, or 3)

    Example: lookup_topological_class("AII", 3)
    → Z2 invariant (3D topological insulator like Bi₂Se₃)
    """
    from noethersolve.topological_invariants import topological_classification
    report = topological_classification(symmetry_class=symmetry_class, dimension=dimension)
    return str(report)


# ── Ergodic Theory ────────────────────────────────────────────────────

@mcp.tool()
def classify_dynamical_system(
    name: str = "",
    level: str = "",
) -> str:
    """Classify a dynamical system in the ergodic hierarchy.

    KEY POINT: The hierarchy has STRICT inclusions:
    Bernoulli ⊊ K-mixing ⊊ mixing ⊊ weak mixing ⊊ ergodic

    Each level implies all levels to the right. LLMs often confuse these
    or claim equivalence where there is none (e.g., "ergodic = mixing" is WRONG).

    name: Known system name (e.g., "bernoulli_shift", "arnolds_cat", "irrational_rotation")
    level: Or specify level directly (bernoulli, k_mixing, mixing, weak_mixing, ergodic)

    Example: classify_dynamical_system(name="horocycle_flow")
    → MIXING but NOT K-mixing (zero entropy counterexample)
    """
    from noethersolve.ergodic_theory import classify_system
    report = classify_system(name=name, level=level)
    return str(report)


@mcp.tool()
def compare_ergodic_levels(
    level_1: str,
    level_2: str,
) -> str:
    """Compare two levels in the ergodic hierarchy.

    Determines implication relationships and provides counterexamples
    showing that implications are strict (not equivalences).

    level_1: First level (bernoulli, k_mixing, mixing, weak_mixing, ergodic)
    level_2: Second level

    Example: compare_ergodic_levels("mixing", "k_mixing")
    → K-mixing ⟹ mixing (strict). Counterexample: horocycle flow
    """
    from noethersolve.ergodic_theory import compare_levels
    report = compare_levels(level_1, level_2)
    return str(report)


@mcp.tool()
def analyze_lyapunov_exponents(
    exponents: list[float],
) -> str:
    """Analyze Lyapunov exponents for chaos and dimension.

    COMPUTES chaos detection (λ_max > 0), Kaplan-Yorke dimension,
    sum of positive exponents (relates to entropy via Pesin), and
    system type (conservative/dissipative).

    exponents: List of Lyapunov exponents (e.g., [0.91, 0.0, -14.57] for Lorenz)

    Example: analyze_lyapunov_exponents([0.91, 0.0, -14.57])
    → CHAOTIC (λ_max > 0), Kaplan-Yorke dim ≈ 2.06, dissipative
    """
    from noethersolve.ergodic_theory import lyapunov_analysis
    report = lyapunov_analysis(exponents)
    return str(report)


@mcp.tool()
def calc_dynamical_entropy(
    ks_entropy: float,
    lyapunov_positive_sum: float = 0.0,
    topological_entropy: float = 0.0,
) -> str:
    """Analyze Kolmogorov-Sinai (metric) entropy.

    COMPUTES whether Pesin formula h = Σλ⁺ is satisfied (SRB measure),
    predictability (h > 0 means chaotic), and variational principle check.

    ks_entropy: Kolmogorov-Sinai entropy (bits/iteration)
    lyapunov_positive_sum: Sum of positive Lyapunov exponents (optional)
    topological_entropy: Topological entropy h_top (optional)

    Example: calc_dynamical_entropy(ks_entropy=0.91, lyapunov_positive_sum=0.91)
    → Pesin satisfied (SRB measure), CHAOTIC (h > 0)
    """
    from noethersolve.ergodic_theory import entropy_analysis
    lps = lyapunov_positive_sum if lyapunov_positive_sum > 0 else None
    top = topological_entropy if topological_entropy > 0 else None
    report = entropy_analysis(ks_entropy=ks_entropy, lyapunov_positive_sum=lps,
                              topological_entropy=top)
    return str(report)


@mcp.tool()
def calc_poincare_recurrence(
    set_measure: float,
    phase_space_volume: float = 1.0,
) -> str:
    """Calculate Poincaré recurrence time.

    COMPUTES expected return time using Kac's lemma: ⟨τ⟩ = 1/μ(A).
    Almost every point in a finite-measure preserving system returns
    arbitrarily close to its starting point — this is guaranteed by theorem.

    set_measure: Measure of the set to return to
    phase_space_volume: Total phase space volume (default 1 for normalized)

    Example: calc_poincare_recurrence(set_measure=1e-10)
    → Expected return time ~ 10^10 iterations (may exceed age of universe!)
    """
    from noethersolve.ergodic_theory import poincare_recurrence
    report = poincare_recurrence(set_measure=set_measure, phase_space_volume=phase_space_volume)
    return str(report)


@mcp.tool()
def analyze_mixing_rate(
    rate_type: str,
    rate_value: float,
) -> str:
    """Analyze mixing rate (correlation decay).

    COMPUTES decay characteristics. Exponential mixing (|C(t)| ~ e^(-λt))
    implies K-system (positive entropy). Polynomial mixing (|C(t)| ~ t^(-α))
    is slower and does not guarantee positive entropy.

    rate_type: "exponential" or "polynomial"
    rate_value: Decay rate λ (for exponential) or exponent α (for polynomial)

    Example: analyze_mixing_rate("exponential", 0.5)
    → Exponential decay, implies K-system, mixing time ~ 2 iterations
    """
    from noethersolve.ergodic_theory import mixing_rate
    report = mixing_rate(rate_type=rate_type, rate_value=rate_value)
    return str(report)


# ── Optimization Convergence ──────────────────────────────────────────

@mcp.tool()
def analyze_gd_convergence(
    L: float,
    mu: float,
    epsilon: float = 1e-6,
) -> str:
    """Compute EXACT convergence rate for gradient descent.

    Rate is (1 - μ/L)^k = (1 - 1/κ)^k EXACTLY, not approximately.
    "Linear convergence" means EXPONENTIAL decay — confusingly named!

    L: Smoothness constant (Lipschitz gradient)
    mu: Strong convexity constant
    epsilon: Target accuracy (default 10⁻⁶)

    Example: analyze_gd_convergence(L=100, mu=1) → κ=100, rate=0.99, ~1300 iters
    """
    from noethersolve.optimization_convergence import gradient_descent_rate
    report = gradient_descent_rate(L=L, mu=mu, epsilon=epsilon)
    return str(report)


@mcp.tool()
def analyze_nesterov_convergence(
    L: float,
    mu: float,
    epsilon: float = 1e-6,
) -> str:
    """Compute EXACT convergence rate for Nesterov accelerated gradient.

    Rate is (1 - √(μ/L))^k = (1 - 1/√κ)^k EXACTLY.
    This is OPTIMAL — achieves the oracle complexity lower bound.

    L: Smoothness constant
    mu: Strong convexity constant
    epsilon: Target accuracy (default 10⁻⁶)

    Example: analyze_nesterov_convergence(L=100, mu=1) → rate=0.9, ~130 iters
    """
    from noethersolve.optimization_convergence import nesterov_rate
    report = nesterov_rate(L=L, mu=mu, epsilon=epsilon)
    return str(report)


@mcp.tool()
def compare_optimization_algorithms(
    L: float,
    mu: float,
) -> str:
    """Compare GD vs Nesterov convergence.

    KEY INSIGHT: Nesterov is √κ times faster in iteration count.
    At κ=1 (perfect conditioning), they are IDENTICAL.
    Acceleration ONLY helps for ill-conditioned problems (κ > 1).

    L: Smoothness constant
    mu: Strong convexity constant

    Example: compare_optimization_algorithms(L=100, mu=1)
    → Nesterov is √100 = 10× faster
    """
    from noethersolve.optimization_convergence import compare_algorithms
    report = compare_algorithms(L=L, mu=mu)
    return str(report)


@mcp.tool()
def analyze_problem_conditioning(
    L: float,
    mu: float,
    epsilon: float = 1e-6,
) -> str:
    """Analyze problem conditioning and acceleration benefit.

    COMPUTES condition number κ = L/μ, classification (well/ill-conditioned),
    iteration counts for GD vs Nesterov, and speedup factor.

    L: Smoothness constant
    mu: Strong convexity constant
    epsilon: Target accuracy

    Example: analyze_problem_conditioning(L=10000, mu=1)
    → Ill-conditioned (κ=10⁴), Nesterov ~100× faster than GD
    """
    from noethersolve.optimization_convergence import analyze_conditioning
    report = analyze_conditioning(L=L, mu=mu, epsilon=epsilon)
    return str(report)


@mcp.tool()
def check_oracle_lower_bound(
    L: float,
    mu: float,
) -> str:
    """Check oracle complexity lower bound for first-order optimization.

    The lower bound rate is (√κ - 1)/(√κ + 1) from Nemirovsky-Yudin (1983).
    Nesterov (1983) ACHIEVES this bound — it's OPTIMAL.
    GD is suboptimal by a factor of √κ.

    L: Smoothness constant
    mu: Strong convexity constant

    Example: check_oracle_lower_bound(L=100, mu=1)
    → GD suboptimal by ~10×, Nesterov achieves lower bound
    """
    from noethersolve.optimization_convergence import oracle_lower_bound
    report = oracle_lower_bound(L=L, mu=mu)
    return str(report)


@mcp.tool()
def calc_optimal_step_size(
    L: float,
    mu: float = 0.0,
    algorithm: str = "gd",
) -> str:
    """Compute optimal step size for convergence.

    For GD on L-smooth: η* = 1/L (or 2/(L+μ) for strongly convex).
    Step size > 2/L causes DIVERGENCE — common error!

    L: Smoothness constant
    mu: Strong convexity constant (0 for just L-smooth)
    algorithm: "gd" or "nesterov"

    Example: calc_optimal_step_size(L=10, mu=0)
    → η* = 0.1, diverges for η > 0.2
    """
    from noethersolve.optimization_convergence import optimal_step_size
    report = optimal_step_size(L=L, mu=mu, algorithm=algorithm)
    return str(report)


# ── Numerical PDE ─────────────────────────────────────────────────────

@mcp.tool()
def check_pde_cfl(
    scheme: str,
    cfl_number: float,
) -> str:
    """Check CFL condition stability for a numerical PDE scheme.

    CFL (Courant-Friedrichs-Lewy) is NECESSARY but NOT SUFFICIENT for
    explicit scheme stability. Leapfrog for diffusion is ALWAYS UNSTABLE.

    scheme: Scheme name (upwind, ftcs, lax_wendroff, crank_nicolson, etc.)
    cfl_number: Your computed CFL number (c×Δt/Δx for wave, D×Δt/Δx² for diffusion)

    Example: check_pde_cfl("ftcs", 0.6) → UNSTABLE (CFL > 0.5 limit)
    """
    from noethersolve.numerical_pde import check_cfl
    report = check_cfl(scheme=scheme, cfl_number=cfl_number)
    return str(report)


@mcp.tool()
def analyze_pde_scheme(
    scheme: str,
) -> str:
    """Get detailed information about a numerical PDE scheme.

    COMPUTES order of accuracy (space and time), stability conditions,
    whether explicit/implicit, CFL limit, and common pitfalls.

    scheme: Scheme name (upwind, ftcs, lax_wendroff, crank_nicolson,
            leapfrog_hyperbolic, leapfrog_parabolic, btcs, adi, etc.)

    Example: analyze_pde_scheme("crank_nicolson")
    → O(Δx², Δt²), implicit, unconditionally stable, gold standard for diffusion
    """
    from noethersolve.numerical_pde import get_scheme_info
    report = get_scheme_info(scheme=scheme)
    return str(report)


@mcp.tool()
def von_neumann_stability(
    scheme: str,
    cfl: float,
    wavenumber_dx: float = 1.57,
) -> str:
    """Perform Von Neumann stability analysis.

    COMPUTES amplification factor G for Fourier modes. Stability requires
    |G| ≤ 1 for all wavenumbers. |G| < 1 means dissipative (damps).

    scheme: Scheme name
    cfl: CFL number
    wavenumber_dx: k×Δx value to check (default π/2 ≈ 1.57)

    Example: von_neumann_stability("leapfrog_hyperbolic", 0.5)
    → |G| = 1 exactly, non-dissipative, stable
    """
    from noethersolve.numerical_pde import von_neumann_analysis
    report = von_neumann_analysis(scheme=scheme, cfl=cfl, wavenumber_dx=wavenumber_dx)
    return str(report)


@mcp.tool()
def check_lax_theorem(
    is_consistent: bool,
    consistency_order: int,
    is_stable: bool,
    is_linear: bool = True,
) -> str:
    """Check Lax equivalence theorem conditions.

    THEOREM: For LINEAR problems, consistency + stability ⟺ convergence.
    CRITICAL: This theorem does NOT apply to nonlinear PDEs!

    is_consistent: Does scheme converge to PDE as Δx,Δt → 0?
    consistency_order: Order of truncation error
    is_stable: Is scheme stable?
    is_linear: Is the PDE linear? (CRITICAL for theorem applicability)

    Example: check_lax_theorem(True, 2, True, is_linear=True)
    → CONVERGENT by Lax equivalence
    """
    from noethersolve.numerical_pde import check_lax_equivalence
    report = check_lax_equivalence(
        is_consistent=is_consistent,
        consistency_order=consistency_order,
        is_stable=is_stable,
        is_linear=is_linear,
    )
    return str(report)


@mcp.tool()
def calc_max_timestep(
    scheme: str,
    c_or_D: float,
    dx: float,
    pde_type: str = "hyperbolic",
) -> str:
    """Calculate maximum stable timestep for a PDE scheme.

    COMPUTES Δt_max from CFL limit. Returns ∞ for unconditionally stable schemes.

    scheme: Scheme name
    c_or_D: Wave speed (hyperbolic) or diffusion coefficient (parabolic)
    dx: Grid spacing
    pde_type: "hyperbolic" or "parabolic"

    Example: calc_max_timestep("ftcs", D=0.1, dx=0.01, pde_type="parabolic")
    → Δt_max = 0.0005 (CFL = D×Δt/Δx² ≤ 0.5)
    """
    from noethersolve.numerical_pde import max_timestep
    dt = max_timestep(scheme=scheme, c_or_D=c_or_D, dx=dx, pde_type=pde_type)
    if dt == float('inf'):
        return f"Scheme '{scheme}' is unconditionally stable — no timestep restriction."
    elif dt == 0.0:
        return f"Scheme '{scheme}' is UNCONDITIONALLY UNSTABLE for {pde_type} PDEs. Use a different scheme!"
    else:
        return f"Maximum stable Δt = {dt:.6g} (with 0.9 safety factor)\nCFL limit for {scheme}: use Δt ≤ {dt/0.9:.6g}"


@mcp.tool()
def analyze_pde_accuracy(
    scheme: str,
) -> str:
    """Analyze truncation error and accuracy of a PDE scheme.

    COMPUTES order of accuracy in space and time, leading error type
    (dissipative vs dispersive), and Richardson extrapolation applicability.

    CRITICAL: Order of accuracy ≠ stability. High-order schemes can be unstable!

    scheme: Scheme name

    Example: analyze_pde_accuracy("lax_wendroff")
    → O(Δx², Δt²), dispersive errors (oscillations near shocks)
    """
    from noethersolve.numerical_pde import analyze_accuracy
    report = analyze_accuracy(scheme=scheme)
    return str(report)


# ── MHD Conservation ──────────────────────────────────────────────────

@mcp.tool()
def check_mhd_helicity(
    helicity_type: str = "magnetic",
    resistivity: float = 0.0,
    viscosity: float = 0.0,
    compressible: bool = False,
) -> str:
    """Check magnetic or cross helicity conservation in MHD.

    CRITICAL FACTS:
    - Magnetic helicity is EXACTLY conserved in ideal MHD (η=0)
    - Magnetic helicity DECAYS in resistive MHD: dH/dt = -2η∫J·B dV
    - Cross helicity requires BOTH η=0 AND incompressibility
    - Hall MHD conserves helicity but transfers between scales

    helicity_type: "magnetic" or "cross"
    resistivity: Magnetic diffusivity η (m²/s), 0 for ideal
    viscosity: Kinematic viscosity ν (for cross helicity)
    compressible: Whether flow is compressible (for cross helicity)

    Example: check_mhd_helicity("magnetic", resistivity=0) → CONSERVED
    Example: check_mhd_helicity("cross", resistivity=0, compressible=True) → NOT CONSERVED
    """
    if helicity_type.lower() == "magnetic":
        from noethersolve.mhd_conservation import check_magnetic_helicity
        report = check_magnetic_helicity(resistivity=resistivity)
    else:
        from noethersolve.mhd_conservation import check_cross_helicity
        report = check_cross_helicity(
            resistivity=resistivity, viscosity=viscosity, compressible=compressible
        )
    return str(report)


@mcp.tool()
def check_frozen_flux_theorem(
    resistivity: float = 0.0,
    length_scale: float = 1.0,
    velocity: float = 1.0,
) -> str:
    """Check validity of frozen-in flux theorem.

    In ideal MHD, field lines are "frozen" into plasma and move with it.
    CRITICAL: Reconnection REQUIRES breaking frozen flux (needs η > 0)!

    resistivity: Magnetic diffusivity η (m²/s)
    length_scale: Characteristic length L (m)
    velocity: Characteristic velocity v (m/s)

    Returns magnetic Reynolds number Rm = vL/η and frozen flux status.

    Example: check_frozen_flux_theorem(resistivity=0) → Flux FROZEN, no reconnection
    Example: check_frozen_flux_theorem(resistivity=1e-4, length_scale=1, velocity=100)
    → Rm=1e6, approximately frozen, reconnection possible at small scales
    """
    from noethersolve.mhd_conservation import check_frozen_flux
    report = check_frozen_flux(
        resistivity=resistivity, length_scale=length_scale, velocity=velocity
    )
    return str(report)


@mcp.tool()
def check_mhd_div_b(
    max_div_B: float,
    B_scale: float = 1e-3,
    dx: float = 0.01,
) -> str:
    """Check ∇·B = 0 constraint satisfaction.

    Maxwell requires ∇·B = 0 (no monopoles). Numerical MHD must PRESERVE
    this — it is NOT automatic! Monopole errors cause spurious forces.

    max_div_B: Maximum |∇·B| in your simulation domain
    B_scale: Characteristic magnetic field strength (T)
    dx: Grid spacing (m)

    Suggests cleaning method if constraint violated.

    Example: check_mhd_div_b(1e-10, B_scale=1e-3, dx=0.01) → SATISFIED
    """
    from noethersolve.mhd_conservation import check_div_B
    report = check_div_B(max_div_B=max_div_B, B_scale=B_scale, dx=dx)
    return str(report)


@mcp.tool()
def check_mhd_conservation(
    invariant: str,
    resistivity: float = 0.0,
    viscosity: float = 0.0,
    compressible: bool = False,
) -> str:
    """Check conservation of any MHD invariant.

    INVARIANTS:
    - mass: ALWAYS conserved (continuity equation)
    - momentum: Conserved without external forces
    - magnetic_flux: Conserved in ideal MHD (frozen flux)
    - magnetic_helicity: Conserved in ideal MHD, decays with η
    - cross_helicity: Conserved in ideal INCOMPRESSIBLE MHD
    - energy: Conserved in ideal MHD, Ohmic/viscous dissipation with η,ν

    invariant: Which invariant to check
    resistivity: Magnetic diffusivity η
    viscosity: Kinematic viscosity ν
    compressible: Whether flow is compressible

    Example: check_mhd_conservation("magnetic_helicity", resistivity=1e-4)
    → NOT CONSERVED, breaking mechanism: resistive dissipation
    """
    from noethersolve.mhd_conservation import check_mhd_invariant
    report = check_mhd_invariant(
        invariant=invariant,
        resistivity=resistivity,
        viscosity=viscosity,
        compressible=compressible,
    )
    return str(report)


# ── GR Constraints (ADM Formalism) ────────────────────────────────────

@mcp.tool()
def check_gr_hamiltonian_constraint(
    value: float = 0.0,
    tolerance: float = 1e-10,
) -> str:
    """Check the Hamiltonian constraint in General Relativity.

    The Hamiltonian constraint H = 0 must be satisfied on each spatial
    hypersurface. In ADM formalism: R + K² - KᵢⱼKⁱʲ - 16πρ = 0

    PHYSICAL MEANING: Energy constraint — GR has no local gravitational
    energy density. The Hamiltonian constraint encodes this.

    value: Computed constraint value (should be ~0 if satisfied)
    tolerance: Numerical tolerance for "satisfied"

    Example: check_gr_hamiltonian_constraint(value=1e-12) → SATISFIED
    Example: check_gr_hamiltonian_constraint(value=1e-5) → VIOLATED
    """
    from noethersolve.gr_constraints import check_hamiltonian_constraint
    report = check_hamiltonian_constraint(value=value, tolerance=tolerance)
    return str(report)


@mcp.tool()
def check_gr_momentum_constraint(
    value: float = 0.0,
    tolerance: float = 1e-10,
    component: str = "",
) -> str:
    """Check the momentum constraint in General Relativity.

    The momentum constraint Mᵢ = 0 must be satisfied. In ADM formalism:
    DⱼKʲᵢ - DᵢK - 8πJᵢ = 0

    PHYSICAL MEANING: Momentum conservation — there are 3 momentum
    constraints (one per spatial direction).

    value: Computed constraint value (should be ~0 if satisfied)
    tolerance: Numerical tolerance for "satisfied"
    component: Optional label ("x", "y", "z") for the constraint component

    Example: check_gr_momentum_constraint(value=1e-12, component="x") → SATISFIED
    """
    from noethersolve.gr_constraints import check_momentum_constraint
    report = check_momentum_constraint(value=value, tolerance=tolerance, component=component)
    return str(report)


@mcp.tool()
def check_gr_mass(
    mass_type: str = "ADM",
    is_asymptotically_flat: bool = True,
    is_isolated: bool = True,
    is_stationary: bool = False,
    has_radiation: bool = False,
    mass_value: float = 0.0,
) -> str:
    """Check applicability and properties of GR mass definitions.

    CRITICAL FACTS LLMs GET WRONG:
    - ADM mass: Total mass at spatial infinity. CONSTANT in time (does NOT
      decrease with gravitational radiation).
    - Bondi mass: Mass at null infinity. DECREASES with gravitational
      radiation via Bondi mass-loss formula.
    - Komar mass: Only defined for STATIONARY spacetimes with timelike
      Killing vector.

    mass_type: "ADM", "Bondi", or "Komar"
    is_asymptotically_flat: Required for ADM and Bondi
    is_isolated: Required for ADM
    is_stationary: Required for Komar (must have timelike Killing vector)
    has_radiation: If True, Bondi mass decreases
    mass_value: Optional mass value to record

    Example: check_gr_mass("Bondi", has_radiation=True)
    → Applicable, DECREASES with radiation
    """
    from noethersolve.gr_constraints import check_adm_mass, check_bondi_mass, check_komar_mass

    mass_type_upper = mass_type.upper()
    if mass_type_upper == "ADM":
        report = check_adm_mass(
            is_asymptotically_flat=is_asymptotically_flat,
            is_isolated=is_isolated,
            mass_value=mass_value if mass_value != 0 else None,
        )
    elif mass_type_upper == "BONDI":
        report = check_bondi_mass(
            is_asymptotically_flat=is_asymptotically_flat,
            has_null_infinity=True,
            has_radiation=has_radiation,
            mass_value=mass_value if mass_value != 0 else None,
        )
    elif mass_type_upper == "KOMAR":
        report = check_komar_mass(
            is_stationary=is_stationary,
            has_killing_vector=is_stationary,
            killing_type="timelike" if is_stationary else "none",
            mass_value=mass_value if mass_value != 0 else None,
        )
    else:
        return f"Unknown mass type '{mass_type}'. Available: ADM, Bondi, Komar"
    return str(report)


@mcp.tool()
def compare_gr_mass_definitions(
    spacetime_type: str = "schwarzschild",
    has_radiation: bool = False,
) -> str:
    """Compare all three mass definitions for a given spacetime.

    CRITICAL: For stationary spacetimes (Schwarzschild, Kerr), all three
    masses agree. For dynamical spacetimes with radiation, ADM ≠ Bondi!

    spacetime_type: One of:
      - "schwarzschild", "kerr", "reissner_nordstrom" (stationary, all agree)
      - "binary_merger", "bbh" (dynamical, Komar N/A, Bondi decreases)
      - "flrw" (cosmological, neither ADM nor Bondi applicable)

    has_radiation: Whether gravitational waves are present

    Example: compare_gr_mass_definitions("schwarzschild") → All 3 agree
    Example: compare_gr_mass_definitions("bbh", has_radiation=True)
    → ADM > Bondi (Bondi decreases with radiation)
    """
    from noethersolve.gr_constraints import compare_mass_definitions
    report = compare_mass_definitions(spacetime_type=spacetime_type, has_radiation=has_radiation)
    return str(report)


# ── Seismic Waves ─────────────────────────────────────────────────────

@mcp.tool()
def calc_seismic_velocity(
    K: float,
    G: float,
    rho: float,
) -> str:
    """Calculate P-wave and S-wave velocities from elastic moduli.

    CRITICAL FORMULAS (LLMs often get these wrong):
    - Vp = sqrt((K + 4G/3) / rho)  [P-wave, compressional]
    - Vs = sqrt(G / rho)           [S-wave, shear]

    Common errors:
    - Forgetting the 4/3 factor in Vp
    - Confusing K (bulk) with E (Young's)

    K: Bulk modulus (Pa or GPa)
    G: Shear modulus (Pa or GPa - same units as K)
    rho: Density (kg/m³ or g/cm³ - consistent with moduli)

    Example: calc_seismic_velocity(50e9, 30e9, 2700) for granite
    → Vp ≈ 5900 m/s, Vs ≈ 3300 m/s
    """
    from noethersolve.seismic_waves import calc_seismic_velocity as _calc
    report = _calc(K, G, rho)
    return str(report)


@mcp.tool()
def calc_poisson_from_velocities(
    Vp: float,
    Vs: float,
) -> str:
    """Calculate Poisson's ratio from measured P and S wave velocities.

    CRITICAL FORMULA:
    nu = (Vp² - 2Vs²) / (2(Vp² - Vs²))

    Common Vp/Vs ratios:
    - √2 ≈ 1.414: nu = 0.0 (unusual)
    - √3 ≈ 1.732: nu = 0.25 (Poisson solid, common reference)
    - 2.0: nu = 0.333
    - >>2: nu → 0.5 (liquids, Vs → 0)

    Thermodynamic bounds: -1 < nu < 0.5
    Most rocks: 0.05 < nu < 0.45

    Vp: P-wave velocity (any units)
    Vs: S-wave velocity (same units, must be < Vp)

    Example: calc_poisson_from_velocities(6000, 3464) → nu ≈ 0.25
    """
    from noethersolve.seismic_waves import poisson_from_velocities
    report = poisson_from_velocities(Vp, Vs)
    return str(report)


@mcp.tool()
def convert_elastic_moduli(
    K: float = 0,
    G: float = 0,
    E: float = 0,
    nu: float = -999,
    lam: float = 0,
) -> str:
    """Convert between any two elastic moduli to get all five.

    Provide exactly two of: K, G, E, nu, lambda (lam)
    Set unused parameters to 0 (except nu, use -999).

    The five moduli:
    - K: Bulk modulus (uniform compression)
    - G (mu): Shear modulus (shape change)
    - E: Young's modulus (tensile stiffness)
    - nu: Poisson's ratio (lateral/axial strain ratio)
    - lambda: Lame's first parameter
    - M: P-wave modulus (computed: K + 4G/3)

    Key relationships:
    - E = 9KG / (3K + G)
    - nu = (3K - 2G) / (6K + 2G)
    - lambda = K - 2G/3

    Example: convert_elastic_moduli(K=50e9, G=30e9)
    → E, nu, lambda, M all computed
    """
    from noethersolve.seismic_waves import convert_elastic_moduli as _convert
    # Build kwargs with only provided values
    kwargs = {}
    if K != 0:
        kwargs['K'] = K
    if G != 0:
        kwargs['G'] = G
    if E != 0:
        kwargs['E'] = E
    if nu != -999:
        kwargs['nu'] = nu
    if lam != 0:
        kwargs['lam'] = lam
    try:
        report = _convert(**kwargs)
        return str(report)
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def calc_reflection_coefficient(
    rho1: float,
    Vp1: float,
    rho2: float,
    Vp2: float,
) -> str:
    """Calculate seismic reflection and transmission coefficients at normal incidence.

    For P-waves at normal incidence:
    - R = (Z₂ - Z₁) / (Z₂ + Z₁)  where Z = ρ × Vp
    - T = 2Z₁ / (Z₂ + Z₁)

    Sign convention:
    - R > 0: Polarity preserved (Z₂ > Z₁, "hard" boundary)
    - R < 0: Polarity reversed (Z₂ < Z₁, "soft" boundary)

    Energy conservation: R² + (Z₂/Z₁)T² = 1

    rho1, Vp1: Density and P-velocity of layer 1 (incident)
    rho2, Vp2: Density and P-velocity of layer 2 (transmitted)

    Example: calc_reflection_coefficient(2500, 4000, 2800, 5000)
    → R = 0.27, polarity preserved (harder layer below)
    """
    from noethersolve.seismic_waves import calc_reflection_coefficient as _calc
    report = _calc(rho1, Vp1, rho2, Vp2)
    return str(report)


@mcp.tool()
def calc_critical_angle(
    V1: float,
    V2: float,
) -> str:
    """Calculate critical angle for total internal reflection.

    sin(θc) = V₁ / V₂ (requires V₂ > V₁)

    At angles >= critical angle, no transmitted wave — all energy reflects.

    V1: Velocity in layer 1 (incident medium)
    V2: Velocity in layer 2 (must be > V1)

    Example: calc_critical_angle(3000, 6000) → θc = 30°
    """
    from noethersolve.seismic_waves import critical_angle
    try:
        theta_c = critical_angle(V1, V2)
        return f"Critical angle: {theta_c:.2f}°\nsin(θc) = {V1/V2:.4f}"
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def calc_snells_law(
    theta1_deg: float,
    V1: float,
    V2: float,
) -> str:
    """Apply Snell's law to find refracted angle.

    sin(θ₁)/V₁ = sin(θ₂)/V₂

    theta1_deg: Incident angle in degrees (0-90)
    V1: Velocity in medium 1
    V2: Velocity in medium 2

    Returns refracted angle. If beyond critical angle, indicates total reflection.

    Example: calc_snells_law(30, 3000, 5000)
    → θ₂ = 56.4° (ray bends away from normal in faster medium)
    """
    from noethersolve.seismic_waves import snells_law
    theta2, transmitted = snells_law(theta1_deg, V1, V2)
    if transmitted:
        return f"Incident angle: {theta1_deg:.1f}°\nRefracted angle: {theta2:.2f}°\nTransmitted: Yes"
    else:
        return f"Incident angle: {theta1_deg:.1f}°\nBeyond critical angle: TOTAL INTERNAL REFLECTION\nTransmitted: No"


# ── Plasma Adiabatic Invariants ───────────────────────────────────────

@mcp.tool()
def calc_magnetic_moment(
    v_perp: float,
    B: float,
    mass_kg: float = 9.109e-31,
    charge_C: float = 1.602e-19,
    field_timescale: float = 0,
) -> str:
    """Calculate the first adiabatic invariant μ = m*v⊥²/(2*B).

    CRITICAL: μ is conserved when field changes slowly vs cyclotron period.
    Breaking condition: τ_field ~ τ_cyclotron → μ breaks

    The magnetic moment is the MOST ROBUST adiabatic invariant.

    v_perp: Perpendicular velocity (m/s)
    B: Magnetic field strength (Tesla)
    mass_kg: Particle mass (kg, default: electron 9.109e-31)
    charge_C: Particle charge magnitude (C, default: 1.602e-19)
    field_timescale: Timescale of field variation (s, 0 = assume slow)

    Example: calc_magnetic_moment(1e7, 1e-5) for 1e7 m/s electron in Earth field
    → μ ≈ 4.6e-16 J/T, cyclotron period ~3.6 μs
    """
    from noethersolve.plasma_adiabatic import calc_magnetic_moment as _calc
    ts = field_timescale if field_timescale > 0 else None
    report = _calc(v_perp, B, mass_kg, charge_C, field_timescale=ts)
    return str(report)


@mcp.tool()
def calc_bounce_invariant(
    v_parallel: float,
    bounce_length: float,
    B_min: float,
    B_max: float,
    field_timescale: float = 0,
) -> str:
    """Calculate the second adiabatic invariant J = ∮ v∥ ds.

    CRITICAL: J is conserved when field changes slowly vs bounce period.
    Breaking condition: τ_field ~ τ_bounce → J breaks (but μ may survive)

    Also computes:
    - Mirror ratio R = B_max/B_min
    - Loss cone angle (particles with smaller pitch angle escape)
    - Bounce period

    v_parallel: Parallel velocity at B_min (m/s)
    bounce_length: Distance between mirror points (m)
    B_min: Minimum field at trap center (T)
    B_max: Maximum field at mirror points (T)
    field_timescale: Timescale of field variation (s, 0 = assume slow)

    Example: calc_bounce_invariant(1e6, 1e7, 1e-5, 5e-5)
    → J, mirror ratio R=5, loss cone 26.6°
    """
    from noethersolve.plasma_adiabatic import calc_bounce_invariant as _calc
    ts = field_timescale if field_timescale > 0 else None
    report = _calc(v_parallel, bounce_length, B_min, B_max, field_timescale=ts)
    return str(report)


@mcp.tool()
def calc_flux_invariant(
    drift_radius: float,
    B_average: float,
    energy_eV: float,
    field_timescale: float = 0,
) -> str:
    """Calculate the third adiabatic invariant Φ = ∮ A·dl.

    CRITICAL: Φ is the MOST FRAGILE invariant — breaks when field changes
    on drift orbit timescale. Typical breaking: magnetic storms, substorms.

    Hierarchy: ω_cyclotron >> ω_bounce >> ω_drift
    So μ is most robust, Φ is most fragile.

    drift_radius: Average radius of drift orbit (m)
    B_average: Average magnetic field over drift orbit (T)
    energy_eV: Particle kinetic energy (eV)
    field_timescale: Timescale of field variation (s, 0 = assume slow)

    Example: calc_flux_invariant(6e7, 1e-5, 1e6) for 1 MeV particle
    → Φ ≈ 0.11 Wb, drift period ~hours
    """
    from noethersolve.plasma_adiabatic import calc_flux_invariant as _calc
    ts = field_timescale if field_timescale > 0 else None
    report = _calc(drift_radius, B_average, energy_eV, field_timescale=ts)
    return str(report)


@mcp.tool()
def check_adiabatic_hierarchy(
    B: float,
    v_total: float,
    pitch_angle_deg: float,
    bounce_length: float,
    drift_radius: float,
    field_timescale: float = 0,
) -> str:
    """Check all three adiabatic invariants and their hierarchy.

    CRITICAL HIERARCHY (LLMs often get this wrong):
    ω_cyclotron >> ω_bounce >> ω_drift
    τ_cyclotron << τ_bounce << τ_drift

    This means: μ is most robust, Φ is most fragile.

    Common regimes:
    - All conserved: truly adiabatic (slow changes)
    - μ, J conserved, Φ broken: geomagnetic storms, substorms
    - Only μ conserved: wave-particle resonance, reconnection
    - μ broken: strong turbulence, shock acceleration

    B: Magnetic field at reference point (T)
    v_total: Total particle velocity (m/s)
    pitch_angle_deg: Pitch angle in degrees (0-90)
    bounce_length: Distance between mirror points (m)
    drift_radius: Average drift orbit radius (m)
    field_timescale: Timescale of field variation (s, 0 = check hierarchy only)

    Example: check_adiabatic_hierarchy(1e-5, 1e7, 45, 1e7, 6e7)
    """
    from noethersolve.plasma_adiabatic import check_adiabatic_hierarchy as _check
    ts = field_timescale if field_timescale > 0 else None
    report = _check(B, v_total, pitch_angle_deg, bounce_length, drift_radius, field_timescale=ts)
    return str(report)


@mcp.tool()
def calc_loss_cone(
    B_min: float,
    B_max: float,
) -> str:
    """Calculate the loss cone angle for a magnetic mirror.

    Particles with pitch angle α < α_loss escape the trap.

    sin²(α_loss) = B_min / B_max

    B_min: Minimum field at trap center (T)
    B_max: Maximum field at mirror points (T)

    Example: calc_loss_cone(3e-5, 6e-5) for Earth-like mirror
    → Loss cone = 45° (R = 2)
    """
    from noethersolve.plasma_adiabatic import loss_cone_angle
    alpha = loss_cone_angle(B_min, B_max)
    R = B_max / B_min
    return f"Loss cone angle: {alpha:.2f}°\nMirror ratio R = B_max/B_min = {R:.2f}\nParticles with pitch angle < {alpha:.1f}° escape the trap"


@mcp.tool()
def calc_cyclotron_params(
    B: float,
    v_perp: float,
    particle: str = "electron",
) -> str:
    """Calculate cyclotron frequency and Larmor radius.

    ω_c = |q|B/m (cyclotron frequency)
    r_L = m*v⊥/(|q|B) = v⊥/ω_c (Larmor/gyro radius)

    B: Magnetic field (T)
    v_perp: Perpendicular velocity (m/s)
    particle: "electron", "proton", "alpha", "O+" (default: electron)

    Example: calc_cyclotron_params(1e-5, 1e6, "proton")
    → ω_c ≈ 960 rad/s, r_L ≈ 1042 m
    """
    from noethersolve.plasma_adiabatic import (
        cyclotron_frequency, larmor_radius, get_particle_mass, ELECTRON_CHARGE
    )
    import math
    mass = get_particle_mass(particle)
    omega_c = cyclotron_frequency(B, mass, ELECTRON_CHARGE)
    r_L = larmor_radius(v_perp, B, mass, ELECTRON_CHARGE)
    tau_c = 2 * math.pi / omega_c

    lines = [
        f"Particle: {particle}",
        f"Magnetic field: {B:.4e} T",
        f"Perpendicular velocity: {v_perp:.4e} m/s",
        "",
        f"Cyclotron frequency: {omega_c:.4e} rad/s",
        f"Cyclotron period: {tau_c:.4e} s",
        f"Larmor radius: {r_L:.4e} m",
    ]
    return "\n".join(lines)


# ── Information-Thermodynamics (Landauer-Shannon Bridge) ─────────────

@mcp.tool()
def calc_landauer_bound(bits_erased: float, temperature: float = 300.0) -> str:
    """Calculate minimum energy required to erase information (Landauer's principle).

    Landauer's principle (1961): Erasing one bit of information requires
    dissipating at least kT·ln(2) of energy as heat. This is the fundamental
    limit connecting information and thermodynamics.

    At room temperature (300 K), erasing 1 bit costs ~2.87 × 10⁻²¹ J.
    Modern computers dissipate ~10⁴ times this limit per bit operation.

    bits_erased: Number of bits to erase (can be fractional)
    temperature: Temperature of heat bath in Kelvin (default: 300 K)

    Example: calc_landauer_bound(1.0, 300)
    → 2.867e-21 J minimum, 0.693 kT per bit
    """
    from noethersolve.info_thermo import calc_landauer_bound as _calc
    return str(_calc(bits_erased, temperature))


@mcp.tool()
def calc_shannon_entropy(probabilities: list[float]) -> str:
    """Calculate Shannon entropy of a probability distribution.

    H = -Σ p_i × log₂(p_i)

    Shannon entropy gives the minimum average number of bits needed to
    encode symbols drawn from this distribution (Shannon's source coding
    theorem). It also bounds the thermodynamic work extractable.

    probabilities: List of probabilities (must sum to 1)

    Example: calc_shannon_entropy([0.5, 0.5])
    → H = 1.000 bits (fair coin)

    Example: calc_shannon_entropy([0.9, 0.1])
    → H = 0.469 bits (biased coin)
    """
    from noethersolve.info_thermo import calc_shannon_entropy as _calc
    return str(_calc(probabilities))


@mcp.tool()
def calc_huffman_landauer(probabilities: list[float], temperature: float = 300.0) -> str:
    """Show the parallel between Huffman coding and Landauer erasure.

    Both optimize the same objective: Σ p_i × cost_i where cost ∝ -log(p_i)
    - Huffman: cost = code length (bits)
    - Landauer: cost = erasure energy (kT × ln(2) per bit)

    Shannon entropy H bounds BOTH:
    - Minimum bits for lossless compression
    - Minimum energy for irreversible erasure

    probabilities: Probability distribution
    temperature: Temperature in Kelvin

    Example: calc_huffman_landauer([0.5, 0.25, 0.125, 0.125], 300)
    → Shows parallel structure with H = 1.75 bits
    """
    from noethersolve.info_thermo import calc_huffman_landauer_parallel
    return str(calc_huffman_landauer_parallel(probabilities, temperature))


# ── Battery Degradation ──────────────────────────────────────────────

@mcp.tool()
def calc_battery_calendar_aging(
    chemistry: str,
    time_days: float,
    temperature_C: float = 25.0,
    soc_storage: float = 0.5,
) -> str:
    """Calculate Li-ion battery calendar (storage) aging.

    CRITICAL: Calendar aging follows sqrt(t), NOT linear!
    This is the #1 LLM error on battery degradation.

    SEI layer growth is diffusion-limited → Q_loss ∝ √t.
    Doubling storage time does NOT double capacity loss.

    chemistry: "NMC", "LFP", or "NCA"
    time_days: Storage time in days
    temperature_C: Storage temperature (default 25°C)
    soc_storage: State of charge during storage (0-1, default 0.5)

    Example: calc_battery_calendar_aging("NMC", 365, 25, 0.5)
    → ~2% loss after 1 year at room temperature
    """
    from noethersolve.battery_degradation import calc_calendar_aging
    return str(calc_calendar_aging(chemistry, time_days, temperature_C, soc_storage))


@mcp.tool()
def calc_battery_cycle_aging(
    chemistry: str,
    cycles: int,
    dod: float = 0.8,
    temperature_C: float = 25.0,
    c_rate: float = 1.0,
) -> str:
    """Calculate Li-ion battery cycle aging.

    Cycle aging is driven by mechanical stress (particle cracking).
    Higher DOD causes EXPONENTIALLY more stress: Q_loss ∝ DOD^n.

    chemistry: "NMC", "LFP", or "NCA"
    cycles: Number of charge/discharge cycles
    dod: Depth of discharge per cycle (0-1, default 0.8)
    temperature_C: Operating temperature (default 25°C)
    c_rate: Charge/discharge rate (default 1.0C)

    Example: calc_battery_cycle_aging("NMC", 500, 0.8)
    → Capacity loss from 500 cycles at 80% DOD
    """
    from noethersolve.battery_degradation import calc_cycle_aging
    return str(calc_cycle_aging(chemistry, cycles, dod, temperature_C, c_rate))


@mcp.tool()
def calc_battery_total_aging(
    chemistry: str,
    time_days: float,
    cycles: int,
    temperature_C: float = 25.0,
    dod: float = 0.8,
    soc_storage: float = 0.5,
) -> str:
    """Calculate total Li-ion battery degradation (calendar + cycle).

    CRITICAL: Calendar and cycle aging are ADDITIVE, not multiplicative!
    Total loss = Calendar loss + Cycle loss

    This is a common LLM error - models often assume multiplicative effects.

    chemistry: "NMC", "LFP", or "NCA"
    time_days: Total time in days
    cycles: Number of cycles performed
    temperature_C: Average operating temperature
    dod: Typical depth of discharge
    soc_storage: Average storage SOC when not cycling

    Example: calc_battery_total_aging("NMC", 730, 600, 25, 0.8)
    → 2 years with 300 cycles/year at 80% DOD
    """
    from noethersolve.battery_degradation import calc_combined_aging
    return str(calc_combined_aging(chemistry, time_days, cycles, temperature_C, dod, soc_storage))


@mcp.tool()
def compare_battery_chemistries(
    time_days: float,
    cycles: int,
    temperature_C: float = 25.0,
    dod: float = 0.8,
) -> str:
    """Compare NMC, LFP, and NCA battery degradation for same usage.

    Useful for battery chemistry selection decisions.

    Key differences:
    - NMC: Balanced calendar/cycle, moderate cost
    - LFP: Better cycle life, worse energy density
    - NCA: High energy density, more sensitive to stress

    time_days: Total time in days
    cycles: Number of cycles
    temperature_C: Operating temperature
    dod: Depth of discharge

    Example: compare_battery_chemistries(730, 600, 25, 0.8)
    → 2-year EV comparison
    """
    from noethersolve.battery_degradation import compare_chemistries
    return compare_chemistries(time_days, cycles, temperature_C, dod)


# ── Blind Spot Detection ──────────────────────────────────────────────

@mcp.tool()
def detect_blind_spots(query: str) -> str:
    """Detect model blind spots in a query and recommend MCP tools.

    CALL THIS FIRST when unsure if a question involves cross-domain
    connections or known blind spots. Returns recommended tools to use.

    Cross-domain blind spots include:
    - Deadlock ↔ Detailed Balance (OS ↔ thermodynamics)
    - PageRank ↔ Equilibrium (web search ↔ stat mech)
    - Database Isolation ↔ Quantum Decoherence
    - Huffman Coding ↔ Landauer's Principle
    - Type Inference ↔ Gauge Fixing

    Single-domain blind spots include:
    - Conjecture/proof status (model hallucinates)
    - Complexity class relationships
    - LLM capability claims

    Example: detect_blind_spots("Is P = NP proven?")
    → Recommends check_conjecture() tool
    """
    from noethersolve.blind_spot_detector import (
        detect_blind_spots as _detect,
        format_blind_spot_warning,
    )
    matches = _detect(query)
    if not matches:
        return "No known blind spots detected. Safe to answer from knowledge."
    return format_blind_spot_warning(matches)


@mcp.tool()
def list_blind_spots(needs_tool_only: bool = False) -> str:
    """List all known model blind spots as research opportunities.

    Returns a complete inventory of:
    - Cross-domain blind spots (model fails to connect separate domains)
    - Single-domain blind spots (model is miscalibrated within domain)

    Each entry shows:
    - Domains involved
    - Key insight (the connection the model misses)
    - Available tools (if any)
    - Tool ideas (for blind spots needing new tools)

    Args:
        needs_tool_only: If True, only show blind spots that need tools built
                        (these are the primary research opportunities)

    Example: list_blind_spots(needs_tool_only=True)
    → Shows Huffman↔Landauer, Type↔Gauge that need calc_landauer_bound(), etc.
    """
    from noethersolve.blind_spot_detector import list_all_blind_spots
    return list_all_blind_spots(needs_tool_only)


# ── Entry Point ───────────────────────────────────────────────────────

def main():
    """Run the NoetherSolve MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
