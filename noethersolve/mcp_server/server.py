"""NoetherSolve MCP Server — expose 87 verified tools to any AI agent.

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
    instructions="87 computational tools for physics, math, genetics, chemistry, pharmacokinetics, "
                 "and LLM science — verified calculators from first principles, not guesses.",
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


# ── Entry Point ───────────────────────────────────────────────────────

def main():
    """Run the NoetherSolve MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
