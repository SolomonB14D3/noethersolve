"""NoetherSolve — find where LLM knowledge ends, build verified tools, serve
them to any AI agent via MCP.

The pipeline: find gaps → flip facts → build tool → add to MCP server.
Every tool added makes every connected agent smarter.

Emmy Noether proved every continuous symmetry corresponds to a conserved quantity.
NoetherSolve finds where LLMs fail to recognize those quantities, builds verified
computational tools for the right answers, and exposes them via Model Context
Protocol (MCP) — 69 tools currently serving physics, math, genetics, complexity
theory, enzyme kinetics, quantum mechanics, pharmacokinetics, drug interactions,
organic chemistry, elliptic curves, intersection theory (adjunction, blow-ups,
ruled surfaces, toric varieties), information theory, topological phases, ergodic
theory, optimization convergence, numerical PDEs, MHD conservation, GR constraints
(ADM formalism), seismic waves (velocity, moduli, reflection), plasma physics
(adiabatic invariants μ, J, Φ), and LLM science.

Package layout:
  noethersolve.mcp_server   — MCP server (69 tools for any AI agent)
  noethersolve.tool_graph   — automatic calculator chaining via metadata tags
  noethersolve.dimension_physics — dimension-dependent physics formulas (2D vs 3D)
  noethersolve.oracle       — model-agnostic MC log-prob scorer (from eval_mc)
  noethersolve.adapter      — snap-on logit adapter architectures (from snap_on)
  noethersolve.train_utils  — LOGIT_SOFTCAP, get_lm_head_fn, apply_adapter
  noethersolve.monitor      — conservation law monitors (VortexMonitor, ChemicalMonitor, GravityMonitor)
  noethersolve.monitor_em   — electromagnetic field conservation monitor (EMMonitor)
  noethersolve.validate     — integrator validation via conservation laws
  noethersolve.audit_chem   — chemical reaction network thermodynamic auditor
  noethersolve.hamiltonian  — Hamiltonian system symplectic structure validator
  noethersolve.learner      — automatic conservation law discovery via optimization
  noethersolve.audit_facts  — oracle fact file quality auditor (token-length bias detection)
  noethersolve.knot         — knot invariant monitor (Reidemeister moves, Jones polynomial)
  noethersolve.audit_sequence — DNA/RNA therapeutic sequence design auditor
  noethersolve.crispr       — CRISPR guide RNA scorer (on-target activity, off-target risk)
  noethersolve.pipeline     — therapeutic pipeline consistency validator
  noethersolve.aggregation  — protein aggregation propensity predictor
  noethersolve.splice       — splice site strength scorer (PWM-based)
  noethersolve.pharmacokinetics — pharmacogenomic CYP interaction checker
  noethersolve.complexity     — complexity class relationship auditor
  noethersolve.conjecture_status — mathematical conjecture status checker
  noethersolve.proof_barriers — proof technique barrier checker
  noethersolve.number_theory  — number theory conjecture numerical verifier
  noethersolve.reductions     — computational reduction chain validator
  noethersolve.pde_regularity — PDE regularity and Sobolev embedding checker
  noethersolve.llm_claims    — LLM claims auditor (benchmark, scaling, misconceptions)
  noethersolve.control       — PID controller simulator and Routh-Hurwitz stability analyzer
  noethersolve.isolation     — SQL transaction isolation anomaly checker
  noethersolve.quantum_circuit — quantum circuit state vector simulator with entanglement detection
  noethersolve.chemistry_calc  — electrochemistry, acid-base, crystal field, semiconductor calculator
  noethersolve.crypto_calc     — cryptographic security level, birthday bound, cipher mode analyzer
  noethersolve.finance_calc    — Black-Scholes, put-call parity, Nash equilibrium, time value of money
  noethersolve.distributed_calc — quorum systems, Byzantine thresholds, vector clocks, consistency models
  noethersolve.network_calc    — bandwidth-delay product, TCP throughput, subnetting, IP fragmentation
  noethersolve.os_calc         — page tables, CPU scheduling, deadlock detection, TLB analysis
  noethersolve.biochemistry    — biochemistry fact checker (enzymes, metabolism, signaling)
  noethersolve.organic_chemistry — organic chemistry fact checker (mechanisms, reactions, synthesis)
  noethersolve.quantum_mechanics — quantum mechanics fact checker (foundations, phenomena, systems)
  noethersolve.elliptic_curves   — elliptic curve arithmetic, invariants, point counting, group structure
  noethersolve.intersection_theory — Bezout, genus-degree, self-intersection, canonical divisors, enumerative geometry
  noethersolve.information_theory  — channel capacity, rate-distortion, source coding, MAC, Fano inequality
  noethersolve.drug_interactions   — CYP450 DDI checker, pharmacogenomics, inhibitor/inducer database
  noethersolve.gr_constraints      — ADM/Bondi/Komar mass, Hamiltonian/momentum constraints, ADM formalism
  noethersolve.seismic_waves       — P/S-wave velocities, elastic moduli, reflection coefficients, Snell's law
  noethersolve.plasma_adiabatic    — Adiabatic invariants (μ, J, Φ), magnetic mirrors, loss cone, cyclotron motion
"""

# MLX-dependent modules — optional, only available on Apple Silicon
try:
    from noethersolve import train_utils  # noqa: F401
except ImportError:
    pass  # MLX not installed — adapter training unavailable, tools still work

from noethersolve.monitor import (  # noqa: F401
    frac_var,
    MonitorReport,
    VortexMonitor,
    ChemicalMonitor,
    GravityMonitor,
)
from noethersolve.monitor_em import EMMonitor  # noqa: F401
from noethersolve.validate import validate_integrator, compare_configs  # noqa: F401
from noethersolve.audit_chem import audit_network, AuditReport  # noqa: F401
from noethersolve.audit_facts import audit_facts, FactAuditReport  # noqa: F401
from noethersolve.hamiltonian import (  # noqa: F401
    HamiltonianMonitor,
    HamiltonianReport,
    harmonic_oscillator,
    kepler_2d,
    henon_heiles,
    coupled_oscillators,
)
from noethersolve.learner import InvariantLearner, LearnerReport  # noqa: F401
from noethersolve.knot import (  # noqa: F401
    KnotMonitor,
    KnotReport,
    KnotDiagram,
    unknot,
    trefoil,
    figure_eight_knot,
)
from noethersolve.audit_sequence import (  # noqa: F401
    audit_sequence,
    SequenceReport,
    SequenceIssue,
    gc_content,
    cpg_observed_expected,
)

from noethersolve.crispr import (  # noqa: F401
    score_guide,
    score_guides,
    check_offtarget_pair,
    GuideReport,
    GuideIssue,
)
from noethersolve.pipeline import (  # noqa: F401
    validate_pipeline,
    validate_pipeline_dict,
    TherapyDesign,
    PipelineReport,
    PipelineIssue,
)
from noethersolve.aggregation import (  # noqa: F401
    predict_aggregation,
    AggregationReport,
    AggregationIssue,
    KYTE_DOOLITTLE,
    AGGRESCAN,
)
from noethersolve.splice import (  # noqa: F401
    score_donor,
    score_acceptor,
    scan_splice_sites,
    pyrimidine_tract_score,
    SpliceSiteReport,
)
from noethersolve.pk_model import (  # noqa: F401
    one_compartment_iv,
    one_compartment_oral,
    half_life,
    steady_state,
    auc_single_dose,
    dose_adjustment,
    IVBolusReport,
    OralDosingReport,
    HalfLifeReport,
    SteadyStateReport,
    AUCReport,
    DoseAdjustmentReport,
)

from noethersolve.complexity import (  # noqa: F401
    audit_complexity,
    check_inclusion,
    check_completeness,
    get_class_info,
    ComplexityReport,
    ComplexityIssue,
)
from noethersolve.conjecture_status import (  # noqa: F401
    check_conjecture,
    check_claim,
    list_conjectures,
    get_conjecture,
    ConjectureReport,
    ConjectureIssue,
    ConjectureInfo,
)
from noethersolve.proof_barriers import (  # noqa: F401
    check_barriers,
    list_barriers,
    list_techniques,
    get_barrier,
    what_works_for,
    BarrierReport,
    BarrierIssue,
    BarrierInfo,
)
from noethersolve.number_theory import (  # noqa: F401
    verify_goldbach,
    verify_collatz,
    verify_twin_primes,
    check_abc_triple,
    verify_legendre,
    prime_gap_analysis,
    is_prime,
    prime_sieve,
    radical,
    GoldbachReport,
    CollatzReport,
    TwinPrimeReport,
    ABCReport,
    LegendreReport,
    PrimeGapReport,
)
from noethersolve.reductions import (  # noqa: F401
    validate_chain,
    check_reduction,
    strongest_reduction,
    list_known_reductions,
    get_reduction_info,
    ChainReport,
    ChainIssue,
    ReductionResult,
)
from noethersolve.pde_regularity import (  # noqa: F401
    check_sobolev_embedding,
    check_pde_regularity,
    critical_exponent,
    check_blowup,
    sobolev_conjugate,
    EmbeddingReport,
    RegularityReport,
    CriticalExponentReport,
    BlowupReport,
    EmbeddingIssue,
)

from noethersolve.llm_claims import (  # noqa: F401
    audit_llm_claims,
    check_llm_claim,
    check_benchmark_score,
    chinchilla_optimal,
    get_llm_topic,
    list_llm_topics,
    list_domains,
    LLMClaimReport,
    LLMClaimResult,
    LLMClaimIssue,
    LLMTopicInfo,
)

from noethersolve.control import (  # noqa: F401
    simulate_pid,
    analyze_stability,
    PIDReport,
    StabilityReport,
)

from noethersolve.isolation import (  # noqa: F401
    check_isolation,
    analyze_schedule,
    list_anomalies,
    IsolationReport,
    ScheduleReport,
)

from noethersolve.quantum_circuit import (  # noqa: F401
    simulate_circuit,
    measure_state,
    CircuitReport,
)

from noethersolve.chemistry_calc import (  # noqa: F401
    nernst_equation,
    henderson_hasselbalch,
    crystal_field_splitting,
    band_gap_analysis,
    balance_redox,
    NernstReport,
    BufferReport,
    CrystalFieldReport,
    BandGapReport,
)

from noethersolve.crypto_calc import (  # noqa: F401
    security_level,
    birthday_bound,
    rsa_key_analysis,
    cipher_mode_analysis,
    SecurityLevelReport,
    BirthdayBoundReport,
    RSAKeyReport,
    CipherModeReport,
)

from noethersolve.finance_calc import (  # noqa: F401
    black_scholes,
    put_call_parity,
    nash_equilibrium_2x2,
    present_value,
    future_value,
    BlackScholesReport,
    PutCallParityReport,
    NashEquilibriumReport,
    PresentValueReport,
)

from noethersolve.distributed_calc import (  # noqa: F401
    quorum_calc,
    byzantine_threshold,
    vector_clock_compare,
    consistency_model,
    gossip_convergence,
    QuorumReport,
    ByzantineReport,
    VectorClockReport,
    ConsistencyReport,
    GossipReport,
)

from noethersolve.network_calc import (  # noqa: F401
    bandwidth_delay_product,
    tcp_throughput,
    subnet_calc,
    ip_fragmentation,
    congestion_window,
    BandwidthDelayReport,
    TCPThroughputReport,
    SubnetReport,
    FragmentationReport,
    CongestionWindowReport,
)

from noethersolve.os_calc import (  # noqa: F401
    page_table_calc,
    schedule_fcfs,
    schedule_sjf,
    schedule_round_robin,
    detect_deadlock,
    tlb_analysis,
    context_switch_cost,
    PageTableReport,
    SchedulingReport,
    DeadlockReport,
    TLBReport,
    ContextSwitchReport,
)

from noethersolve.enzyme_kinetics import (  # noqa: F401
    michaelis_menten,
    inhibition,
    catalytic_efficiency,
    lineweaver_burk,
    ph_rate_profile,
    cooperativity,
    MMReport,
    InhibitionReport,
    EfficiencyReport,
    LineweaverBurkReport,
    PHProfileReport,
    CooperativityReport,
    DIFFUSION_LIMIT,
)

try:
    from noethersolve.reaction_engine import (  # noqa: F401
        analyze_molecule,
        predict_selectivity,
        predict_mechanism,
        validate_synthesis,
        check_baldwin,
        check_woodward_hoffmann,
        list_mayr_nucleophiles,
        list_mayr_electrophiles,
        list_reaction_templates,
        get_reaction_template,
        MoleculeAnalysis,
        SelectivityReport,
        MechanismReport,
        SynthesisReport,
        BaldwinReport,
        WoodwardHoffmannReport,
    )
except ImportError:
    pass  # RDKit not installed — reaction engine unavailable, other tools still work

from noethersolve.qm_calculator import (  # noqa: F401
    particle_in_box,
    hydrogen_energy,
    uncertainty_check,
    tunneling_probability,
    harmonic_oscillator,
    angular_momentum_addition,
    ParticleInBoxReport,
    HydrogenEnergyReport,
    UncertaintyReport,
    TunnelingReport,
    HarmonicOscillatorReport,
    AngularMomentumReport,
)

from noethersolve.elliptic_curves import (  # noqa: F401
    # Core arithmetic
    is_on_curve,
    point_add,
    point_double,
    point_negate,
    scalar_mult,
    point_order,
    # Invariants
    discriminant,
    j_invariant,
    is_singular,
    # Point counting
    hasse_bounds,
    count_points_naive,
    verify_hasse,
    find_points,
    # Torsion
    is_valid_torsion_order,
    # Reports
    analyze_curve,
    analyze_point_arithmetic,
    EllipticCurveReport,
    PointArithmeticReport,
)

from noethersolve.intersection_theory import (  # noqa: F401
    # Bezout
    bezout_intersection,
    BezoutReport,
    # Genus
    genus_degree_formula,
    GenusReport,
    # Self-intersection
    self_intersection_Pn,
    self_intersection_line_P2,
    self_intersection_exceptional,
    SelfIntersectionReport,
    # Canonical divisor
    canonical_P2,
    canonical_cubic_surface,
    del_pezzo_degree,
    CanonicalReport,
    # Noether formula
    noether_formula,
    NoetherReport,
    # Classical enumerative
    lines_on_cubic_surface,
    bitangents_to_quartic,
    conics_through_5_points,
    lines_meeting_4_general_lines_P3,
    plane_cubics_through_9_points,
    rational_curves_on_quintic_threefold,
    EnumerativeReport,
    # Chow ring
    chow_ring_Pn,
    segre_embedding_degree,
    # Intersection multiplicity
    intersection_multiplicity_formula,
    compute_multiplicity_smooth_transverse,
    compute_multiplicity_tangent,
    # Adjunction formula
    adjunction_formula,
    adjunction_complete_intersection,
    AdjunctionReport,
    # Blow-up formulas
    blowup_K_squared,
    blowup_P2,
    blowup_transform_divisor,
    BlowupReport,
    # Ruled surfaces
    ruled_surface,
    hirzebruch_surface,
    RuledSurfaceReport,
    # Toric varieties
    toric_canonical,
    toric_Pn_canonical,
    ToricCanonicalReport,
)

from noethersolve.information_theory import (  # noqa: F401
    # Entropy functions
    binary_entropy,
    entropy,
    relative_entropy,
    mutual_information,
    # Channel capacity
    capacity_bsc,
    capacity_bec,
    capacity_awgn,
    capacity_z_channel,
    ChannelCapacityReport,
    # Rate-distortion
    rate_distortion_binary,
    rate_distortion_gaussian,
    RateDistortionReport,
    # Source coding
    source_coding_bound,
    SourceCodingReport,
    # MAC
    mac_capacity_region_2user,
    MACRegionReport,
    # Data processing
    check_data_processing,
    DataProcessingReport,
    # Fano
    fano_inequality,
    FanoReport,
    # Typical sets
    typical_set_bounds,
    TypicalSetReport,
)

from noethersolve.drug_interactions import (  # noqa: F401
    # Drug profiles
    get_drug_profile,
    DrugProfileReport,
    # CYP info
    get_cyp_info,
    CYPInfoReport,
    # Interactions
    check_interaction,
    predict_auc_change,
    InteractionReport,
    # Pharmacogenomics
    check_pharmacogenomics,
    PharmacogenomicsReport,
    # List functions
    list_cyp_enzymes,
    list_substrates,
    list_inhibitors,
    list_inducers,
    # Enums
    Strength,
    InteractionType,
    Severity,
)

from noethersolve.epidemiology import (  # noqa: F401
    herd_immunity_threshold,
    reproduction_number,
    doubling_time,
    attack_rate,
    sir_model,
    vaccine_impact,
    generation_interval,
    seir_parameters,
    get_disease_R0,
    list_diseases,
    HerdImmunityReport,
    ReproductionNumberReport,
    DoublingTimeReport,
    AttackRateReport,
    SIRReport,
    VaccineImpactReport,
    GenerationIntervalReport,
)

from noethersolve.radiative_transfer import (  # noqa: F401
    radiative_forcing,
    planck_response,
    climate_sensitivity,
    stefan_boltzmann,
    effective_temperature,
    analyze_feedback,
    list_feedbacks,
    RadiativeForcingReport,
    PlanckResponseReport,
    ClimateSensitivityReport,
    StefanBoltzmannReport,
    EffectiveTemperatureReport,
    FeedbackAnalysisReport,
)

from noethersolve.turbulence import (  # noqa: F401
    kolmogorov_45_law,
    energy_spectrum,
    length_scales,
    structure_function_exponent,
    intermittency_analysis,
    is_in_inertial_range,
    inertial_range_extent,
    Kolmogorov45Report,
    EnergySpectrumReport,
    LengthScalesReport,
    StructureFunctionReport,
    IntermittencyReport,
)

from noethersolve.topological_invariants import (  # noqa: F401
    chern_number,
    z2_invariant,
    bulk_boundary_correspondence,
    quantum_hall,
    berry_phase,
    topological_classification,
    list_symmetry_classes,
    ChernNumberReport,
    Z2InvariantReport,
    BulkBoundaryReport,
    QuantumHallReport,
    BerryPhaseReport,
    TopologicalClassReport,
    VON_KLITZING,
    CONDUCTANCE_QUANTUM,
)

from noethersolve.ergodic_theory import (  # noqa: F401
    classify_system,
    compare_levels,
    lyapunov_analysis,
    entropy_analysis,
    poincare_recurrence,
    mixing_rate,
    list_systems as list_dynamical_systems,
    list_levels as list_ergodic_levels,
    is_stronger,
    implies,
    HierarchyReport,
    LyapunovReport,
    EntropyReport,
    RecurrenceReport,
    ComparisonReport,
    MixingRateReport,
)

from noethersolve.optimization_convergence import (  # noqa: F401
    gradient_descent_rate,
    nesterov_rate,
    compare_algorithms,
    analyze_conditioning,
    oracle_lower_bound,
    optimal_step_size,
    non_convex_rate,
    list_algorithms,
    iterations_needed,
    ConvergenceReport,
    ComparisonReport as OptimizationComparisonReport,
    ConditionReport,
    LowerBoundReport,
    StepSizeReport,
    NonConvexReport,
)

from noethersolve.numerical_pde import (  # noqa: F401
    check_cfl,
    cfl_hyperbolic,
    cfl_parabolic,
    max_timestep,
    von_neumann_analysis,
    get_scheme_info,
    list_schemes,
    check_lax_equivalence,
    analyze_accuracy,
    check_common_error,
    CFLReport,
    VonNeumannReport,
    SchemeReport,
    LaxEquivalenceReport,
    AccuracyReport,
)

from noethersolve.mhd_conservation import (  # noqa: F401
    check_magnetic_helicity,
    check_cross_helicity,
    check_mhd_energy,
    check_frozen_flux,
    check_div_B,
    check_mhd_invariant,
    list_mhd_invariants,
    HelicityReport,
    MHDEnergyReport,
    FrozenFluxReport,
    DivBReport,
    InvariantReport as MHDInvariantReport,
)

from noethersolve.gr_constraints import (  # noqa: F401
    check_hamiltonian_constraint,
    check_momentum_constraint,
    check_adm_mass,
    check_bondi_mass,
    check_komar_mass,
    compare_mass_definitions,
    analyze_adm_formalism,
    list_gr_concepts,
    ConstraintReport,
    MassReport,
    MassComparisonReport,
    ADMReport,
)

from noethersolve.seismic_waves import (  # noqa: F401
    calc_seismic_velocity,
    calc_velocity_from_poisson,
    poisson_from_velocities,
    convert_elastic_moduli,
    calc_reflection_coefficient,
    critical_angle,
    snells_law,
    vp_vs_ratio_bounds,
    SeismicVelocityReport,
    PoissonRatioReport,
    ElasticModuliReport,
    ReflectionReport,
)

from noethersolve.plasma_adiabatic import (  # noqa: F401
    calc_magnetic_moment,
    calc_bounce_invariant,
    calc_flux_invariant,
    check_adiabatic_hierarchy,
    mirror_force,
    loss_cone_angle,
    cyclotron_frequency,
    larmor_radius,
    get_particle_mass,
    MagneticMomentReport,
    BounceInvariantReport,
    FluxInvariantReport,
    AdiabaticHierarchyReport,
    ELECTRON_MASS,
    PROTON_MASS,
    ELECTRON_CHARGE,
)

from noethersolve.dimension_physics import (  # noqa: F401
    check_dimension_dependence,
    get_formula,
    list_dimension_dependent_concepts,
    DimensionalFormula,
    DimensionCheckResult,
    DIMENSIONAL_PHYSICS,
)

from noethersolve.tool_graph import (  # noqa: F401
    calculator,
    get_registry,
    find_tool_chain,
    execute_chain,
    ToolRegistry,
    CalculatorMeta,
)

from noethersolve.meta_router import (  # noqa: F401
    MetaRouter,
    MetaRouterConfig,
    OutcomeRecord,
    FactEmbedder,
)

from noethersolve.stage_discovery import (  # noqa: F401
    StageDiscoverer,
    DiscoveryConfig,
    EvalResult,
    StageSequence,
)

from noethersolve.outcome_logger import (  # noqa: F401
    OutcomeLogger,
    get_logger as get_outcome_logger,
    log_outcome,
    log_batch,
)

__version__ = "1.18.0"
