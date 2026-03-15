"""NoetherSolve — find where LLM knowledge ends, build verified tools, serve
them to any AI agent via MCP.

The pipeline: find gaps → flip facts → build tool → add to MCP server.
Every tool added makes every connected agent smarter.

Emmy Noether proved every continuous symmetry corresponds to a conserved quantity.
NoetherSolve finds where LLMs fail to recognize those quantities, builds verified
computational tools for the right answers, and exposes them via Model Context
Protocol (MCP) — 46 tools currently serving physics, math, genetics, complexity
theory, pharmacogenomics, biochemistry, organic chemistry, quantum mechanics,
and LLM science. 30 are calculators (derive answers from first principles),
16 are lookup tables (reference databases).

Package layout:
  noethersolve.mcp_server   — MCP server (46 tools for any AI agent)
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
from noethersolve.pharmacokinetics import (  # noqa: F401
    audit_drug_list,
    check_drug_interactions,
    check_phenotype,
    check_hla,
    get_enzyme_for_drug,
    get_interactions,
    PharmReport,
    DrugInteraction,
    PharmIssue,
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

from noethersolve.biochemistry import (  # noqa: F401
    check_biochemistry,
    get_biochemistry_topic,
    list_biochemistry_topics,
    BiochemistryReport,
    BiochemistryIssue,
    BiochemistryInfo,
)

from noethersolve.organic_chemistry import (  # noqa: F401
    check_organic_chemistry,
    get_organic_chemistry_topic,
    list_organic_chemistry_topics,
    OrganicChemistryReport,
    OrganicChemistryIssue,
    OrganicChemistryTopic,
)

from noethersolve.quantum_mechanics import (  # noqa: F401
    check_quantum_mechanics,
    get_quantum_mechanics_topic,
    list_quantum_mechanics_topics,
    QuantumMechanicsReport,
    QuantumMechanicsIssue,
    QMTopicInfo,
)

__version__ = "1.1.0"
