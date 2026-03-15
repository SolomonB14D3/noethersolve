"""NoetherSolve — find physical/mathematical structures that are numerically
conserved but not recognized by LLMs, then close those gaps with targeted adapters.

Emmy Noether proved every continuous symmetry corresponds to a conserved quantity.
NoetherSolve finds where LLMs fail to recognize those quantities and fixes it.

Package layout:
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
"""

from noethersolve import train_utils  # noqa: F401
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

__version__ = "0.7.0"
