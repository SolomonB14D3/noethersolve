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
from noethersolve.hamiltonian import (  # noqa: F401
    HamiltonianMonitor,
    HamiltonianReport,
    harmonic_oscillator,
    kepler_2d,
    henon_heiles,
    coupled_oscillators,
)
from noethersolve.learner import InvariantLearner, LearnerReport  # noqa: F401

__version__ = "0.4.0"
