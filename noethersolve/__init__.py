"""NoetherSolve — find physical/mathematical structures that are numerically
conserved but not recognized by LLMs, then close those gaps with targeted adapters.

Emmy Noether proved every continuous symmetry corresponds to a conserved quantity.
NoetherSolve finds where LLMs fail to recognize those quantities and fixes it.

Package layout:
  noethersolve.oracle       — model-agnostic MC log-prob scorer (from eval_mc)
  noethersolve.adapter      — snap-on logit adapter architectures (from snap_on)
  noethersolve.train_utils  — LOGIT_SOFTCAP, get_lm_head_fn, apply_adapter
"""

from noethersolve import train_utils  # noqa: F401

__version__ = "0.1.0"
