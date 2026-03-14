# Papers

The two papers that directly underpin NoetherSolve:

| File | Title | DOI |
|------|-------|-----|
| `paper9_stem_truth_oracle.pdf` | STEM Truth Oracle: Log-Prob MC Ranking Reveals and Corrects Scale-Invariant Factual Biases | [10.5281/zenodo.19005729](https://doi.org/10.5281/zenodo.19005729) |
| `paper8_snap_on_communication_modules.pdf` | Snap-On Communication Modules: Frozen Logit-Space Adapters | [10.5281/zenodo.18902616](https://doi.org/10.5281/zenodo.18902616) |

**Paper 9** defines the oracle (log-prob margin as a perfect binary classifier),
the four diagnostic quadrants, and the repair methodology used in `oracle_wrapper.py`.

**Paper 8** defines the snap-on logit adapter architecture (SwiGLU in logit space,
zero-init output, zero-mean centering) vendored in `noethersolve/adapter.py`.
