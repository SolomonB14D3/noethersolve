# NoetherSolve

**Verified computational tools for AI agents, served via [Model Context Protocol](https://modelcontextprotocol.io/).**

300+ calculators spanning physics, mathematics, chemistry, pharmacokinetics, epidemiology, cryptography, finance, distributed systems, and more. Each tool derives answers from first principles and is validated by 2,265 tests.

---

## Install

```bash
pip install noethersolve
```

## Use directly

```python
from noethersolve import check_conjecture

result = check_conjecture("Riemann")
# → Status: OPEN, Clay Millennium Problem, key facts, common errors
```

## Serve to AI agents via MCP

```bash
noethersolve-mcp
```

Any MCP-compatible agent (Claude, etc.) auto-discovers the tools. In Claude Code, the `.mcp.json` is picked up automatically.

---

## What the tools do

Tools **verify and compute** — they don't generate. Give them inputs, get verified outputs.

| Category | Examples |
|----------|----------|
| Conservation laws | `check_vortex_conservation`, `check_hamiltonian_system`, `check_em_conservation` |
| Mathematics | `check_conjecture`, `check_complexity_inclusion`, `verify_goldbach`, `check_sobolev_embedding` |
| Quantum mechanics | `calc_particle_in_box`, `calc_hydrogen_energy`, `calc_tunneling` |
| Chemistry | `calc_nernst`, `calc_buffer_ph`, `analyze_molecule`, `predict_reaction_mechanism` |
| Pharmacokinetics | `calc_iv_bolus`, `calc_oral_dose`, `calc_half_life`, `calc_steady_state` |
| Epidemiology | `calc_sir_model`, `calc_reproduction_number`, `calc_herd_immunity` |
| Cryptography | `calc_security_level`, `calc_birthday_bound`, `calc_cipher_mode` |
| Finance | `calc_black_scholes`, `calc_put_call_parity`, `calc_nash_equilibrium` |
| Distributed systems | `calc_quorum`, `calc_byzantine`, `calc_vector_clock` |
| And 25+ more domains | See `noethersolve/mcp_server/server.py` for the full list |

---

## Papers

- **Correcting Suppressed Log-Probabilities in Language Models with Post-Transformer Adapters**
  Bryan Sanchez, 2026.
  [Code](https://github.com/SolomonB14D3/qwen-adapter-correction) · [Weights](https://huggingface.co/bsanch52/qwen3-adapter-correction)

---

## Cite

```bibtex
@software{noethersolve2026,
  author = {Sanchez, Bryan},
  title = {NoetherSolve: Verified Computational Tools for AI Agents},
  year = {2026},
  url = {https://github.com/SolomonB14D3/noethersolve}
}
```

## License

MIT
