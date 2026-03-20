# Scripts

## 27B Work Scripts (local compute — the workhorse)

| Script | Purpose |
|--------|---------|
| `adapter_trainer.py` | **PRIMARY** — Train 4B adapters on failing domains using escalation ladder |
| `train_from_facts.py` | Train single 4B LoRA adapter from a fact file |
| `train_missing_adapters.py` | Fill gaps in 4B adapter coverage |
| `train_with_proven_methods.py` | Analyze domain + recommend training technique (staged/orthogonal/etc.) |
| `research_runner.py` | Oracle evaluation — **DONE** (111 domains complete, do not restart) |

## Claude Code Scripts (needs internet/reasoning)

| Script | Purpose |
|--------|---------|
| `paper_prospector.py` | Scan for paper-ready clusters across domains |
| `paper_watchdog.py` | Monitor paper pipeline progress |
| `discovery_grader.py` | Grade discovery clusters for paper readiness |
| `research_dashboard.py` | Serve live HTML dashboard on port 8050 |
| `hardware_profile.py` | Detect hardware capabilities (MLX, CUDA, CPU) |
| `job_tracker.py` | Track long-running job status |

## Usage

```bash
# Train 4B adapters on all failing domains (the 27B's main job now)
python scripts/adapter_trainer.py

# Train one domain and exit
python scripts/adapter_trainer.py --once

# Check which domains need adapter training
python scripts/adapter_trainer.py --status

# Train specific domain
python scripts/adapter_trainer.py --domain knot_invariants
```

## Archive

Superseded scripts in `archive/`. Kept for reference only — do not run.
