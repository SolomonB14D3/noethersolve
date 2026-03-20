# Scripts

## Role Separation

| Actor | Role | What it does with scripts |
|-------|------|--------------------------|
| **27B (Apple Silicon)** | Local compute workhorse | Runs `adapter_trainer.py` to train 4B adapters. Provides GPU compute via MLX. |
| **4B (Qwen3-4B-Base)** | Student model | Gets adapters trained FOR it. Forward/backward passes go through 4B. |
| **Claude Code** | Manager | Creates V2 fact files, writes papers, handles escalations, writes code. |

**The 27B does NOT get adapters.** It trains adapters FOR the 4B.
**The 4B does NOT train itself.** The 27B's compute trains it.
**Claude Code does NOT train models.** It creates the data and manages the pipeline.

## 27B Work Scripts (local compute — the workhorse)

| Script | Purpose |
|--------|---------|
| `adapter_trainer.py` | **PRIMARY** — Train 4B adapters on failing domains using escalation ladder |
| `train_from_facts.py` | Train single 4B Snap-On adapter from a facts file |
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
# The 27B provides compute; adapters target the 4B model
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
