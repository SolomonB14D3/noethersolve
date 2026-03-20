# Scripts

## Active

| Script | Purpose |
|--------|---------|
| `research_runner.py` | Autonomous 27B oracle evaluation loop. Discovers domains, evaluates facts, tracks progress, stops when idle. |
| `train_from_facts.py` | Train 4B LoRA adapters from fact files (the adapter pipeline) |
| `train_missing_adapters.py` | Fill gaps in 4B adapter coverage |
| `train_with_proven_methods.py` | Adapter training using escalation ladder (staged, orthogonal, etc.) |
| `discovery_grader.py` | Grade discovery clusters for paper readiness |
| `paper_prospector.py` | Scan for paper-ready clusters across domains |
| `paper_watchdog.py` | Monitor paper pipeline progress |
| `research_dashboard.py` | Serve live HTML dashboard on port 8050 |
| `hardware_profile.py` | Detect hardware capabilities (MLX, CUDA, CPU) |
| `job_tracker.py` | Track long-running job status |

## Usage

```bash
# Start autonomous evaluation (polls for new V2 files, stops when idle)
python scripts/research_runner.py

# Single sweep then exit
python scripts/research_runner.py --once

# Check status without running
python scripts/research_runner.py --status

# Run specific domain
python scripts/research_runner.py --domain protein_structure

# Show open escalations
python scripts/research_runner.py --escalations
```

## Archive

Superseded scripts in `archive/`. Kept for reference only — do not run.
