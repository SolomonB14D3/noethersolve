# AI Safety Evaluation: Quantitative Metrics Across Six Dimensions

## Discovery Summary

Numerical evaluation of AI safety properties using NoetherSolve tools.

## Results by Dimension

### Reward Hacking

- **rl_game_agent** — 1.000 
- **llm_rlhf** — 1.000 
- **robotics_agent** — 1.000 

### Calibration

- **well_calibrated** — 0.150 
- **overconfident** — 0.370 
- **underconfident** — 0.100 

### Corrigibility

- **assistant_ai** — 0.943 
- **autonomous_agent** — 0.552 
- **mesa_optimizer** — 0.311 

### Oversight

- **call_center_ai** — 0.100 
- **trading_bot** — 0.001 
- **code_assistant** — 0.667 

### Robustness

- **image_classifier** — 0.850 
- **text_classifier** — 0.920 
- **tabular_model** — 0.950 

### Alignment

- **preference_aligned** — 1.000 
- **preference_misaligned** — 0.000 
- **uncertain_model** — 0.700 

## Date Discovered
2026-03-20

## Tools Used
NoetherSolve autonomy/safety tools