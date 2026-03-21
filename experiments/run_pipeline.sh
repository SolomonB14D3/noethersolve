#!/bin/bash
# Pipeline: finish vector extraction, then train adapters on failures
# Run with: nohup bash experiments/run_pipeline.sh > logs/pipeline.log 2>&1 &

cd "/Volumes/4TB SD/ClaudeCode/noethersolve"

echo "=== Waiting for vector extraction (PID 73710) to finish ==="
while kill -0 73710 2>/dev/null; do
    sleep 60
    DONE=$(tail -1 logs/extract_vectors_v2.log | grep -oP '\[\K[0-9]+' | head -1)
    echo "  $(date): vectors at $DONE/569"
done
echo "=== Vector extraction complete at $(date) ==="

echo ""
echo "=== Starting adapter training on steering failures ==="
PYTHONUNBUFFERED=1 python3 experiments/train_steering_failures.py
echo "=== Adapter training complete at $(date) ==="
