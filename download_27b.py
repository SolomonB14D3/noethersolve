import os
os.environ["HF_HOME"] = "/Volumes/4TB SD/ml_cache/huggingface"

from huggingface_hub import snapshot_download
import time

print(f"Downloading mlx-community/Qwen3.5-27B-4bit to {os.environ['HF_HOME']}")
start = time.time()

try:
    path = snapshot_download(
        "mlx-community/Qwen3.5-27B-4bit",
        local_files_only=False,
        resume_download=True,
    )
    print(f"Download complete! Path: {path}")
    print(f"Time: {time.time() - start:.1f}s")
except Exception as e:
    print(f"Error: {e}")
