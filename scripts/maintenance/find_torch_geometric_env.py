"""Check all environments for torch_geometric"""
import os
import subprocess
import glob

envs_dir = r"E:\anaconda3\envs"

# Get all environment directories
envs = [d for d in os.listdir(envs_dir) if os.path.isdir(os.path.join(envs_dir, d))]

print("Checking environments for torch_geometric...")
for env in sorted(envs):
    env_python = os.path.join(envs_dir, env, "python.exe")
    if not os.path.exists(env_python):
        continue

    result = subprocess.run([env_python, "-c",
        "import torch; import torch_geometric; print(torch_geometric.__version__)"],
        capture_output=True, text=True)
    if result.returncode == 0:
        torch_version = subprocess.run([env_python, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True).stdout.strip()
        print(f"  {env}: torch={torch_version}, torch_geometric={result.stdout.strip()}")
