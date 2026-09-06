"""Check all environments for dcor"""
import os
import subprocess

envs_dir = r"E:\anaconda3\envs"

# Get all environment directories
envs = [d for d in os.listdir(envs_dir) if os.path.isdir(os.path.join(envs_dir, d))]

print("Checking environments for dcor...")
for env in sorted(envs):
    env_python = os.path.join(envs_dir, env, "python.exe")
    if not os.path.exists(env_python):
        continue

    result = subprocess.run([env_python, "-c",
        "import dcor; print(dcor.__version__)"],
        capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  {env}: dcor={result.stdout.strip()}")
