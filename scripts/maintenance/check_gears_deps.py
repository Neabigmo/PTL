"""Check GEARS dependencies"""
import sys
import subprocess

env_python = r"E:\anaconda3\envs\pytorch-clean\python.exe"

# Check torch
result = subprocess.run([env_python, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True)
print(f"torch: {result.stdout.strip()}")

# Check torch_geometric
result = subprocess.run([env_python, "-c", "import torch_geometric; print(torch_geometric.__version__)"], capture_output=True, text=True)
if result.returncode == 0:
    print(f"torch_geometric: {result.stdout.strip()}")
else:
    print(f"torch_geometric: NOT INSTALLED")
    print(f"  Error: {result.stderr.strip()}")

# Check scanpy
result = subprocess.run([env_python, "-c", "import scanpy; print(scanpy.__version__)"], capture_output=True, text=True)
if result.returncode == 0:
    print(f"scanpy: {result.stdout.strip()}")
else:
    print(f"scanpy: NOT INSTALLED")