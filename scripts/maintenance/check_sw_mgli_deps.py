"""Check sw_mgli environment for all GEARS dependencies"""
import os
import subprocess

env_python = r"E:\anaconda3\envs\sw_mgli\python.exe"

packages = [
    "torch",
    "torch_geometric",
    "numpy",
    "pandas",
    "tqdm",
    "scikit-learn",
    "scanpy",
    "networkx",
    "dcor",
    "scipy",
]

print(f"Checking sw_mgli environment ({env_python}):")
print("-" * 50)

for pkg in packages:
    result = subprocess.run([env_python, "-c",
        f"import {pkg.replace('-', '_')}; print({pkg.replace('-', '_')}.__version__)"],
        capture_output=True, text=True)
    if result.returncode == 0:
        version = result.stdout.strip()
        print(f"  {pkg}: {version}")
    else:
        print(f"  {pkg}: NOT INSTALLED")

# Check GEARS directly
print("-" * 50)
result = subprocess.run([env_python, "-c", "import GEARS; print('GEARS version:', GEARS.__version__)"],
    capture_output=True, text=True)
if result.returncode == 0:
    print(f"  GEARS: {result.stdout.strip()}")
else:
    print(f"  GEARS: NOT INSTALLED")
    if result.stderr:
        print(f"    Error: {result.stderr.strip()[:200]}")