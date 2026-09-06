"""Test GEARS import in sw_mgli"""
import subprocess
import sys

env_python = r"E:\anaconda3\envs\sw_mgli\python.exe"
cmd = [
    env_python, "-c",
    "import sys; sys.path.insert(0, r'E:\\anaconda3\\envs\\sw_mgli\\lib\\site-packages'); "
    "from gears import GEARS, PertData; print('OK: GEARS imported successfully')"
]

result = subprocess.run(cmd, capture_output=True, text=True)
print("stdout:", result.stdout)
print("stderr:", result.stderr[:500] if result.stderr else "")
print("returncode:", result.returncode)