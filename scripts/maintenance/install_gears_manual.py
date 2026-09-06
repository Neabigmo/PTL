"""Copy GEARS source to sw_mgli site-packages"""
import shutil
import os

src_dir = r"h:/2026try/4.24/third_party/GEARS/gears"
dest_dir = r"E:\anaconda3\envs\sw_mgli\Lib\site-packages\gears"

# Remove existing gears directory if exists
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

# Copy gears module
shutil.copytree(src_dir, dest_dir)
print(f"Copied GEARS to {dest_dir}")

# Verify by listing files
for root, dirs, files in os.walk(dest_dir):
    for f in files:
        print(f"  {os.path.join(root, f)}")
