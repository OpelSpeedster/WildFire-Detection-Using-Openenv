import os

os.environ["PYTHONIOENCODING"] = "utf-8"
import subprocess

result = subprocess.run(
    [
        "python",
        "-m",
        "openenv.cli",
        "push",
        "--repo-id",
        "OpelSpeedster/Wildfire-Detection_Openenv",
    ],
    cwd=r"C:\Users\ASUS\OneDrive\Desktop\RL\multipen",
    capture_output=True,
    text=True,
    encoding="utf-8",
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
