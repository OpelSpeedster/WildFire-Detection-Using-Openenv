import sys
import os
import numpy as np

sys.path.insert(0, r"C:\Users\ASUS\OneDrive\Desktop\RL\multipen")

from environments.wildfire_detection.wildfire_env import WildfireDetectionEnv

env = WildfireDetectionEnv()
actions = ["Alert", "Scan", "Ignore", "Deploy"]

print("=== Analyzing Environment ===")

for episode in range(2):
    print(f"\n--- Episode {episode + 1} ---")
    obs = env.reset()

    for step in range(3):
        gt = obs["ground_truth"]
        if hasattr(gt, "__len__") and gt.ndim > 0:
            gt_idx = int(gt[0])
        else:
            gt_idx = int(gt)
        gt_label = ["fire", "smoke", "no_fire"][gt_idx]

        print(f"Step {step + 1}: GT={gt_label}, Pred={obs['prediction']}")

        for a_idx, a_name in enumerate(actions):
            obs2, r, done, info = env.step(a_idx)
            print(f"  {a_name}: reward={r:.2f}")

        if step < 2:
            obs = obs2

env.close()
