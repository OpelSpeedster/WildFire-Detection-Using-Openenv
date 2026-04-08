import sys
import os

sys.path.insert(0, r"C:\Users\ASUS\OneDrive\Desktop\RL\multipen")

from environments.wildfire_detection.wildfire_env import WildfireDetectionEnv

print("Creating environment...")
env = WildfireDetectionEnv()
print("Environment created successfully")

obs = env.reset()
print(f"Reset successful, step: {obs['step']}")

action = 0  # Alert
obs, reward, done, info = env.step(action)
print(f"Step taken: action={info['action']}, reward={reward}, done={done}")

print("Environment test passed!")
