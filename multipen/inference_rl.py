"""
Wildfire Detection RL Inference Script
=======================================
Uses trained Q-learning model for decision making.
"""

import os
import sys
import pickle
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environments.wildfire_detection.wildfire_env import WildfireDetectionEnv


class TrainedQLearningAgent:
    """Trained Q-learning agent for inference."""

    def __init__(self, model_path=None):
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), "q_model.pkl"
        )
        self.q_table = None
        self.actions = ["Alert", "Scan", "Ignore", "Deploy"]
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0], pickle.load(f))
            print(f"Loaded trained model from: {self.model_path}")
            print(f"Q-table: {dict(self.q_table)}")
        else:
            print("WARNING: No trained model found!")
            self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def get_state_key(self, obs):
        """Convert observation to discrete state key."""
        prediction = obs.get("prediction", np.zeros(3))
        if hasattr(prediction, "tolist"):
            p = prediction.tolist()
        elif hasattr(prediction, "__iter__"):
            p = list(prediction)
        else:
            p = [0, 0, 0]

        # Match training: fire_high, smoke_high
        fire_high = 0 if p[0] < 0.5 else 1
        smoke_high = 0 if p[1] < 0.5 else 1

        return (fire_high, smoke_high)

    def choose_action(self, state):
        """Choose best action based on Q-values."""
        q_values = self.q_table.get(state, [0.0, 0.0, 0.0, 0.0])
        if hasattr(q_values, "tolist"):
            q_values = q_values.tolist()
        return q_values.index(max(q_values))

    def get_action_name(self, action_idx):
        return self.actions[action_idx]


def run_inference():
    """Run inference with trained RL agent."""
    print(
        "[START] task=wildfire_detection env=wildfire_detection model=trained_q_learning"
    )

    env = WildfireDetectionEnv()
    agent = TrainedQLearningAgent()

    rewards_list = []
    steps_executed = 0

    obs = env.reset()

    for step in range(1, 21):
        steps_executed = step

        state = agent.get_state_key(obs)
        action_idx = agent.choose_action(state)
        action_str = agent.get_action_name(action_idx)

        obs, reward, done, info = env.step(action_idx)
        rewards_list.append(reward)

        error_msg = "null"
        print(
            f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}"
        )

        if done:
            break

    env.close()

    total_reward = sum(rewards_list)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)
    success = total_reward > 0

    print(
        f"[END] success={str(success).lower()} steps={steps_executed} rewards={rewards_str}"
    )

    return {
        "success": success,
        "steps": steps_executed,
        "rewards": rewards_list,
        "total_reward": total_reward,
    }


if __name__ == "__main__":
    result = run_inference()
    print(
        f"\nResult: success={result['success']}, total_reward={result['total_reward']:.2f}"
    )
