"""
Wildfire Detection RL Training (Fast Version)
==============================================
"""

import os
import sys
import numpy as np
from collections import defaultdict
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pre-computed frame data to avoid slow TensorFlow loading
FRAME_DATA = [
    {"prediction": [0.9, 0.1, 0.0], "ground_truth": 0, "frame": 0},  # fire
    {"prediction": [0.85, 0.15, 0.0], "ground_truth": 0, "frame": 1},  # fire
    {"prediction": [0.8, 0.2, 0.0], "ground_truth": 0, "frame": 2},  # fire
    {"prediction": [0.7, 0.25, 0.05], "ground_truth": 0, "frame": 3},  # fire
    {"prediction": [0.6, 0.3, 0.1], "ground_truth": 0, "frame": 4},  # fire
    {"prediction": [0.5, 0.4, 0.1], "ground_truth": 1, "frame": 5},  # smoke
    {"prediction": [0.3, 0.6, 0.1], "ground_truth": 1, "frame": 6},  # smoke
    {"prediction": [0.2, 0.7, 0.1], "ground_truth": 1, "frame": 7},  # smoke
    {"prediction": [0.1, 0.8, 0.1], "ground_truth": 1, "frame": 8},  # smoke
    {"prediction": [0.1, 0.85, 0.05], "ground_truth": 1, "frame": 9},  # smoke
    {"prediction": [0.0, 0.1, 0.9], "ground_truth": 2, "frame": 10},  # no_fire
    {"prediction": [0.0, 0.05, 0.95], "ground_truth": 2, "frame": 11},  # no_fire
    {"prediction": [0.05, 0.05, 0.9], "ground_truth": 2, "frame": 12},  # no_fire
    {"prediction": [0.0, 0.0, 1.0], "ground_truth": 2, "frame": 13},  # no_fire
    {"prediction": [0.0, 0.0, 1.0], "ground_truth": 2, "frame": 14},  # no_fire
    {"prediction": [0.9, 0.1, 0.0], "ground_truth": 0, "frame": 15},  # fire (repeat)
]


class FastEnv:
    """Fast environment without TensorFlow."""

    ACTIONS = ["Alert", "Scan", "Ignore", "Deploy"]
    CLASS_LABELS = ["fire", "smoke", "no_fire"]

    def __init__(self):
        self.frame_idx = 0
        self.n_frames = len(FRAME_DATA)

    def reset(self):
        self.frame_idx = 0
        return self._get_obs()

    def step(self, action):
        frame = FRAME_DATA[self.frame_idx]
        gt = frame["ground_truth"]
        action_str = self.ACTIONS[action]

        # Reward function
        if gt == 2:  # no_fire
            if action_str in ["Alert", "Deploy"]:
                reward = -0.75
            elif action_str == "Ignore":
                reward = 0.1
            else:
                reward = 0.0
        else:  # fire or smoke
            if action_str == "Ignore":
                reward = -0.50
            elif action_str in ["Alert", "Deploy"]:
                reward = 0.50
            else:  # Scan
                reward = 0.0

        # Move to next frame BEFORE checking done
        self.frame_idx += 1
        done = self.frame_idx >= self.n_frames

        # If we moved past valid frames, return last observation but mark done
        if done:
            obs = {
                "prediction": np.array([0, 0, 0], dtype=np.float32),
                "ground_truth": np.array([2], dtype=np.int32),
                "frame_id": np.array([self.frame_idx - 1], dtype=np.int32),
            }
        else:
            obs = self._get_obs()

        return obs, reward, done, {"ground_truth": self.CLASS_LABELS[gt]}

    def _get_obs(self):
        frame = FRAME_DATA[self.frame_idx]
        return {
            "prediction": np.array(frame["prediction"], dtype=np.float32),
            "ground_truth": np.array([frame["ground_truth"]], dtype=np.int32),
            "frame_id": np.array([self.frame_idx], dtype=np.int32),
        }


class QLearningAgent:
    def __init__(
        self,
        n_actions=4,
        learning_rate=0.4,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.98,
        epsilon_min=0.05,
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def get_state_key(self, obs):
        p = obs["prediction"]
        fire_high = 0 if p[0] < 0.5 else 1
        smoke_high = 0 if p[1] < 0.5 else 1
        return (fire_high, smoke_high)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * max_next_q
        self.q_table[state][action] = current_q + self.lr * (td_target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(num_episodes=200):
    print("Initializing fast environment...")
    env = FastEnv()

    agent = QLearningAgent()

    print(f"\nTraining for {num_episodes} episodes...")

    for episode in range(num_episodes):
        obs = env.reset()
        state = agent.get_state_key(obs)
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            obs, reward, done, info = env.step(action)
            next_state = agent.get_state_key(obs)

            agent.learn(state, action, reward, next_state)
            total_reward += reward
            state = next_state

        agent.decay_epsilon()

        if (episode + 1) % 20 == 0:
            print(
                f"Episode {episode + 1}/{num_episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}"
            )

    # Save model
    model_path = os.path.join(os.path.dirname(__file__), "q_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(dict(agent.q_table), f)

    print(f"\nTraining complete!")
    print(f"Q-table: {dict(agent.q_table)}")
    print(f"Model saved to: {model_path}")

    return agent


def evaluate():
    env = FastEnv()

    model_path = os.path.join(os.path.dirname(__file__), "q_model.pkl")
    with open(model_path, "rb") as f:
        q_table = pickle.load(f)

    agent = QLearningAgent()
    agent.q_table = defaultdict(lambda: np.zeros(4), q_table)
    agent.epsilon = 0

    print("\n=== Evaluation ===")

    for ep in range(3):
        obs = env.reset()
        state = agent.get_state_key(obs)
        total_reward = 0
        done = False
        actions = []

        while not done:
            action = agent.choose_action(state)
            actions.append(["Alert", "Scan", "Ignore", "Deploy"][action])
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            state = agent.get_state_key(obs)

        print(f"Episode: {actions}")
        print(f"Reward: {total_reward:.2f}")


if __name__ == "__main__":
    train_agent(200)
    evaluate()
