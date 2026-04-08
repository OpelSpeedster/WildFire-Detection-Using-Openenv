"""
Wildfire Detection Inference Script
====================================
For LOCAL TESTING: Uses trained RL model (no API key required)
For SUBMISSION: Uses OpenAI client with API key injected by judges
"""

import os
import sys
import base64
import pickle
from io import BytesIO
from typing import List, Optional, Dict, Any
from collections import defaultdict

from openai import OpenAI
import numpy as np
from PIL import Image

# Environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

API_KEY = os.getenv("API_KEY", HF_TOKEN)

# Check if we can use LLM mode
USE_LLM = API_KEY is not None

if USE_LLM:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
else:
    client = None
    print("NOTE: Running in LOCAL MODE (no API_KEY). Using trained RL model.")


# RL Model class for local mode
class TrainedQLearningAgent:
    """Trained Q-learning agent for local inference."""

    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "q_model.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0], pickle.load(f))
            print(f"Loaded RL model from: {model_path}")
        else:
            print("WARNING: No trained model found, using fallback!")
            self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.actions = ["Alert", "Scan", "Ignore", "Deploy"]

    def get_state_key(self, obs):
        prediction = obs.get("prediction", np.zeros(3))
        if hasattr(prediction, "tolist"):
            p = prediction.tolist()
        else:
            p = list(prediction)
        fire_high = 0 if p[0] < 0.5 else 1
        smoke_high = 0 if p[1] < 0.5 else 1
        return (fire_high, smoke_high)

    def choose_action(self, state):
        q_values = self.q_table.get(state, [0.0, 0.0, 0.0, 0.0])
        if hasattr(q_values, "tolist"):
            q_values = q_values.tolist()
        return q_values.index(max(q_values))

    def get_action_name(self, idx):
        return self.actions[idx]


# Import environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environments.wildfire_detection.wildfire_env import WildfireDetectionEnv


def encode_image_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy image to base64 data URI."""
    image = Image.fromarray(image_array)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    data = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def build_observation_prompt(step: int, obs: Dict[str, Any]) -> str:
    """Build user prompt from observation."""
    prediction = obs.get("prediction", {})
    fire_prob = prediction[0] if hasattr(prediction, "__getitem__") else 0
    smoke_prob = prediction[1] if hasattr(prediction, "__getitem__") else 0
    no_fire_prob = prediction[2] if hasattr(prediction, "__getitem__") else 0

    gradcam = obs.get("gradcam_summary", "N/A")
    frame_id = (
        obs.get("frame_id", [0])[0]
        if hasattr(obs.get("frame_id", [0]), "__len__")
        else 0
    )

    prompt = f"""Frame {frame_id} - Step {step}
Prediction probabilities:
  - Fire: {fire_prob:.2%}
  - Smoke: {smoke_prob:.2%}
  - No Fire: {no_fire_prob:.2%}
  
Grad-CAM analysis: {gradcam}

What is your action? (Alert/Scan/Ignore/Deploy)
"""
    return prompt.strip()


def parse_action(response_text: str) -> str:
    """Parse LLM response to extract action."""
    if not response_text:
        return "Scan"
    response = response_text.strip().lower()
    if "alert" in response:
        return "Alert"
    elif "deploy" in response:
        return "Deploy"
    elif "ignore" in response:
        return "Ignore"
    return "Scan"


def action_to_index(action: str) -> int:
    actions = ["Alert", "Scan", "Ignore", "Deploy"]
    return actions.index(action) if action in actions else 1


MAX_STEPS = 20
TEMPERATURE = 0.2
MAX_TOKENS = 500

SYSTEM_PROMPT = """You are an autonomous wildfire detection agent. You control a drone monitoring for fires and smoke.

Given camera frame observations with prediction probabilities and Grad-CAM analysis, decide the best action.

Actions available:
- Alert: Raise immediate alert if fire/smoke detected with high confidence
- Scan: Continue scanning/observing the current frame more carefully  
- Ignore: No action needed, clear conditions
- Deploy: Deploy resources to the detected location

Output ONLY the action name: Alert, Scan, Ignore, or Deploy

Consider:
- Prediction confidence levels
- Grad-CAM heatmap alignment (strong hotspot = more reliable detection)
- Balance early detection vs false alarms
- Sequential frame context
"""


def run_episode(env) -> Dict[str, Any]:
    """Run a single episode and return results."""
    rewards_list = []
    steps_executed = 0

    # Use appropriate model name based on mode
    model_used = MODEL_NAME if USE_LLM else "trained_q_learning"
    print(f"[START] task=wildfire_detection env=wildfire_detection model={model_used}")

    # Initialize RL agent for local mode
    rl_agent = TrainedQLearningAgent() if not USE_LLM else None

    obs = env.reset()

    for step in range(1, MAX_STEPS + 1):
        steps_executed = step

        if USE_LLM:
            # LLM mode
            user_prompt = build_observation_prompt(step, obs)

            image_b64 = ""
            if "image" in obs and obs["image"] is not None:
                image_b64 = encode_image_to_base64(obs["image"])

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_b64}}
                        if image_b64
                        else {"type": "text", "text": "(no image)"},
                    ],
                },
            ]

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as e:
                print(f"Model request failed: {e}. Using fallback.")
                response_text = "Scan"

            action_str = parse_action(response_text)
        else:
            # RL mode
            state = rl_agent.get_state_key(obs)
            action_idx = rl_agent.choose_action(state)
            action_str = rl_agent.get_action_name(action_idx)

        action_idx = action_to_index(action_str)

        obs, reward, done, info = env.step(action_idx)
        rewards_list.append(reward)

        error_msg = "null"
        print(
            f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}"
        )

        if done:
            break

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


def main():
    """Main entry point."""
    model_path = os.path.join(
        os.path.dirname(__file__),
        "FirenetCNN1.h5",
    )

    env = WildfireDetectionEnv(model_path=model_path)

    try:
        result = run_episode(env)
        print(
            f"Episode completed: success={result['success']}, steps={result['steps']}, total_reward={result['total_reward']:.2f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
