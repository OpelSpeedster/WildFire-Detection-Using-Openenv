import numpy as np
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import WildfireAction, WildfireObservation
except ImportError:
    from models import WildfireAction, WildfireObservation

CLASS_LABELS = {0: "fire", 1: "smoke", 2: "no_fire"}

TASK_FRAMES = {
    "easy_clear_detection": [
        {"prediction": [0.95, 0.04, 0.01], "ground_truth": 0, "gradcam_intensity": 0.9},
        {"prediction": [0.85, 0.10, 0.05], "ground_truth": 0, "gradcam_intensity": 0.8},
        {"prediction": [0.01, 0.02, 0.97], "ground_truth": 2, "gradcam_intensity": 0.1},
        {"prediction": [0.02, 0.05, 0.93], "ground_truth": 2, "gradcam_intensity": 0.2},
    ],
    "medium_smoke_ambiguous": [
        {"prediction": [0.10, 0.80, 0.10], "ground_truth": 1, "gradcam_intensity": 0.5},
        {"prediction": [0.40, 0.50, 0.10], "ground_truth": 1, "gradcam_intensity": 0.6},
        {"prediction": [0.70, 0.20, 0.10], "ground_truth": 0, "gradcam_intensity": 0.8},
    ],
    "hard_multi_threat": [
        {"prediction": [0.05, 0.15, 0.80], "ground_truth": 2, "gradcam_intensity": 0.2},
        {"prediction": [0.20, 0.70, 0.10], "ground_truth": 1, "gradcam_intensity": 0.6},
        {"prediction": [0.90, 0.05, 0.05], "ground_truth": 0, "gradcam_intensity": 0.9},
        {"prediction": [0.02, 0.01, 0.97], "ground_truth": 2, "gradcam_intensity": 0.1},
        {"prediction": [0.85, 0.10, 0.05], "ground_truth": 0, "gradcam_intensity": 0.85},
    ],
}

def compute_reward(action: str, ground_truth_idx: int, probs: list) -> float:
    gt_label = CLASS_LABELS[ground_truth_idx]
    if gt_label in ("fire", "smoke"):
        if action == "Alert":  return +1.0   # correct
        if action == "Deploy": return +0.8   # good but expensive
        if action == "Scan":   return +0.1 if max(probs[:2]) < 0.6 else -0.2
        if action == "Ignore": return -1.0   # dangerous miss
    else:  # no_fire
        if action == "Ignore": return +0.5   # correct clear
        if action == "Scan":   return +0.2   # cautious ok
        if action == "Alert":  return -0.75  # false alarm
        if action == "Deploy": return -0.5   # wasteful
    return 0.0

def gradcam_summary(probs, gradcam_intensity) -> str:
    dominant = CLASS_LABELS[int(np.argmax(probs))]
    intensity = "strong" if gradcam_intensity > 0.6 else "moderate" if gradcam_intensity > 0.4 else "weak"
    if dominant == "fire":
        return f"Grad-CAM shows {intensity} hotspot. Fire {probs[0]*100:.0f}%."
    elif dominant == "smoke":
        return f"Grad-CAM shows {intensity} diffuse activation. Smoke {probs[1]*100:.0f}%."
    else:
        return f"Minimal activation. Clear conditions {probs[2]*100:.0f}%."

class WildfireDetectionEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_name="easy_clear_detection"):
        self.task_name = task_name if task_name else "easy_clear_detection"
        self._frames = TASK_FRAMES.get(self.task_name, TASK_FRAMES["easy_clear_detection"])
        self._frame_idx = 0
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._episode_id = str(uuid4())

    def reset(self) -> WildfireObservation:
        self._frame_idx = 0
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._episode_id = str(uuid4())
        return self._get_observation()

    def _get_observation(self, reward: float = 0.0) -> WildfireObservation:
        frame = self._frames[self._frame_idx]
        probs = frame["prediction"]
        intensity = frame["gradcam_intensity"]
        gt_idx = frame["ground_truth"]

        predicted_class = CLASS_LABELS[int(np.argmax(probs))]
        summary = gradcam_summary(probs, intensity)
        
        return WildfireObservation(
            fire_prob=probs[0],
            smoke_prob=probs[1],
            no_fire_prob=probs[2],
            predicted_class=predicted_class,
            gradcam_summary=summary,
            gradcam_intensity=intensity,
            frame_id=self._frame_idx,
            total_frames=len(self._frames),
            step=self._step_count,
            ground_truth=CLASS_LABELS[gt_idx],
            reward=reward,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
            metadata={"task": self.task_name}
        )

    def step(self, action: WildfireAction) -> WildfireObservation:
        if self._done:
            return self._get_observation()
            
        frame = self._frames[self._frame_idx]
        probs = frame["prediction"]
        gt_idx = frame["ground_truth"]
        
        reward = compute_reward(action.action, gt_idx, probs)
        self._cumulative_reward += reward
        self._step_count += 1
        
        self._frame_idx += 1
        if self._frame_idx >= len(self._frames):
            self._frame_idx = len(self._frames) - 1
            self._done = True
            
        return self._get_observation(reward=reward)

    @property
    def state(self) -> State:
        return State(episode_id=self._episode_id, step_count=self._step_count)
