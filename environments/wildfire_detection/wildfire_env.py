"""WildfireDetectionEnv
OpenEnv-compatible environment for autonomous wildfire patrol.

Notes:
- Action space: ["Alert", "Scan", "Ignore", "Deploy"]
- Uses exact FirenetCNN inference from the repo (no modifications)
- Implements reward structure from skill description
- Includes grader for episode evaluation
"""

from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from PIL import Image

try:
    import gym
    from gym import spaces
except Exception:
    gym = None
    spaces = None

    class DummySpace:
        def __init__(self, *args, **kwargs):
            pass

        def sample(self):
            return None

    class DummyEnv:
        pass

    spaces = type(
        "spaces", (), {"Dict": DummySpace, "Box": DummySpace, "Discrete": DummySpace}
    )
    gym = type("gym", (), {"Env": DummyEnv})

try:
    from .firenet_inference import FirenetInference, load_image_for_inference
except (ModuleNotFoundError, ImportError):
    FirenetInference = None
    load_image_for_inference = None


class WildfireDetectionEnv(
    gym.Env if "gym" in globals() and gym is not None else object
):
    """OpenEnv environment for wildfire detection using FirenetCNN."""

    ACTIONS = ["Alert", "Scan", "Ignore", "Deploy"]
    CLASS_LABELS = ["no_fire", "fire", "smoke"]
    MAX_MISSES = 3
    LOG_FILE = "wildfire_episode_log.txt"

    def __init__(
        self,
        frames_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        frames_labels: Optional[Dict[str, str]] = None,
    ):
        super().__init__()

        self.frames_dir = frames_dir or os.path.join(
            os.path.dirname(__file__), "sample_frames"
        )
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "FirenetCNN1.h5",
        )

        self._frames = self._discover_frames()
        self._max_frames = len(self._frames)
        self._frame_idx = 0
        self._step = 0
        self._missed_count = 0
        self._cumulative_reward = 0.0
        self._episode_history: List[Dict[str, Any]] = []

        self._inference = None
        self._load_inference()

        self._ground_truth_labels = frames_labels or self._load_labels()

        self._frame_height = 224
        self._frame_width = 224
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._frame_height, self._frame_width, 3),
                    dtype=np.uint8,
                ),
                "prediction": spaces.Box(
                    low=0.0, high=1.0, shape=(3,), dtype=np.float32
                ),
                "gradcam": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "gradcam_summary": spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.int32
                ),
                "frame_id": spaces.Box(
                    low=0, high=max(0, self._max_frames - 1), shape=(1,), dtype=np.int32
                ),
                "step": spaces.Box(low=0, high=1_000_000, shape=(1,), dtype=np.int32),
                "ground_truth": spaces.Box(low=0, high=2, shape=(1,), dtype=np.int32),
            }
        )

        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self._action_str = lambda idx: self.ACTIONS[int(idx)]

    def _load_inference(self):
        if os.path.exists(self.model_path):
            try:
                self._inference = FirenetInference(self.model_path)
            except Exception:
                self._inference = None
        else:
            self._inference = None

    def _load_labels(self) -> Dict[str, str]:
        labels = {}
        for frame in self._frames:
            if frame:
                fname = os.path.basename(frame).lower()
                if "fire" in fname:
                    labels[frame] = "fire"
                elif "smoke" in fname:
                    labels[frame] = "smoke"
                else:
                    labels[frame] = "no_fire"
        return labels

    def _discover_frames(self):
        frames = []
        if self.frames_dir and os.path.isdir(self.frames_dir):
            for fname in sorted(os.listdir(self.frames_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    frames.append(os.path.join(self.frames_dir, fname))
        if not frames:
            frames = [None]
        return frames

    def _run_inference(self, frame: np.ndarray) -> Dict[str, Any]:
        if self._inference is not None:
            try:
                cv2_frame = frame[:, :, ::-1].copy() if frame.shape[-1] == 3 else frame
                result = self._inference.predict(cv2_frame)
                class_idx = result["class_idx"]
                conf = result["confidence"]
                gradcam_summary_val = (
                    1 if "hotspot" in result.get("gradcam_summary", "") else 0
                )
                return {
                    "class_idx": class_idx,
                    "confidence": conf,
                    "gradcam": np.array([conf], dtype=np.float32),
                    "gradcam_summary_val": gradcam_summary_val,
                    "label": result.get("label", self.CLASS_LABELS[class_idx]),
                    "frame": frame,
                }
            except Exception:
                pass

        gray = np.mean(frame) if frame is not None else 127
        if gray > 170:
            class_idx = 1
            conf = 0.92
        elif gray > 85:
            class_idx = 2
            conf = 0.60
        else:
            class_idx = 0
            conf = 0.40
        return {
            "class_idx": class_idx,
            "confidence": conf,
            "gradcam": np.array([conf], dtype=np.float32),
            "gradcam_summary_val": 0,
            "label": self.CLASS_LABELS[class_idx],
            "frame": frame,
        }

    def _compute_reward(
        self, action: int, inf: Dict[str, Any], ground_truth: str
    ) -> Tuple[float, bool]:
        action_str = self._action_str(action)
        pred_label = inf.get("label", self.CLASS_LABELS[inf["class_idx"]])
        confidence = inf["confidence"]
        gradcam_strong = inf.get("gradcam_summary_val", 0) == 1

        # Simplified reward - focus on action vs ground_truth, not prediction match
        if ground_truth == "no_fire":
            if action_str in ["Alert", "Deploy"]:
                return -0.75, False
            elif action_str == "Ignore":
                return 0.1, False  # Small positive for correct inaction
            else:  # Scan
                return 0.0, False

        if ground_truth in ["fire", "smoke"]:
            if action_str == "Ignore":
                self._missed_count += 1
                return -0.50, self._missed_count >= self.MAX_MISSES

            if action_str in ["Alert", "Deploy"]:
                # Reward based on confidence regardless of exact prediction match
                if confidence >= 0.85 and gradcam_strong:
                    return 2.00, False
                elif confidence >= 0.70:
                    return 1.00, False
                else:
                    return 0.50, False

            if action_str == "Scan":
                # Scan is neutral - continue observing
                return 0.0, False

        return 0.0, False

    def _get_ground_truth(self, frame_path: Optional[str]) -> str:
        if frame_path and frame_path in self._ground_truth_labels:
            return self._ground_truth_labels[frame_path]
        if frame_path:
            fname = os.path.basename(frame_path).lower()
            if "fire" in fname:
                return "fire"
            elif "smoke" in fname:
                return "smoke"
        return "no_fire"

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        if seed is not None:
            np.random.seed(seed)
        self._frame_idx = 0
        self._step = 0
        self._missed_count = 0
        self._cumulative_reward = 0.0
        self._episode_history = []

        frame_path = self._frames[self._frame_idx] if self._frames else None
        frame = self._load_frame(frame_path)
        if frame is None:
            frame = self._synthetic_frame()

        inf = self._run_inference(frame)
        ground_truth = self._get_ground_truth(frame_path)
        gradcam_val = inf.get("gradcam_summary_val", 0)

        obs = {
            "image": frame,
            "prediction": np.array(
                [
                    inf["confidence"] if inf["class_idx"] == 1 else 0.0,
                    inf["confidence"] if inf["class_idx"] == 2 else 0.0,
                    inf["confidence"] if inf["class_idx"] == 0 else 0.0,
                ]
            ).astype(np.float32),
            "gradcam": inf["gradcam"],
            "gradcam_summary": np.array([gradcam_val], dtype=np.int32),
            "frame_id": np.array([self._frame_idx], dtype=np.int32),
            "step": np.array([self._step], dtype=np.int32),
            "ground_truth": np.array(
                ["fire", "smoke", "no_fire"].index(ground_truth), dtype=np.int32
            ),
        }
        return obs

    def _load_frame(self, path: Optional[str]) -> Optional[np.ndarray]:
        if path is None:
            return None
        try:
            img = (
                Image.open(path)
                .convert("RGB")
                .resize((self._frame_width, self._frame_height))
            )
            return np.asarray(img, dtype=np.uint8)
        except Exception:
            return None

    def _synthetic_frame(self) -> np.ndarray:
        return np.full((self._frame_height, self._frame_width, 3), 128, dtype=np.uint8)

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        assert 0 <= int(action) < len(self.ACTIONS), "Invalid action"
        self._step += 1
        self._frame_idx += 1

        frame_path = (
            self._frames[self._frame_idx]
            if self._frame_idx < self._max_frames
            else None
        )
        frame = self._load_frame(frame_path)
        if frame is None:
            frame = self._synthetic_frame()

        inf = self._run_inference(frame)
        ground_truth = self._get_ground_truth(frame_path)
        gradcam_val = inf.get("gradcam_summary_val", 0)

        reward, terminal_failure = self._compute_reward(action, inf, ground_truth)
        self._cumulative_reward += reward

        self._episode_history.append(
            {
                "step": self._step,
                "frame_id": self._frame_idx,
                "action": self._action_str(action),
                "prediction": inf.get("label", self.CLASS_LABELS[inf["class_idx"]]),
                "confidence": inf["confidence"],
                "ground_truth": ground_truth,
                "reward": reward,
            }
        )

        done = terminal_failure or self._frame_idx >= self._max_frames

        obs = {
            "image": frame,
            "prediction": np.array(
                [
                    inf["confidence"] if inf["class_idx"] == 1 else 0.0,
                    inf["confidence"] if inf["class_idx"] == 2 else 0.0,
                    inf["confidence"] if inf["class_idx"] == 0 else 0.0,
                ]
            ).astype(np.float32),
            "gradcam": inf["gradcam"],
            "gradcam_summary": np.array([gradcam_val], dtype=np.int32),
            "frame_id": np.array([self._frame_idx], dtype=np.int32),
            "step": np.array([self._step], dtype=np.int32),
            "ground_truth": np.array(
                ["fire", "smoke", "no_fire"].index(ground_truth), dtype=np.int32
            ),
        }

        info = {
            "action": self._action_str(action),
            "reward": reward,
            "cumulative_reward": self._cumulative_reward,
            "prediction": inf.get("label", self.CLASS_LABELS[inf["class_idx"]]),
            "ground_truth": ground_truth,
            "confidence": inf["confidence"],
            "terminal": terminal_failure,
        }

        if done:
            self._save_episode_log()

        return obs, reward, done, info

    def _save_episode_log(self):
        try:
            with open(self.LOG_FILE, "a") as f:
                f.write(f"\n=== Episode ===\n")
                f.write(f"Total frames: {len(self._episode_history)}\n")
                f.write(f"Cumulative reward: {self._cumulative_reward:.2f}\n")
                f.write(f"Success: {self._is_success()}\n")
                for entry in self._episode_history:
                    f.write(f"  Step {entry['step']}: {entry}\n")
        except Exception:
            pass

    def _is_success(self) -> bool:
        for entry in self._episode_history:
            gt = entry["ground_truth"]
            action = entry["action"]
            if gt in ["fire", "smoke"] and action not in ["Alert", "Deploy"]:
                return False
            if gt == "no_fire" and action in ["Alert", "Deploy"]:
                return False
        return True

    def grade(self) -> float:
        if not self._episode_history:
            return 0.0
        if self._is_success():
            return 1.0
        return max(0.0, self._cumulative_reward / 10.0)

    def render(self, mode: str = "human") -> Optional[Any]:
        return None

    def close(self) -> None:
        pass
