# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Wildfire Detection Environment Implementation.

This environment uses the FirenetCNN model for wildfire detection.
"""

import os
import base64
import numpy as np
from uuid import uuid4
from io import BytesIO

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import WildfireAction, WildfireObservation
except ImportError:
    from models import WildfireAction, WildfireObservation

try:
    from environments.wildfire_detection.wildfire_env import WildfireDetectionEnv
except ModuleNotFoundError:
    from multipen.environments.wildfire_detection.wildfire_env import (
        WildfireDetectionEnv,
    )


class WildfireEnvironment(Environment):
    """OpenEnv environment for wildfire detection using FirenetCNN."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._env = None
        self._init_env()

    def _init_env(self):
        model_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "Forest-Fire-Detection-Using-FirenetCNN-and-XAI-Techniques",
            "FirenetCNN1.h5",
        )
        self._env = WildfireDetectionEnv(model_path=model_path)

    def reset(self) -> WildfireObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        obs = self._env.reset()

        return self._make_observation(obs, 0.0, False)

    def step(self, action: WildfireAction) -> WildfireObservation:
        self._state.step_count += 1

        action_idx = ["Alert", "Scan", "Ignore", "Deploy"].index(action.action)
        obs, reward, done, info = self._env.step(action_idx)

        return self._make_observation(obs, reward, done)

    def _make_observation(
        self, obs: dict, reward: float, done: bool
    ) -> WildfireObservation:
        img = obs.get("image")
        img_b64 = ""
        if img is not None:
            from PIL import Image
            import numpy as np

            pil_img = Image.fromarray(img)
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()

        return WildfireObservation(
            image=img_b64,
            prediction={
                "fire": float(obs.get("prediction", [0, 0, 0])[0]),
                "smoke": float(obs.get("prediction", [0, 0, 0])[1]),
                "no_fire": float(obs.get("prediction", [0, 0, 0])[2]),
            },
            gradcam_summary=f"Grad-CAM: {obs.get('gradcam_summary', [0])[0]}",
            frame_id=int(obs.get("frame_id", [0])[0]),
            step=int(obs.get("step", [0])[0]),
            ground_truth="no_fire",
            reward=reward,
            done=done,
            metadata={},
        )

    @property
    def state(self) -> State:
        return self._state
