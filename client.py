# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wildfire Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import WildfireAction, WildfireObservation


class WildfireEnv(
    EnvClient[WildfireAction, WildfireObservation, State]
):
    """
    Client for the Wildfire Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.
    """

    def _step_payload(self, action: WildfireAction) -> Dict:
        """
        Convert WildfireAction to JSON payload for step message.
        """
        return {"action": action.action}

    def _parse_result(self, payload: Dict) -> StepResult[WildfireObservation]:
        """
        Parse server response into StepResult[WildfireObservation].
        """
        obs_data = payload.get("observation", payload)
        
        observation = WildfireObservation(
            fire_prob=obs_data.get("fire_prob", 0.0),
            smoke_prob=obs_data.get("smoke_prob", 0.0),
            no_fire_prob=obs_data.get("no_fire_prob", 1.0),
            predicted_class=obs_data.get("predicted_class", "no_fire"),
            gradcam_summary=obs_data.get("gradcam_summary", ""),
            gradcam_intensity=obs_data.get("gradcam_intensity", 0.0),
            frame_id=obs_data.get("frame_id", 0),
            total_frames=obs_data.get("total_frames", 1),
            step=obs_data.get("step", 0),
            ground_truth=obs_data.get("ground_truth", ""),
            reward=payload.get("reward", 0.0),
            cumulative_reward=obs_data.get("cumulative_reward", 0.0),
            done=payload.get("done", False),
            llm_explanation=obs_data.get("llm_explanation", ""),
            alert_raised=obs_data.get("alert_raised", False),
            alert_level=obs_data.get("alert_level", "NONE"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
