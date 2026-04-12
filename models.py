# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Wildfire Env Environment.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional, Dict, Any

class WildfireAction(Action):
    action: str = Field(..., description="Alert | Scan | Ignore | Deploy")

class WildfireObservation(Observation):
    fire_prob: float = Field(default=0.0)
    smoke_prob: float = Field(default=0.0)
    no_fire_prob: float = Field(default=1.0)
    predicted_class: str = Field(default="no_fire")
    gradcam_summary: str = Field(default="")
    gradcam_intensity: float = Field(default=0.0)
    frame_id: int = Field(default=0)
    total_frames: int = Field(default=1)
    step: int = Field(default=0)
    ground_truth: str = Field(default="")
    reward: float = Field(default=0.0)
    cumulative_reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    llm_explanation: str = Field(default="")
    alert_raised: bool = Field(default=False)
    alert_level: str = Field(default="NONE")   # NONE|LOW|MEDIUM|HIGH|CRITICAL
    metadata: Dict[str, Any] = Field(default_factory=dict)
