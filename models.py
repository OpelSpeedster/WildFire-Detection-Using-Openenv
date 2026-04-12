# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Wildfire Detection Environment.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional, Dict, Any


class WildfireAction(Action):
    """Action for the Wildfire Detection environment."""

    action: str = Field(
        ..., description="Action to take: Alert, Scan, Ignore, or Deploy"
    )


class WildfireObservation(Observation):
    """Observation from the Wildfire Detection environment."""

    image: str = Field(default="", description="Base64 encoded image or file path")
    prediction: Dict[str, float] = Field(
        default_factory=dict,
        description="Prediction probabilities for fire, smoke, no_fire",
    )
    gradcam_summary: str = Field(default="", description="Grad-CAM heatmap summary")
    frame_id: int = Field(default=0, description="Current frame ID")
    step: int = Field(default=0, description="Current step")
    ground_truth: str = Field(
        default="", description="Ground truth label (for training)"
    )
    reward: float = Field(default=0.0, description="Reward for the last action")
    done: bool = Field(default=False, description="Whether episode is done")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
