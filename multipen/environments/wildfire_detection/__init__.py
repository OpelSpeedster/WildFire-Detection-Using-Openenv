"""Wildfire Detection OpenEnv Environment

This package provides a minimal OpenEnv-compatible RL environment
that follows the WildfireDetection skill specification. It is designed
to be extended with the exact FirenetCNN inference pipeline from the
opposed GitHub repo without modifying that repository.
"""

from .wildfire_env import WildfireDetectionEnv

__all__ = ["WildfireDetectionEnv"]
