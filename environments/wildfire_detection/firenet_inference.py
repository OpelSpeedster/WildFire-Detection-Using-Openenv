"""
FirenetCNN Inference Module

This module provides the exact inference and Grad-CAM implementation
from the Forest-Fire-Detection-Using-FirenetCNN-and-XAI-Techniques repository.
No modifications to the original repo code.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from typing import Tuple, Dict, Any, Optional


MODEL_PATH = os.getenv("FIRENET_MODEL_PATH", "FirenetCNN1.h5")
TARGET_SIZE = (224, 224)
CLASS_LABELS = ["no_fire", "fire", "smoke"]
LAST_CONV_LAYER_NAME = "out_relu"


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index):
    """Creates a Grad-CAM heatmap."""
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """Applies the heatmap to the original image."""
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + original_img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img


class FirenetInference:
    """Wrapper class for FirenetCNN inference with Grad-CAM."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or MODEL_PATH
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the FirenetCNN model."""
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path, compile=False)
        else:
            raise FileNotFoundError(f"Model not found: {self.model_path}")

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on an image and return prediction + Grad-CAM info.

        Args:
            image: BGR image (from cv2.imread) or RGB image

        Returns:
            dict with keys:
                - label: 'fire', 'smoke', or 'no_fire'
                - confidence: float (0.0-1.0)
                - class_idx: int (0=no_fire, 1=fire, 2=smoke)
                - heatmap: numpy array or None
                - gradcam_summary: str describing heatmap location
        """
        original_image = image.copy()

        if image.shape[:2] != TARGET_SIZE[::-1]:
            image = cv2.resize(image, TARGET_SIZE)

        input_image_scaled = image.astype(np.float32) / 255.0
        input_image_batch = np.expand_dims(input_image_scaled, axis=0)

        preds = self.model.predict(input_image_batch, verbose=0)
        confidence = float(np.max(preds[0]))
        class_idx = int(np.argmax(preds[0]))
        label = CLASS_LABELS[class_idx]

        heatmap = None
        gradcam_summary = "N/A"

        if label != "no_fire":
            heatmap = make_gradcam_heatmap(
                input_image_batch, self.model, LAST_CONV_LAYER_NAME, class_idx
            )
            gradcam_summary = self._summarize_heatmap(heatmap)

        return {
            "label": label,
            "confidence": confidence,
            "class_idx": class_idx,
            "heatmap": heatmap,
            "gradcam_summary": gradcam_summary,
        }

    def _summarize_heatmap(self, heatmap: np.ndarray) -> str:
        """
        Summarize the heatmap location as center/left/right/top/bottom.
        """
        h, w = heatmap.shape
        threshold = 0.5

        heatmap_thresh = (heatmap > threshold).astype(np.float32)

        h_half = h // 2
        w_half = w // 2

        top_region = heatmap_thresh[:h_half, :].sum()
        bottom_region = heatmap_thresh[h_half:, :].sum()
        left_region = heatmap_thresh[:, :w_half].sum()
        right_region = heatmap_thresh[:, w_half:].sum()
        center_region = heatmap_thresh[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].sum()

        total = heatmap_thresh.sum()
        if total == 0:
            return "no significant hotspot"

        regions = {
            "center": center_region,
            "top": top_region,
            "bottom": bottom_region,
            "left": left_region,
            "right": right_region,
        }

        dominant = max(regions, key=regions.get)
        return f"hotspot at {dominant}"


def load_image_for_inference(image_path: str) -> Optional[np.ndarray]:
    """Load and preprocess an image for inference."""
    if not os.path.exists(image_path):
        return None
    image = cv2.imread(image_path)
    return image


def extract_video_frames(video_path: str, frame_interval: int = 10) -> list:
    """
    Extract frames from a video at regular intervals.

    Args:
        video_path: Path to video file
        frame_interval: Extract every N frames

    Returns:
        List of frames as numpy arrays
    """
    frames = []
    if not os.path.exists(video_path):
        return frames

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    return frames
