"""
Image preprocessing utilities for Deep Reinforcement Learning.

This module provides image processing functions commonly used in Deep Q-Learning
and other vision-based reinforcement learning algorithms. It handles camera input
preprocessing, frame stacking, and normalization for neural network consumption.

Key Features:
    - **Image Resizing**: Scale images to fixed dimensions for CNN input
    - **Grayscale Conversion**: Convert RGB images to grayscale to reduce dimensionality
    - **Normalization**: Scale pixel values from [0, 255] to [0.0, 1.0] for better training
    - **Frame Stacking**: Concatenate sequential frames to capture temporal information
"""

from collections import deque

import cv2
import numpy as np


def format_image(
    image: np.ndarray, shape: tuple[int, int] = None, grayscale: bool = False, normalize: bool = False
) -> np.ndarray:
    """
    Preprocess an image for deep reinforcement learning neural networks.

    Applies a sequence of transformations to prepare raw camera images for CNN consumption.
    Common preprocessing pipeline for vision-based RL includes resizing to fixed dimensions,
    converting to grayscale to reduce input dimensionality, and normalizing pixel values.

    Args:
        image (np.ndarray): Input image array from camera.
            Expected shapes:
            - RGB: (height, width, 3) with values in [0, 255]
            - Grayscale: (height, width) with values in [0, 255]
        shape (tuple[int, int], optional): Target (width, height) for resizing.
            Format: (width, height) following OpenCV convention.
            Common values: (84, 84) for Atari DQN, (64, 64) for robotics.
            If None, no resizing is performed.
        grayscale (bool, optional): Whether to convert RGB image to grayscale.
            Reduces dimensionality from 3 channels to 1.
            Default: False (keep original color space).
        normalize (bool, optional): Whether to normalize pixel values to [0.0, 1.0].
            Divides all values by 255.0 for better neural network training.
            Default: False (keep [0, 255] range).

    Returns:
        np.ndarray: Processed image with transformations applied.
            Shape depends on parameters:
            - With shape=(w, h), grayscale=True: (h, w)
            - With shape=(w, h), grayscale=False: (h, w, 3)
            - Without shape, grayscale=True: (original_h, original_w)
            Values: [0, 255] if normalize=False, [0.0, 1.0] if normalize=True
    """
    if shape is not None:
        image = cv2.resize(image, shape)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if normalize:
        image = image / 255.0
    return image


def concatenate_frames(frames: deque, size: int) -> np.ndarray:
    """
    Stack sequential frames along the channel dimension for temporal context.

    Combines multiple consecutive frames into a single multi-channel array, allowing
    neural networks to learn temporal dynamics and motion information. This is essential
    for partially observable environments where a single frame doesn't capture velocity
    or direction of movement.

    The function handles cases where fewer frames are available than requested by
    duplicating the last frame to fill the stack. This is useful at the start of episodes
    before enough frames have been collected.

    Args:
        frames (deque): Queue of preprocessed image frames in temporal order (oldest to newest).
            Each frame should have the same shape.
            Expected frame shapes:
            - Grayscale: (height, width)
            - RGB: (height, width, 3)
        size (int): Target number of frames to stack.
            Common values: 4 (Atari DQN standard), 3, or 8 for complex dynamics.
            Must be >= 1.

    Returns:
        np.ndarray: Stacked frames concatenated along last dimension.
            Shape: (*frame_shape, size)
            Examples:
            - Input: 4 grayscale frames (84, 84) → Output: (84, 84, 4)
            - Input: 4 RGB frames (64, 64, 3) → Output: (64, 64, 3, 4)

    Raises:
        ValueError: If frames deque is empty or contains more than `size` frames.
            Valid range: 1 <= len(frames) <= size
    """
    len_frames = len(frames)
    if len_frames < 1 or len_frames > size:
        raise ValueError(f"Frames deque has invalid length. {len_frames} frames found, expected 1 to {size}.")
    elif len_frames < size:
        for i in range(size - len_frames):
            frames.append(frames[-1])
    return np.stack(frames, axis=-1)
