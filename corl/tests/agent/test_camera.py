"""
Unit tests for Camera.format_image and Camera.concatenate_frames.

Camera.__init__ requires a live Webots CameraDevice, so both methods are
tested by instantiating Camera with a mock device. The `controller` module
is also mocked at import time as it is only available inside Webots.
"""

import sys
from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

# Mock the Webots controller module before importing Camera
sys.modules.setdefault("controller", MagicMock())

from corl.agent.camera import Camera  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_camera(
    frame_size: int = 4,
    image_shape: tuple[int, int] | None = (42, 42),
    grayscale: bool = True,
    normalize: bool = True,
) -> Camera:
    """Return a Camera instance with a mocked Webots device."""
    mock_device = MagicMock()
    cam = Camera.__new__(Camera)
    cam.camera = mock_device
    cam.frame_size = frame_size
    cam.image_shape = image_shape
    cam.grayscale = grayscale
    cam.normalize = normalize
    cam.camera_frame_buffer = deque(maxlen=frame_size)
    return cam


def bgra_image(h: int = 64, w: int = 64) -> NDArray[np.uint8]:
    """Return a synthetic BGRA image."""
    return np.random.randint(0, 256, (h, w, 4), dtype=np.uint8)


# ---------------------------------------------------------------------------
# format_image
# ---------------------------------------------------------------------------


class TestFormatImage:
    def test_output_dtype_is_float32(self):
        cam = make_camera()
        result = cam.format_image(bgra_image())
        assert result.dtype == np.float32

    def test_grayscale_removes_channel_dim(self):
        cam = make_camera(grayscale=True, image_shape=None)
        result = cam.format_image(bgra_image(64, 64))
        assert result.ndim == 2

    def test_no_grayscale_preserves_channels(self):
        cam = make_camera(grayscale=False, image_shape=None)
        result = cam.format_image(bgra_image(64, 64))
        assert result.ndim == 3

    def test_resize_output_shape(self):
        cam = make_camera(grayscale=False, image_shape=(32, 16))
        result = cam.format_image(bgra_image(64, 64))
        assert result.shape[:2] == (32, 16)  # (H, W)

    def test_normalize_scales_to_0_1(self):
        cam = make_camera(normalize=True, image_shape=None, grayscale=False)
        image = np.full((4, 4, 4), 255, dtype=np.uint8)
        result = cam.format_image(image)
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_no_normalize_preserves_range(self):
        cam = make_camera(normalize=False, image_shape=None, grayscale=False)
        image = np.full((4, 4, 4), 200, dtype=np.uint8)
        result = cam.format_image(image)
        assert result.max() == pytest.approx(200.0)

    def test_no_resize_when_image_shape_is_none(self):
        cam = make_camera(grayscale=False, normalize=False, image_shape=None)
        image = bgra_image(64, 32)
        result = cam.format_image(image)
        assert result.shape[:2] == (64, 32)

    def test_grayscale_and_resize_combined(self):
        cam = make_camera(grayscale=True, normalize=False, image_shape=(20, 30))
        result = cam.format_image(bgra_image(64, 64))
        assert result.shape == (20, 30)


# ---------------------------------------------------------------------------
# concatenate_frames
# ---------------------------------------------------------------------------


class TestConcatenateFrames:
    def _push_frames(self, cam: Camera, n: int, h: int = 42, w: int = 42) -> None:
        for _ in range(n):
            cam.camera_frame_buffer.append(np.zeros((h, w), dtype=np.float32))

    def test_raises_when_buffer_empty(self):
        cam = make_camera(frame_size=4)
        with pytest.raises(ValueError, match="empty"):
            cam.concatenate_frames()

    def test_output_shape_full_buffer(self):
        cam = make_camera(frame_size=4)
        self._push_frames(cam, 4)
        result = cam.concatenate_frames()
        assert result.shape == (42, 42, 4)

    def test_output_dtype_is_float32(self):
        cam = make_camera(frame_size=4)
        self._push_frames(cam, 4)
        assert cam.concatenate_frames().dtype == np.float32

    def test_partial_buffer_pads_with_zeros(self):
        cam = make_camera(frame_size=4)
        frame = np.ones((42, 42), dtype=np.float32)
        cam.camera_frame_buffer.append(frame)
        result = cam.concatenate_frames()
        assert result.shape == (42, 42, 4)
        np.testing.assert_array_equal(result[:, :, 0], 0.0)  # first 3 are padding
        np.testing.assert_array_equal(result[:, :, 1], 0.0)
        np.testing.assert_array_equal(result[:, :, 2], 0.0)
        np.testing.assert_array_equal(result[:, :, 3], 1.0)  # last is the real frame

    def test_padding_frames_are_zero(self):
        cam = make_camera(frame_size=3)
        cam.camera_frame_buffer.append(np.ones((42, 42), dtype=np.float32))
        result = cam.concatenate_frames()
        np.testing.assert_array_equal(result[:, :, :2], 0.0)

    def test_frames_stacked_on_last_axis(self):
        cam = make_camera(frame_size=2)
        cam.camera_frame_buffer.append(np.full((4, 4), 1.0, dtype=np.float32))
        cam.camera_frame_buffer.append(np.full((4, 4), 2.0, dtype=np.float32))
        result = cam.concatenate_frames()
        np.testing.assert_array_equal(result[:, :, 0], 1.0)
        np.testing.assert_array_equal(result[:, :, 1], 2.0)

    def test_frame_size_one(self):
        cam = make_camera(frame_size=1)
        cam.camera_frame_buffer.append(np.ones((8, 8), dtype=np.float32))
        result = cam.concatenate_frames()
        assert result.shape == (8, 8, 1)
