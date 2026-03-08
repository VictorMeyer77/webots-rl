"""
Unit tests for corl.utils.image

Tests cover:
- format_image: resizing, grayscale conversion, normalization, and combinations
- concatenate_frames: full buffer, partial buffer (padding), single frame, and
  invalid inputs
"""

from collections import deque

import numpy as np
import pytest

from corl.utils.image import concatenate_frames, format_image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rgb_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a deterministic uint8 RGB image."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _gray_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a deterministic uint8 grayscale image."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _make_frames(n: int, h: int = 42, w: int = 42) -> deque:
    """Return a deque of ``n`` distinct grayscale frames."""
    rng = np.random.default_rng(42)
    return deque(rng.integers(0, 256, (h, w), dtype=np.uint8) for _ in range(n))


# ===========================================================================
# format_image
# ===========================================================================


class TestFormatImageNoOp:
    def test_returns_ndarray(self):
        img = _rgb_image()
        result = format_image(img)
        assert isinstance(result, np.ndarray)

    def test_no_transforms_preserves_shape(self):
        img = _rgb_image(64, 64)
        result = format_image(img)
        assert result.shape == img.shape

    def test_no_transforms_preserves_values(self):
        img = _rgb_image()
        result = format_image(img)
        np.testing.assert_array_equal(result, img)

    def test_no_transforms_preserves_dtype(self):
        img = _rgb_image()
        result = format_image(img)
        assert result.dtype == img.dtype


class TestFormatImageResize:
    def test_resizes_to_target_shape(self):
        img = _rgb_image(64, 64)
        result = format_image(img, shape=(42, 42))
        # cv2 shape is (width, height) → result is (height, width, channels)
        assert result.shape == (42, 42, 3)

    def test_resize_non_square(self):
        img = _rgb_image(64, 128)
        result = format_image(img, shape=(32, 16))
        assert result.shape == (16, 32, 3)

    def test_resize_upscale(self):
        img = _rgb_image(32, 32)
        result = format_image(img, shape=(64, 64))
        assert result.shape == (64, 64, 3)

    def test_none_shape_skips_resize(self):
        img = _rgb_image(64, 64)
        result = format_image(img, shape=None)
        assert result.shape == img.shape


class TestFormatImageGrayscale:
    def test_grayscale_removes_channel_dim(self):
        img = _rgb_image(64, 64)
        result = format_image(img, grayscale=True)
        assert result.ndim == 2

    def test_grayscale_output_shape(self):
        img = _rgb_image(64, 64)
        result = format_image(img, grayscale=True)
        assert result.shape == (64, 64)

    def test_grayscale_false_preserves_channels(self):
        img = _rgb_image(64, 64)
        result = format_image(img, grayscale=False)
        assert result.shape == (64, 64, 3)


class TestFormatImageNormalize:
    def test_normalize_max_value_is_one(self):
        img = np.full((4, 4, 3), 255, dtype=np.uint8)
        result = format_image(img, normalize=True)
        assert result.max() == pytest.approx(1.0)

    def test_normalize_min_value_is_zero(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        result = format_image(img, normalize=True)
        assert result.min() == pytest.approx(0.0)

    def test_normalize_values_in_range(self):
        img = _rgb_image()
        result = format_image(img, normalize=True)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_normalize_false_preserves_range(self):
        img = np.full((4, 4, 3), 200, dtype=np.uint8)
        result = format_image(img, normalize=False)
        assert result.max() == 200

    def test_normalize_dtype_is_float(self):
        img = _rgb_image()
        result = format_image(img, normalize=True)
        assert np.issubdtype(result.dtype, np.floating)


class TestFormatImageCombined:
    def test_resize_and_grayscale(self):
        img = _rgb_image(64, 64)
        result = format_image(img, shape=(42, 42), grayscale=True)
        assert result.shape == (42, 42)

    def test_resize_and_normalize(self):
        img = _rgb_image(64, 64)
        result = format_image(img, shape=(32, 32), normalize=True)
        assert result.shape == (32, 32, 3)
        assert result.max() <= 1.0

    def test_all_transforms(self):
        img = _rgb_image(64, 64)
        result = format_image(img, shape=(42, 42), grayscale=True, normalize=True)
        assert result.shape == (42, 42)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ===========================================================================
# concatenate_frames
# ===========================================================================


class TestConcatenateFramesFull:
    def test_returns_ndarray(self):
        frames = _make_frames(4)
        result = concatenate_frames(frames, 4)
        assert isinstance(result, np.ndarray)

    def test_full_buffer_output_shape(self):
        frames = _make_frames(4, h=42, w=42)
        result = concatenate_frames(frames, 4)
        assert result.shape == (42, 42, 4)

    def test_full_buffer_different_size(self):
        frames = _make_frames(8, h=84, w=84)
        result = concatenate_frames(frames, 8)
        assert result.shape == (84, 84, 8)

    def test_frames_stacked_in_order(self):
        """Last frame should appear at index -1 along the channel axis."""
        f0 = np.zeros((4, 4), dtype=np.uint8)
        f1 = np.ones((4, 4), dtype=np.uint8) * 128
        f2 = np.ones((4, 4), dtype=np.uint8) * 200
        f3 = np.ones((4, 4), dtype=np.uint8) * 255
        frames = deque([f0, f1, f2, f3])
        result = concatenate_frames(frames, 4)
        np.testing.assert_array_equal(result[:, :, 0], f0)
        np.testing.assert_array_equal(result[:, :, 3], f3)


class TestConcatenateFramesPartial:
    def test_single_frame_pads_to_size(self):
        frames = _make_frames(1, h=42, w=42)
        result = concatenate_frames(frames, 4)
        assert result.shape == (42, 42, 4)

    def test_partial_buffer_pads_to_size(self):
        frames = _make_frames(2, h=42, w=42)
        result = concatenate_frames(frames, 4)
        assert result.shape == (42, 42, 4)

    def test_padding_duplicates_last_frame(self):
        """When buffer has 1 frame, all 4 channels should be identical."""
        frame = np.ones((4, 4), dtype=np.uint8) * 99
        frames = deque([frame])
        result = concatenate_frames(frames, 4)
        for i in range(4):
            np.testing.assert_array_equal(result[:, :, i], frame)

    def test_partial_padding_last_frame_repeated(self):
        """With 2 frames the last frame fills the remaining slots."""
        f0 = np.zeros((4, 4), dtype=np.uint8)
        f1 = np.ones((4, 4), dtype=np.uint8) * 255
        frames = deque([f0, f1])
        result = concatenate_frames(frames, 4)
        np.testing.assert_array_equal(result[:, :, 0], f0)
        np.testing.assert_array_equal(result[:, :, 1], f1)
        np.testing.assert_array_equal(result[:, :, 2], f1)
        np.testing.assert_array_equal(result[:, :, 3], f1)

    def test_size_one_single_frame(self):
        frames = _make_frames(1, h=8, w=8)
        result = concatenate_frames(frames, 1)
        assert result.shape == (8, 8, 1)


class TestConcatenateFramesInvalid:
    def test_empty_deque_raises_value_error(self):
        with pytest.raises(ValueError):
            concatenate_frames(deque(), 4)

    def test_too_many_frames_raises_value_error(self):
        frames = _make_frames(5)
        with pytest.raises(ValueError):
            concatenate_frames(frames, 4)

    def test_error_message_contains_counts(self):
        frames = _make_frames(6)
        with pytest.raises(ValueError, match="6"):
            concatenate_frames(frames, 4)
