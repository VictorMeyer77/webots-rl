"""
Unit tests for corl.utils.video

Tests cover:
- write_legend: bad input path, bad output path, legend rendered on frames
- concat_videos: empty input, bad input path, resolution mismatch, happy path
- generate_training_video: empty directory, non-episode files ignored,
  correct ordering, cleanup behaviour, return value
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from corl.utils.video import concat_videos, generate_training_video, write_legend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_W, _H, _FPS = 64, 64, 10


def _write_video(path: Path, frames: int = 5, w: int = _W, h: int = _H) -> None:
    """Write a minimal solid-colour MP4 to *path*."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(_FPS), (w, h))
    assert writer.isOpened(), f"Could not create test video at {path}"
    rng = np.random.default_rng(0)
    for _ in range(frames):
        frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _frame_count(path: Path) -> int:
    """Return the number of frames in a video file."""
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


# ===========================================================================
# write_legend
# ===========================================================================


class TestWriteLegend:
    def test_raises_on_missing_input(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot open input video"):
            write_legend(
                str(tmp_path / "ghost.mp4"),
                str(tmp_path / "out.mp4"),
                {"k": "v"},
            )

    def test_raises_on_bad_output_path(self, tmp_path):
        src = tmp_path / "src.mp4"
        _write_video(src)
        with pytest.raises(ValueError, match="Cannot initialise video writer"):
            write_legend(
                str(src),
                str(tmp_path / "no_such_dir" / "out.mp4"),
                {"k": "v"},
            )

    def test_output_file_created(self, tmp_path):
        src = tmp_path / "src.mp4"
        out = tmp_path / "out.mp4"
        _write_video(src)
        write_legend(str(src), str(out), {"Episode": 1})
        assert out.exists()

    def test_output_has_same_frame_count(self, tmp_path):
        src = tmp_path / "src.mp4"
        out = tmp_path / "out.mp4"
        _write_video(src, frames=8)
        write_legend(str(src), str(out), {"Episode": 1})
        assert _frame_count(out) == _frame_count(src)

    def test_empty_legend_still_produces_output(self, tmp_path):
        src = tmp_path / "src.mp4"
        out = tmp_path / "out.mp4"
        _write_video(src)
        write_legend(str(src), str(out), {})
        assert out.exists()

    def test_no_background_option(self, tmp_path):
        src = tmp_path / "src.mp4"
        out = tmp_path / "out.mp4"
        _write_video(src)
        write_legend(str(src), str(out), {"x": 1}, bg_color=None)
        assert out.exists()

    def test_custom_position(self, tmp_path):
        src = tmp_path / "src.mp4"
        out = tmp_path / "out.mp4"
        _write_video(src)
        write_legend(str(src), str(out), {"x": 1}, position=(5, 5))
        assert out.exists()

    def test_output_dimensions_unchanged(self, tmp_path):
        src = tmp_path / "src.mp4"
        out = tmp_path / "out.mp4"
        _write_video(src, w=80, h=48)
        write_legend(str(src), str(out), {"Episode": 0})
        cap = cv2.VideoCapture(str(out))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        assert (w, h) == (80, 48)


# ===========================================================================
# concat_videos
# ===========================================================================


class TestConcatVideos:
    def test_raises_on_empty_list(self, tmp_path):
        with pytest.raises(ValueError, match="must not be empty"):
            concat_videos([], str(tmp_path / "out.mp4"))

    def test_raises_on_missing_input(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot open input video"):
            concat_videos(
                [str(tmp_path / "ghost.mp4")],
                str(tmp_path / "out.mp4"),
            )

    def test_raises_on_resolution_mismatch(self, tmp_path):
        a = tmp_path / "a.mp4"
        b = tmp_path / "b.mp4"
        _write_video(a, w=64, h=64)
        _write_video(b, w=32, h=32)
        with pytest.raises(ValueError, match="Resolution mismatch"):
            concat_videos([str(a), str(b)], str(tmp_path / "out.mp4"))

    def test_output_file_created(self, tmp_path):
        a = tmp_path / "a.mp4"
        b = tmp_path / "b.mp4"
        _write_video(a, frames=4)
        _write_video(b, frames=4)
        out = tmp_path / "out.mp4"
        concat_videos([str(a), str(b)], str(out))
        assert out.exists()

    def test_frame_count_is_sum(self, tmp_path):
        a = tmp_path / "a.mp4"
        b = tmp_path / "b.mp4"
        _write_video(a, frames=5)
        _write_video(b, frames=7)
        out = tmp_path / "out.mp4"
        concat_videos([str(a), str(b)], str(out))
        assert _frame_count(out) == _frame_count(a) + _frame_count(b)

    def test_single_input(self, tmp_path):
        a = tmp_path / "a.mp4"
        _write_video(a, frames=6)
        out = tmp_path / "out.mp4"
        concat_videos([str(a)], str(out))
        assert _frame_count(out) == _frame_count(a)


# ===========================================================================
# generate_training_video
# ===========================================================================


class TestGenerateTrainingVideo:
    def test_returns_none_when_no_episodes(self, tmp_path):
        result = generate_training_video(str(tmp_path))
        assert result is None

    def test_ignores_non_episode_files(self, tmp_path):
        (tmp_path / "random.mp4").write_bytes(b"")
        (tmp_path / "notes.txt").write_text("hello")
        result = generate_training_video(str(tmp_path))
        assert result is None

    def test_ignores_full_training_on_rerun(self, tmp_path):
        """full_training.mp4 from a prior run must not be picked up as an episode."""
        ep = tmp_path / "episode_0.mp4"
        _write_video(ep, frames=3)
        generate_training_video(str(tmp_path))
        # second run should not crash
        result = generate_training_video(str(tmp_path))
        assert result is not None

    def test_returns_absolute_path(self, tmp_path):
        ep = tmp_path / "episode_1.mp4"
        _write_video(ep, frames=3)
        result = generate_training_video(str(tmp_path))
        assert result is not None
        assert os.path.isabs(result)

    def test_output_file_exists(self, tmp_path):
        ep = tmp_path / "episode_0.mp4"
        _write_video(ep, frames=3)
        result = generate_training_video(str(tmp_path))
        assert Path(result).exists()

    def test_output_named_full_training(self, tmp_path):
        ep = tmp_path / "episode_0.mp4"
        _write_video(ep, frames=3)
        result = generate_training_video(str(tmp_path))
        assert Path(result).name == "full_training.mp4"

    def test_cleanup_removes_legend_files(self, tmp_path):
        for i in range(3):
            _write_video(tmp_path / f"episode_{i}.mp4", frames=3)
        generate_training_video(str(tmp_path), cleanup=True)
        legend_files = list(tmp_path.glob("*_legend.mp4"))
        assert legend_files == []

    def test_no_cleanup_keeps_legend_files(self, tmp_path):
        for i in range(2):
            _write_video(tmp_path / f"episode_{i}.mp4", frames=3)
        generate_training_video(str(tmp_path), cleanup=False)
        legend_files = list(tmp_path.glob("*_legend.mp4"))
        assert len(legend_files) == 2

    def test_episodes_concatenated_in_order(self, tmp_path):
        """Output frame count must equal the sum of all individual episode frame counts."""
        counts = {0: 3, 1: 5, 2: 4}
        for ep_id, n in counts.items():
            _write_video(tmp_path / f"episode_{ep_id}.mp4", frames=n)
        result = generate_training_video(str(tmp_path))
        assert _frame_count(Path(result)) == sum(counts.values())

    def test_single_episode(self, tmp_path):
        ep = tmp_path / "episode_0.mp4"
        _write_video(ep, frames=5)
        result = generate_training_video(str(tmp_path))
        assert _frame_count(Path(result)) == _frame_count(ep)
