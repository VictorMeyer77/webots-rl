"""
Video processing utilities for Deep Reinforcement Learning.

This module provides video manipulation functions used to annotate and post-process
recorded episodes from reinforcement learning training and evaluation runs.

Public API:
    - :func:`write_legend`         — Burn a key-value legend onto every frame of a video.
    - :func:`concat_videos`        — Concatenate an ordered list of video files into one.
    - :func:`generate_training_video` — Annotate and merge all episode videos in a directory.
"""

import logging
import re
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

_EPISODE_RE = re.compile(r"^episode_(\d+)\.mp4$")


def write_legend(
    input_path: str,
    output_path: str,
    legend: dict,
    position: tuple[int, int] | None = None,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] | None = (0, 0, 0),
    bg_alpha: float = 0.5,
    line_spacing: int = 20,
) -> None:
    """
    Overlay a legend onto every frame of a video and write the result to a new file.

    Renders each key-value pair in *legend* as a separate text line, optionally backed
    by a semi-transparent rectangle so the text remains readable on any background.

    Args:
        input_path (str): Path to the source video file (any format supported by OpenCV).
        output_path (str): Destination path for the annotated video (e.g. ``"out.mp4"``).
        legend (dict): Mapping of label → value to display.
            Keys and values are converted to strings via ``str()``.
            Example: ``{"episode": 42, "reward": 3.14, "epsilon": 0.1}``
        position (tuple[int, int] | None, optional): (x, y) pixel coordinate of the
            top-left corner of the first legend line. Pass ``None`` (default) to
            automatically place the legend in the top-right corner of the frame.
        font_scale (float, optional): OpenCV font scale factor. Default: ``0.5``.
        font_thickness (int, optional): Stroke thickness in pixels. Default: ``1``.
        text_color (tuple[int, int, int], optional): BGR text colour. Default: white
            ``(255, 255, 255)``.
        bg_color (tuple[int, int, int] | None, optional): BGR colour of the background
            rectangle. Pass ``None`` to disable the background. Default: black
            ``(0, 0, 0)``.
        bg_alpha (float, optional): Opacity of the background rectangle in ``[0, 1]``.
            Ignored when *bg_color* is ``None``. Default: ``0.5``.
        line_spacing (int, optional): Vertical gap between consecutive legend lines in
            pixels. Default: ``20``.

    Raises:
        ValueError: If the input video cannot be opened.
        ValueError: If the output video writer cannot be initialised.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open input video: {input_path!r}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Cannot initialise video writer for: {output_path!r}")

    lines = [f"{k}: {v}" for k, v in legend.items()]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Pre-compute per-line text sizes once.
    line_sizes = [
        cv2.getTextSize(line, font, font_scale, font_thickness)[0] for line in lines
    ]

    # Resolve default top-right position using the widest line.
    if position is None:
        max_text_w = max((w for w, _ in line_sizes), default=0)
        x = width - max_text_w - 15
        y = line_spacing
        position = (x, y)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if bg_color is not None and lines:
                # Compute bounding box that covers all legend lines.
                box_w = max(w for w, _ in line_sizes) + 10
                box_h = len(lines) * line_spacing + 5
                x0, y0 = position[0] - 5, position[1] - line_spacing + 5

                overlay = frame.copy()
                cv2.rectangle(
                    overlay,
                    (x0, y0),
                    (x0 + box_w, y0 + box_h),
                    bg_color,
                    thickness=-1,
                )
                cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)

            for i, line in enumerate(lines):
                y = position[1] + i * line_spacing
                cv2.putText(
                    frame,
                    line,
                    (position[0], y),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA,
                )

            writer.write(frame)
    finally:
        cap.release()
        writer.release()


def concat_videos(
    input_paths: list[str],
    output_path: str,
) -> None:
    """
    Concatenate a list of video files into a single output video.

    All input videos must share the same resolution. The frame rate and codec are
    taken from the first video in the list.

    Args:
        input_paths (list[str]): Ordered list of paths to the source video files.
        output_path (str): Destination path for the concatenated video (e.g. ``"full.mp4"``).

    Raises:
        ValueError: If *input_paths* is empty.
        ValueError: If any input video cannot be opened.
        ValueError: If the output video writer cannot be initialised.
        ValueError: If an input video has a different resolution than the first one.
    """
    if not input_paths:
        raise ValueError("input_paths must not be empty.")

    # Read metadata from the first video.
    first_cap = cv2.VideoCapture(input_paths[0])
    if not first_cap.isOpened():
        raise ValueError(f"Cannot open input video: {input_paths[0]!r}")

    fps = first_cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Cannot initialise video writer for: {output_path!r}")

    try:
        for path in input_paths:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open input video: {path!r}")

            v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if v_width != width or v_height != height:
                cap.release()
                raise ValueError(
                    f"Resolution mismatch in {path!r}: "
                    f"expected ({width}x{height}), got ({v_width}x{v_height})."
                )

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    writer.write(frame)
            finally:
                cap.release()
    finally:
        writer.release()


def generate_training_video(videos_dir: str, cleanup: bool = True) -> str | None:
    """
    Annotate every episode video in *videos_dir* with its episode number and
    concatenate them in order into a single ``full_training.mp4`` file.

    The function expects video files named ``episode_<id>.mp4`` inside *videos_dir*.
    For each one it burns an episode legend via :func:`write_legend`, then joins all
    annotated clips in ascending episode order with :func:`concat_videos`.

    Args:
        videos_dir (str): Directory that contains the raw ``episode_*.mp4`` files.
        cleanup (bool, optional): Remove intermediate ``*_legend.mp4`` files after
            concatenation. Default: ``True``.

    Returns:
        str | None: Absolute path to the generated ``full_training.mp4``, or ``None``
        if no episode videos were found in *videos_dir*.
    """
    videos_path = Path(videos_dir)
    episode_files: list[tuple[int, Path]] = []
    legend_files: list[Path] = []

    for episode_path in videos_path.iterdir():
        match = _EPISODE_RE.match(episode_path.name)
        if not match:
            continue

        episode_id = int(match.group(1))
        legend_path = episode_path.with_stem(episode_path.stem + "_legend")
        write_legend(str(episode_path), str(legend_path), {"Episode": episode_id})
        episode_files.append((episode_id, legend_path))
        legend_files.append(legend_path)

    if not episode_files:
        logger.debug("No videos found to concatenate for full video generation.")
        return None

    ordered_paths = [str(p) for _, p in sorted(episode_files, key=lambda x: x[0])]
    output_path = videos_path / "full_training.mp4"

    try:
        concat_videos(ordered_paths, str(output_path))
    finally:
        if cleanup:
            for legend_file in legend_files:
                legend_file.unlink(missing_ok=True)

    logger.debug(
        f"Generated full training video at {output_path} with {len(episode_files)} episodes."
    )
    return str(output_path)
