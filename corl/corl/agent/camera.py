from collections import deque

import cv2
import numpy as np
from controller import Camera as CameraDevice
from numpy.typing import NDArray


class Camera:
    """
    Wraps a Webots camera device with preprocessing and temporal frame stacking.

    Captures raw BGRA frames from the Webots ``Camera`` device, applies
    optional grayscale conversion, resizing, and normalisation, then
    maintains a circular buffer of the last ``frame_size`` processed frames.
    The stacked buffer forms a single observation tensor suitable for CNN input.

    Attributes:
        camera: The underlying Webots ``Camera`` device.
        image_shape: Target ``(height, width)`` to resize frames to, or ``None``
            to keep the native camera resolution.
        grayscale: When ``True``, frames are converted from BGRA to grayscale.
        normalize: When ``True``, pixel values are scaled from ``[0, 255]`` to
            ``[0.0, 1.0]``.
        frame_size: Number of consecutive frames to stack into one observation.
        camera_frame_buffer: Circular buffer holding the last ``frame_size``
            processed frames.
    """

    camera: CameraDevice
    image_shape: tuple[int, int] | None
    grayscale: bool
    normalize: bool
    frame_size: int
    camera_frame_buffer: deque[NDArray[np.float32]]

    def __init__(
        self,
        camera: CameraDevice,
        timestep: int,
        frame_size: int = 4,
        image_shape: tuple[int, int] | None = None,
        grayscale: bool = True,
        normalize: bool = True,
    ):
        """
        Initialise the camera wrapper and enable the Webots device.

        Args:
            camera: Webots ``Camera`` device to wrap.
            timestep: Simulation timestep in milliseconds passed to
                ``camera.enable()``.
            frame_size: Number of frames to stack per observation. Defaults
                to ``4``.
            image_shape: Target ``(height, width)`` for resizing, or ``None``
                to use the native camera resolution. Defaults to ``None``.
            grayscale: Convert BGRA frames to grayscale when ``True``.
                Defaults to ``True``.
            normalize: Scale pixel values to ``[0.0, 1.0]`` when ``True``.
                Defaults to ``True``.
        """

        self.camera = camera
        self.camera.enable(timestep)
        self.frame_size = frame_size
        self.image_shape = image_shape
        self.grayscale = grayscale
        self.normalize = normalize
        self.camera_frame_buffer = deque(maxlen=frame_size)

    def process_camera_image(self) -> NDArray[np.float32]:
        """
        Capture, preprocess, and buffer the current camera frame.

        Reads the latest frame from the Webots camera, applies
        :meth:`format_image`, appends the result to the internal buffer,
        and returns the full stacked observation via :meth:`concatenate_frames`.

        Returns:
            NDArray[np.float32]: Stacked frame tensor of shape
            ``(*frame_shape, frame_size)``.
        """
        observation = np.array(self.camera.getImageArray(), dtype=np.uint8)
        frame = self.format_image(observation)
        self.camera_frame_buffer.append(frame)
        return self.concatenate_frames()

    def format_image(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """
        Preprocess a raw BGRA image according to the instance's settings.

        Applies, in order: grayscale conversion (BGRA → gray), resizing to
        ``image_shape`` (width, height convention passed to OpenCV), and
        normalisation to ``[0.0, 1.0]``. Steps are skipped when the
        corresponding flag is ``False`` or ``image_shape`` is ``None``.

        Args:
            image: Raw BGRA image array of shape ``(H, W, 4)`` as returned
                by the Webots camera.

        Returns:
            NDArray[np.float32]: Processed image of shape ``(H, W)`` when
            grayscale is ``True``, or ``(H, W, C)`` otherwise.
        """

        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        if self.image_shape is not None:
            image = cv2.resize(image, (self.image_shape[1], self.image_shape[0]))
        if self.normalize:
            image = image.astype(np.float32) / 255.0
        return image.astype(np.float32)

    def concatenate_frames(self) -> NDArray[np.float32]:
        """
        Build a stacked observation tensor from the frame buffer.

        If the buffer holds fewer than ``frame_size`` frames (e.g. at the
        start of an episode), zero-valued frames are prepended so the
        output always has a fixed shape. Frames are stacked along the last
        axis, producing a tensor of shape ``(*frame_shape, frame_size)``.

        Returns:
            NDArray[np.float32]: Stacked tensor of shape
            ``(*frame_shape, frame_size)``.

        Raises:
            ValueError: If the buffer is completely empty.
        """

        frames = list(self.camera_frame_buffer)

        if len(frames) == 0:
            raise ValueError("Camera frame buffer is empty. Cannot concatenate frames.")

        if len(frames) < self.frame_size:
            padding = [np.zeros_like(frames[0])] * (self.frame_size - len(frames))
            frames = padding + frames

        return np.stack(frames, axis=-1)
