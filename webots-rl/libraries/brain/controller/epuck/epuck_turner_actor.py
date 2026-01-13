
import os

import numpy as np
import tensorflow as tf
from brain.controller.epuck import EpuckTurner
from brain.model import MODEL_PATH, Model
from brain.utils.logger import logger
from brain.utils.register_tf import dueling_combine_streams
from controller import Robot


class EpuckTurnerActor(EpuckTurner):

    model: tf.keras.models.Model | None
    stochastic: bool

    def __init__(self, robot: Robot, timestep: int, max_speed: float, stochastic: bool = False) -> None:

        super().__init__(robot=robot, timestep=timestep, max_speed=max_speed)
        self.model = None
        self.stochastic = stochastic

    def set_model(self, model: Model) -> None:

        raise NotImplementedError("Used load_model instead")

    def load_model(self, name: str) -> None:

        path = os.path.join(MODEL_PATH, name + ".keras")
        self.model = tf.keras.models.load_model(
            path, custom_objects={"dueling_combine_streams": dueling_combine_streams}
        )
        self.model.summary()
        logger().info(f"Loaded TensorFlow model from {path}")

    def policy(self, observation: dict) -> int:

        if self.camera is not None:
            observation = np.array(observation["camera"]).astype(np.uint8)
            observation = self.format_camera_image(observation)
        elif self.distance_sensors is not None:
            observation = np.atleast_2d(np.array(observation["distance_sensors"]).astype(np.float32))
        else:
            raise RuntimeError("No valid observation source available.")

        observation = tf.convert_to_tensor(observation, dtype=tf.float32)

        logits, _ = self.model(observation, training=False) # not work for reinforce

        if self.stochastic:
            probs = tf.nn.softmax(logits)
            action = tf.random.categorical(tf.math.log(probs), 1)[0, 0].numpy()
        else:
            action = int(tf.argmax(logits, axis=-1)[0])

        return action
