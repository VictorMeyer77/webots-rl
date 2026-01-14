
import os

import numpy as np
import tensorflow as tf
from brain.controller.epuck import EpuckTurner
from brain.model import MODEL_PATH, Model
from brain.utils.logger import logger
#from brain.utils.register_tf import dueling_combine_streams
from controller import Robot


class EpuckTurnerActor(EpuckTurner):

    model: tf.keras.Model | None
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
            path#, custom_objects={"dueling_combine_streams": dueling_combine_streams}
        )
        self.model.summary()
        logger().info(f"Loaded TensorFlow model from {path}")


    @staticmethod
    def _extract_logits(model_out):
        if isinstance(model_out, (tuple, list)):
            return model_out[0]
        return model_out

    def policy(self, observation: dict) -> int:

        if getattr(self, "camera", None) is not None:
            observation = np.array(observation["camera"], dtype=np.uint8)
            observation = self.format_camera_image(observation)
        elif getattr(self, "distance_sensors", None) is not None:
            observation = np.array(observation["distance_sensors"], dtype=np.float32)
            observation = np.atleast_2d(observation)
        else:
            raise RuntimeError("No valid observation source available.")

        x = tf.convert_to_tensor(observation, dtype=tf.float32)
        out = self.model(x, training=False)
        logits = self._extract_logits(out)

        # test with
        """
        if self.stochastic:
            action = tf.random.categorical(logits, 1)[0, 0].numpy()
            return int(action)

        return int(tf.argmax(logits, axis=-1)[0].numpy())
        """

        if self.stochastic:
            probs = tf.nn.softmax(logits)
            action = tf.random.categorical(tf.math.log(probs), 1)[0, 0].numpy()
        else:
            action = int(tf.argmax(logits, axis=-1)[0])

        return action
