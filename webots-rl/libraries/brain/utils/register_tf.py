"""
TensorFlow Custom Object Registration Module.

This module provides custom TensorFlow/Keras objects that need to be registered
for proper model serialization and deserialization. All custom layers, functions,
and objects used in Lambda layers or other custom components must be defined here
and decorated with @register_keras_serializable() to ensure they can be saved
and loaded correctly.

Purpose:
    - Centralized location for all custom TensorFlow objects
    - Ensures custom objects are registered at import time
    - Prevents deserialization errors when loading models
    - Enables sharing of custom objects across training and production code

Custom Objects:
    - dueling_combine_streams: Lambda function for dueling Q-network architecture
      Combines value and advantage streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
"""

from tensorflow import reduce_mean
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable()
def dueling_combine_streams(inputs):
    """
    Combines value and advantage streams for dueling Q-network.

    Implements the dueling architecture where Q-values are computed from separate
    value (state worth) and advantage (action benefit) streams. This decomposition
    helps the network learn which states are valuable independent of action choice.

    Formula:
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))

    Args:
        inputs (tuple): Tuple of two tensors (value, advantage)
            - value: Tensor of shape (batch_size, 1) - state value estimates
            - advantage: Tensor of shape (batch_size, num_actions) - action advantages

    Returns:
        Tensor: Q-values of shape (batch_size, num_actions)
    """
    v, a = inputs
    return v + (a - reduce_mean(a, axis=1, keepdims=True))
