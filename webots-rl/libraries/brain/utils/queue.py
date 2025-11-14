"""
Queue utilities for JSON message passing over Webots `Emitter` and `Receiver`.

This module defines a `Queue` abstraction that:
- Sends Python `dict` objects serialized as UTF-8 JSON.
- Reads all pending packets from a `Receiver`, decoding them into Python objects.
"""

import json
from typing import Any, List

from brain.utils.logger import logger
from controller import Emitter, Receiver


class Queue:
    """
    Lightweight JSON message queue over Webots controller devices.

    Messages are serialized with `json.dumps` and sent as UTF-8 bytes.
    The `read` method drains all currently queued packets and returns
    the decoded Python objects (usually `dict` instances).
    """

    emitter: Emitter
    receiver: Receiver
    message_buffer: List[dict]

    def __init__(self, timestep: int, emitter: Emitter, receiver: Receiver):
        """
        Initialize the queue and enable the receiver.

        Parameters:
            timestep: Simulation timestep in milliseconds used to enable the receiver.
            emitter: Webots `Emitter` device used to send messages.
            receiver: Webots `Receiver` device used to receive messages.
        """
        self.emitter = emitter
        self.receiver = receiver
        self.receiver.enable(timestep)
        logger().debug(f"Queue initialized with timestep {timestep} ms")

        self.message_buffer = []

    def send(self, message: dict) -> None:
        """
        Send a JSON-serializable dictionary.

        Parameters:
            message: Dictionary to serialize and transmit.
        """
        self.emitter.send(json.dumps(message).encode("utf-8"))

    def read(self) -> List[Any]:
        """
        Read and decode all pending messages.

        Returns:
            A list of decoded JSON objects (usually dictionaries). Empty if no messages.
        """
        messages: List[Any] = []
        while self.receiver.getQueueLength() > 0:
            data = self.receiver.getString()
            messages.append(json.loads(data))
            self.receiver.nextPacket()
        return messages

    def search_message(self, key: str) -> List[dict]:
        """
        Search buffered messages for a given key.

        Parameters:
            key: The key to search for in buffered messages.

        Returns:
            A list of messages containing the specified key.
        """
        self.message_buffer.extend(self.read())
        search_messages = [message for message in self.message_buffer if key in message.keys()]
        return search_messages

    def clear_buffer(self):
        """
        Clear the internal message buffer.
        """
        self.message_buffer = []
