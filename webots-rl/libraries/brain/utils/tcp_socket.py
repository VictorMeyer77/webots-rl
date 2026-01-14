"""
TCP Socket Utilities for RL Training Communication

This module provides low-level TCP socket communication utilities for exchanging
string messages between Webots supervisor controllers and external training processes.
It implements a simple length-prefixed message protocol to ensure reliable message
transmission over TCP connections.

Protocol Design:
    Messages are transmitted using a length-prefix protocol:
    1. First 4 bytes: Message length as big-endian unsigned integer (>I)
    2. Remaining bytes: UTF-8 encoded message content

    This protocol ensures:
    - Complete message reception (no partial reads)
    - Support for variable-length messages
    - Binary-safe transmission of text data
    - Cross-platform compatibility (big-endian byte order)

Message Format:
    [4 bytes: length][N bytes: UTF-8 data]
    Example: "hello" → [0x00, 0x00, 0x00, 0x05, 'h', 'e', 'l', 'l', 'o']

Use Cases:
    - Sending observation data from Webots to trainer
    - Receiving action commands from trainer to Webots
    - Exchanging control signals (reset, done, config)
    - Synchronizing episode boundaries between processes

Thread Safety:
    These functions are NOT thread-safe. Concurrent calls on the same socket
    may result in interleaved messages or protocol errors. Use external locking
    if multiple threads access the same connection.

Performance Considerations:
    - Uses blocking I/O (synchronous communication)
    - recv() loops until complete message is received
    - No internal buffering beyond Python's socket buffers
    - Suitable for moderate message rates (<1000 msg/sec)

Protocol Limitations:
    - Maximum message size: 4GB (2^32-1 bytes)
    - No message compression or encryption
    - No automatic reconnection on failure
    - Single message per call (no batching)
"""

import socket
import struct


def send(conn: socket.socket, message: str) -> None:
    """
    Send a UTF-8 encoded string message with a 4-byte length prefix.

    Args:
        message (str): Message to send.
    Raises:
        ConnectionError: If socket is not initialized.
    """
    if not conn:
        raise ConnectionError("Socket not initialized")
    buffer = message.encode("utf-8")
    msg_len = struct.pack(">I", len(buffer))
    conn.sendall(msg_len + buffer)


def read(conn: socket.socket) -> str | None:
    """
    Read a length-prefixed UTF-8 string message.

    Returns:
        str: Received message, or empty string on timeout.
    Raises:
        ConnectionError: If socket is not initialized or connection lost.
    """
    if not conn:
        raise ConnectionError("Socket not initialized")
    try:
        raw_len = conn.recv(4)
        if not raw_len:
            return None
        msg_len = struct.unpack(">I", raw_len)[0]
        message = b""
        while len(message) < msg_len:
            chunk = conn.recv(msg_len - len(message))
            if not chunk:
                raise ConnectionError("Connection lost during reception")
            message += chunk
        message = message.decode("utf-8")
        return message
    except socket.timeout:
        return None
