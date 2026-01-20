"""
TCP socket utilities for framed message exchange.

This module provides helper functions to send and receive length-prefixed
messages over a TCP socket. The framing format uses a 4-byte big-endian unsigned
integer to encode the payload length, followed by the payload bytes (UTF-8
encoded JSON).

Framing protocol:
    [4 bytes: payload length (big-endian)] + [payload bytes (UTF-8 JSON)]

The framing is necessary because TCP is a stream protocol that does not preserve
message boundaries. Without framing, the receiver cannot distinguish where one
message ends and another begins. A fixed-size header allows the receiver to know
exactly how many bytes to read for the complete message.

Functions:
    send(conn, message): Serialize a dict to JSON and send with length prefix.
    read(conn): Read one framed message and return the decoded JSON object.

Error handling:
    Both functions catch network errors (TimeoutError, ConnectionError, OSError)
    and log them via the project logger. The `read()` function also catches
    json.JSONDecodeError and returns None on any error.

Notes:
    - The `recv()` call may return fewer bytes than requested due to TCP
      fragmentation. The `read()` function loops until the full payload is
      received to handle this correctly.
    - If the connection is closed during reception, `read()` raises
      ConnectionError with a descriptive message.
    - The `send()` function uses `sendall()` to ensure all bytes are transmitted,
      but may raise an exception if the connection fails mid-transmission.
    - Both functions expect an active socket connection; they raise
      ConnectionError if the socket is not initialized.
"""

import json
import socket
import struct
from brain.utils.logger import logger

def send(conn: socket.socket, message: dict) -> None:
    """
    Serialize a dictionary to JSON and send it with a 4-byte length prefix.

    The message is serialized to JSON, encoded as UTF-8, and sent with a
    4-byte big-endian length header followed by the payload bytes.

    Args:
        conn: Active TCP socket connection.
        message: Dictionary to serialize and send as JSON.

    Raises:
        ConnectionError: If the socket is not initialized.
    """
    if not conn:
        raise ConnectionError("Socket not initialized")
    message_json = json.dumps(message)
    buffer = message_json.encode("utf-8")
    msg_len = struct.pack(">I", len(buffer))
    try:
        conn.sendall(msg_len + buffer)
    except (TimeoutError, ConnectionError, OSError) as e:
        logger().debug(f"TCP send error: {e}")

def read(conn: socket.socket) -> dict | None:
    """
    Read one framed message and return the decoded JSON object.

    Reads a 4-byte big-endian length header, then reads exactly that many
    payload bytes, decodes UTF-8, and parses JSON. Handles TCP fragmentation
    by looping until the full payload is received.

    Args:
        conn: Active TCP socket connection.

    Returns:
        Decoded JSON object (dict), or None if the connection is closed or
        an error occurs (network error or JSON decode error).

    Raises:
        ConnectionError: If the socket is not initialized or if the connection
                        is lost during reception (no data received when expected).

    Notes:
        This function returns None instead of raising on errors to allow the
        caller to handle connection failures gracefully (e.g., retry logic).
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
        message = json.loads(message.decode("utf-8"))
        return message
    except (TimeoutError, ConnectionError, OSError, json.JSONDecodeError) as e:
        logger().debug(f"TCP read error: {e}")
        return None
