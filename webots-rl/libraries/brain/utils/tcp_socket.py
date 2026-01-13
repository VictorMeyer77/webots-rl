import socket
import struct


def send(conn: socket.socket, message: str):
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
