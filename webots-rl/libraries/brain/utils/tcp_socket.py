import socket
import struct
import time

from brain.utils.logger import logger

HOST = "127.0.0.1"
PORT = 65432
SOCKET_TIMEOUT = 0.001


class TcpSocket:
    """
    TCP socket utility for client-server communication.

    Attributes:
        is_client (bool): Whether the socket acts as a client or a server.
        side (str): "client" or "server" based on the mode.
        server (socket.socket | None): Server socket (server mode only).
        connection (socket.socket): Active connection socket.
    """

    is_client: bool
    side: str
    server: socket.socket | None
    connection: socket.socket

    def __init__(self, is_client: bool = False):
        """
        Initialize the TCP socket as client or server.

        Args:
            is_client (bool, optional): If True, acts as client, else as server.
            Defaults to False.
        """
        self.is_client = is_client
        if is_client:
            self.side = "client"
            self._init_client()
        else:
            self.side = "server"
            self._init_server()

    def __del__(self):
        """
        Clean up sockets on deletion.
        """
        if not self.is_client:
            self.server.close()
        self.connection.close()
        logger().info(f"TCP {self.side}: socket closed")

    def _init_client(self):
        """
        Initialize the TCP socket as a client.

        Attempts to connect to the server at HOST and PORT, retrying up to 10 times.
        Sets a timeout for the connection.

        Raises:
            ConnectionError: If unable to connect after multiple attempts.
        """
        for _ in range(10):
            try:
                self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.connection.connect((HOST, PORT))
                self.connection.settimeout(SOCKET_TIMEOUT)
                logger().info("TCP client: connected with server")
                return
            except ConnectionRefusedError:
                logger().info("TCP client: waiting for TCP server...")
                time.sleep(1)
        raise ConnectionError(f"Could not connect to TCP server ({HOST}:{PORT}) after multiple attempts")

    def _init_server(self):
        """
        Initialize the TCP socket as a server.

        Binds the server socket to HOST and PORT, listens for a single client,
        accepts the connection, and sets a timeout.
        """
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((HOST, PORT))
        self.server.listen(1)
        logger().info(f"TCP server: listening on {HOST}:{PORT}")
        self.connection, addr = self.server.accept()
        self.connection.settimeout(SOCKET_TIMEOUT)
        logger().info(f"TCP server: connected by {addr}")

    def send(self, message: str):
        """
        Send a UTF-8 encoded string message with a 4-byte length prefix.

        Args:
            message (str): Message to send.
        Raises:
            ConnectionError: If socket is not initialized.
        """
        if not self.connection:
            raise ConnectionError("Socket not initialized")
        buffer = message.encode("utf-8")
        msg_len = struct.pack(">I", len(buffer))
        self.connection.sendall(msg_len + buffer)
        logger().debug(f"TCP {self.side}: sent message: {message}")

    def read(self) -> str:
        """
        Read a length-prefixed UTF-8 string message.

        Returns:
            str: Received message, or empty string on timeout.
        Raises:
            ConnectionError: If socket is not initialized or connection lost.
        """
        if not self.connection:
            raise ConnectionError("Socket not initialized")
        try:
            raw_len = self.connection.recv(4)
            if not raw_len:
                logger().debug(f"TCP {self.side}: read an empty message")
                return ""
            msg_len = struct.unpack(">I", raw_len)[0]
            message = b""
            while len(message) < msg_len:
                chunk = self.connection.recv(msg_len - len(message))
                if not chunk:
                    raise ConnectionError("Connection lost during reception")
                message += chunk
            message = message.decode("utf-8")
            logger().debug(f"TCP {self.side}: received message: {message}")
            return message
        except socket.timeout:
            logger().debug(f"TCP {self.side}: read timeout")
            return ""
