import logging
import socket

logger = logging.getLogger(__name__)


def find_available_port(host: str = "127.0.0.1", port: int = None) -> int:
    """
    Find a free port on the host if the port is not provided. If the port is provided, check if it is available.

    Args:
        host: The host to bind to. Default is "127.0.0.1".
        port: The port to bind to. Default is None, which means a free port will be found.

    Returns:
        The available port.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return port
    except OSError:
        pass
    logger.warning(f"Port {port} is already in use on {host}.")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        free_port = s.getsockname()[1]
    logger.warning(f"Using port {free_port} instead.")
    return free_port
