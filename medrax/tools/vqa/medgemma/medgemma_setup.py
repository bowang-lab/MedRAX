import os
from pathlib import Path
import subprocess
import socket
from contextlib import closing
import venv

def _resolve_writable_cache_dir(preferred: str | None) -> str:
    """Return a writable cache directory, falling back to user cache if needed."""
    # Preferred path first
    if preferred:
        try:
            os.makedirs(preferred, exist_ok=True)
            if os.access(preferred, os.W_OK):
                return preferred
        except Exception:
            pass
    # Fallback path under user's home
    fallback = os.path.join(Path.home(), ".cache", "medrax", "medgemma")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _is_port_free(host: str, port: int) -> bool:
    """Return True if (host, port) is free to bind on this machine."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _find_free_loopback_and_port(start_octet: int = 2, end_octet: int = 254, base_port: int = 8002, max_port_tries: int = 50) -> tuple[str, int]:
    """Find a free 127.0.0.X address and port combination.

    Tries 127.0.0.2..127.0.0.254 each with ports base_port..base_port+max_port_tries
    until a free pair is found. Falls back to 127.0.0.1 if none found for other octets.
    """
    # Try alternate loopback IPs first
    for last_octet in range(start_octet, end_octet + 1):
        host = f"127.0.0.{last_octet}"
        for port in range(base_port, base_port + max_port_tries):
            if _is_port_free(host, port):
                return host, port
    # Fallback: use 127.0.0.1 with port scan
    host = "127.0.0.1"
    for port in range(base_port, base_port + max_port_tries):
        if _is_port_free(host, port):
            return host, port
    # Last resort: system-chosen ephemeral on 127.0.0.1
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind((host, 0))
        return host, sock.getsockname()[1]


def setup_medgemma_env(cache_dir: str | None = None, device: str | None = None) -> str:
    """Set up MedGemma virtual environment and launch the FastAPI service.
    
    This function performs the following steps:
    1. Creates a virtual environment for MedGemma if it doesn't exist
    2. Installs MedGemma-specific dependencies from requirements.txt
    3. Launches the MedGemma FastAPI service in the isolated environment
    
    Returns:
        None: Launches MedGemma service as a background process
        
    Raises:
        subprocess.CalledProcessError: If pip installation fails
        FileNotFoundError: If required files are missing
        OSError: If virtual environment creation fails
    """
    # Get the directory containing this script
    current_dir = Path(__file__).resolve().parent
    
    # Define paths for MedGemma components
    medgemma_path = current_dir / "medgemma.py"
    requirements_path = current_dir / "medgemma_requirements_standard.txt"
    env_dir = current_dir / "medgemma_env"

    # Determine executable paths based on operating system
    if os.name == "nt":  # Windows
        pip_executable = env_dir / "Scripts" / "pip"
        python_executable = env_dir / "Scripts" / "python"
    else:  # Unix/Linux/macOS
        pip_executable = env_dir / "bin" / "pip"
        python_executable = env_dir / "bin" / "python"

    # Create virtual environment if it doesn't exist
    if not env_dir.exists():
        print("Creating MedGemma virtual environment...")
        venv.create(env_dir, with_pip=True)
        
        # Install MedGemma dependencies
        print("Installing MedGemma dependencies...")
        subprocess.check_call([
            str(pip_executable), 
            "install", 
            "-r", 
            str(requirements_path)
        ])

    # Ensure environment exists before accessing executables
    if not env_dir.exists():
        raise RuntimeError("Failed to create MedGemma virtual environment")

    # Decide host/port to avoid collisions when multiple instances run
    medgemma_host = os.getenv("MEDGEMMA_HOST")
    medgemma_port_env = os.getenv("MEDGEMMA_PORT")
    chosen_host: str
    chosen_port: int
    if medgemma_host and medgemma_port_env:
        try:
            port_val = int(medgemma_port_env)
        except ValueError:
            port_val = 8002
        # If explicit host/port are provided, prefer them; if taken, try incrementing the port on the same host
        chosen_host = medgemma_host
        chosen_port = None
        for p in range(port_val, port_val + 50):
            if _is_port_free(medgemma_host, p):
                chosen_port = p
                break
        if chosen_port is None:
            print(f"No free ports in range {port_val}-{port_val+49} on {medgemma_host}; selecting a free loopback IP/port...")
            chosen_host, chosen_port = _find_free_loopback_and_port()
    else:
        # Auto-pick a free loopback IP and port
        chosen_host, chosen_port = _find_free_loopback_and_port()

    print(f"Launching MedGemma FastAPI service on {chosen_host}:{chosen_port} ...")
    env = os.environ.copy()
    resolved_cache = _resolve_writable_cache_dir(cache_dir)
    env["MEDGEMMA_CACHE_DIR"] = resolved_cache
    if device:
        env["MEDGEMMA_DEVICE"] = device
    # Pass the chosen binding to the server via env
    env["MEDGEMMA_HOST"] = chosen_host
    env["MEDGEMMA_PORT"] = str(chosen_port)
    subprocess.Popen([
        str(python_executable), 
        str(medgemma_path)
    ], env=env)

    # Return the base URL so callers can use it. If bound to 0.0.0.0, use 127.0.0.1 for local client access.
    chosen_client_host = "127.0.0.1" if chosen_host in ("0.0.0.0", "::") else chosen_host
    return f"http://{chosen_client_host}:{chosen_port}"
    # Note: stdout and stderr redirection commented out for debugging
    # stdout=subprocess.DEVNULL,
    # stderr=subprocess.DEVNULL,