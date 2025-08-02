import os
from pathlib import Path
import subprocess
import venv

def setup_medgemma_env():
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

    # Launch MedGemma FastAPI service
    print("Launching MedGemma FastAPI service...")
    subprocess.Popen([
        str(python_executable), 
        str(medgemma_path)
    ])
    # Note: stdout and stderr redirection commented out for debugging
    # stdout=subprocess.DEVNULL,
    # stderr=subprocess.DEVNULL,