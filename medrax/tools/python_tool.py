"""Python sandbox tool for executing Python code in a secure environment.

This module provides a pre-configured PyodideSandboxTool instance for the medical agent.

Note: Deno must be installed on the host system for this tool to work.
https://docs.deno.com/runtime/getting_started/installation

The python_sandbox instance is configured with:
- pip_packages: Pre-installs a list of common data analysis packages for efficiency.
- stateful=True: Enables the sandbox to maintain variables, functions, and
  imports between calls. The agent state must be configured to handle the
  session_bytes and session_metadata for this to work.
- allow_net=True: Allows the sandbox to make network requests, which is
  necessary for `micropip` to install packages from PyPI. This should be
  used with caution in production environments.
"""

from langchain_sandbox import PyodideSandboxTool
from typing import Optional, List


def create_python_sandbox(
    pip_packages: Optional[List[str]] = None,
    stateful: bool = True,
    allow_net: bool = True,
    **kwargs,
) -> PyodideSandboxTool:
    """Create a pre-configured PyodideSandboxTool instance for the medical agent.

    Args:
        pip_packages (list, optional): List of pip packages to pre-install.
            Defaults to common medical analysis packages.
        stateful (bool, optional): Whether to maintain state between calls.
            Defaults to True.
        allow_net (bool, optional): Whether to allow network requests.
            Defaults to True.
        deno_path (str, optional): Path to the Deno executable.
            Defaults to "/home/adib/deno-2.1.0-x86_64-unknown-linux-gnu/bin/deno".
        **kwargs: Additional keyword arguments passed to PyodideSandboxTool.

    Returns:
        PyodideSandboxTool: Configured sandbox with medical analysis packages
    """
    if pip_packages is None:
        pip_packages = [
            "pandas",
            "numpy",
            "pydicom",
            "SimpleITK",
            "scikit-image",
            "Pillow",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "openpyxl",
        ]

    return PyodideSandboxTool(
        pip_packages=pip_packages,
        stateful=stateful,
        allow_net=allow_net,
        **kwargs,
    )
