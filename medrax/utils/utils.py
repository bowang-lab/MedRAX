import os
import json
import numpy as np
from typing import Dict, List, Union, Tuple


def preprocess_medical_image(
    image: np.ndarray, 
    target_range: Tuple[float, float] = (0.0, 1.0),
    clip_values: bool = True
) -> np.ndarray:
    """
    Preprocess medical images by auto-detecting bit depth and normalizing appropriately.
    
    This function handles both 8-bit (0-255) and 16-bit (0-65535) images automatically,
    normalizing them to the target range. It's designed for medical imaging tools that
    expect consistent input ranges regardless of the original image bit depth.
    
    Args:
        image (np.ndarray): Input image array (2D or 3D)
        target_range (Tuple[float, float]): Target range for normalization (default: (0.0, 1.0))
        clip_values (bool): Whether to clip values to target range (default: True)
    
    Returns:
        np.ndarray: Normalized image in the target range
        
    Raises:
        ValueError: If image is empty or has invalid values
        ValueError: If target_range is invalid
    """
    if image.size == 0:
        raise ValueError("Input image is empty")
    
    if len(target_range) != 2 or target_range[0] >= target_range[1]:
        raise ValueError("target_range must be a tuple of (min, max) where min < max")
    
    # Convert to float for processing
    image = image.astype(np.float32)
    
    # Auto-detect bit depth based on maximum value
    max_val = np.max(image)
    min_val = np.min(image)
    
    # Determine the expected maximum value based on bit depth
    if max_val <= 255:
        # 8-bit image
        expected_max = 255.0
    elif max_val <= 65535:
        # 16-bit image
        expected_max = 65535.0
    else:
        # Higher bit depth or already normalized, use actual max
        expected_max = max_val
    
    # Normalize to 0-1 range first
    if expected_max > 0:
        image = (image - min_val) / (expected_max - min_val)
    else:
        # Handle edge case where image has no contrast
        image = np.zeros_like(image)
    
    # Scale to target range
    target_min, target_max = target_range
    image = image * (target_max - target_min) + target_min
    
    # Clip values if requested
    if clip_values:
        image = np.clip(image, target_min, target_max)
    
    return image


def normalize_medical_image_for_torchxrayvision(image: np.ndarray) -> np.ndarray:
    """
    Normalize medical images specifically for TorchXRayVision models.
    
    This function is a convenience wrapper around preprocess_medical_image
    that normalizes images to the -1024 to 1024 range expected by TorchXRayVision models.
    This range corresponds to the Hounsfield Unit scale adapted for X-ray images.
    
    Args:
        image (np.ndarray): Input image array (2D or 3D)
    
    Returns:
        np.ndarray: Normalized image in -1024 to 1024 range
    """
    return preprocess_medical_image(image, target_range=(-1024.0, 1024.0))


def load_prompts_from_file(file_path: str) -> Dict[str, str]:
    """
    Load multiple prompts from a file.

    Args:
    file_path (str): Path to the file containing prompts.

    Returns:
    Dict[str, str]: A dictionary of prompt names and their content.

    Raises:
    FileNotFoundError: If the specified file is not found.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompts file not found: {file_path}")

    prompts = {}
    current_prompt = None
    current_content = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                if current_prompt:
                    prompts[current_prompt] = "\n".join(current_content).strip()
                current_prompt = line[1:-1]
                current_content = []
            elif line:
                current_content.append(line)

    if current_prompt:
        prompts[current_prompt] = "\n".join(current_content).strip()

    return prompts


def load_tool_prompts(tools: List[str], tools_json_path: str) -> str:
    """
    Load prompts for specified tools from the tools.json file.

    Args:
    tools (List[str]): List of tool names to load prompts for.
    tools_json_path (str): Path to the tools.json file.

    Returns:
    str: A string containing prompts for the specified tools.

    Raises:
    FileNotFoundError: If the tools.json file is not found.
    """
    if not os.path.exists(tools_json_path):
        raise FileNotFoundError(f"Tools JSON file not found: {tools_json_path}")

    with open(tools_json_path, "r") as file:
        tools_data = json.load(file)

    tool_prompts = []
    for tool in tools:
        if tool in tools_data:
            tool_info = tools_data[tool]
            tool_prompt = f"Tool: {tool}\n"
            tool_prompt += f"Description: {tool_info['description']}\n"
            tool_prompt += f"Usage: {tool_info['prompt']}\n"
            tool_prompt += f"Input type: {tool_info['input_type']}\n"
            tool_prompt += f"Return type: {tool_info['return_type']}\n\n"
            tool_prompts.append(tool_prompt)

    return "\n".join(tool_prompts)


def load_system_prompt(
    system_prompts_file: str,
    system_prompt_type: str,
    tools: List[str],
    tools_json_path: str,
) -> str:
    """
    Load the system prompt by combining the system prompt and tool information.

    Args:
    system_prompts_file (str): Path to the file containing system prompts.
    system_prompt_type (str): The type of system prompt to use.
    tools (List[str]): List of tool names to include in the prompt.
    tools_json_path (str): Path to the tools.json file.

    Returns:
    str: The system prompt combining system prompt and tool information.
    """
    prompts = load_prompts_from_file(system_prompts_file)
    system_prompt = prompts.get(system_prompt_type, "GENERAL_ASSISTANT")
    tool_prompts = load_tool_prompts(tools, tools_json_path)

    return f"{system_prompt}\n\nTools:\n{tool_prompts}".strip()
