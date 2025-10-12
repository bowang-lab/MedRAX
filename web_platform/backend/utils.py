"""
Utility functions for MedRAX Backend
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


def ensure_json_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy/torch types to Python natives for JSON serialization.

    Args:
        obj: Object to convert (can be dict, list, numpy/torch types, etc.)

    Returns:
        JSON-serializable version of the object

    Usage:
        result = {"accuracy": np.float32(0.95), "counts": np.array([1, 2, 3])}
        clean_result = ensure_json_serializable(result)
        json.dumps(clean_result)  # Works!
    """
    # Handle None
    if obj is None:
        return None

    # Handle numpy types
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    # Handle PyTorch tensors
    elif hasattr(obj, 'detach') and hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):  # torch.Tensor
        return obj.detach().cpu().numpy().tolist()

    # Handle datetime
    elif isinstance(obj, datetime):
        return obj.isoformat()

    # Handle Path
    elif isinstance(obj, Path):
        return str(obj)

    # Handle dictionaries
    elif isinstance(obj, dict):
        return {key: ensure_json_serializable(value) for key, value in obj.items()}

    # Handle lists and tuples
    elif isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item) for item in obj]

    # Handle sets
    elif isinstance(obj, set):
        return list(obj)

    # Return as-is for already JSON-serializable types
    elif isinstance(obj, (str, int, float, bool)):
        return obj

    # For other objects, try to convert to string
    else:
        try:
            return str(obj)
        except Exception:
            return f"<non-serializable: {type(obj).__name__}>"


def validate_image_path(path: str, temp_dir: str = "temp") -> bool:
    """
    Validate that an image path is safe and within allowed directory.

    Args:
        path: Path to validate
        temp_dir: Allowed base directory

    Returns:
        True if path is valid and safe
    """
    if not path:
        return False

    try:
        # Convert to Path object
        path_obj = Path(path)
        temp_path = Path(temp_dir).resolve()

        # Check if path is within temp directory
        resolved = path_obj.resolve()

        # Must be within temp directory
        if not str(resolved).startswith(str(temp_path)):
            return False

        # Check file exists and is a file
        if not resolved.exists() or not resolved.is_file():
            return False

        # Check extension is image
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.dcm'}
        if resolved.suffix.lower() not in allowed_extensions:
            return False

        return True

    except Exception:
        return False


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize a filename to prevent path traversal and other issues.

    Args:
        filename: Original filename
        max_length: Maximum allowed length

    Returns:
        Sanitized filename
    """
    # Remove path separators
    filename = filename.replace('/', '_').replace('\\', '_')

    # Remove null bytes
    filename = filename.replace('\0', '')

    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')

    # Truncate if too long
    if len(filename) > max_length:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_length = max_length - len(ext) - 1
        filename = f"{name[:max_name_length]}.{ext}" if ext else name[:max_length]

    return filename


def validate_chat_message(message: str, max_length: int = 10000) -> tuple[bool, str]:
    """
    Validate chat message input.

    Args:
        message: Message to validate
        max_length: Maximum allowed length

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not message:
        return False, "Message cannot be empty"

    if not isinstance(message, str):
        return False, "Message must be a string"

    if len(message) > max_length:
        return False, f"Message too long (max {max_length} characters)"

    # Check for suspicious patterns
    if '\0' in message:
        return False, "Message contains null bytes"

    return True, ""


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def cleanup_old_files(directory: Path, max_age_hours: int = 24):
    """
    Clean up files older than specified age.

    Args:
        directory: Directory to clean
        max_age_hours: Maximum age in hours
    """
    if not directory.exists():
        return

    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    deleted_count = 0
    freed_space = 0

    for file_path in directory.iterdir():
        if file_path.is_file():
            try:
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff_time:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    deleted_count += 1
                    freed_space += file_size
            except Exception as e:
                # Log but don't fail
                print(f"Failed to delete {file_path}: {e}")

    if deleted_count > 0:
        print(f"🧹 Cleaned up {deleted_count} old files, freed {format_file_size(freed_space)}")


# Example usage
if __name__ == "__main__":
    # Test JSON serialization
    test_data = {
        "float32": np.float32(0.95),
        "int64": np.int64(42),
        "array": np.array([1, 2, 3]),
        "nested": {
            "value": np.float64(3.14),
            "list": [np.int32(1), np.int32(2)]
        }
    }

    clean_data = ensure_json_serializable(test_data)
    print("Cleaned data:", clean_data)

    # Test path validation
    print("Valid path:", validate_image_path("temp/test.jpg"))
    print("Invalid path:", validate_image_path("../etc/passwd"))

    # Test message validation
    valid, error = validate_chat_message("Hello, analyze this image")
    print(f"Message valid: {valid}")

