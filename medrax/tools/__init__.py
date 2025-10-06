"""Tools for the Medical Agent."""

# Import tools with graceful error handling for optional dependencies

# Core tools (should always work)
try:
    from .utils import *
except ImportError as e:
    print(f"Warning: Failed to import utils tools: {e}")

try:
    from .classification import *
except ImportError as e:
    print(f"Warning: Failed to import classification tools: {e}")

try:
    from .report_generation import *
except ImportError as e:
    print(f"Warning: Failed to import report generation tools: {e}")

try:
    from .segmentation import *
except ImportError as e:
    print(f"Warning: Failed to import segmentation tools: {e}")

try:
    from .xray_vqa import *
except ImportError as e:
    print(f"Warning: Failed to import VQA tools: {e}")

# Optional tools (may require additional dependencies)
try:
    from .llava_med import *
except ImportError as e:
    print(f"Info: LLaVA-Med not available: {e}")

try:
    from .grounding import *
except ImportError as e:
    print(f"Info: Grounding tool not available: {e}")

try:
    from .generation import *
except ImportError as e:
    print(f"Info: Generation tool not available: {e}")

try:
    from .dicom import *
except ImportError as e:
    print(f"Info: DICOM processor not available: {e}")

try:
    from .rag import *
except ImportError as e:
    print(f"Info: RAG tool not available (requires langchain_cohere): {e}")