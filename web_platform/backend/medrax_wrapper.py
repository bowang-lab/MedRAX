"""
Minimal MedRAX wrapper for web backend - loads only essential components
"""

import sys
from pathlib import Path

# Add MedRAX to path
medrax_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(medrax_root))

# Import logger
from logger_config import get_logger

logger = get_logger(__name__)

def create_mock_agent():
    """Create a mock agent for development when full MedRAX isn't available"""
    class MockAgent:
        def __init__(self):
            self.workflow = MockWorkflow()

    class MockWorkflow:
        def stream(self, messages, config):
            # Mock response for development
            yield {
                "process": {
                    "messages": [MockMessage("Hello! This is a mock response. The MedRAX agent is not fully initialized yet.")]
                }
            }

    class MockMessage:
        def __init__(self, content):
            self.content = content

    return MockAgent(), {"MockTool": "Available"}

def initialize_medrax_agent(
    prompt_file="medrax/docs/system_prompts.txt",
    tools_to_use=None,
    model_dir="/model-weights",
    temp_dir="temp",
    device="cpu",  # Default to CPU for compatibility
    model="gpt-4o",
    temperature=0.7,
    top_p=0.95,
    openai_kwargs={}
):
    """
    Initialize MedRAX agent with fallback to mock for development
    """
    try:
        # Try to import and initialize full MedRAX
        from dotenv import load_dotenv
        load_dotenv()

        # Import MedRAX components (at top of function to avoid circular imports)
        from langchain_openai import ChatOpenAI
        from langgraph.checkpoint.memory import MemorySaver
        from medrax.agent import Agent

        # Import tools
        from medrax.tools import (
            ChestXRayClassifierTool,
            ChestXRayReportGeneratorTool,
            ChestXRaySegmentationTool,
            DicomProcessorTool,
            ImageVisualizerTool,
            XRayVQATool,
        )
        from medrax.utils import load_prompts_from_file

        logger.info("success", message="MedRAX components imported successfully")

        # Load prompts - fix path
        prompt_path = medrax_root / prompt_file
        if not prompt_path.exists():
            logger.warning("warning", message=f"Prompts file not found at {prompt_path}, using default prompt")
            prompt = "You are a medical AI assistant specialized in analyzing chest X-rays."
        else:
            prompts = load_prompts_from_file(str(prompt_path))
            prompt = prompts.get("MEDICAL_ASSISTANT", "You are a medical AI assistant.")

        logger.info("message", text=f"📝 Using system prompt (length: {len(prompt)})")

        # Initialize tools
        logger.info("message", text=f"🔧 Initializing tools on device: {device}")
        tools_list = []
        tools_dict = {}

        try:
            logger.info("message", text="   Loading ImageVisualizerTool...")
            tool = ImageVisualizerTool()
            tools_list.append(tool)
            tools_dict[tool.name] = tool
            logger.info("message", text=f"   ✅ {tool.name}")
        except Exception as e:
            logger.info("message", text=f"   ⚠️  ImageVisualizerTool failed: {e}")

        try:
            logger.info("message", text="   Loading ChestXRayClassifierTool...")
            tool = ChestXRayClassifierTool(device=device)
            tools_list.append(tool)
            tools_dict[tool.name] = tool
            logger.info("message", text=f"   ✅ {tool.name}")
        except Exception as e:
            logger.info("message", text=f"   ⚠️  ChestXRayClassifierTool failed: {e}")

        try:
            logger.info("message", text="   Loading ChestXRaySegmentationTool...")
            tool = ChestXRaySegmentationTool(device=device)
            tools_list.append(tool)
            tools_dict[tool.name] = tool
            logger.info("message", text=f"   ✅ {tool.name}")
        except Exception as e:
            logger.info("message", text=f"   ⚠️  ChestXRaySegmentationTool failed: {e}")

        try:
            logger.info("message", text="   Loading XRayVQATool...")
            tool = XRayVQATool(device=device)
            tools_list.append(tool)
            tools_dict[tool.name] = tool
            logger.info("message", text=f"   ✅ {tool.name}")
        except Exception as e:
            logger.info("message", text=f"   ⚠️  XRayVQATool failed: {e}")

        try:
            logger.info("message", text="   Loading ChestXRayReportGeneratorTool...")
            tool = ChestXRayReportGeneratorTool(device=device)
            tools_list.append(tool)
            tools_dict[tool.name] = tool
            logger.info("message", text=f"   ✅ {tool.name}")
        except Exception as e:
            logger.info("message", text=f"   ⚠️  ChestXRayReportGeneratorTool failed: {e}")

        # Grounding tool disabled - requires MAIRA-2 model (very large, not available)
        # try:
        #     logger.info("message", text="   Loading XRayPhraseGroundingTool...")
        #     tool = XRayPhraseGroundingTool(device=device)
        #     tools_list.append(tool)
        #     tools_dict[tool.name] = tool
        #     logger.info("message", text=f"   ✅ {tool.name}")
        # except Exception as e:
        #     logger.info("message", text=f"   ⚠️  XRayPhraseGroundingTool failed: {e}")

        try:
            logger.info("message", text="   Loading DicomProcessorTool...")
            tool = DicomProcessorTool()
            tools_list.append(tool)
            tools_dict[tool.name] = tool
            logger.info("message", text=f"   ✅ {tool.name}")
        except Exception as e:
            logger.info("message", text=f"   ⚠️  DicomProcessorTool failed: {e}")

        if not tools_list:
            logger.error("error", message="No tools loaded successfully, falling back to mock")
            return create_mock_agent()

        logger.info("success", message=f"Loaded {len(tools_list)} tools successfully")

        # Create agent
        checkpointer = MemorySaver()
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            top_p=top_p,
            **openai_kwargs
        )

        agent = Agent(
            model=llm,
            tools=tools_list,
            checkpointer=checkpointer,
            system_prompt=prompt,
            log_tools=True,
            log_dir=temp_dir
        )

        logger.info("success", message="MedRAX agent initialized successfully with real tools!")
        return agent, tools_dict

    except Exception as e:
        import traceback
        logger.error("error", message=f"Failed to initialize MedRAX agent: {e}")
        logger.info("message", text=f"   Traceback: {traceback.format_exc()}")
        logger.info("message", text="🔧 Using mock agent for development")
        return create_mock_agent()

def check_medrax_availability():
    """Check if MedRAX components are available"""
    try:
        import importlib.util
        spec = importlib.util.find_spec("medrax.agent")
        return spec is not None
    except ImportError:
        return False
