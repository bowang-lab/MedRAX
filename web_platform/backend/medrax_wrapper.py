"""
Minimal MedRAX wrapper for web backend - loads only essential components
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add MedRAX to path
medrax_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(medrax_root))

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
        
        # Import MedRAX components
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_openai import ChatOpenAI
        from medrax.agent import Agent
        from medrax.utils import load_prompts_from_file
        
        # Import tools
        from medrax.tools import (
            ImageVisualizerTool,
            ChestXRayClassifierTool,
            ChestXRaySegmentationTool,
            ChestXRayReportGeneratorTool,
            XRayVQATool,
            XRayPhraseGroundingTool,
            DicomProcessorTool
        )
        
        print("✅ MedRAX components imported successfully")
        
        # Load prompts - fix path
        prompt_path = medrax_root / prompt_file
        if not prompt_path.exists():
            print(f"⚠️  Prompts file not found at {prompt_path}, using default prompt")
            prompt = "You are a medical AI assistant specialized in analyzing chest X-rays."
        else:
            prompts = load_prompts_from_file(str(prompt_path))
            prompt = prompts.get("MEDICAL_ASSISTANT", "You are a medical AI assistant.")
        
        print(f"📝 Using system prompt (length: {len(prompt)})")
        
        # Initialize tools
        print(f"🔧 Initializing tools on device: {device}")
        tools_list = []
        tools_dict = {}
        
        try:
            print("   Loading ImageVisualizerTool...")
            tool = ImageVisualizerTool()
            tools_list.append(tool)
            tools_dict[tool.name] = tool
            print(f"   ✅ {tool.name}")
        except Exception as e:
            print(f"   ⚠️  ImageVisualizerTool failed: {e}")
        
        try:
            print("   Loading ChestXRayClassifierTool...")
            tool = ChestXRayClassifierTool(device=device)
            tools_list.append(tool)
            tools_dict[tool.name] = tool
            print(f"   ✅ {tool.name}")
        except Exception as e:
            print(f"   ⚠️  ChestXRayClassifierTool failed: {e}")
        
        try:
            print("   Loading ChestXRaySegmentationTool...")
            tool = ChestXRaySegmentationTool(device=device)
            tools_list.append(tool)
            tools_dict[tool.name] = tool
            print(f"   ✅ {tool.name}")
        except Exception as e:
            print(f"   ⚠️  ChestXRaySegmentationTool failed: {e}")
        
        try:
            print("   Loading XRayVQATool...")
            tool = XRayVQATool(device=device)
            tools_list.append(tool)
            tools_dict[tool.name] = tool
            print(f"   ✅ {tool.name}")
        except Exception as e:
            print(f"   ⚠️  XRayVQATool failed: {e}")
        
        try:
            print("   Loading ChestXRayReportGeneratorTool...")
            tool = ChestXRayReportGeneratorTool(device=device)
            tools_list.append(tool)
            tools_dict[tool.name] = tool
            print(f"   ✅ {tool.name}")
        except Exception as e:
            print(f"   ⚠️  ChestXRayReportGeneratorTool failed: {e}")
        
        # Grounding tool disabled - requires MAIRA-2 model (very large, not available)
        # try:
        #     print("   Loading XRayPhraseGroundingTool...")
        #     tool = XRayPhraseGroundingTool(device=device)
        #     tools_list.append(tool)
        #     tools_dict[tool.name] = tool
        #     print(f"   ✅ {tool.name}")
        # except Exception as e:
        #     print(f"   ⚠️  XRayPhraseGroundingTool failed: {e}")
        
        try:
            print("   Loading DicomProcessorTool...")
            tool = DicomProcessorTool()
            tools_list.append(tool)
            tools_dict[tool.name] = tool
            print(f"   ✅ {tool.name}")
        except Exception as e:
            print(f"   ⚠️  DicomProcessorTool failed: {e}")
        
        if not tools_list:
            print("❌ No tools loaded successfully, falling back to mock")
            return create_mock_agent()
        
        print(f"✅ Loaded {len(tools_list)} tools successfully")
        
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
        
        print("✅ MedRAX agent initialized successfully with real tools!")
        return agent, tools_dict
        
    except Exception as e:
        import traceback
        print(f"❌ Failed to initialize MedRAX agent: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        print("🔧 Using mock agent for development")
        return create_mock_agent()

def check_medrax_availability():
    """Check if MedRAX components are available"""
    try:
        from medrax.agent import Agent
        return True
    except ImportError:
        return False
