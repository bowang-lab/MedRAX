"""MedRAX LLM provider implementation."""

import os
import time
import tempfile
import shutil
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from .base import LLMProvider, LLMRequest, LLMResponse

# Import MedRAX components
from medrax.agent import Agent
from medrax.tools import *
from medrax.utils import load_prompts_from_file, RAGConfig
from medrax.models import ModelFactory
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage


class MedRAXProvider(LLMProvider):
    """MedRAX LLM provider that uses the full MedRAX agent system."""

    def __init__(self, model_name: str, **kwargs):
        """Initialize MedRAX provider.
        
        Args:
            model_name (str): Base LLM model name (e.g., "gpt-4.1-2025-04-14")
            **kwargs: Additional configuration parameters
        """
        # MedRAX-specific configuration
        self.tools_to_use = kwargs.get("tools_to_use", [
            "WebBrowserTool",
            "MedicalRAGTool", 
            "PythonSandboxTool"
        ])
        self.model_dir = kwargs.get("model_dir", "/model-weights")
        self.temp_dir = kwargs.get("temp_dir", "temp")
        self.device = kwargs.get("device", "cuda")
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 0.95)
        self.rag_config = kwargs.get("rag_config")
        self.prompt_file = kwargs.get("prompt_file", "medrax/docs/system_prompts.txt")
        
        # Initialize agent as None, will be created in _setup
        self.agent = None
        self.tools_dict = None
        
        super().__init__(model_name, **kwargs)

    def _setup(self) -> None:
        """Set up MedRAX agent system."""
        try:
            # Load system prompts
            prompts = load_prompts_from_file(self.prompt_file)
            prompt = prompts["MEDICAL_ASSISTANT"]
            
            # Initialize tools
            all_tools = {
                "TorchXRayVisionClassifierTool": lambda: TorchXRayVisionClassifierTool(device=self.device),
                "ArcPlusClassifierTool": lambda: ArcPlusClassifierTool(cache_dir=self.model_dir, device=self.device),
                "ChestXRaySegmentationTool": lambda: ChestXRaySegmentationTool(device=self.device),
                "LlavaMedTool": lambda: LlavaMedTool(cache_dir=self.model_dir, device=self.device, load_in_8bit=True),
                "XRayVQATool": lambda: XRayVQATool(cache_dir=self.model_dir, device=self.device),
                "ChestXRayReportGeneratorTool": lambda: ChestXRayReportGeneratorTool(
                    cache_dir=self.model_dir, device=self.device
                ),
                "XRayPhraseGroundingTool": lambda: XRayPhraseGroundingTool(
                    cache_dir=self.model_dir, temp_dir=self.temp_dir, load_in_8bit=True, device=self.device
                ),
                "ChestXRayGeneratorTool": lambda: ChestXRayGeneratorTool(
                    model_path=f"{self.model_dir}/roentgen", temp_dir=self.temp_dir, device=self.device
                ),
                "ImageVisualizerTool": lambda: ImageVisualizerTool(),
                "DicomProcessorTool": lambda: DicomProcessorTool(temp_dir=self.temp_dir),
                "MedicalRAGTool": lambda: RAGTool(config=self.rag_config) if self.rag_config else None,
                "WebBrowserTool": lambda: WebBrowserTool(),
            }
            
            # Add PythonSandboxTool if available
            try:
                all_tools["PythonSandboxTool"] = lambda: create_python_sandbox()
            except Exception as e:
                print(f"Warning: PythonSandboxTool not available: {e}")
            
            # Initialize selected tools
            self.tools_dict = {}
            for tool_name in self.tools_to_use:
                if tool_name in all_tools:
                    try:
                        tool_instance = all_tools[tool_name]()
                        if tool_instance is not None:
                            self.tools_dict[tool_name] = tool_instance
                    except Exception as e:
                        print(f"Warning: Failed to initialize {tool_name}: {e}")
            
            # Set up checkpointing
            checkpointer = MemorySaver()
            
            # Create the language model
            llm = ModelFactory.create_model(
                model_name=self.model_name,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            # Create the agent
            self.agent = Agent(
                llm,
                tools=list(self.tools_dict.values()),
                log_tools=False,  # Disable logging for benchmarking
                system_prompt=prompt,
                checkpointer=checkpointer,
                debug=False,
            )
            
            # Create temporary directory for this session
            self.session_temp_dir = Path(tempfile.mkdtemp(prefix="medrax_bench_"))
            
            print(f"MedRAX agent initialized with tools: {list(self.tools_dict.keys())}")
            
        except Exception as e:
            print(f"Error initializing MedRAX agent: {e}")
            raise

    def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using MedRAX agent.
        
        Args:
            request (LLMRequest): The request containing text, images, and parameters
            
        Returns:
            LLMResponse: The response from MedRAX agent
        """
        start_time = time.time()
        
        if self.agent is None:
            return LLMResponse(
                content="Error: MedRAX agent not initialized",
                duration=time.time() - start_time,
                raw_response=None
            )
        
        try:
            # Build messages for the agent
            messages = []
            thread_id = str(int(time.time() * 1000))  # Unique thread ID
            
            # Copy images to session temp directory and provide paths
            image_paths = []
            if request.images:
                valid_images = self._validate_image_paths(request.images)
                for i, image_path in enumerate(valid_images):
                    # Copy image to session temp directory
                    dest_path = self.session_temp_dir / f"image_{i}_{Path(image_path).name}"
                    shutil.copy2(image_path, dest_path)
                    image_paths.append(str(dest_path))
                    
                    # Add image path message for tools
                    messages.append({
                        "role": "user",
                        "content": f"image_path: {dest_path}"
                    })
                    
                    # Add image content for multimodal LLM
                    with open(image_path, "rb") as img_file:
                        img_base64 = self._encode_image(image_path)
                    
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                        }]
                    })
            
            # Add text message
            messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": request.text
                }]
            })
            
            # Run the agent
            response_content = ""
            for chunk in self.agent.workflow.stream(
                {"messages": messages},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="updates"
            ):
                if isinstance(chunk, dict):
                    for node_name, node_output in chunk.items():
                        if "messages" in node_output:
                            for msg in node_output["messages"]:
                                if hasattr(msg, 'content') and msg.content:
                                    response_content += str(msg.content)
            
            duration = time.time() - start_time
            
            # Clean up temporary files
            self._cleanup_temp_files()
            
            return LLMResponse(
                content=response_content.strip(),
                usage={"agent_tools": list(self.tools_dict.keys())},
                duration=duration,
                raw_response={"thread_id": thread_id, "image_paths": image_paths}
            )
            
        except Exception as e:
            self._cleanup_temp_files()
            return LLMResponse(
                content=f"Error: {str(e)}",
                duration=time.time() - start_time,
                raw_response=None
            )

    def _cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        try:
            if hasattr(self, 'session_temp_dir') and self.session_temp_dir.exists():
                shutil.rmtree(self.session_temp_dir)
        except Exception as e:
            print(f"Warning: Failed to cleanup temp files: {e}")
