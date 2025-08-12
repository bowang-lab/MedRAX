"""
MedRAX Application Main Module

This module serves as the entry point for the MedRAX medical imaging AI assistant.
It provides functionality to initialize an AI agent with various medical imaging tools
and launch a web interface for interacting with the system.

The system uses OpenAI's language models for reasoning and can be configured
with different model weights, tools, and parameters.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from interface import create_demo
from medrax.agent import *
from medrax.tools import *
from medrax.utils import *

# Suppress unnecessary warnings and logging
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Load environment variables from .env file
_ = load_dotenv()


def initialize_agent(
    prompt_file: str,
    tools_to_use: Optional[List[str]] = None,
    model_dir: str = "/model-weights",
    temp_dir: str = "temp",
    device: str = "cuda",
    model: str = "gpt-4o",
    temperature: float = 0.7,
    top_p: float = 0.95,
    rag_config: Optional[RAGConfig] = None,
    openai_kwargs: Dict[str, Any] = {},
) -> Tuple[Agent, Dict[str, BaseTool]]:
    """Initialize the MedRAX agent with specified tools and configuration.

    Args:
        prompt_file (str): Path to file containing system prompts
        tools_to_use (List[str], optional): List of tool names to initialize. If None, all tools are initialized.
        model_dir (str, optional): Directory containing model weights. Defaults to "/model-weights".
        temp_dir (str, optional): Directory for temporary files. Defaults to "temp".
        device (str, optional): Device to run models on. Defaults to "cuda".
        model (str, optional): Model to use. Defaults to "chatgpt-4o-latest".
        temperature (float, optional): Temperature for the model. Defaults to 0.7.
        top_p (float, optional): Top P for the model. Defaults to 0.95.
        rag_config (RAGConfig, optional): Configuration for the RAG tool. Defaults to None.
        openai_kwargs (dict, optional): Additional keyword arguments for OpenAI API, such as API key and base URL.

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: Initialized agent and dictionary of tool instances
    """
    # Load system prompts from file
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts["MEDICAL_ASSISTANT"]

    # Define all available tools with their initialization functions
    all_tools: Dict[str, callable] = {
        "ChestXRayClassifierTool": lambda: ChestXRayClassifierTool(device=device),
        "ChestXRaySegmentationTool": lambda: ChestXRaySegmentationTool(device=device),
        "LlavaMedTool": lambda: LlavaMedTool(cache_dir=model_dir, device=device, load_in_8bit=True),
        "XRayVQATool": lambda: XRayVQATool(cache_dir=model_dir, device=device),
        "ChestXRayReportGeneratorTool": lambda: ChestXRayReportGeneratorTool(
            cache_dir=model_dir, device=device
        ),
        "XRayPhraseGroundingTool": lambda: XRayPhraseGroundingTool(
            cache_dir=model_dir, temp_dir=temp_dir, load_in_8bit=True, device=device
        ),
        "ChestXRayGeneratorTool": lambda: ChestXRayGeneratorTool(
            model_path=f"{model_dir}/roentgen", temp_dir=temp_dir, device=device
        ),
        "ImageVisualizerTool": lambda: ImageVisualizerTool(),
        "DicomProcessorTool": lambda: DicomProcessorTool(temp_dir=temp_dir),
        "MedicalRAGTool": lambda: RAGTool(config=rag_config),
    }

    # Initialize only selected tools or all if none specified
    tools_dict: Dict[str, BaseTool] = {}
    tools_to_use = tools_to_use or all_tools.keys()
    for tool_name in tools_to_use:
        if tool_name in all_tools:
            tools_dict[tool_name] = all_tools[tool_name]()

    # Set up checkpointing for conversation state
    checkpointer = MemorySaver()

    # Initialize the language model
    model = ChatOpenAI(model=model, temperature=temperature, top_p=top_p, **openai_kwargs)

    # Create the agent with the specified model, tools, and configuration
    agent = Agent(
        model,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="logs",
        system_prompt=prompt,
        checkpointer=checkpointer,
    )
    print("Agent initialized")

    return agent, tools_dict


if __name__ == "__main__":
    """
    This is the main entry point for the MedRAX application.
    It initializes the agent with the selected tools and creates the demo.
    """
    print("Starting server...")

    # Define which tools to use in the application
    # Each tool provides specific medical imaging functionality
    # You can uncomment the tools you dont want to use
    tools_to_use = [
        "ImageVisualizerTool",  # For displaying images in the UI
        "DicomProcessorTool",  # For processing DICOM medical image files
        "ChestXRayClassifierTool",  # For classifying chest X-ray images
        "ChestXRaySegmentationTool",  # For segmenting anatomical regions in chest X-rays
        "ChestXRayReportGeneratorTool",  # For generating medical reports from X-rays
        "XRayVQATool",  # For visual question answering on X-rays
        "LlavaMedTool",  # For multimodal medical image understanding
        "XRayPhraseGroundingTool",  # For locating described features in X-rays
        "ChestXRayGeneratorTool",  # For generating synthetic chest X-rays
        "MedicalRAGTool",  # For retrieval-augmented generation with medical knowledge
    ]

    # Configure the Retrieval Augmented Generation (RAG) system
    # This allows the agent to access and use medical knowledge documents
    rag_config = RAGConfig(
        model="command-a-03-2025",  # Set COHERE_API_KEY in .env
        temperature=0.7,
        persist_dir="medrax/rag/vectorDB",  # Change this to the target path of the vector database
        chunk_size=1000,
        chunk_overlap=100,
        retriever_k=3,
        local_docs_dir="medrax/rag/docs",  # Change this to the path of the documents for RAG
        use_medrag_textbooks=False,  # Set to True if you want to use the MedRAG textbooks dataset
    )

    # Prepare OpenAI API configuration from environment variables
    openai_kwargs: Dict[str, str] = {}
    if api_key := os.getenv("OPENAI_API_KEY"):
        openai_kwargs["api_key"] = api_key

    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url

    # Initialize the agent with all configured components
    agent, tools_dict = initialize_agent(
        prompt_file="medrax/docs/system_prompts.txt",  # File containing system instructions
        tools_to_use=tools_to_use,
        model_dir="/model-weights",  # Change this to the path of the model weights
        temp_dir="temp",  # Change this to the path of the temporary directory
        device="cuda",  # Change this to the device you want to use
        model="gpt-4o",  # Change this to the model you want to use, e.g. gpt-4o-mini
        temperature=0.7,
        top_p=0.95,
        rag_config=rag_config,
        openai_kwargs=openai_kwargs,
    )

    # Create and launch the web interface
    demo = create_demo(agent, tools_dict)
    demo.launch(server_name="0.0.0.0", server_port=8585, share=True)
