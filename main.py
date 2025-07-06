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
from medrax.models import ModelFactory

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
    device: str = "cpu",
    model: str = "gpt-4.1-2025-04-14",
    temperature: float = 0.7,
    top_p: float = 0.95,
    rag_config: Optional[RAGConfig] = None,
    model_kwargs: Dict[str, Any] = {},
):
    """Initialize the MedRAX agent with specified tools and configuration.

    Args:
        prompt_file (str): Path to file containing system prompts
        tools_to_use (List[str], optional): List of tool names to initialize. If None, all tools are initialized.
        model_dir (str, optional): Directory containing model weights. Defaults to "/model-weights".
        temp_dir (str, optional): Directory for temporary files. Defaults to "temp".
        device (str, optional): Device to run models on. Defaults to "cuda".
        model (str, optional): Model to use. Defaults to "gpt-4o".
        temperature (float, optional): Temperature for the model. Defaults to 0.7.
        top_p (float, optional): Top P for the model. Defaults to 0.95.
        rag_config (RAGConfig, optional): Configuration for the RAG tool. Defaults to None.
        model_kwargs (dict, optional): Additional keyword arguments for model.

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: Initialized agent and dictionary of tool instances
    """
    # Load system prompts from file
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts["MEDICAL_ASSISTANT"]

    all_tools = {
        "TorchXRayVisionClassifierTool": lambda: TorchXRayVisionClassifierTool(device=device),
        "ArcPlusClassifierTool": lambda: ArcPlusClassifierTool(cache_dir=model_dir, device=device),
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
        "WebBrowserTool": lambda: WebBrowserTool(),
    }

    try:
        tools_dict["PythonSandboxTool"] = create_python_sandbox()
    except Exception as e:
        print(f"Error creating PythonSandboxTool: {e}")
        print("Skipping PythonSandboxTool")

    # Initialize only selected tools or all if none specified
    tools_dict: Dict[str, BaseTool] = {}
    tools_to_use = tools_to_use or all_tools.keys()
    for tool_name in tools_to_use:
        if tool_name in all_tools:
            tools_dict[tool_name] = all_tools[tool_name]()

    # Set up checkpointing for conversation state
    checkpointer = MemorySaver()

    # Create the language model using the factory
    try:
        llm = ModelFactory.create_model(
            model_name=model, temperature=temperature, top_p=top_p, **model_kwargs
        )
    except ValueError as e:
        print(f"Error creating language model: {e}")
        print(f"Available model providers: {list(ModelFactory._model_providers.keys())}")
        raise

    agent = Agent(
        llm,
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

    # Example: initialize with only specific tools
    # Here three tools are commented out, you can uncomment them to use them
    selected_tools = [
        # "ImageVisualizerTool",  # For displaying images in the UI
        # "DicomProcessorTool",  # For processing DICOM medical image files
        # "TorchXRayVisionClassifierTool",  # For classifying chest X-ray images using TorchXRayVision
        # "ArcPlusClassifierTool",  # For advanced chest X-ray classification using ArcPlus
        # "ChestXRaySegmentationTool",  # For segmenting anatomical regions in chest X-rays
        # "ChestXRayReportGeneratorTool",  # For generating medical reports from X-rays
        # "XRayVQATool",  # For visual question answering on X-rays
        # "LlavaMedTool",  # For multimodal medical image understanding
        # "XRayPhraseGroundingTool",  # For locating described features in X-rays
        # "ChestXRayGeneratorTool",  # For generating synthetic chest X-rays
        "WebBrowserTool",  # For web browsing and search capabilities
        "MedicalRAGTool",  # For retrieval-augmented generation with medical knowledge
        "PythonSandboxTool",  # Add the Python sandbox tool
    ]

    # Configure the Retrieval Augmented Generation (RAG) system
    # This allows the agent to access and use medical knowledge documents
    rag_config = RAGConfig(
        model="command-a-03-2025",  # Chat model for generating responses
        embedding_model="embed-v4.0",  # Embedding model for the RAG system
        rerank_model="rerank-v3.5",  # Reranking model for the RAG system
        temperature=0.3,
        pinecone_index_name="medrax2",  # Name for the Pinecone index
        chunk_size=1500,
        chunk_overlap=300,
        retriever_k=7,
        local_docs_dir="rag_docs",  # Change this to the path of the documents for RAG
        huggingface_datasets=["VictorLJZ/medrax2"],  # List of HuggingFace datasets to load
        dataset_split="train",  # Which split of the datasets to use
    )

    # Prepare any additional model-specific kwargs
    model_kwargs = {}

    # Set up API keys for the web browser tool
    # You'll need to set these environment variables:
    # - GOOGLE_SEARCH_API_KEY: Your Google Custom Search API key
    # - GOOGLE_SEARCH_ENGINE_ID: Your Google Custom Search Engine ID
    # - COHERE_API_KEY: Your Cohere API key
    # - OPENAI_API_KEY: Your OpenAI API key
    # - PINECONE_API_KEY: Your Pinecone API key

    agent, tools_dict = initialize_agent(
        prompt_file="medrax/docs/system_prompts.txt",
        tools_to_use=selected_tools,
        model_dir="/model-weights",
        temp_dir="temp",  # Change this to the path of the temporary directory
        device="cuda",
        model="gpt-4.1-2025-04-14",  # Change this to the model you want to use, e.g. gpt-4.1-2025-04-14, gemini-2.5-pro
        temperature=0.7,
        top_p=0.95,
        model_kwargs=model_kwargs,
        rag_config=rag_config,
    )

    # Create and launch the web interface
    demo = create_demo(agent, tools_dict)
    demo.launch(server_name="0.0.0.0", server_port=8585, share=True)
