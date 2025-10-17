"""
MedRAX Application Main Module

This module serves as the entry point for the MedRAX medical imaging AI assistant.
It provides functionality to initialize an AI agent with various medical imaging tools
and launch a web interface for interacting with the system.

The system uses OpenAI's language models for reasoning and can be configured
with different model weights, tools, and parameters.
"""

import warnings
import os
import argparse
from pyngrok import ngrok
import threading
import uvicorn
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from medrax.models import ModelFactory

from interface import create_demo
from api import create_api
from medrax.agent import *
from medrax.tools import *
from medrax.utils import *

# Suppress unnecessary warnings and logging
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Load environment variables from .env file
_ = load_dotenv()


def resolve_medgemma_api_url_from_value(value: Optional[str]) -> str:
    """Resolve the MedGemma API base URL using CLI value, env var, and SLURM-aware fallback.

    Resolution order:
    1) Explicit provided value (e.g., CLI flag)
    2) MEDGEMMA_API_URL environment variable
    3) If on SLURM, require explicit URL (raise)
    4) Otherwise, default to localhost for single-box setups
    """
    if value:
        return value

    env_url = os.getenv("MEDGEMMA_API_URL")
    if env_url:
        return env_url

    if os.getenv("SLURM_JOB_ID") or os.getenv("SLURM_NODEID"):
        raise RuntimeError(
            "MEDGEMMA_API_URL not set and --medgemma-api-url not provided. "
            "On SLURM, the client usually runs on a different node, "
            "so you must point to the server’s reachable IP, e.g. http://<node-ip>:8002"
        )

    return "http://127.0.0.1:8002"


def resolve_medgemma_api_url(args) -> str:
    """Helper that reads from an argparse Namespace if available."""
    return resolve_medgemma_api_url_from_value(getattr(args, "medgemma_api_url", None))


def initialize_agent(
    prompt_file: str,
    tools_to_use: Optional[List[str]] = None,
    model_dir: str = "/model-weights",
    temp_dir: str = "temp",
    device: str = "cuda",
    model: str = "gpt-4.1",
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 5000,
    rag_config: Optional[RAGConfig] = None,
    model_kwargs: Dict[str, Any] = {},
    system_prompt: str = "MEDICAL_ASSISTANT",
    medgemma_api_url: Optional[str] = None,
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
        rag_config (RAGConfig, optional): Configuration for the RAG tool. Defaults to None.
        model_kwargs (dict, optional): Additional keyword arguments for model.
        system_prompt (str, optional): System prompt to use. Defaults to "MEDICAL_ASSISTANT".
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: Initialized agent and dictionary of tool instances
    """
    # Load system prompts from file
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts[system_prompt]

    all_tools = {
        "TorchXRayVisionClassifierTool": lambda: TorchXRayVisionClassifierTool(device=device),
        "ArcPlusClassifierTool": lambda: ArcPlusClassifierTool(cache_dir=model_dir, device=device),
        "ChestXRaySegmentationTool": lambda: ChestXRaySegmentationTool(device=device),
        "LlavaMedTool": lambda: LlavaMedTool(cache_dir=model_dir, device=device, load_in_8bit=True),
        "CheXagentXRayVQATool": lambda: CheXagentXRayVQATool(cache_dir=model_dir, device=device),
        "ChestXRayReportGeneratorTool": lambda: ChestXRayReportGeneratorTool(cache_dir=model_dir, device=device),
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
        "DuckDuckGoSearchTool": lambda: DuckDuckGoSearchTool(),
        "MedSAM2Tool": lambda: MedSAM2Tool(device=device, cache_dir=model_dir, temp_dir=temp_dir),
        "MedGemmaVQATool": lambda: MedGemmaAPIClientTool(
            cache_dir=model_dir,
            device=device,
            load_in_8bit=True,
            api_url=resolve_medgemma_api_url_from_value(medgemma_api_url),
        ),
    }

    # Initialize only selected tools or all if none specified
    tools_dict: Dict[str, BaseTool] = {}

    if tools_to_use is None:
        tools_to_use = []

    for tool_name in tools_to_use:
        if tool_name == "PythonSandboxTool":
            try:
                tools_dict["PythonSandboxTool"] = create_python_sandbox()
            except Exception as e:
                print(f"Error creating PythonSandboxTool: {e}")
                print("Skipping PythonSandboxTool")
        if tool_name in all_tools:
            tools_dict[tool_name] = all_tools[tool_name]()

    # Set up checkpointing for conversation state
    checkpointer = MemorySaver()

    # Create the language model using the factory
    try:
        llm = ModelFactory.create_model(
            model_name=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens, **model_kwargs
        )
    except ValueError as e:
        print(f"Error creating language model: {e}")
        print(f"Available model providers: {list(ModelFactory._model_providers.keys())}")
        raise

    agent = Agent(
        llm,
        tools=list(tools_dict.values()),
        system_prompt=prompt,
        checkpointer=checkpointer,
    )
    print("Agent initialized")

    return agent, tools_dict


def run_gradio_interface(agent, tools_dict, host="0.0.0.0", port=8686, 
                        auth=None, share=False):
    """
    Run the Gradio web interface.

    Args:
        agent: The initialized MedRAX agent
        tools_dict: Dictionary of available tools
        host (str): Host to bind the server to
        port (int): Port to run the server on
        auth: Authentication credentials (tuple)
        share (bool): Whether to create a shareable public link
    """
    print(f"Starting Gradio interface on {host}:{port}")
    
    if auth:
        print(f"🔐 Authentication enabled for user: {auth[0]}")
    else:
        print("⚠️  Running without authentication (public access)")
    
    if share:
        print("🌍 Creating shareable public link (expires in 1 week)...")
    
    demo = create_demo(agent, tools_dict)
    
    # Prepare launch parameters
    launch_kwargs = {
        "server_name": host,
        "server_port": port,
        "share": share
    }
    
    if auth:
        launch_kwargs["auth"] = auth
        
    demo.launch(**launch_kwargs)


def run_api_server(agent, tools_dict, host="0.0.0.0", port=8585, public=False):
    """
    Run the FastAPI server.

    Args:
        agent: The initialized MedRAX agent
        tools_dict: Dictionary of available tools
        host (str): Host to bind the server to
        port (int): Port to run the server on
        public (bool): Whether to expose via ngrok tunnel
    """
    print(f"Starting API server on {host}:{port}")

    if public:
        try:
            public_tunnel = ngrok.connect(port)
            public_url = public_tunnel.public_url
            print(
                f"🌍 Public URL: {public_url}\n🌍 API Documentation: {public_url}/docs\n🌍 Share this URL with your friend!\n{'=' * 60}"
            )
        except ImportError:
            print("⚠️  pyngrok not installed. Install with: pip install pyngrok\nRunning locally only...")
            public = False
        except Exception as e:
            print(f"⚠️  Failed to create public tunnel: {e}\nRunning locally only...")
            public = False

    app = create_api(agent, tools_dict)

    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        if public:
            try:
                ngrok.disconnect(public_tunnel.public_url)
                ngrok.kill()
            except:
                pass


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MedRAX - Medical Reasoning Agent for Chest X-ray")

    # Run mode
    parser.add_argument(
        "--mode",
        choices=["gradio", "api", "both"],
        default="gradio",
        help="Run mode: 'gradio' for web interface, 'api' for REST API, 'both' for both services",
    )
    
    # Gradio interface options
    parser.add_argument("--gradio-host", default="0.0.0.0", help="Gradio host address")
    parser.add_argument("--gradio-port", type=int, default=8686, help="Gradio port")
    parser.add_argument("--auth", nargs=2, metavar=("USERNAME", "PASSWORD"), 
                       default=["admin", "adibjun"],
                       help="Enable password authentication (default: admin adibjun)")
    parser.add_argument("--no-auth", action="store_true", 
                       help="Disable authentication (public access)")
    parser.add_argument("--share", action="store_true", 
                       help="Create a temporary shareable link (expires in 1 week)")
    
    # API server options
    parser.add_argument("--api-host", default="0.0.0.0", help="API host address")
    parser.add_argument("--api-port", type=int, default=8000, help="API port")
    parser.add_argument("--public", action="store_true", help="Make API publicly accessible via ngrok tunnel")

    # Model and system configuration
    parser.add_argument(
        "--model-dir",
        default="/model-weights",
        help="Directory containing model weights (default: uses MODEL_WEIGHTS_DIR env var or '/model-weights')",
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to run models on (default: uses MEDRAX_DEVICE env var or 'cuda:1')"
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1",
        help="Model to use (default: gpt-4.1). Examples: gpt-4.1-2025-04-14, gemini-2.5-pro, gpt-5",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the model (default: 1.0)")
    parser.add_argument("--temp-dir", default="temp2", help="Directory for temporary files (default: temp2)")
    parser.add_argument(
        "--prompt-file",
        default="medrax/docs/system_prompts.txt",
        help="Path to file containing system prompts (default: medrax/docs/system_prompts.txt)",
    )
    parser.add_argument(
        "--system-prompt", default="MEDICAL_ASSISTANT", help="System prompt to use (default: MEDICAL_ASSISTANT)"
    )

    # RAG configuration
    parser.add_argument(
        "--rag-model", default="command-a-03-2025", help="Chat model for RAG responses (default: command-a-03-2025)"
    )
    parser.add_argument(
        "--rag-embedding-model", default="embed-v4.0", help="Embedding model for RAG system (default: embed-v4.0)"
    )
    parser.add_argument(
        "--rag-rerank-model", default="rerank-v3.5", help="Reranking model for RAG system (default: rerank-v3.5)"
    )
    parser.add_argument("--rag-temperature", type=float, default=0.3, help="Temperature for RAG model (default: 0.3)")
    parser.add_argument("--pinecone-index", default="medrax2", help="Pinecone index name (default: medrax2)")
    parser.add_argument("--chunk-size", type=int, default=1500, help="RAG chunk size (default: 1500)")
    parser.add_argument("--chunk-overlap", type=int, default=300, help="RAG chunk overlap (default: 300)")
    parser.add_argument("--retriever-k", type=int, default=3, help="Number of documents to retrieve (default: 3)")
    parser.add_argument("--rag-docs-dir", default="rag_docs", help="Directory for RAG documents (default: rag_docs)")

    # Tools configuration
    parser.add_argument(
        "--tools",
        nargs="*",
        help="Specific tools to enable (if not provided, uses default set). Available tools: "
        + "ImageVisualizerTool, DicomProcessorTool, MedSAM2Tool, ChestXRaySegmentationTool, "
        + "ChestXRayGeneratorTool, TorchXRayVisionClassifierTool, ArcPlusClassifierTool, "
        + "ChestXRayReportGeneratorTool, XRayPhraseGroundingTool, MedGemmaVQATool, "
        + "XRayVQATool, LlavaMedTool, MedicalRAGTool, WebBrowserTool, DuckDuckGoSearchTool, "
        + "PythonSandboxTool",
    )

    # MedGemma API configuration
    parser.add_argument(
        "--medgemma-api-url",
        default=None,
        help="MedGemma API base URL, e.g. http://127.0.0.1:8002 or http://<node-ip>:8002"
    )

    return parser.parse_args()


if __name__ == "__main__":
    """
    This is the main entry point for the MedRAX application.
    It initializes the agent with the selected tools and creates the demo/API.
    """
    args = parse_arguments()
    print(f"Starting MedRAX in {args.mode} mode...")

    # Configure tools based on arguments
    if args.tools is not None:
        # Use tools specified via command line
        selected_tools = args.tools
    else:
        # Use default tools selection
        selected_tools = [
            # Image Processing Tools
            "ImageVisualizerTool",  # For displaying images in the UI
            # "DicomProcessorTool",  # For processing DICOM medical image files
            # Segmentation Tools
            "MedSAM2Tool",  # For advanced medical image segmentation using MedSAM2
            "ChestXRaySegmentationTool",  # For segmenting anatomical regions in chest X-rays
            # Generation Tools
            # "ChestXRayGeneratorTool",  # For generating synthetic chest X-rays
            # Classification Tools
            "TorchXRayVisionClassifierTool",  # For classifying chest X-ray images using TorchXRayVision
            "ArcPlusClassifierTool",  # For advanced chest X-ray classification using ArcPlus
            # Report Generation Tools
            "ChestXRayReportGeneratorTool",  # For generating medical reports from X-rays
            # Grounding Tools
            "XRayPhraseGroundingTool",  # For locating described features in X-rays
            # VQA Tools
            # "MedGemmaVQATool",  # Google MedGemma VQA tool
            "XRayVQATool",  # For visual question answering on X-rays
            # "LlavaMedTool",  # For multimodal medical image understanding
            # RAG Tools
            "MedicalRAGTool",  # For retrieval-augmented generation with medical knowledge
            # Search Tools
            # "WebBrowserTool",  # For web browsing and search capabilities
            "DuckDuckGoSearchTool",  # For privacy-focused web search using DuckDuckGo
            # Development Tools
            # "PythonSandboxTool",  # Add the Python sandbox tool
        ]

    # Configure model directory and device
    model_dir = args.model_dir or os.getenv("MODEL_WEIGHTS_DIR", "/model-weights")
    device = args.device or os.getenv("MEDRAX_DEVICE", "cuda:0")

    print(f"Using model directory: {model_dir}")
    print(f"Using device: {device}")
    print(f"Using model: {args.model}")
    print(f"Selected tools: {selected_tools}")
    print(f"Using system prompt: {args.system_prompt}")
    
    # Set up authentication (simplified with argparse defaults)
    if args.no_auth:
        auth_credentials = None
        print("⚠️  Authentication disabled (public access)")
    else:
        auth_credentials = tuple(args.auth)  # Uses default ["admin", "adibjun"] if not specified
        print(f"✅ Authentication enabled for user: {auth_credentials[0]}")

    # Setup the MedGemma environment if the MedGemmaVQATool is selected
    medgemma_base_url_from_setup: Optional[str] = None
    medgemma_api_url_effective: Optional[str] = args.medgemma_api_url
    if "MedGemmaVQATool" in selected_tools:
        # Launch server and capture its URL if no explicit URL/ENV provided
        try:
            if medgemma_api_url_effective is None and os.getenv("MEDGEMMA_API_URL") is None:
                medgemma_base_url_from_setup = setup_medgemma_env(cache_dir=model_dir, device=device)
                # If we auto-launched, use this URL unless overridden later
                if medgemma_base_url_from_setup:
                    medgemma_api_url_effective = medgemma_base_url_from_setup
                    print(f"MedGemma API auto-launched at {medgemma_api_url_effective}")
            else:
                # Still ensure environment is set up; it will bind to provided host/port
                setup_medgemma_env(cache_dir=model_dir, device=device)
        except Exception as e:
            print(f"Warning: Failed to launch MedGemma service automatically: {e}")

    # Configure the Retrieval Augmented Generation (RAG) system
    # This allows the agent to access and use medical knowledge documents
    rag_config = RAGConfig(
        model=args.rag_model,
        embedding_model=args.rag_embedding_model,
        rerank_model=args.rag_rerank_model,
        temperature=args.rag_temperature,
        pinecone_index_name=args.pinecone_index,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        retriever_k=args.retriever_k,
        local_docs_dir=args.rag_docs_dir,
        huggingface_datasets=["VictorLJZ/medrax2"],  # List of HuggingFace datasets to load
        dataset_split="train",  # Which split of the datasets to use
    )

    # Prepare any additional model-specific kwargs
    model_kwargs = {}

    agent, tools_dict = initialize_agent(
        prompt_file=args.prompt_file,
        tools_to_use=selected_tools,
        model_dir=model_dir,
        temp_dir=args.temp_dir,
        device=device,
        model=args.model,
        temperature=args.temperature,
        model_kwargs=model_kwargs,
        rag_config=rag_config,
        system_prompt=args.system_prompt,
        medgemma_api_url=medgemma_api_url_effective,
    )

    # Launch based on selected mode
    if args.mode == "gradio":
        run_gradio_interface(
            agent, tools_dict, 
            host=args.gradio_host, 
            port=args.gradio_port,
            auth=auth_credentials,
            share=args.share
        )

    elif args.mode == "api":
        run_api_server(agent, tools_dict, args.api_host, args.api_port, args.public)

    elif args.mode == "both":
        # Run both services in separate threads
        api_thread = threading.Thread(
            target=run_api_server, 
            args=(agent, tools_dict, args.api_host, args.api_port, args.public)
        )
        api_thread.daemon = True
        api_thread.start()

        # Run Gradio in main thread with authentication and sharing
        run_gradio_interface(
            agent, tools_dict, 
            host=args.gradio_host, 
            port=args.gradio_port,
            auth=auth_credentials,
            share=args.share
        )
