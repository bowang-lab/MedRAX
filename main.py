import os
import warnings
from typing import *
from dotenv import load_dotenv
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from medrax.models import ModelFactory

from interface import create_demo
from medrax.agent import *
from medrax.tools import *
from medrax.utils import *

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
_ = load_dotenv()


def initialize_agent(
    prompt_file,
    tools_to_use=None,
    model_dir="/model-weights",
    temp_dir="temp",
    device="cuda",
    model="gpt-4.1-2025-04-14",
    temperature=0.7,
    top_p=0.95,
    model_kwargs={}
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
        model_kwargs (dict, optional): Additional keyword arguments for model.

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: Initialized agent and dictionary of tool instances
    """
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts["MEDICAL_ASSISTANT"]

    all_tools = {
        "TorchXRayVisionClassifierTool": lambda: TorchXRayVisionClassifierTool(device=device),
        "ArcPlusClassifierTool": lambda: ArcPlusClassifierTool(
            cache_dir=model_dir,
            device=device
        ),
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
        "WebBrowserTool": lambda: WebBrowserTool(),
        "PythonSandboxTool": lambda: create_python_sandbox(),
    }

    # Initialize only selected tools or all if none specified
    tools_dict = {}
    tools_to_use = tools_to_use or all_tools.keys()
    for tool_name in tools_to_use:
        if tool_name in all_tools:
            tools_dict[tool_name] = all_tools[tool_name]()

    checkpointer = MemorySaver()
    
    # Create the language model using the factory
    try:
        llm = ModelFactory.create_model(
            model_name=model,
            temperature=temperature,
            top_p=top_p,
            **model_kwargs
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
        # "ImageVisualizerTool",
        # "DicomProcessorTool",
        # "TorchXRayVisionClassifierTool",
        # "ArcPlusClassifierTool",
        # "ChestXRaySegmentationTool",
        # "ChestXRayReportGeneratorTool",
        # "XRayVQATool",
        "WebBrowserTool",  # Add the web browser tool
        "PythonSandboxTool",  # Add the Python sandbox tool
        # "LlavaMedTool",
        # "XRayPhraseGroundingTool",
        # "ChestXRayGeneratorTool",
    ]

    # Prepare any additional model-specific kwargs
    model_kwargs = {}
    
    # Set up API keys for the web browser tool
    # You'll need to set these environment variables:
    # - GOOGLE_SEARCH_API_KEY: Your Google Custom Search API key
    # - GOOGLE_SEARCH_ENGINE_ID: Your Google Custom Search Engine ID
    
    agent, tools_dict = initialize_agent(
        "medrax/docs/system_prompts.txt",
        tools_to_use=selected_tools,
        model_dir="/model-weights",
        temp_dir="temp",  # Change this to the path of the temporary directory
        device="cuda",
        model="gpt-4.1-2025-04-14",  # Change this to the model you want to use, e.g. gpt-4.1-2025-04-14, gemini-2.5-pro
        temperature=0.7,
        top_p=0.95,
        model_kwargs=model_kwargs
    )
    demo = create_demo(agent, tools_dict)

    demo.launch(server_name="0.0.0.0", server_port=8585, share=True)
