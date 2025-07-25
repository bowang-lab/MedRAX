"""MedRAX LLM provider implementation."""

import time
import shutil
from pathlib import Path

from .base import LLMProvider, LLMRequest, LLMResponse

from medrax.rag.rag import RAGConfig
from main import initialize_agent


class MedRAXProvider(LLMProvider):
    """MedRAX LLM provider that uses the full MedRAX agent system."""

    def __init__(self, model_name: str, system_prompt: str, **kwargs):
        """Initialize MedRAX provider.
        
        Args:
            model_name (str): Base LLM model name (e.g., "gpt-4.1-2025-04-14")
            system_prompt (str): System prompt to use
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.agent = None
        self.tools_dict = None

        super().__init__(model_name, system_prompt, **kwargs)

    def _setup(self) -> None:
        """Set up MedRAX agent system."""
        try:
            print("Starting server...")

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
                # "PythonSandboxTool",  # Add the Python sandbox tool
            ]

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

            agent, tools_dict = initialize_agent(
                prompt_file="medrax/docs/system_prompts.txt",
                tools_to_use=selected_tools,
                model_dir="/model-weights",
                temp_dir="temp",  # Change this to the path of the temporary directory
                device="cpu",
                model=self.model_name,  # Change this to the model you want to use, e.g. gpt-4.1-2025-04-14, gemini-2.5-pro
                temperature=0.7,
                top_p=0.95,
                model_kwargs=model_kwargs,
                rag_config=rag_config,
                system_prompt=self.prompt_name,
                debug=True,
            )
            
            self.agent = agent
            self.tools_dict = tools_dict
            
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
                print(f"Processing {len(valid_images)} images")
                for i, image_path in enumerate(valid_images):
                    print(f"Original image path: {image_path}")
                    # Copy image to session temp directory
                    dest_path = Path("temp") / f"image_{i}_{Path(image_path).name}"
                    print(f"Destination path: {dest_path}")
                    shutil.copy2(image_path, dest_path)
                    image_paths.append(str(dest_path))
                    
                    # Verify file exists after copy
                    if not dest_path.exists():
                        print(f"ERROR: File not found after copy: {dest_path}")
                    else:
                        print(f"File successfully copied: {dest_path}")
                    
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
            
            return LLMResponse(
                content=response_content.strip(),
                usage={"agent_tools": list(self.tools_dict.keys())},
                duration=duration,
                raw_response={"thread_id": thread_id, "image_paths": image_paths}
            )
            
        except Exception as e:
            return LLMResponse(
                content=f"Error: {str(e)}",
                duration=time.time() - start_time,
                raw_response=None
            )
