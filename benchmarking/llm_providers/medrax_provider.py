"""MedRAX LLM provider implementation."""

import time
import re

from .base import LLMProvider, LLMRequest, LLMResponse
from langchain_core.messages import AIMessage, HumanMessage

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
                "TorchXRayVisionClassifierTool",  # For classifying chest X-ray images using TorchXRayVision
                "ArcPlusClassifierTool",  # For advanced chest X-ray classification using ArcPlus
                "ChestXRayReportGeneratorTool",  # For generating medical reports from X-rays
                "XRayPhraseGroundingTool",  # For locating described features in X-rays
                "MedGemmaVQATool",
                # "XRayVQATool",  # For visual question answering on X-rays
                # "MedicalRAGTool",  # For retrieval-augmented generation with medical knowledge
                # "WebBrowserTool",  # For web browsing and search capabilities
                # "DuckDuckGoSearchTool",  # For privacy-focused web search using DuckDuckGo
            ]

            rag_config = RAGConfig(
                model="command-a-03-2025",  # Chat model for generating responses
                embedding_model="embed-v4.0",  # Embedding model for the RAG system
                rerank_model="rerank-v3.5",  # Reranking model for the RAG system
                temperature=0.3,
                pinecone_index_name="medrax2",  # Name for the Pinecone index
                chunk_size=1500,
                chunk_overlap=300,
                retriever_k=3,
                local_docs_dir="rag_docs",  # Change this to the path of the documents for RAG
                huggingface_datasets=["VictorLJZ/medrax2"],  # List of HuggingFace datasets to load
                dataset_split="train",  # Which split of the datasets to use
            )

            # Prepare any additional model-specific kwargs
            model_kwargs = {}

            agent, tools_dict = initialize_agent(
                prompt_file="medrax/docs/system_prompts.txt",
                tools_to_use=selected_tools,
                model_dir="/scratch/ssd004/scratch/victorli/model-weights",
                temp_dir="temp",  # Change this to the path of the temporary directory
                device="cuda:0",
                model=self.model_name,  # Change this to the model you want to use, e.g. gpt-4.1-2025-04-14, gemini-2.5-pro
                temperature=1.0,
                top_p=0.95,
                model_kwargs=model_kwargs,
                rag_config=rag_config,
                system_prompt=self.prompt_name,
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
                duration=time.time() - start_time
            )
        
        try:
            # Build messages for the agent
            messages = []
            thread_id = str(int(time.time() * 1000))  # Unique thread ID
            
            if request.images:
                valid_images = self._validate_image_paths(request.images)
                print(f"Processing {len(valid_images)} images")
                for i, image_path in enumerate(valid_images):
                    # Add image path message for tools
                    messages.append(HumanMessage(content=f"image_path: {image_path}"))
                    
                    # Add image content for multimodal LLM
                    try:
                        with open(image_path, "rb") as img_file:
                            img_base64 = self._encode_image(image_path)
                        
                        messages.append(HumanMessage(content=[{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                        }]))
                    except Exception as e:
                        print(f"ERROR: Image encoding failed for {image_path}: {e}")
                        raise
            
            # Add text message
            if request.images:
                # If there are images, add text as part of multimodal content
                messages.append(HumanMessage(content=[{
                    "type": "text",
                    "text": request.text
                }]))
            else:
                # If no images, add text as simple string
                messages.append(HumanMessage(content=request.text))
            
            # Run the agent with proper message type handling
            final_response = ""
            chunk_history = []
            
            for chunk in self.agent.workflow.stream(
                {"messages": messages},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="updates"
            ):
                if not isinstance(chunk, dict):
                    continue
                    
                for node_name, node_output in chunk.items():
                    # Log chunk and get serializable version
                    serializable_chunk = self._log_chunk(node_output, node_name)
                    chunk_history.append(serializable_chunk)

                    if "messages" not in node_output:
                        continue
                        
                    for msg in node_output["messages"]:
                        if isinstance(msg, AIMessage) and msg.content:
                            # Handle case where content is a list
                            content = msg.content
                            if isinstance(content, list):
                                content = " ".join(content)
                            # Clean up the content (remove temp paths, etc.)
                            final_response = re.sub(r"temp/[^\s]*", "", content).strip()
            
            # Determine the final response
            if final_response:
                response_content = final_response
            else:
                # Fallback if no LLM response was received
                response_content = "No response generated"
            
            duration = time.time() - start_time
            
            return LLMResponse(
                content=response_content,
                usage={"agent_tools": list(self.tools_dict.keys())},
                duration=duration,
                chunk_history=chunk_history
            )
            
        except Exception as e:
            print(f"ERROR: MedRAX agent failed: {e}")
            return LLMResponse(
                content=f"Error: {str(e)}",
                duration=time.time() - start_time
            )

    def _log_chunk(self, chunk: dict, node_name: str) -> dict:
        """Log and process a chunk from the agent workflow.
        
        Args:
            chunk (dict): The chunk data from the agent workflow
            node_name (str): Name of the node that produced the chunk
            
        Returns:
            dict: Serializable version of the chunk for debugging
        """
        # Log every chunk for debugging
        print(f"Chunk from node '{node_name}': {type(chunk)}")
        
        # Store serializable version of chunk for debugging
        serializable_chunk = {
            "node_name": node_name,
            "node_type": type(chunk).__name__,
        }
        
        # Log messages in this chunk
        if "messages" in chunk and isinstance(chunk, dict):
            chunk_messages = []
            for msg in chunk["messages"]:
                msg_info = {
                    "type": type(msg).__name__,
                    "content": str(msg.content) if hasattr(msg, 'content') else str(msg)
                }
                chunk_messages.append(msg_info)
                print(f"Message in chunk: {msg_info}")
            serializable_chunk["messages"] = chunk_messages
        
        return serializable_chunk
