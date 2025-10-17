"""MedRAX LLM provider implementation."""

import os
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
        # Set provider name
        self.provider_name = "medrax"

        self.agent = None
        self.tools_dict = None

        super().__init__(model_name, system_prompt, **kwargs)

    def _setup(self) -> None:
        """Set up MedRAX agent system."""
        try:
            print("Starting server...")

            selected_tools = [
                # "TorchXRayVisionClassifierTool",  # For classifying chest X-ray images using TorchXRayVision
                # "ArcPlusClassifierTool",  # For advanced chest X-ray classification using ArcPlus
                # "ChestXRayReportGeneratorTool",  # For generating medical reports from X-rays
                # "XRayPhraseGroundingTool",  # For locating described features in X-rays
                # "MedGemmaVQATool", # Google MedGemma VQA tool
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
                prompt_file="benchmarking/system_prompts.txt",
                tools_to_use=selected_tools,
                model_dir="/home/lijunzh3/scratch/MedRAX2/model-weights",
                temp_dir="temp",  # Change this to the path of the temporary directory
                device=os.getenv("MEDRAX_DEVICE", "cuda:0"),
                model=self.model_name,  # Change this to the model you want to use, e.g. gpt-4.1-2025-04-14, gemini-2.5-pro
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
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
                # Build multimodal content with text and images
                content = [{"type": "text", "text": request.text}]

                # Validate image paths
                valid_images = self._validate_image_paths(request.images)
                print(f"Processing {len(valid_images)} images")
                
                # Add image paths for tools
                for image_path in valid_images:
                    content.append({"type": "text", "text": f"image_path: {image_path}"})
                
                # Add image content for multimodal LLM
                for image_path in valid_images:
                    try:
                        img_base64 = self._encode_image(image_path)
                        mime_type = self._get_image_mime_type(image_path)
                        
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}
                        })
                    except Exception as e:
                        print(f"ERROR: Image encoding failed for {image_path}: {e}")
                        raise
                
                # Create single multimodal message
                messages.append(HumanMessage(content=content))
                
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
                
                # Extract response metadata (reasoning/thinking traces)
                if hasattr(msg, 'response_metadata') and msg.response_metadata:
                    try:
                        msg_info["response_metadata"] = dict(msg.response_metadata)
                        
                        # Extract specific reasoning fields for easier access
                        # Gemini 2.0 Flash Thinking uses 'thoughts'
                        if "thoughts" in msg.response_metadata:
                            msg_info["thinking"] = msg.response_metadata["thoughts"]
                        
                        # DeepSeek-R1 and similar models use 'reasoning_content'
                        if "reasoning_content" in msg.response_metadata:
                            msg_info["reasoning"] = msg.response_metadata["reasoning_content"]
                        
                        # Some models expose thinking in other fields
                        if "extended_thinking" in msg.response_metadata:
                            msg_info["extended_thinking"] = msg.response_metadata["extended_thinking"]
                    except Exception as e:
                        print(f"Warning: Could not serialize response_metadata: {e}")
                
                # Extract usage metadata (reasoning tokens for o1/o3 models)
                if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                    try:
                        msg_info["usage_metadata"] = dict(msg.usage_metadata)
                        
                        # Highlight reasoning tokens if present
                        if isinstance(msg.usage_metadata, dict) and "reasoning_tokens" in msg.usage_metadata:
                            msg_info["reasoning_tokens"] = msg.usage_metadata["reasoning_tokens"]
                    except Exception as e:
                        print(f"Warning: Could not serialize usage_metadata: {e}")
                
                # Extract additional kwargs (some models put reasoning here)
                if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                    try:
                        # Filter for reasoning-related fields
                        reasoning_kwargs = {}
                        for key in ['thinking', 'reasoning', 'thoughts', 'chain_of_thought']:
                            if key in msg.additional_kwargs:
                                reasoning_kwargs[key] = msg.additional_kwargs[key]
                        
                        if reasoning_kwargs:
                            msg_info["additional_reasoning"] = reasoning_kwargs
                        
                        # Include full additional_kwargs for completeness (may contain other useful info)
                        msg_info["additional_kwargs"] = dict(msg.additional_kwargs)
                    except Exception as e:
                        print(f"Warning: Could not serialize additional_kwargs: {e}")
                
                chunk_messages.append(msg_info)
                
                # Enhanced logging for debugging
                log_msg = f"Message in chunk: type={msg_info['type']}"
                if "thinking" in msg_info:
                    log_msg += f", has_thinking=True (length={len(str(msg_info['thinking']))})"
                if "reasoning" in msg_info:
                    log_msg += f", has_reasoning=True (length={len(str(msg_info['reasoning']))})"
                if "reasoning_tokens" in msg_info:
                    log_msg += f", reasoning_tokens={msg_info['reasoning_tokens']}"
                print(log_msg)
                
            serializable_chunk["messages"] = chunk_messages
        
        return serializable_chunk
