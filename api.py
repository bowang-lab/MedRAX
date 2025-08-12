"""
MedRAX API Module

This module provides a FastAPI-based REST API for the MedRAX medical imaging AI assistant.
It offers endpoints for processing medical images with text queries using the same agent
architecture as the Gradio interface.

The API supports:
- Text-only queries
- Single or multiple image inputs
- Optional custom system prompts
- Automatic thread management for each request
- Tool execution and result aggregation
"""

import uuid
import base64
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from io import BytesIO

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, ToolMessage

# Import MedRAX components
from medrax.agent import Agent


class QueryRequest(BaseModel):
    """
    Request model for text-only queries.
    
    Attributes:
        question (str): The question or query to ask the agent
        system_prompt (Optional[str]): Custom system prompt to override default
        thread_id (Optional[str]): Optional thread ID for conversation continuity
    """
    question: str = Field(..., description="The question or query to ask the agent")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt to override default")
    thread_id: Optional[str] = Field(None, description="Optional thread ID for conversation continuity")


class QueryResponse(BaseModel):
    """
    Response model for API queries.
    
    Attributes:
        response (str): The agent's text response
        thread_id (str): The thread ID used for this conversation
        tools_used (List[str]): List of tools that were executed
        processing_time (float): Time taken to process the request in seconds
    """
    response: str = Field(..., description="The agent's text response")
    thread_id: str = Field(..., description="The thread ID used for this conversation")
    tools_used: List[str] = Field(..., description="List of tools that were executed")
    processing_time: float = Field(..., description="Time taken to process the request in seconds")


class MedRAXAPI:
    """
    FastAPI application wrapper for the MedRAX agent.
    
    This class provides a clean interface for creating and managing the API endpoints
    while maintaining separation of concerns from the core agent functionality.
    """
    
    def __init__(self, agent: Agent, tools_dict: Dict[str, Any], temp_dir: str = "temp_api"):
        """
        Initialize the MedRAX API.
        
        Args:
            agent (Agent): The initialized MedRAX agent
            tools_dict (Dict[str, Any]): Dictionary of available tools
            temp_dir (str): Directory for temporary file storage
        """
        self.agent = agent
        self.tools_dict = tools_dict
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="MedRAX API",
            description="Medical Reasoning Agent for Chest X-ray Analysis",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register all API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "service": "MedRAX API"}
        
        @self.app.get("/tools")
        async def list_tools():
            """List available tools."""
            return {
                "available_tools": list(self.tools_dict.keys()),
                "total_count": len(self.tools_dict)
            }
        
        @self.app.post("/query", response_model=QueryResponse)
        async def query_text_only(request: QueryRequest):
            """
            Process a text-only query without images.
            
            Args:
                request (QueryRequest): The query request
                
            Returns:
                QueryResponse: The agent's response
            """
            return await self._process_query(
                question=request.question,
                system_prompt=request.system_prompt,
                thread_id=request.thread_id,
                images=None
            )
        
        @self.app.post("/query-with-images", response_model=QueryResponse)
        async def query_with_images(
            question: str = Form(..., description="The question or query to ask the agent"),
            system_prompt: Optional[str] = Form(None, description="Custom system prompt to override default"),
            thread_id: Optional[str] = Form(None, description="Optional thread ID for conversation continuity"),
            images: List[UploadFile] = File(..., description="One or more medical images to analyze")
        ):
            """
            Process a query with one or more images.
            
            Args:
                question (str): The question or query to ask the agent
                system_prompt (Optional[str]): Custom system prompt to override default
                thread_id (Optional[str]): Optional thread ID for conversation continuity
                images (List[UploadFile]): List of uploaded image files
                
            Returns:
                QueryResponse: The agent's response
            """
            # Validate image files
            if not images or len(images) == 0:
                raise HTTPException(status_code=400, detail="At least one image is required")
            
            # Validate file types
            allowed_types = {'image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff', 'application/dicom'}
            for image in images:
                if image.content_type not in allowed_types:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Unsupported file type: {image.content_type}. Allowed types: {allowed_types}"
                    )
            
            return await self._process_query(
                question=question,
                system_prompt=system_prompt,
                thread_id=thread_id,
                images=images
            )
    
    async def _process_query(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        thread_id: Optional[str] = None,
        images: Optional[List[UploadFile]] = None
    ) -> QueryResponse:
        """
        Internal method to process queries through the agent.
        
        Args:
            question (str): The question to ask
            system_prompt (Optional[str]): Custom system prompt
            thread_id (Optional[str]): Thread ID for conversation
            images (Optional[List[UploadFile]]): List of images
            
        Returns:
            QueryResponse: The processed response
        """
        import time
        start_time = time.time()
        
        # Generate thread ID if not provided
        if not thread_id:
            thread_id = str(uuid.uuid4())
        
        try:
            # Prepare messages
            messages = []
            image_paths = []
            
            # Handle image uploads
            if images:
                for i, image in enumerate(images):
                    # Save uploaded file temporarily
                    temp_path = self.temp_dir / f"{thread_id}_{i}_{image.filename}"
                    
                    with open(temp_path, "wb") as buffer:
                        content = await image.read()
                        buffer.write(content)
                    
                    image_paths.append(str(temp_path))
                    
                    # Add image path for tools
                    messages.append({"role": "user", "content": f"image_path: {temp_path}"})
                    
                    # Add base64 encoded image for multimodal processing
                    image_base64 = base64.b64encode(content).decode("utf-8")
                    
                    # Determine MIME type
                    mime_type = "image/jpeg"  # Default
                    if image.content_type:
                        mime_type = image.content_type
                    elif temp_path.suffix.lower() in ['.png']:
                        mime_type = "image/png"
                    
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                            }
                        ],
                    })
            
            # Add text question
            messages.append({"role": "user", "content": [{"type": "text", "text": question}]})
            
            # Process through agent workflow
            response_text = ""
            tools_used = []
            
            # Temporarily update system prompt if provided
            original_prompt = None
            if system_prompt:
                original_prompt = self.agent.system_prompt
                self.agent.system_prompt = system_prompt
            
            try:
                async for chunk in self._stream_agent_response(messages, thread_id):
                    if chunk.get("type") == "text":
                        response_text += chunk.get("content", "")
                    elif chunk.get("type") == "tool":
                        tools_used.append(chunk.get("tool_name", ""))
            finally:
                # Restore original system prompt
                if original_prompt is not None:
                    self.agent.system_prompt = original_prompt
            
            # Clean up temporary files
            for image_path in image_paths:
                try:
                    Path(image_path).unlink(missing_ok=True)
                except Exception:
                    pass  # Ignore cleanup errors
            
            processing_time = time.time() - start_time
            
            return QueryResponse(
                response=response_text.strip(),
                thread_id=thread_id,
                tools_used=list(set(tools_used)),  # Remove duplicates
                processing_time=processing_time
            )
            
        except Exception as e:
            # Clean up on error
            for image_path in image_paths:
                try:
                    Path(image_path).unlink(missing_ok=True)
                except Exception:
                    pass
            
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
    async def _stream_agent_response(self, messages: List[Dict], thread_id: str):
        """
        Stream responses from the agent workflow.
        
        Args:
            messages (List[Dict]): Messages to process
            thread_id (str): Thread ID for the conversation
            
        Yields:
            Dict: Response chunks with type and content
        """
        try:
            for chunk in self.agent.workflow.stream(
                {"messages": messages},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="updates",
            ):
                if not isinstance(chunk, dict):
                    continue
                
                for node_name, node_output in chunk.items():
                    if "messages" not in node_output:
                        continue
                    
                    for msg in node_output["messages"]:
                        if isinstance(msg, AIMessage) and msg.content:
                            # Clean up temp paths from response
                            import re
                            clean_content = re.sub(r"temp[^\s]*", "", msg.content).strip()
                            if clean_content:
                                yield {"type": "text", "content": clean_content}
                        
                        elif isinstance(msg, ToolMessage):
                            # Extract tool name from the message
                            tool_call_id = msg.tool_call_id
                            # We'll track tool usage but not include detailed output in API response
                            yield {"type": "tool", "tool_name": "tool_executed"}
        
        except Exception as e:
            yield {"type": "error", "content": str(e)}


def create_api(agent: Agent, tools_dict: Dict[str, Any], temp_dir: str = "temp_api") -> FastAPI:
    """
    Create and configure the MedRAX FastAPI application.
    
    Args:
        agent (Agent): The initialized MedRAX agent
        tools_dict (Dict[str, Any]): Dictionary of available tools
        temp_dir (str): Directory for temporary file storage
        
    Returns:
        FastAPI: Configured FastAPI application
    """
    api = MedRAXAPI(agent, tools_dict, temp_dir)
    return api.app
