"""
Simple FastAPI backend that wraps the existing MedRAX agent functionality.
Similar to Google's ADK but specialized for medical agents.
"""

import os
import sys
import asyncio
import base64
import uuid
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Import MedRAX wrapper and utilities
from medrax_wrapper import initialize_medrax_agent, check_medrax_availability
from chat_interface_minimal import MinimalChatInterface
from dotenv import load_dotenv

load_dotenv()

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj

app = FastAPI(title="MedRAX Agent Platform", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (temp directory for images)
temp_dir = Path("temp")
temp_dir.mkdir(exist_ok=True)
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

# Global storage for agent sessions (in production, use Redis/Database)
agent_sessions: Dict[str, MinimalChatInterface] = {}
global_agent = None
global_tools = None

# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str
    image_path: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    image_path: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: List[Dict] = []
    images: List[str] = []  # Tool-generated images

class SessionInfo(BaseModel):
    session_id: str
    status: str
    created_at: str

@app.on_event("startup")
async def startup_event():
    """Initialize the MedRAX agent on startup"""
    global global_agent, global_tools
    
    print("Initializing MedRAX agent...")
    
    # Enable more tools to support frontend features
    # Note: Some tools may not be available if dependencies aren't installed
    selected_tools = [
        "ImageVisualizerTool",
        "ChestXRayClassifierTool",
        "ChestXRayReportGeneratorTool",
        "ChestXRaySegmentationTool",
        "XRayVQATool",  # CheXagent - VQA expert
        # "XRayPhraseGroundingTool",  # Maira-2 - Disabled: model too large, not available
        # "DicomProcessorTool",  # May not work without gdcm on ARM64 macOS
        # "LlavaMedTool",  # LLaVA-Med - requires more dependencies
    ]
    
    try:
        global_agent, global_tools = initialize_medrax_agent(
            "medrax/docs/system_prompts.txt",
            tools_to_use=selected_tools,
            model_dir="/model-weights",  # Adjust path as needed
            temp_dir="temp",
            device="cpu",  # Force CPU for compatibility
            model="gpt-4o",
            temperature=0.7,
            openai_kwargs={
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": os.getenv("OPENAI_BASE_URL")
            }
        )
        print("✅ MedRAX agent initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize MedRAX agent: {e}")
        # For demo purposes, continue without agent
        global_agent = None
        global_tools = {}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "MedRAX Agent Platform API", 
        "status": "running",
        "agent_available": global_agent is not None
    }

@app.get("/api/health")
async def health():
    """Health check with more details"""
    return {
        "status": "healthy",
        "agent_initialized": global_agent is not None,
        "available_tools": list(global_tools.keys()) if global_tools else [],
        "active_sessions": len(agent_sessions)
    }

@app.post("/api/sessions", response_model=SessionInfo)
async def create_session():
    """Create a new agent session"""
    # Allow session creation even with mock agent for development
    session_id = str(uuid.uuid4())
    
    # Create a chat interface for this session
    chat_interface = MinimalChatInterface(global_agent, global_tools or {})
    agent_sessions[session_id] = chat_interface
    
    return SessionInfo(
        session_id=session_id,
        status="active" if global_agent else "mock", 
        created_at=datetime.now().isoformat()
    )

@app.post("/api/upload/{session_id}")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    """Upload medical image or DICOM file"""
    if session_id not in agent_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Sanitize filename to prevent path traversal
        safe_filename = Path(file.filename).name  # Gets only the filename, removes any path components
        if not safe_filename or safe_filename.startswith('.'):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Create upload directory
        upload_dir = Path("temp") / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        file_path = upload_dir / safe_filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process file through existing MedRAX interface
        chat_interface = agent_sessions[session_id]
        display_path = chat_interface.handle_upload(str(file_path))
        
        return {
            "message": "File uploaded successfully",
            "file_path": str(file_path),
            "display_path": display_path,
            "filename": safe_filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/api/chat/{session_id}", response_model=ChatResponse)
async def chat_with_agent(session_id: str, request: ChatRequest):
    """Send message to MedRAX agent"""
    if session_id not in agent_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_interface = agent_sessions[session_id]
    
    try:
        # Process message through minimal chat interface
        responses = []
        async for response in chat_interface.process_message(
            request.message, 
            request.image_path
        ):
            responses.append(response)
        
        # Combine all responses
        final_response = "\n\n".join(responses) if responses else "No response generated"
        
        return ChatResponse(
            response=final_response,
            session_id=session_id,
            tool_calls=[]  # Can be enhanced to return actual tool calls
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/api/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific session"""
    if session_id not in agent_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_interface = agent_sessions[session_id]
    
    return {
        "session_id": session_id,
        "status": "active",
        "has_image": len(chat_interface.uploaded_files) > 0,
        "current_image": chat_interface.uploaded_files[0] if chat_interface.uploaded_files else None,
        "thread_id": chat_interface.current_thread_id
    }

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete an agent session"""
    if session_id not in agent_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del agent_sessions[session_id]
    
    # Clean up temp files
    session_dir = Path("temp") / session_id
    if session_dir.exists():
        try:
            shutil.rmtree(session_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up session directory: {e}")
    
    return {"message": "Session deleted successfully"}

@app.get("/api/tools")
async def get_available_tools():
    """Get list of available medical tools"""
    if not global_tools:
        return {"tools": [], "message": "No tools available"}
    
    tools_info = []
    for tool_name, tool in global_tools.items():
        tools_info.append({
            "name": tool_name,
            "description": getattr(tool, 'description', 'No description available'),
            "type": str(type(tool).__name__)
        })
    
    return {"tools": tools_info}

@app.post("/api/tools/{tool_name}/run/{session_id}")
async def run_specific_tool(tool_name: str, session_id: str, request: Optional[dict] = None):
    """Run a specific tool with parameters"""
    if session_id not in agent_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not global_tools or tool_name not in global_tools:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    chat_interface = agent_sessions[session_id]
    image_paths = chat_interface.uploaded_files
    
    if not image_paths:
        raise HTTPException(status_code=400, detail="No images uploaded for this session")
    
    # Use the first image for single-image tools, or all images for multi-image tools
    primary_image_path = image_paths[0]
    
    try:
        tool = global_tools[tool_name]
        
        # Run the tool directly
        if tool_name == "ChestXRayClassifierTool":
            result, metadata = tool._run(primary_image_path)
            result = convert_numpy_types(result)
            metadata = convert_numpy_types(metadata)
            return {"tool": tool_name, "result": result, "metadata": metadata}
            
        elif tool_name == "ChestXRayReportGeneratorTool":
            result, metadata = tool._run(primary_image_path)
            result = convert_numpy_types(result)
            metadata = convert_numpy_types(metadata)
            return {"tool": tool_name, "result": result, "metadata": metadata}
            
        elif tool_name == "ChestXRaySegmentationTool":
            organs = request.get("organs", None) if request else None
            result, metadata = tool._run(primary_image_path, organs)
            result = convert_numpy_types(result)
            metadata = convert_numpy_types(metadata)
            return {"tool": tool_name, "result": result, "metadata": metadata}
            
        elif tool_name == "XRayVQATool":
            # This tool supports multiple images
            prompt = request.get("prompt", "Analyze these chest X-ray images") if request else "Analyze these chest X-ray images"
            result, metadata = tool._run(image_paths, prompt)
            result = convert_numpy_types(result)
            metadata = convert_numpy_types(metadata)
            return {"tool": tool_name, "result": result, "metadata": metadata}
            
        else:
            result, metadata = tool._run(primary_image_path)
            result = convert_numpy_types(result)
            metadata = convert_numpy_types(metadata)
            return {"tool": tool_name, "result": result, "metadata": metadata}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

@app.get("/api/analysis/{session_id}")
async def get_analysis_results(session_id: str):
    """Get comprehensive analysis results for a session"""
    if session_id not in agent_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_interface = agent_sessions[session_id]
    
    # Return stored tool results from the last agent execution
    results = {}
    
    # Get results from chat interface's latest_tool_results
    for tool_name, tool_data in chat_interface.latest_tool_results.items():
        # Always store with original tool name for frontend compatibility
        results[tool_name] = convert_numpy_types(tool_data)
    
    return {"session_id": session_id, "results": results}

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
