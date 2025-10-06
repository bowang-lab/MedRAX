"""
MedRAX FastAPI Backend
Production-grade backend for MedRAX medical imaging AI agent.
"""

import os
import sys
import json
import asyncio
import base64
import uuid
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

# Set environment variables to suppress verbose library output (BEFORE imports)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformers info
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizers warning

# Redirect stderr temporarily to suppress bitsandbytes print statements
import io
_original_stderr = sys.stderr
sys.stderr = io.StringIO()  # Capture stderr during imports

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Suppress various third-party warnings
warnings.filterwarnings('ignore', message='.*bitsandbytes.*')
warnings.filterwarnings('ignore', message='.*cadam32bit_grad_fp32.*')
warnings.filterwarnings('ignore', message='.*pylibjpeg.*')
warnings.filterwarnings('ignore', message='.*No plugins found.*')
warnings.filterwarnings('ignore', message='.*GenerationMixin.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning)

# Import logging configuration FIRST
from logger_config import get_logger, log_tool_execution, log_api_request
from utils import (
    ensure_json_serializable, 
    validate_image_path, 
    sanitize_filename,
    validate_chat_message,
    cleanup_old_files
)

# Import MedRAX wrapper and utilities
from medrax_wrapper import initialize_medrax_agent, check_medrax_availability
from chat_interface_minimal import MinimalChatInterface
from tool_manager import ToolManager, ToolStatus

# Import LangChain components (used in multiple places)
from langchain_openai import ChatOpenAI
from medrax.agent import Agent
from medrax.utils import load_prompts_from_file

# Import SSE for streaming
from sse_starlette.sse import EventSourceResponse

# Import session manager
from session_manager import SessionManager, get_session_manager

# Restore stderr after noisy imports
sys.stderr = _original_stderr

load_dotenv()

# Initialize logger
logger = get_logger(__name__)

# Note: ensure_json_serializable is now imported from utils.py

app = FastAPI(title="MedRAX Agent Platform", version="2.0.0")

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

# Global storage
global_agent = None
global_tools = None
global_tool_manager: Optional[ToolManager] = None
global_checkpointer = None  # Will be MemorySaver or None

# Use SessionManager instead of plain dict
session_manager: SessionManager = get_session_manager()

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

# Background cleanup task
async def cleanup_temp_files_periodically():
    """Periodically clean up old temporary files"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            logger.info("cleanup_start", message="Starting cleanup of old temporary files")
            cleanup_old_files(temp_dir, max_age_hours=24)
            logger.info("cleanup_complete", message="Temporary file cleanup completed")
        except Exception as e:
            logger.error("cleanup_error", error=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize the MedRAX agent with tool manager"""
    global global_agent, global_tools, global_tool_manager, global_checkpointer
    
    logger.info("separator", line="="*60)
    logger.info("app_startup", message="MedRAX Platform with Advanced Tool Manager")
    logger.info("separator", line="="*60)
    
    try:
        # Initialize tool manager
        device = "cpu"  # Use CPU for Mac compatibility
        
        # Use writable cache directory (defaults to HuggingFace cache or local)
        default_cache = os.path.expanduser("~/.cache/huggingface")
        model_dir = os.getenv("MODEL_DIR", default_cache)
        temp_dir = "temp"
        
        # Ensure cache directory exists and is writable
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        global_tool_manager = ToolManager(
            device=device,
            model_dir=model_dir,
            temp_dir=temp_dir
        )
        
        logger.info("message", text="\n📋 Tool Registry:")
        logger.info("output", data="-" * 60)
        for tool_id, tool_info in global_tool_manager.get_all_tools().items():
            status_icon = {
                "available": "🟢",
                "loaded": "✅",
                "unavailable": "❌",
                "error": "⚠️"
            }.get(tool_info["status"], "❓")
            logger.info("message", text=f"{status_icon} {tool_info['display_name']: <30} [{tool_info['status']}]")
            if tool_info["error_message"]:
                logger.info("message", text=f"   → {tool_info['error_message']}")
        
        # Load default tools
        logger.info("message", text="\n🔧 Loading Default Tools...")
        logger.info("output", data="-" * 60)
        load_results = global_tool_manager.load_default_tools()
        
        loaded_count = sum(1 for success in load_results.values() if success)
        total_count = len(load_results)
        
        logger.info("message", text=f"\n✅ Loaded {loaded_count}/{total_count} default tools successfully!")
        
        # Get loaded tools for agent
        tools_list = global_tool_manager.get_loaded_tools()
        global_tools = global_tool_manager.get_loaded_tools_dict()
        
        if not tools_list:
            logger.warning("warning", message="No tools loaded, agent will have limited functionality")
            global_agent = None
            return
        
        # Initialize LangChain agent
        # Load system prompt
        medrax_root = Path(__file__).parent.parent.parent
        prompt_path = medrax_root / "medrax/docs/system_prompts.txt"
        
        if prompt_path.exists():
            prompts = load_prompts_from_file(str(prompt_path))
            prompt = prompts.get("MEDICAL_ASSISTANT", "You are a medical AI assistant.")
        else:
            prompt = "You are a medical AI assistant specialized in analyzing chest X-rays."
        
        # Create agent with checkpointer
        # NOTE: AsyncSqliteSaver causes "threads can only be started once" error
        # Using MemorySaver instead for in-memory state (no persistence across restarts)
        from langgraph.checkpoint.memory import MemorySaver
        
        global_checkpointer = MemorySaver()  # In-memory checkpointer (no threading issues)
        
        logger.info("checkpointer_status", type="MemorySaver", persistent=False, reason="Avoiding threading issues")
        
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        
        global_agent = Agent(
            model=llm,
            tools=tools_list,
            checkpointer=global_checkpointer,
            system_prompt=prompt,
            log_tools=True,
            log_dir=temp_dir
        )
        
        # Start background cleanup tasks
        asyncio.create_task(cleanup_temp_files_periodically())
        asyncio.create_task(session_manager.cleanup_old_sessions_periodically())
        logger.info("background_tasks", message="Started cleanup background tasks")
        
        logger.info("output", data="\n" + "="*60)
        logger.info("success", message="MedRAX Platform Initialized Successfully!")
        logger.info("separator", line="="*60)
        
    except Exception as e:
        import traceback
        logger.info("message", text=f"\n❌ Failed to initialize MedRAX platform: {e}")
        logger.info("output", data=traceback.format_exc())
        logger.warning("warning", message="Running with limited functionality")
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
        "active_sessions": len(session_manager.sessions)  # Fixed: was agent_sessions
    }

@app.post("/api/sessions", response_model=SessionInfo)
async def create_session():
    """Create a new agent session"""
    # Allow session creation even with mock agent for development
    session_id = str(uuid.uuid4())
    
    # Create a chat interface for this session
    # Pass the global agent reference (not a copy) so it gets updates when tools are loaded
    chat_interface = MinimalChatInterface(global_agent, global_tools or {})
    chat_interface.current_thread_id = session_id  # Use session_id as thread_id for memory persistence
    session_manager.create_session(session_id, chat_interface)
    
    logger.info("session_created", session_id=session_id[:8])
    
    return SessionInfo(
        session_id=session_id,
        status="active" if global_agent else "mock", 
        created_at=datetime.now(timezone.utc).isoformat()
    )

@app.post("/api/sessions/{session_id}/clear")
async def clear_chat(session_id: str):
    """Clear chat history but keep session alive"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.uploaded_files = []
    session.display_files = []
    session.latest_tool_results = {}
    logger.info("chat_cleared", session_id=session_id[:8])
    return {"success": True}

@app.post("/api/sessions/{session_id}/new-thread")
async def new_thread(session_id: str):
    """Start new conversation thread"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    new_thread_id = str(uuid.uuid4())
    session.current_thread_id = new_thread_id
    logger.info("new_thread", session_id=session_id[:8], thread_id=new_thread_id[:8])
    return {"success": True, "thread_id": new_thread_id}

@app.post("/api/upload/{session_id}")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    """Upload medical image or DICOM file"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Validate and sanitize filename
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename cannot be empty")
        
        safe_filename = sanitize_filename(file.filename)
        if not safe_filename or safe_filename.startswith('.'):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.dcm'}
        file_ext = Path(safe_filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"File type {file_ext} not allowed. Allowed types: {', '.join(allowed_extensions)}")
        
        # Read file content
        content = await file.read()
        
        # Validate file size (max 50 MB)
        max_size = 50 * 1024 * 1024  # 50 MB
        if len(content) > max_size:
            raise HTTPException(status_code=400, detail=f"File too large. Maximum size is 50 MB")
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Create upload directory
        upload_dir = Path("temp") / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        file_path = upload_dir / safe_filename
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Process file through existing MedRAX interface
        chat_interface = session_manager.get_session(session_id)
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
    session = session_manager.get_session(session_id)
    if not session:  # Fixed: was "if session:"
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Validate message
    is_valid, error_msg = validate_chat_message(request.message)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Validate image path if provided
    if request.image_path and not validate_image_path(request.image_path):
        raise HTTPException(status_code=400, detail="Invalid or unsafe image path")
    
    chat_interface = session_manager.get_session(session_id)
    
    try:
        # Process message through minimal chat interface
        responses = []
        tool_events = []  # Track tool execution
        async for response in chat_interface.process_message(
            request.message, 
            request.image_path
        ):
            # Parse tool events (special markers)
            if response.startswith("__TOOL_START__"):
                tool_name = response.replace("__TOOL_START__", "").replace("__", "")
                tool_events.append({"tool_name": tool_name, "status": "running"})
            elif response.startswith("__TOOL_DONE__"):
                tool_name = response.replace("__TOOL_DONE__", "").replace("__", "")
                # Update status
                for event in tool_events:
                    if event["tool_name"] == tool_name:
                        event["status"] = "completed"
            else:
                responses.append(response)
        
        # Combine all responses
        final_response = "\n\n".join(responses) if responses else "No response generated"
        
        # Get tool calls from chat interface
        tool_calls_data = chat_interface.last_tool_calls if hasattr(chat_interface, 'last_tool_calls') else []
        
        return ChatResponse(
            response=final_response,
            session_id=session_id,
            tool_calls=tool_calls_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/api/chat/{session_id}/stream")
async def stream_analysis(session_id: str, image_path: str):
    """Stream analysis progress using Server-Sent Events"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_interface = session_manager.get_session(session_id)
    
    async def event_generator():
        """Generate SSE events for analysis progress"""
        try:
            # Send start event
            yield {
                "event": "status",
                "data": json.dumps({
                    "type": "start",
                    "message": "🤖 Starting comprehensive AI analysis..."
                })
            }
            
            # Define the analysis tools to run in order
            # NOTE: Use actual tool.name values, not tool_manager IDs
            analysis_tools = [
                ("chest_xray_classifier", "Pathology Classifier"),
                ("chest_xray_segmentation", "Anatomical Segmentation"),
                ("chest_xray_report_generator", "Report Generator"),  # Fixed: was "chest_xray_report"
            ]
            
            for tool_id, tool_name in analysis_tools:
                # Check if tool is loaded
                if tool_id not in global_tools:
                    yield {
                        "event": "status",
                        "data": json.dumps({
                            "type": "skip",
                            "tool": tool_name,
                            "message": f"⏭️ Skipping {tool_name} (not loaded)"
                        })
                    }
                    continue
                
                # Send progress event
                yield {
                    "event": "status",
                    "data": json.dumps({
                        "type": "progress",
                        "tool": tool_name,
                        "message": f"🔧 Running {tool_name}..."
                    })
                }
                
                try:
                    # Run the tool
                    tool = global_tools[tool_id]
                    timestamp = datetime.now(timezone.utc).isoformat()
                    
                    if tool_id == "chest_xray_classifier":
                        result, metadata = tool._run(image_path)
                    elif tool_id == "chest_xray_segmentation":
                        result, metadata = tool._run(image_path)
                    elif tool_id == "chest_xray_report_generator":  # Fixed: was "chest_xray_report"
                        result, metadata = tool._run(image_path, chat_interface.uploaded_files)
                    else:
                        continue
                    
                    metadata["timestamp"] = timestamp
                    
                    # Store result directly in latest_tool_results
                    chat_interface.latest_tool_results[tool_id] = {
                        "result": ensure_json_serializable(result),
                        "metadata": ensure_json_serializable(metadata)
                    }
                    
                    # Debug logging
                    logger.info("tool_result_stored", 
                               tool_id=tool_id, 
                               has_result=bool(result),
                               result_type=type(result).__name__)
                    
                    # Send completion event
                    yield {
                        "event": "status",
                        "data": json.dumps({
                            "type": "complete",
                            "tool": tool_name,
                            "message": f"✅ {tool_name} completed"
                        })
                    }
                    
                except Exception as e:
                    logger.error("tool_error", tool=tool_name, error=str(e))
                    yield {
                        "event": "status",
                        "data": json.dumps({
                            "type": "error",
                            "tool": tool_name,
                            "message": f"❌ {tool_name} failed: {str(e)}"
                        })
                    }
            
            # Send final completion event
            yield {
                "event": "status",
                "data": json.dumps({
                    "type": "done",
                    "message": "✅ Analysis complete!"
                })
            }
            
        except Exception as e:
            logger.error("stream_error", error=str(e))
            yield {
                "event": "status",
                "data": json.dumps({
                    "type": "error",
                    "message": f"❌ Stream error: {str(e)}"
                })
            }
    
    return EventSourceResponse(event_generator())

@app.get("/api/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific session"""
    session = session_manager.get_session(session_id)
    if not session:  # Fixed: was "if session:"
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_interface = session_manager.get_session(session_id)
    
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
    session = session_manager.get_session(session_id)
    if not session:  # Fixed: was "if session:"
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Delete session using session_manager
    session_manager.delete_session(session_id)
    
    # Clean up temp files
    session_dir = Path("temp") / session_id
    if session_dir.exists():
        try:
            shutil.rmtree(session_dir)
        except Exception as e:
            logger.info("message", text=f"Warning: Failed to clean up session directory: {e}")
    
    return {"message": "Session deleted successfully"}

@app.get("/api/tools")
async def get_available_tools():
    """Get comprehensive list of all MedRAX tools with status"""
    if not global_tool_manager:
        return {"tools": [], "message": "Tool manager not initialized"}
    
    all_tools = global_tool_manager.get_all_tools()
    recommendations = global_tool_manager.get_tool_recommendations()
    
    return {
        "tools": all_tools,
        "recommendations": recommendations,
        "loaded_count": sum(1 for t in all_tools.values() if t["is_loaded"]),
        "available_count": sum(1 for t in all_tools.values() if t["status"] in ["available", "loaded"])
    }

@app.post("/api/tools/{tool_id}/load")
async def load_tool(tool_id: str):
    """Load a specific tool"""
    if not global_tool_manager:
        raise HTTPException(status_code=503, detail="Tool manager not initialized")
    
    success, error = global_tool_manager.load_tool(tool_id)
    
    if not success:
        raise HTTPException(status_code=400, detail=error or "Failed to load tool")
    
    # Update global tools dict and reinitialize agent with new tool
    global global_tools, global_agent
    global_tools = global_tool_manager.get_loaded_tools_dict()
    
    # CRITICAL: Update agent with new tools
    if global_agent:
        try:
            tools_list = global_tool_manager.get_loaded_tools()
            
            # Update agent's tools dict (line 101 in agent.py)
            global_agent.tools = {t.name: t for t in tools_list}
            
            # Rebind model with new tools (line 102 in agent.py)
            # Get the base model without bindings
            base_model = ChatOpenAI(
                model="gpt-4o",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL")
            )
            global_agent.model = base_model.bind_tools(tools_list)
            
            logger.info("success", message=f"Agent updated with new tool: {tool_id}")
            logger.info("message", text=f"📋 Agent now has {len(tools_list)} tools: {[t.name for t in tools_list]}")
            
            # Update all existing session interfaces to use updated agent
            for sid in list(session_manager.sessions.keys()):  # Fixed: was using undefined session_id
                chat_interface = session_manager.get_session(sid)
                if chat_interface:
                    chat_interface.agent = global_agent
                    logger.debug("session_updated", session_id=sid[:8])
        except Exception as e:
            logger.warning("warning", message=f"Failed to update agent tools: {e}")
    
    return {
        "success": True,
        "tool_id": tool_id,
        "message": f"Tool '{tool_id}' loaded successfully",
        "tool_info": global_tool_manager.get_all_tools()[tool_id]
    }

@app.post("/api/tools/{tool_id}/unload")
async def unload_tool(tool_id: str):
    """Unload a specific tool to free memory"""
    if not global_tool_manager:
        raise HTTPException(status_code=503, detail="Tool manager not initialized")
    
    success, error = global_tool_manager.unload_tool(tool_id)
    
    if not success:
        raise HTTPException(status_code=400, detail=error or "Failed to unload tool")
    
    # Update global tools dict and agent
    global global_tools, global_agent
    global_tools = global_tool_manager.get_loaded_tools_dict()
    
    # Update agent with reduced tools list
    if global_agent:
        try:
            tools_list = global_tool_manager.get_loaded_tools()
            
            # Update agent's tools dict
            global_agent.tools = {t.name: t for t in tools_list}
            
            # Rebind model with updated tools
            base_model = ChatOpenAI(
                model="gpt-4o",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL")
            )
            global_agent.model = base_model.bind_tools(tools_list)
            
            logger.info("success", message=f"Agent updated after unloading: {tool_id}")
            logger.info("message", text=f"📋 Agent now has {len(tools_list)} tools")
            
            # Update all existing sessions
            for sid in list(session_manager.sessions.keys()):  # Fixed: was using undefined session_id
                chat_interface = session_manager.get_session(sid)
                if chat_interface:
                    chat_interface.agent = global_agent
                    logger.debug("session_updated", session_id=sid[:8])
        except Exception as e:
            logger.warning("warning", message=f"Failed to update agent tools: {e}")
    
    return {
        "success": True,
        "tool_id": tool_id,
        "message": f"Tool '{tool_id}' unloaded successfully",
        "tool_info": global_tool_manager.get_all_tools()[tool_id]
    }

@app.get("/api/tools/{tool_id}/info")
async def get_tool_info(tool_id: str):
    """Get detailed information about a specific tool"""
    if not global_tool_manager:
        raise HTTPException(status_code=503, detail="Tool manager not initialized")
    
    all_tools = global_tool_manager.get_all_tools()
    
    if tool_id not in all_tools:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found")
    
    return {"tool": all_tools[tool_id]}

@app.post("/api/tools/{tool_name}/run/{session_id}")
async def run_specific_tool(tool_name: str, session_id: str, request: Optional[dict] = None):
    """Run a specific tool with parameters"""
    session = session_manager.get_session(session_id)
    if not session:  # Fixed: was "if session:"
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not global_tools or tool_name not in global_tools:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    chat_interface = session_manager.get_session(session_id)
    image_paths = chat_interface.uploaded_files
    
    if not image_paths:
        raise HTTPException(status_code=400, detail="No images uploaded for this session")
    
    # Use the first image for single-image tools, or all images for multi-image tools
    primary_image_path = image_paths[0]
    
    try:
        tool = global_tools[tool_name]
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Run the tool directly
        # NOTE: tool_name here is the actual tool.name (e.g., "chest_xray_classifier", "chest_xray_report_generator")
        if tool_name == "chest_xray_classifier":
            result, metadata = tool._run(primary_image_path)
            metadata["timestamp"] = timestamp
            result = ensure_json_serializable(result)
            metadata = ensure_json_serializable(metadata)
            return {"tool": tool_name, "result": result, "metadata": metadata}
            
        elif tool_name == "chest_xray_report_generator":  # Fixed: was "ChestXRayReportGeneratorTool"
            result, metadata = tool._run(primary_image_path, image_paths)
            metadata["timestamp"] = timestamp
            result = ensure_json_serializable(result)
            metadata = ensure_json_serializable(metadata)
            return {"tool": tool_name, "result": result, "metadata": metadata}
            
        elif tool_name == "chest_xray_segmentation":  # Fixed: was "ChestXRaySegmentationTool"
            organs = request.get("organs", None) if request else None
            result, metadata = tool._run(primary_image_path, organs)
            metadata["timestamp"] = timestamp
            result = ensure_json_serializable(result)
            metadata = ensure_json_serializable(metadata)
            return {"tool": tool_name, "result": result, "metadata": metadata}
            
        elif tool_name == "chest_xray_expert":  # Fixed: was "XRayVQATool"
            # This tool supports multiple images
            prompt = request.get("prompt", "Analyze these chest X-ray images") if request else "Analyze these chest X-ray images"
            result, metadata = tool._run(image_paths, prompt)
            metadata["timestamp"] = timestamp
            result = ensure_json_serializable(result)
            metadata = ensure_json_serializable(metadata)
            return {"tool": tool_name, "result": result, "metadata": metadata}
            
        elif tool_name == "xray_phrase_grounding":  # Added: handle grounding tool
            phrase = request.get("phrase", "Cardiomegaly") if request else "Cardiomegaly"
            result, metadata = tool._run(primary_image_path, phrase)
            metadata["timestamp"] = timestamp
            result = ensure_json_serializable(result)
            metadata = ensure_json_serializable(metadata)
            return {"tool": tool_name, "result": result, "metadata": metadata}
            
        elif tool_name == "llava_med_qa":  # Added: handle LLaVA-Med (note: uses tool.name, not tool_id)
            prompt = request.get("prompt", "Analyze this medical image") if request else "Analyze this medical image"
            result, metadata = tool._run(primary_image_path, prompt)
            metadata["timestamp"] = timestamp
            result = ensure_json_serializable(result)
            metadata = ensure_json_serializable(metadata)
            return {"tool": tool_name, "result": result, "metadata": metadata}
            
        else:
            # Generic fallback for other tools
            result, metadata = tool._run(primary_image_path)
            metadata["timestamp"] = timestamp
            result = ensure_json_serializable(result)
            metadata = ensure_json_serializable(metadata)
            return {"tool": tool_name, "result": result, "metadata": metadata}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

@app.get("/api/analysis/{session_id}")
async def get_analysis_results(session_id: str):
    """Get comprehensive analysis results for a session"""
    session = session_manager.get_session(session_id)
    if not session:  # Fixed: was "if session:"
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_interface = session_manager.get_session(session_id)
    
    # Return stored tool results from the last agent execution
    results = {}
    
    # Get results from chat interface's latest_tool_results
    for tool_name, tool_data in chat_interface.latest_tool_results.items():
        # Always store with original tool name for frontend compatibility
        results[tool_name] = ensure_json_serializable(tool_data)
    
    # Debug logging
    logger.info("analysis_results", 
                session_id=session_id[:8], 
                tool_count=len(results),
                tool_names=list(results.keys()))
    
    return {"session_id": session_id, "results": results}

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
