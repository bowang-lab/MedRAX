"""
MedRAX FastAPI Backend
Production-grade backend for MedRAX medical imaging AI agent.

Note: Some imports appear after initial setup code. This is intentional to suppress
verbose output from ML libraries (TensorFlow, transformers, bitsandbytes). The setup
must occur before importing these libraries.
"""

import asyncio
import json
import os
import shutil
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from medrax.agent import Agent
from medrax.utils import load_prompts_from_file
from pydantic import BaseModel

# Local imports
from auth import SimpleAuthManager, get_auth_manager
from chat_interface import ChatInterface
from logger_config import get_logger
from session_manager import SessionManager, get_session_manager
from tool_manager import ToolManager
from utils import (
    cleanup_old_files as util_cleanup_old_files,
)
from utils import (
    ensure_json_serializable,
    sanitize_filename,
    validate_chat_message,
)


# Helper function to verify user authentication
def verify_user_token(token: Optional[str], expected_user_id: str) -> bool:
    """
    Verify that the provided token belongs to the expected user.
    Returns True if valid, raises HTTPException if not.
    """
    if not token:
        raise HTTPException(status_code=401, detail="Authentication token required")

    actual_user_id = auth_manager.verify_token(token)
    if not actual_user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    if actual_user_id != expected_user_id:
        raise HTTPException(status_code=403, detail="Access denied: user_id mismatch")

    return True

# Configure environment variables for ML libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configure warning filters
warnings.filterwarnings('ignore', message='.*bitsandbytes.*')
warnings.filterwarnings('ignore', message='.*cadam32bit_grad_fp32.*')
warnings.filterwarnings('ignore', message='.*pylibjpeg.*')
warnings.filterwarnings('ignore', message='.*No plugins found.*')
warnings.filterwarnings('ignore', message='.*GenerationMixin.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning)

# Load environment variables
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

# Use AuthManager for user authentication
auth_manager: SimpleAuthManager = get_auth_manager()

# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str
    image_path: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    image_path: Optional[str] = None

class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

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
            util_cleanup_old_files(temp_dir, max_age_hours=24)
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

# ========== Authentication Endpoints ==========

@app.post("/api/auth/register")
async def register(request: RegisterRequest):
    """
    Register a new user.
    Simple registration - just username and password (no strength requirements).
    """
    success, message = auth_manager.register_user(
        username=request.username,
        password=request.password,
        display_name=request.display_name
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    logger.info("user_registered", username=request.username)

    return {
        "success": True,
        "message": message,
        "username": request.username
    }

@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """
    Login user and get session token.
    Returns token to be used in Authorization header for subsequent requests.
    """
    success, token, message = auth_manager.login(
        username=request.username,
        password=request.password
    )

    if not success:
        raise HTTPException(status_code=401, detail=message)

    user_info = auth_manager.get_user_info(request.username)

    logger.info("user_logged_in", username=request.username)

    return {
        "success": True,
        "message": message,
        "token": token,
        "user": user_info
    }

@app.post("/api/auth/logout")
async def logout(token: str = Query(..., description="Session token")):
    """Logout user by invalidating session token"""
    success = auth_manager.logout(token)

    if not success:
        raise HTTPException(status_code=400, detail="Invalid token")

    return {
        "success": True,
        "message": "Logged out successfully"
    }

@app.get("/api/auth/verify")
async def verify_token(token: str = Query(..., description="Session token")):
    """Verify if token is valid and return user info"""
    user_id = auth_manager.verify_token(token)

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_info = auth_manager.get_user_info(user_id)

    return {
        "valid": True,
        "user": user_info
    }

@app.get("/api/auth/users")
async def list_users():
    """List all registered users (admin endpoint)"""
    users = auth_manager.list_users()

    return {
        "users": users,
        "count": len(users)
    }

# LEGACY ENDPOINT REMOVED - Use POST /api/users/{user_id}/chats instead
# Old: POST /api/sessions → created session-based chat
# New: POST /api/users/{user_id}/chats → creates chat under specific user

# LEGACY ENDPOINT REMOVED - Create new chat instead (automatic fresh context)
# Old: POST /api/sessions/{id}/clear → cleared messages but kept session
# New: POST /api/users/{uid}/chats → new chat = fresh context

# LEGACY ENDPOINT REMOVED - Create new chat instead
# Old: POST /api/sessions/{id}/new-thread → new thread in same session
# New: POST /api/users/{uid}/chats → new chat = new thread

# LEGACY ENDPOINT REMOVED - Use POST /api/users/{uid}/chats/{cid}/upload instead
# Old: POST /api/upload/{session_id} → uploaded to session
# New: POST /api/users/{uid}/chats/{cid}/upload → uploads to specific chat

# LEGACY ENDPOINT REMOVED - Use POST /api/users/{uid}/chats/{cid}/messages instead
# Old: POST /api/chat/{session_id} → sent message to session
# New: POST /api/users/{uid}/chats/{cid}/messages → sends to specific chat

# LEGACY ENDPOINT REMOVED - Use GET /api/users/{uid}/chats/{cid}/stream instead
# Old: GET /api/chat/{session_id}/stream → streamed session analysis
# New: GET /api/users/{uid}/chats/{cid}/stream → streams chat analysis

# LEGACY ENDPOINT REMOVED - Use GET /api/users/{uid}/chats/{cid} instead
# Old: GET /api/sessions/{session_id} → got session info
# New: GET /api/users/{uid}/chats/{cid} → gets chat info with metadata

# LEGACY ENDPOINT REMOVED - Use DELETE /api/users/{uid}/chats/{cid} instead
# Old: DELETE /api/sessions/{session_id} → deleted session
# New: DELETE /api/users/{uid}/chats/{cid} → deletes chat with cleanup

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

# LEGACY ENDPOINT REMOVED - Tools now run automatically via agent
# Old: POST /api/tools/{name}/run/{session_id} → manually ran specific tool
# New: Tools execute automatically when agent processes messages in chat context

# LEGACY ENDPOINT REMOVED - Use GET /api/users/{uid}/chats/{cid}/results instead
# Old: GET /api/analysis/{session_id} → got session analysis results
# New: GET /api/users/{uid}/chats/{cid}/results → gets chat-specific results


# ========== NEW: Multi-Chat Per User API Endpoints ==========

@app.post("/api/users/{user_id}/chats")
async def create_user_chat(user_id: str, chat_name: Optional[str] = None, token: Optional[str] = Query(None)):
    """Create a new chat for a user"""
    # Verify authentication
    verify_user_token(token, user_id)

    chat_id = str(uuid.uuid4())

    # Create chat interface
    chat_interface = ChatInterface(global_agent, global_tools or {})
    chat_interface.current_thread_id = chat_id

    # Create metadata
    metadata = {
        "name": chat_name or f"Chat {datetime.now().strftime('%b %d, %I:%M %p')}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message_count": 0,
        "image_count": 0
    }
    chat_interface.chat_metadata = metadata

    # Store in session manager
    session_manager.create_chat(user_id, chat_id, chat_interface, metadata)

    logger.info("user_chat_created", user_id=user_id[:8], chat_id=chat_id[:8])

    return {
        "user_id": user_id,
        "chat_id": chat_id,
        "name": metadata["name"],
        "created_at": metadata["created_at"]
    }

@app.get("/api/users/{user_id}/chats")
async def list_user_chats(user_id: str, token: Optional[str] = Query(None)):
    """List all chats for a user"""
    # Verify authentication
    verify_user_token(token, user_id)

    chats = session_manager.list_chats(user_id)

    logger.info("chats_listed", user_id=user_id[:8], count=len(chats))

    return {
        "user_id": user_id,
        "chats": chats,
        "total": len(chats)
    }

@app.get("/api/users/{user_id}/chats/{chat_id}")
async def get_user_chat(user_id: str, chat_id: str, token: Optional[str] = Query(None)):
    """Get details of a specific chat"""
    # Verify authentication
    verify_user_token(token, user_id)

    chat_interface = session_manager.get_chat(user_id, chat_id)
    if not chat_interface:
        raise HTTPException(status_code=404, detail="Chat not found")

    metadata = chat_interface.get_metadata()

    return {
        "user_id": user_id,
        "chat_id": chat_id,
        "metadata": metadata,
        "uploaded_images": chat_interface.uploaded_files,
        "message_history": chat_interface.get_message_history() if hasattr(chat_interface, 'get_message_history') else [],
        "has_results": len(chat_interface.latest_tool_results) > 0
    }

@app.post("/api/users/{user_id}/chats/{chat_id}/messages")
async def send_chat_message(user_id: str, chat_id: str, request: ChatRequest, token: Optional[str] = Query(None)):
    """Send a message to a specific chat"""
    # Verify authentication
    verify_user_token(token, user_id)

    chat_interface = session_manager.get_chat(user_id, chat_id)
    if not chat_interface:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Validate message
    is_valid, error_msg = validate_chat_message(request.message)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    try:
        # Process message through chat interface
        responses = []
        tool_events = []
        async for response in chat_interface.process_message(
            request.message,
            request.image_path
        ):
            # Parse tool events
            if response.startswith("__TOOL_START__"):
                tool_name = response.replace("__TOOL_START__", "").replace("__", "")
                tool_events.append({"tool_name": tool_name, "status": "running"})
            elif response.startswith("__TOOL_DONE__"):
                tool_name = response.replace("__TOOL_DONE__", "").replace("__", "")
                for event in tool_events:
                    if event["tool_name"] == tool_name:
                        event["status"] = "completed"
            else:
                responses.append(response)

        # Update chat metadata in session manager
        session_manager.update_chat_metadata(chat_id, {
            "message_count": chat_interface.chat_metadata["message_count"],
            "image_count": chat_interface.chat_metadata["image_count"]
        })

        # Combine responses
        final_response = "\n\n".join(responses) if responses else "No response generated"

        return ChatResponse(
            response=final_response,
            session_id=chat_id,  # For backward compatibility
            tool_calls=chat_interface.last_tool_calls if hasattr(chat_interface, 'last_tool_calls') else []
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/api/users/{user_id}/chats/{chat_id}/stream")
async def stream_chat_analysis(user_id: str, chat_id: str, token: Optional[str] = Query(None)):
    """
    Stream comprehensive analysis results for ALL images in the chat using Server-Sent Events.
    
    This endpoint analyzes ALL uploaded images in the chat session, not just one.
    The analysis includes:
    - Pathology classification for each image
    - Anatomical segmentation
    - Detailed radiology report generation
    - Visual question answering capabilities
    """
    # Verify authentication
    verify_user_token(token, user_id)

    chat_interface = session_manager.get_chat(user_id, chat_id)
    if not chat_interface:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Check if images are uploaded
    if not chat_interface.uploaded_files:
        raise HTTPException(status_code=400, detail="No images uploaded for analysis")

    async def event_generator():
        try:
            # Send initial status with image count
            image_count = len(chat_interface.uploaded_files)
            yield f"event: status\ndata: {json.dumps({'type': 'status', 'message': f'🤖 Starting AI analysis of {image_count} image(s)...'})}\n\n"
            await asyncio.sleep(0.1)  # Small delay to ensure message is sent

            # Process the message with a comprehensive prompt
            responses = []
            analysis_prompt = f"""Please perform a comprehensive medical analysis on all {image_count} uploaded chest X-ray image(s).

For each image, provide:
1. **Pathology Classification**: Identify and quantify any abnormalities (pneumonia, atelectasis, consolidation, edema, etc.)
2. **Anatomical Segmentation**: Identify and segment key anatomical structures (lungs, heart, etc.)
3. **Detailed Findings**: Describe notable findings, their locations, and clinical significance
4. **Comparative Analysis**: If multiple images are present, compare and contrast findings across images
5. **Clinical Impression**: Provide an overall clinical assessment and recommendations

Use all available diagnostic tools to provide the most thorough analysis possible."""

            try:
                async for response in chat_interface.process_message(
                    analysis_prompt,
                    None  # display_image parameter is deprecated, uses self.uploaded_files
                ):
                    # Parse tool events
                    if response.startswith("__TOOL_START__"):
                        tool_name = response.replace("__TOOL_START__", "").replace("__", "")
                        yield f"event: status\ndata: {json.dumps({'type': 'progress', 'message': f'🔧 Running: {tool_name}...'})}\n\n"
                        await asyncio.sleep(0.1)
                    elif response.startswith("__TOOL_DONE__"):
                        tool_name = response.replace("__TOOL_DONE__", "").replace("__", "")
                        yield f"event: status\ndata: {json.dumps({'type': 'complete', 'message': f'✅ Completed: {tool_name}'})}\n\n"
                        await asyncio.sleep(0.1)
                    else:
                        responses.append(response)
            except Exception as process_error:
                logger.error("process_message_error", error=str(process_error), exc_info=True)
                yield f"event: status\ndata: {json.dumps({'type': 'error', 'message': f'❌ Processing error: {str(process_error)}'})}\n\n"
                return

            # Send completion status
            yield f"event: status\ndata: {json.dumps({'type': 'done', 'message': '✅ Analysis complete!'})}\n\n"
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error("stream_error", error=str(e), exc_info=True)
            yield f"event: status\ndata: {json.dumps({'type': 'error', 'message': f'❌ Error: {str(e)}'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        }
    )

@app.get("/api/users/{user_id}/chats/{chat_id}/results")
async def get_chat_analysis_results(user_id: str, chat_id: str, token: Optional[str] = Query(None)):
    """Get analysis results for a specific chat"""
    # Verify authentication
    verify_user_token(token, user_id)

    chat_interface = session_manager.get_chat(user_id, chat_id)
    if not chat_interface:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Return stored tool results from the chat interface
    results = {}
    for tool_name, tool_data in chat_interface.latest_tool_results.items():
        results[tool_name] = ensure_json_serializable(tool_data)

    logger.info("results_fetched",
               chat_id=chat_id[:8],
               tool_count=len(results),
               tools=list(results.keys()))

    return {
        "results": results,
        "chat_id": chat_id,
        "user_id": user_id,
        "count": len(results)
    }

@app.get("/api/users/{user_id}/chats/{chat_id}/tool-history")
async def get_tool_execution_history(
    user_id: str,
    chat_id: str,
    filter_by_image: Optional[str] = Query(None, description="Filter by image path"),
    filter_by_request: Optional[str] = Query(None, description="Filter by request ID"),
    latest_only: bool = Query(False, description="Return only latest execution per tool"),
    token: Optional[str] = Query(None)
):
    """
    Get full tool execution history for a chat with optional filtering.
    
    Supports three filtering modes:
    - latest_only=true: Show only the most recent execution per tool
    - filter_by_request=<request_id>: Show only executions from a specific analysis request
    - filter_by_image=<image_path>: Show only executions that used a specific image
    
    Returns a list of all tool executions with timestamps, image_paths, and results.
    """
    # Verify authentication
    verify_user_token(token, user_id)

    # Load tool results from database
    from database import SessionLocal, ToolResult
    db = SessionLocal()
    try:
        # Query tool results for this chat
        query = db.query(ToolResult).filter(ToolResult.chat_id == chat_id)
        
        # Apply filters
        if filter_by_request:
            query = query.filter(ToolResult.request_id == filter_by_request)
        
        if filter_by_image:
            # Filter by image path in metadata JSON
            query = query.filter(ToolResult.metadata.contains(filter_by_image))
        
        # Order by creation time (newest first)
        query = query.order_by(ToolResult.created_at.desc())
        
        tool_results = query.all()
        
        # Convert to dict format
        history = []
        seen_tools = set()
        
        for result in tool_results:
            # If latest_only, skip if we've already seen this tool
            if latest_only and result.tool_name in seen_tools:
                continue
            
            history.append({
                "execution_id": result.execution_id,
                "timestamp": result.created_at.isoformat() if result.created_at else None,
                "request_id": result.request_id,
                "tool_name": result.tool_name,
                "image_paths": result.metadata.get("image_paths", []) if result.metadata else [],
                "result": result.result_data,
                "metadata": result.metadata or {}
            })
            
            if latest_only:
                seen_tools.add(result.tool_name)
        
        logger.info("tool_history_fetched_from_db",
                   chat_id=chat_id[:8],
                   total_executions=len(history),
                   filter_image=filter_by_image is not None,
                   filter_request=filter_by_request is not None,
                   latest_only=latest_only)
        
        return {
            "history": history,
            "count": len(history),
            "filters": {
                "image": filter_by_image,
                "request": filter_by_request,
                "latest_only": latest_only
            }
        }
    finally:
        db.close()

@app.delete("/api/users/{user_id}/chats/{chat_id}")
async def delete_user_chat(user_id: str, chat_id: str, token: Optional[str] = Query(None)):
    """Delete a specific chat"""
    # Verify authentication
    verify_user_token(token, user_id)

    success = session_manager.delete_chat(user_id, chat_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Clean up temp files for this chat
    chat_dir = Path("temp") / chat_id
    if chat_dir.exists():
        try:
            shutil.rmtree(chat_dir)
        except Exception as e:
            logger.warning("cleanup_warning", message=f"Failed to clean up chat directory: {e}")

    logger.info("chat_deleted", user_id=user_id[:8], chat_id=chat_id[:8])

    return {"success": True, "message": "Chat deleted successfully"}

# ========== Memory Management Endpoints ==========

@app.get("/api/system/memory")
async def get_memory_stats():
    """Get comprehensive memory statistics for all sessions"""
    stats = session_manager.get_memory_stats()
    return stats

@app.post("/api/system/cleanup/memory")
async def cleanup_memory():
    """Trigger memory cleanup for all active chats"""
    stats = session_manager.cleanup_all_memory()
    return {
        "success": True,
        "message": "Memory cleanup completed",
        "stats": stats
    }

@app.post("/api/system/cleanup/files")
async def cleanup_old_files_endpoint(max_age_hours: int = 24):
    """Clean up old temporary files"""
    stats = session_manager.cleanup_old_files(max_age_hours)
    return {
        "success": True,
        "message": "File cleanup completed",
        "stats": stats
    }

@app.post("/api/users/{user_id}/chats/{chat_id}/cleanup")
async def cleanup_chat(user_id: str, chat_id: str, token: Optional[str] = Query(None)):
    """Clean up memory and temp files for a specific chat"""
    # Verify authentication
    verify_user_token(token, user_id)

    chat_interface = session_manager.get_chat(user_id, chat_id)
    if not chat_interface:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Clean up memory
    memory_stats = chat_interface.cleanup_memory()

    # Clean up temp files
    file_stats = chat_interface.cleanup_temp_files()

    return {
        "success": True,
        "message": "Chat cleanup completed",
        "memory_stats": memory_stats,
        "file_stats": file_stats
    }

@app.post("/api/users/{user_id}/chats/{chat_id}/upload")
async def upload_to_chat(user_id: str, chat_id: str, file: UploadFile = File(...), token: Optional[str] = Query(None)):
    """Upload an image to a specific chat"""
    # Verify authentication
    verify_user_token(token, user_id)

    chat_interface = session_manager.get_chat(user_id, chat_id)
    if not chat_interface:
        raise HTTPException(status_code=404, detail="Chat not found")

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
            raise HTTPException(status_code=400, detail=f"File type {file_ext} not allowed")

        # Read file content
        content = await file.read()

        # Validate file size (max 50 MB)
        max_size = 50 * 1024 * 1024
        if len(content) > max_size:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 50 MB")

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        # Create upload directory for this chat
        upload_dir = Path("temp") / chat_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        file_path = upload_dir / safe_filename
        with open(file_path, "wb") as f:
            f.write(content)

        # Process through chat interface
        display_path = chat_interface.handle_upload(str(file_path))

        # Update metadata in session manager
        session_manager.update_chat_metadata(chat_id, {
            "image_count": chat_interface.chat_metadata["image_count"]
        })

        return {
            "message": "File uploaded successfully",
            "file_path": str(file_path),
            "display_path": display_path,
            "filename": safe_filename,
            "total_images": len(chat_interface.uploaded_files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
