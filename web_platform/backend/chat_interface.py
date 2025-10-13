"""
ChatInterface for web backend - handles agent interactions and file management.
"""

import ast
import base64
import re
import shutil
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

# Import logger
from logger_config import get_logger

logger = get_logger(__name__)

def parse_numpy_repr(s: str):
    """Parse a string containing numpy type constructors like 'np.float32(0.5)'"""
    # Replace numpy type constructors with their values
    # np.float32(0.5) -> 0.5
    s = re.sub(r'np\.float32\(([-\d.e]+)\)', r'\1', s)
    s = re.sub(r'np\.float64\(([-\d.e]+)\)', r'\1', s)
    s = re.sub(r'np\.int32\(([-\d]+)\)', r'\1', s)
    s = re.sub(r'np\.int64\(([-\d]+)\)', r'\1', s)
    return s

class ChatInterface:
    """
    Chat interface for web backend handling agent interactions and file management.
    """

    def __init__(self, agent, tools_dict, chat_metadata=None):
        """Initialize the chat interface."""
        self.agent = agent
        self.tools_dict = tools_dict
        self.upload_dir = Path("temp")
        self.upload_dir.mkdir(exist_ok=True)
        self.current_thread_id = None
        self.uploaded_files = []  # List of uploaded file paths
        self.display_files = []   # List of display file paths
        self.display_file_path = None  # For compatibility with image visualizer
        
        # NEW: Tool execution history (list of all executions)
        self.tool_execution_history = []  # List of {execution_id, timestamp, tool_name, image_paths, result, metadata, request_id}
        
        # KEEP: Latest results for backward compatibility
        self.latest_tool_results = {}  # Store latest tool execution results per tool
        
        self.last_tool_calls = []  # Track tool calls from last message
        self.current_request_id = None  # Track current analysis request

        # Message history storage
        self.message_history = []  # List of {role, content, timestamp} dicts

        # Chat metadata for multi-chat support
        self.chat_metadata = chat_metadata or {
            "name": "New Chat",
            "description": "",
            "created_at": None,
            "message_count": 0,
            "image_count": 0
        }

    def handle_upload(self, file_path: str) -> str:
        """Handle file upload and return display path."""
        if not file_path:
            return None

        source = Path(file_path)
        timestamp = int(time.time())

        # Save original file with proper suffix
        suffix = source.suffix.lower()
        saved_path = self.upload_dir / f"upload_{timestamp}{suffix}"
        shutil.copy2(file_path, saved_path)

        # Add to uploaded files list
        self.uploaded_files.append(str(saved_path))

        # Update chat metadata
        self.chat_metadata["image_count"] = len(self.uploaded_files)

        # Handle DICOM conversion for display only (if tool available)
        if suffix == ".dcm" and "DicomProcessorTool" in self.tools_dict:
            try:
                output, _ = self.tools_dict["DicomProcessorTool"]._run(str(saved_path))
                display_path = output["image_path"]
                logger.info("dicom_converted", path=str(saved_path))
            except Exception as e:
                logger.info("message", text=f"DICOM processing failed: {e}")
                display_path = str(saved_path)
        else:
            display_path = str(saved_path)

        # Add to display files list
        self.display_files.append(display_path)

        logger.info("image_uploaded",
                   total_images=len(self.uploaded_files),
                   display_path=display_path,
                   file_type=suffix)

        return display_path

    async def process_message(
        self, message: str, display_image: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Process a message and generate responses.
        Simplified version that yields text responses.

        Args:
            message: The user message to process
            display_image: (Deprecated) Image path - uses self.uploaded_files instead
        """
        import uuid
        from datetime import datetime, timezone
        
        # Generate unique request ID for this analysis
        self.current_request_id = str(uuid.uuid4())
        
        # Initialize thread if needed
        if not self.current_thread_id:
            self.current_thread_id = str(time.time())

        messages = []

        # Handle multiple images
        if self.uploaded_files:
            # Log multi-image processing
            logger.info("multi_image_processing",
                       image_count=len(self.uploaded_files),
                       paths=self.uploaded_files)

            # Send all image paths for tools
            image_paths_str = ", ".join(self.uploaded_files)
            messages.append({"role": "user", "content": f"image_paths: {image_paths_str}"})

            # Load and encode all images for multimodal
            for idx, image_path in enumerate(self.uploaded_files):
                try:
                    with open(image_path, "rb") as img_file:
                        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                            }
                        ],
                    })
                    logger.info("image_encoded", index=idx+1, path=image_path)
                except Exception as e:
                    logger.error("image_encoding_error", image_path=image_path, error=str(e))

        if message is not None:
            messages.append({"role": "user", "content": [{"type": "text", "text": message}]})
            # Update message count
            self.chat_metadata["message_count"] += 1

        try:
            responses = []
            self.last_tool_calls = []  # Reset for this message

            # Check if agent is available
            if self.agent is None:
                error_msg = "❌ Agent not initialized. Please check backend logs and ensure OpenAI API key is set."
                yield error_msg
                return

            # Process through the agent
            # CRITICAL: Use astream() not stream() for AsyncSqliteSaver
            async for event in self.agent.workflow.astream(
                {"messages": messages}, {"configurable": {"thread_id": self.current_thread_id}}
            ):
                if isinstance(event, dict):
                    if "process" in event:
                        content = event["process"]["messages"][-1].content
                        if content:
                            # Clean up temp paths from content
                            import re
                            content = re.sub(r"temp/[^\s]*", "", content)
                            responses.append(content)
                            yield content

                    elif "execute" in event:
                        for message in event["execute"]["messages"]:
                            tool_name = message.name

                            # Yield tool start event
                            tool_start_msg = f"__TOOL_START__{tool_name}__"
                            yield tool_start_msg

                            # Safely parse tool result
                            # message.content contains string like: "[({'key': np.float32(0.5)}, {'meta': 'data'})]"
                            try:
                                # First, clean up numpy repr from the string
                                cleaned_content = parse_numpy_repr(message.content) if message.content else None

                                # Then evaluate the string to get Python objects
                                parsed = ast.literal_eval(cleaned_content) if cleaned_content else None

                                # Check if it's a list with one element (common LangChain format)
                                if isinstance(parsed, list) and len(parsed) > 0:
                                    tool_result = parsed[0]
                                else:
                                    tool_result = parsed

                                logger.info("message", text=f"🔍 Parsed {tool_name}: type={type(tool_result).__name__}")
                            except (ValueError, SyntaxError) as e:
                                # If parsing fails, it might contain numpy types - try string manipulation
                                logger.warning("warning", message=f"ast.literal_eval failed for {tool_name}, trying manual parse: {str(e)[:100]}")
                                # Store as-is, frontend will handle
                                tool_result = message.content

                            # Store the tool result
                            if tool_result:
                                result_data = None
                                metadata_data = {}
                                
                                # If parsing failed and we got a string, try to extract the tuple manually
                                if isinstance(tool_result, str) and tool_result.startswith("(") and ", {" in tool_result:
                                    logger.warning("warning", message=f"{tool_name} result is a string, storing as-is")
                                    result_data = tool_result
                                    metadata_data = {}
                                # Parse result and metadata if it's a tuple
                                elif isinstance(tool_result, tuple) and len(tool_result) >= 2:
                                    result_data, metadata_data = tool_result[0], tool_result[1]
                                    logger.info("success", message=f"Stored {tool_name} result: {type(result_data).__name__} with {len(metadata_data)} metadata fields")
                                elif isinstance(tool_result, dict):
                                    # Some tools return just a dict
                                    result_data = tool_result
                                    metadata_data = {}
                                else:
                                    result_data = tool_result
                                    metadata_data = {}
                                
                                # Store in latest_tool_results (backward compatibility)
                                self.latest_tool_results[tool_name] = {
                                    "result": result_data,
                                    "metadata": metadata_data
                                }
                                
                                # NEW: Append to execution history
                                execution_record = {
                                    "execution_id": str(uuid.uuid4()),
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "request_id": self.current_request_id,
                                    "tool_name": tool_name,
                                    "image_paths": self.uploaded_files.copy(),  # Snapshot of images at execution
                                    "result": result_data,
                                    "metadata": metadata_data
                                }
                                self.tool_execution_history.append(execution_record)
                                
                                logger.info("tool_execution_stored",
                                          execution_id=execution_record["execution_id"][:8],
                                          tool_name=tool_name,
                                          request_id=self.current_request_id[:8],
                                          image_count=len(self.uploaded_files))

                                # Track this tool call
                                self.last_tool_calls.append({
                                    "tool_name": tool_name,
                                    "status": "completed"
                                })

                                # Format tool response based on tool type (don't show raw data)
                                if "classifier" in tool_name.lower():
                                    tool_response = f"🔧 {tool_name}: Classification completed"
                                elif "segmentation" in tool_name.lower():
                                    tool_response = f"🔧 {tool_name}: Segmentation completed"
                                elif "report" in tool_name.lower():
                                    tool_response = f"🔧 {tool_name}: Report generated"
                                elif "visualizer" in tool_name.lower():
                                    tool_response = f"🔧 {tool_name}: Visualization created"
                                else:
                                    # For other tools, show a brief summary
                                    tool_response = f"🔧 {tool_name}: Completed"

                                # Yield tool completion event
                                tool_done_msg = f"__TOOL_DONE__{tool_name}__"
                                yield tool_done_msg

                                responses.append(tool_response)
                                yield tool_response

                            # Update display path for image visualizer
                            if tool_name == "image_visualizer" and tool_result:
                                if isinstance(tool_result, dict):
                                    self.display_file_path = tool_result.get("image_path", self.display_file_path)
                                elif isinstance(tool_result, tuple) and len(tool_result) > 0:
                                    # Extract dict from tuple
                                    result_data = tool_result[0] if isinstance(tool_result[0], dict) else {}
                                    if isinstance(result_data, dict):
                                        self.display_file_path = result_data.get("image_path", self.display_file_path)

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            yield error_msg

        # Store message history after processing
        from datetime import datetime, timezone
        if message:
            self.message_history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        if responses:
            full_response = "\n\n".join(responses)
            self.message_history.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

    def get_message_history(self) -> list:
        """Get the message history for this chat."""
        return self.message_history
    
    def get_tool_execution_history(
        self, 
        filter_by_image: Optional[str] = None,
        filter_by_request: Optional[str] = None,
        latest_only: bool = False
    ) -> list:
        """
        Get tool execution history with optional filtering.
        
        Args:
            filter_by_image: Only return executions that used this image path
            filter_by_request: Only return executions from this request_id
            latest_only: Only return the latest execution per tool
            
        Returns:
            List of execution records matching the filters
        """
        history = self.tool_execution_history
        
        # Filter by image if specified
        if filter_by_image:
            history = [
                exec_record for exec_record in history
                if filter_by_image in exec_record.get("image_paths", [])
            ]
        
        # Filter by request if specified
        if filter_by_request:
            history = [
                exec_record for exec_record in history
                if exec_record.get("request_id") == filter_by_request
            ]
        
        # If latest_only, keep only the most recent execution per tool
        if latest_only:
            latest_by_tool = {}
            for exec_record in reversed(history):  # Reverse to get latest first
                tool_name = exec_record["tool_name"]
                if tool_name not in latest_by_tool:
                    latest_by_tool[tool_name] = exec_record
            history = list(latest_by_tool.values())
        
        return history

    def clear_images(self) -> None:
        """Clear all uploaded images from this chat."""
        self.uploaded_files = []
        self.display_files = []
        self.chat_metadata["image_count"] = 0
        logger.info("images_cleared", chat=self.chat_metadata.get("name", "Unknown"))

    def get_metadata(self) -> dict:
        """Get current chat metadata."""
        return {
            **self.chat_metadata,
            "current_image_count": len(self.uploaded_files),
            "has_tool_results": len(self.latest_tool_results) > 0
        }

    def cleanup_memory(self) -> dict:
        """
        Clean up memory by removing cached results and forcing garbage collection.

        Returns:
            Dictionary with cleanup statistics
        """
        import gc

        stats = {
            "tool_results_cleared": len(self.latest_tool_results),
            "tool_calls_cleared": len(self.last_tool_calls),
            "files_retained": len(self.uploaded_files)
        }

        # Clear cached tool results
        self.latest_tool_results.clear()
        self.last_tool_calls.clear()

        # Force garbage collection
        collected = gc.collect()
        stats["objects_collected"] = collected

        logger.info("memory_cleanup", **stats)
        return stats

    def cleanup_temp_files(self) -> dict:
        """
        Clean up temporary files for this chat.

        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            "files_deleted": 0,
            "space_freed_mb": 0,
            "errors": []
        }

        files_to_delete = self.uploaded_files.copy()

        for file_path in files_to_delete:
            try:
                path = Path(file_path)
                if path.exists():
                    size_mb = path.stat().st_size / (1024 * 1024)
                    path.unlink()
                    stats["files_deleted"] += 1
                    stats["space_freed_mb"] += size_mb
            except Exception as e:
                stats["errors"].append(f"{file_path}: {str(e)}")
                logger.warning("file_cleanup_error", file=file_path, error=str(e))

        # Clear file lists after deletion
        self.uploaded_files.clear()
        self.display_files.clear()
        self.chat_metadata["image_count"] = 0

        logger.info("temp_files_cleanup", **{k: v for k, v in stats.items() if k != "errors"})
        return stats

    def get_memory_info(self) -> dict:
        """
        Get memory usage information for this chat.

        Returns:
            Dictionary with memory statistics
        """

        return {
            "uploaded_files_count": len(self.uploaded_files),
            "tool_results_count": len(self.latest_tool_results),
            "tool_calls_count": len(self.last_tool_calls),
            "thread_id": self.current_thread_id,
            "chat_name": self.chat_metadata.get("name", "Unknown")
        }

