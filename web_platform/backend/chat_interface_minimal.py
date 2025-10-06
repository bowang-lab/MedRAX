"""
Minimal ChatInterface for web backend - without Gradio dependencies
"""

import time
import shutil
import base64
import numpy as np
import ast
import re
from pathlib import Path
from typing import Optional, AsyncGenerator, List, Tuple

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

class MinimalChatInterface:
    """
    A minimal chat interface for web backend that mimics the original ChatInterface
    without Gradio dependencies.
    """
    
    def __init__(self, agent, tools_dict):
        """Initialize the minimal chat interface."""
        self.agent = agent
        self.tools_dict = tools_dict
        self.upload_dir = Path("temp")
        self.upload_dir.mkdir(exist_ok=True)
        self.current_thread_id = None
        self.uploaded_files = []  # List of uploaded file paths
        self.display_files = []   # List of display file paths
        self.display_file_path = None  # For compatibility with image visualizer
        self.latest_tool_results = {}  # Store latest tool execution results
        self.last_tool_calls = []  # Track tool calls from last message
    
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

        # Handle DICOM conversion for display only (if tool available)
        if suffix == ".dcm" and "DicomProcessorTool" in self.tools_dict:
            try:
                output, _ = self.tools_dict["DicomProcessorTool"]._run(str(saved_path))
                display_path = output["image_path"]
            except Exception as e:
                logger.info("message", text=f"DICOM processing failed: {e}")
                display_path = str(saved_path)
        else:
            display_path = str(saved_path)
        
        # Add to display files list
        self.display_files.append(display_path)
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
        # Initialize thread if needed
        if not self.current_thread_id:
            self.current_thread_id = str(time.time())

        messages = []
        
        # Handle multiple images
        if self.uploaded_files:
            # Send all image paths for tools
            image_paths_str = ", ".join(self.uploaded_files)
            messages.append({"role": "user", "content": f"image_paths: {image_paths_str}"})

            # Load and encode all images for multimodal
            for image_path in self.uploaded_files:
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
                except Exception as e:
                    logger.info("message", text=f"Error processing image {image_path}: {e}")

        if message is not None:
            messages.append({"role": "user", "content": [{"type": "text", "text": message}]})

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
                                # If parsing failed and we got a string, try to extract the tuple manually
                                if isinstance(tool_result, str) and tool_result.startswith("(") and ", {" in tool_result:
                                    logger.warning("warning", message=f"{tool_name} result is a string, storing as-is")
                                    # Store the string as result - frontend will show it for debugging
                                    self.latest_tool_results[tool_name] = {
                                        "result": tool_result,
                                        "metadata": {}
                                    }
                                # Parse result and metadata if it's a tuple
                                elif isinstance(tool_result, tuple) and len(tool_result) >= 2:
                                    result_data, metadata = tool_result[0], tool_result[1]
                                    self.latest_tool_results[tool_name] = {
                                        "result": result_data,
                                        "metadata": metadata
                                    }
                                    logger.info("success", message=f"Stored {tool_name} result: {type(result_data).__name__} with {len(metadata)} metadata fields")
                                elif isinstance(tool_result, dict):
                                    # Some tools return just a dict
                                    self.latest_tool_results[tool_name] = {
                                        "result": tool_result,
                                        "metadata": {}
                                    }
                                else:
                                    self.latest_tool_results[tool_name] = {
                                        "result": tool_result,
                                        "metadata": {}
                                    }
                                
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

