import re
import base64
import json
import ast
import gradio as gr
from pathlib import Path
import time
import shutil
from typing import AsyncGenerator, List, Optional, Tuple
from gradio import ChatMessage
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage


class ChatInterface:
    """
    A chat interface for interacting with a medical AI agent through Gradio.

    Handles file uploads, message processing, and chat history management.
    Supports both regular image files and DICOM medical imaging files.
    """

    def __init__(self, agent, tools_dict):
        """
        Initialize the chat interface.

        Args:
            agent: The medical AI agent to handle requests
            tools_dict (dict): Dictionary of available tools for image processing
        """
        self.agent = agent
        self.tools_dict = tools_dict
        self.upload_dir = Path("temp")
        self.upload_dir.mkdir(exist_ok=True)
        self.current_thread_id = None
        # Separate storage for original and display paths
        self.original_file_path = None  # For LLM (.dcm or other)
        self.display_file_path = None  # For UI (always viewable format)
        self.pending_tool_calls = {}

    def handle_upload(self, file_path: str) -> str:
        """
        Handle new file upload and set appropriate paths.

        Args:
            file_path (str): Path to the uploaded file

        Returns:
            str: Display path for UI, or None if no file uploaded
        """
        if not file_path:
            return None

        source = Path(file_path)
        timestamp = int(time.time())

        # Save original file with proper suffix
        suffix = source.suffix.lower()
        saved_path = self.upload_dir / f"upload_{timestamp}{suffix}"
        shutil.copy2(file_path, saved_path)  # Use file_path directly instead of source
        self.original_file_path = str(saved_path)

        # Handle DICOM conversion for display only
        if suffix == ".dcm":
            output, _ = self.tools_dict["DicomProcessorTool"]._run(str(saved_path))
            self.display_file_path = output["image_path"]
        else:
            self.display_file_path = str(saved_path)

        return self.display_file_path

    def add_message(
        self, message: str, display_image: str, history: List[dict]
    ) -> Tuple[List[dict], gr.Textbox]:
        """
        Add a new message to the chat history.

        Args:
            message (str): Text message to add
            display_image (str): Path to image being displayed
            history (List[dict]): Current chat history

        Returns:
            Tuple[List[dict], gr.Textbox]: Updated history and textbox component
        """
        image_path = self.original_file_path or display_image
        if image_path is not None:
            history.append({"role": "user", "content": {"path": image_path}})
        if message is not None:
            history.append({"role": "user", "content": message})
        return history, gr.Textbox(value=message, interactive=False)

    async def process_message(
        self, message: str, display_image: Optional[str], chat_history: List[ChatMessage]
    ) -> AsyncGenerator[Tuple[List[ChatMessage], Optional[str], str], None]:
        """
        Process a message and generate responses.

        Args:
            message (str): User message to process
            display_image (Optional[str]): Path to currently displayed image
            chat_history (List[ChatMessage]): Current chat history

        Yields:
            Tuple[List[ChatMessage], Optional[str], str]: Updated chat history, display path, and empty string
        """
        chat_history = chat_history or []

        # Initialize thread if needed
        if not self.current_thread_id:
            self.current_thread_id = str(time.time())

        messages = []
        image_path = self.original_file_path or display_image

        if image_path is not None:
            # Send path for tools
            messages.append({"role": "user", "content": f"image_path: {image_path}"})

            # Load and encode image for multimodal
            with open(image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        }
                    ],
                }
            )

        if message is not None:
            messages.append({"role": "user", "content": [{"type": "text", "text": message}]})

        try:
            accumulated_content = ""
            final_message = None

            for chunk in self.agent.workflow.stream(
                {"messages": messages},
                {"configurable": {"thread_id": self.current_thread_id}},
                stream_mode="updates",
            ):
                if not isinstance(chunk, dict):
                    continue

                for node_name, node_output in chunk.items():
                    if "messages" not in node_output:
                        continue

                    for msg in node_output["messages"]:
                        if isinstance(msg, AIMessageChunk) and msg.content:
                            accumulated_content += msg.content
                            if final_message is None:
                                final_message = ChatMessage(
                                    role="assistant", content=accumulated_content
                                )
                                chat_history.append(final_message)
                            else:
                                final_message.content = accumulated_content
                            yield chat_history, self.display_file_path, ""

                        elif isinstance(msg, AIMessage):
                            if msg.content:
                                final_content = re.sub(r"temp/[^\s]*", "", msg.content).strip()
                                if final_message:
                                    final_message.content = final_content
                                else:
                                    chat_history.append(
                                        ChatMessage(role="assistant", content=final_content)
                                    )
                                yield chat_history, self.display_file_path, ""

                            if msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    self.pending_tool_calls[tool_call["id"]] = {
                                        "name": tool_call["name"],
                                        "args": tool_call["args"],
                                    }

                            final_message = None
                            accumulated_content = ""

                        elif isinstance(msg, ToolMessage):
                            tool_call_id = msg.tool_call_id
                            if tool_call_id in self.pending_tool_calls:
                                pending_call = self.pending_tool_calls.pop(tool_call_id)
                                tool_name = pending_call["name"]
                                tool_args = pending_call["args"]

                                try:
                                    tool_output_json = json.loads(msg.content)
                                    tool_output_str = json.dumps(tool_output_json, indent=2)
                                except (json.JSONDecodeError, TypeError):
                                    tool_output_str = str(msg.content)

                                tool_args_str = json.dumps(tool_args, indent=2)

                                description = f"**Input:**\n```json\n{tool_args_str}\n```\n\n**Output:**\n```json\n{tool_output_str}\n```"

                                metadata = {
                                    "title": f"⚒️ Tool: {tool_name}",
                                    "description": description,
                                    "status": "done",
                                }
                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content=description,
                                        metadata=metadata,
                                    )
                                )
                                yield chat_history, self.display_file_path, ""

                                if tool_name == "image_visualizer":
                                    try:
                                        result = json.loads(msg.content)
                                        if isinstance(result, dict) and "image_path" in result:
                                            self.display_file_path = result["image_path"]
                                            chat_history.append(
                                                ChatMessage(
                                                    role="assistant",
                                                    content={"path": self.display_file_path},
                                                )
                                            )
                                            yield chat_history, self.display_file_path, ""
                                    except (json.JSONDecodeError, TypeError):
                                        pass

        except Exception as e:
            chat_history.append(
                ChatMessage(
                    role="assistant", content=f"❌ Error: {str(e)}", metadata={"title": "Error"}
                )
            )
            yield chat_history, self.display_file_path, ""


def create_demo(agent, tools_dict):
    """
    Create a Gradio demo interface for the medical AI agent.

    Args:
        agent: The medical AI agent to handle requests
        tools_dict (dict): Dictionary of available tools for image processing

    Returns:
        gr.Blocks: Gradio Blocks interface
    """
    interface = ChatInterface(agent, tools_dict)

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Column():
            gr.Markdown(
                """
            # 🏥 MedRAX-2
            Medical Reasoning Agent for Chest X-ray
            """
            )

            with gr.Row():
                with gr.Column(scale=5):
                    chatbot = gr.Chatbot(
                        [],
                        height=1000,
                        container=True,
                        show_label=True,
                        elem_classes="chat-box",
                        type="messages",
                        label="Agent",
                        avatar_images=(
                            None,
                            "assets/medrax_logo.jpg",
                        ),
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            txt = gr.Textbox(
                                show_label=False,
                                placeholder="Ask about the X-ray...",
                                container=False,
                            )

                with gr.Column(scale=3):
                    image_display = gr.Image(
                        label="Image", type="filepath", height=600, container=True
                    )
                    with gr.Row():
                        upload_button = gr.UploadButton(
                            "📎 Upload X-Ray",
                            file_types=["image"],
                        )
                        dicom_upload = gr.UploadButton(
                            "📄 Upload DICOM",
                            file_types=["file"],
                        )
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat")
                        new_thread_btn = gr.Button("New Thread")

        # Event handlers
        def clear_chat():
            interface.original_file_path = None
            interface.display_file_path = None
            return [], None

        def new_thread():
            interface.current_thread_id = str(time.time())
            return [], interface.display_file_path

        def handle_file_upload(file):
            return interface.handle_upload(file.name)

        chat_msg = txt.submit(
            interface.add_message, inputs=[txt, image_display, chatbot], outputs=[chatbot, txt]
        )
        bot_msg = chat_msg.then(
            interface.process_message,
            inputs=[txt, image_display, chatbot],
            outputs=[chatbot, image_display, txt],
        )
        bot_msg.then(lambda: gr.Textbox(interactive=True), None, [txt])

        upload_button.upload(handle_file_upload, inputs=upload_button, outputs=image_display)

        dicom_upload.upload(handle_file_upload, inputs=dicom_upload, outputs=image_display)

        clear_btn.click(clear_chat, outputs=[chatbot, image_display])
        new_thread_btn.click(new_thread, outputs=[chatbot, image_display])

    return demo
