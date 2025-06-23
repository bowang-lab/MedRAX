import re
import base64
import gradio as gr
from pathlib import Path
import time
import shutil
from typing import AsyncGenerator, List, Optional, Tuple
from gradio import ChatMessage


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
        self.last_uploaded_filename_for_chat_display = None # To show in chat

    def handle_upload(self, file_path: str) -> str:
        """
        Handle new file upload and set appropriate paths.
        It also stores the original filename for potential display in chat.

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
        self.last_uploaded_filename_for_chat_display = source.name # Store original name

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
            history.append({"role": "user", "content": {"path": image_path}}) # For agent to get path

        # Display "Uploaded: filename" in chat if a new file was just handled by handle_upload
        if self.last_uploaded_filename_for_chat_display:
            history.append({
                "role": "user",
                "content": f"(System notification: User uploaded '{self.last_uploaded_filename_for_chat_display}')"
            })
            self.last_uploaded_filename_for_chat_display = None # Reset after displaying

        if message is not None and message.strip() != "": # Add user's actual typed message
            history.append({"role": "user", "content": message})

        # If only a file was uploaded and no text message, the history will contain the path and upload notification.
        # If there's also a text message, it's appended after.

        return history, gr.Textbox(value="", interactive=False) # Clear textbox, set to non-interactive

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
            for event in self.agent.workflow.stream(
                {"messages": messages}, {"configurable": {"thread_id": self.current_thread_id}}
            ):
                if isinstance(event, dict):
                    if "process" in event:
                        content = event["process"]["messages"][-1].content
                        if content:
                            content = re.sub(r"temp/[^\s]*", "", content)
                            chat_history.append(ChatMessage(role="assistant", content=content))
                            yield chat_history, self.display_file_path, ""

                    elif "execute" in event:
                        for message in event["execute"]["messages"]:
                            tool_name = message.name
                            tool_result_obj = eval(message.content)[0] # The actual Python object

                            handled_specifically = False

                            if tool_name == "image_visualizer" and isinstance(tool_result_obj, dict) and "image_path" in tool_result_obj:
                                self.display_file_path = tool_result_obj["image_path"]
                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content={"path": self.display_file_path},
                                        metadata={"title": "Visualized Image"}
                                    )
                                )
                                handled_specifically = True

                            elif tool_name == "Medical Imaging Series Analyzer" and isinstance(tool_result_obj, dict):
                                formatted_series_info = f"**{tool_name} Results:**\n"
                                if tool_result_obj.get("status"):
                                    formatted_series_info += f"- Status: {tool_result_obj['status']}\n"
                                if tool_result_obj.get("number_of_slices") is not None:
                                    formatted_series_info += f"- Slices Found: {tool_result_obj['number_of_slices']}\n"

                                series_meta = tool_result_obj.get("series_metadata", {})
                                if series_meta:
                                    formatted_series_info += "- Key Series Metadata:\n"
                                    for k, v_val in series_meta.items():
                                        if v_val and v_val != "N/A" and k in ["Modality", "SeriesDescription", "BodyPartExamined", "StudyDescription", "PatientID"]:
                                            formatted_series_info += f"  - {k}: {v_val}\n"

                                rep_slice_meta = tool_result_obj.get("representative_slice_details", {})
                                if rep_slice_meta and rep_slice_meta.get("error") is None :
                                     formatted_series_info += "- Representative Slice Info:\n"
                                     for k, v_val in rep_slice_meta.items():
                                         if v_val and v_val != "N/A" and k in ["InstanceNumber", "SliceThickness", "ImagePositionPatient"]:
                                             formatted_series_info += f"  - {k}: {v_val}\n"

                                # Add the textual summary to chat
                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content=formatted_series_info.strip(),
                                        metadata={"title": "Imaging Series Analysis"}
                                    )
                                )

                                # Handle image display if path is present
                                if tool_result_obj.get("representative_image_path"):
                                    self.display_file_path = tool_result_obj["representative_image_path"]
                                    chat_history.append(
                                        ChatMessage(
                                            role="assistant",
                                            content={"path": self.display_file_path},
                                            metadata={"title": "Representative Slice"}
                                        )
                                    )
                                    # Add a small note about the image in the main text message if not already covered
                                    # This is now implicitly handled by the image appearing.
                                elif tool_result_obj.get("status", "").startswith("completed_with_errors") or tool_result_obj.get("error"):
                                    # If image path is missing but there were errors, mention it.
                                    # The error from tool_result_obj.get("error") or rep_slice_meta.get("png_error")
                                    # could be added to formatted_series_info earlier.
                                    pass # Error details should be part of formatted_series_info

                                handled_specifically = True

                            elif tool_name == "Ophthalmic Imaging Analyzer" and isinstance(tool_result_obj, dict):
                                formatted_ophth_info = f"**{tool_name} Results:**\n"
                                if tool_result_obj.get("status"):
                                    formatted_ophth_info += f"- Status: {tool_result_obj['status']}\n"
                                if tool_result_obj.get("image_type"):
                                    formatted_ophth_info += f"- Image Type: {tool_result_obj['image_type']}\n"

                                metadata = tool_result_obj.get("metadata", {}) # For single image or video
                                series_metadata = tool_result_obj.get("series_metadata", {}) # For DICOM series
                                rep_slice_details = tool_result_obj.get("representative_slice_details", {})

                                if tool_result_obj.get("number_of_images_in_series") is not None:
                                    formatted_ophth_info += f"- Images in Series: {tool_result_obj['number_of_images_in_series']}\n"

                                if series_metadata:
                                    formatted_ophth_info += "- Key Series Metadata:\n"
                                    for k, v in series_metadata.items():
                                        if v and v != "N/A" and k in ["Modality", "Laterality", "SeriesDescription", "PatientID"]:
                                            formatted_ophth_info += f"  - {k}: {v}\n"
                                if rep_slice_details and not rep_slice_details.get("error"):
                                    formatted_ophth_info += "- Representative Slice/Image Details:\n"
                                    for k, v in rep_slice_details.items():
                                         if v and v != "N/A" and k in ["Laterality", "ImageType", "ImageComments", "InstanceNumber"]:
                                            formatted_ophth_info += f"  - {k}: {v}\n"
                                if metadata: # For single images or video
                                    formatted_ophth_info += "- File Metadata:\n"
                                    for k, v in metadata.items():
                                        if v and v != "N/A":
                                            formatted_ophth_info += f"  - {k}: {v}\n"

                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content=formatted_ophth_info.strip(),
                                        metadata={"title": "Ophthalmic Analysis"}
                                    )
                                )

                                # Handle image display
                                display_image_path = tool_result_obj.get("representative_image_path") or tool_result_obj.get("image_path")
                                if display_image_path:
                                    self.display_file_path = display_image_path
                                    chat_history.append(
                                        ChatMessage(
                                            role="assistant",
                                            content={"path": self.display_file_path},
                                            metadata={"title": "Ophthalmic Image/Frame"}
                                        )
                                    )

                                # Handle video frames if any (just list them for now)
                                extracted_frames = tool_result_obj.get("extracted_frames_paths", [])
                                if extracted_frames and len(extracted_frames) > 1: # Already showed the first one
                                    frames_info = f"- Additional Extracted Frames ({len(extracted_frames)-1}):\n"
                                    for frame_p in extracted_frames[1:3]: # Show next 2 paths as example
                                        frames_info += f"  - {Path(frame_p).name}\n"
                                    if len(extracted_frames) > 3:
                                        frames_info += "  - ... and more.\n"
                                    chat_history.append(
                                        ChatMessage(
                                            role="assistant",
                                            content=frames_info.strip(),
                                            metadata={"title": "Extracted Video Frames"}
                                        )
                                    )
                                handled_specifically = True


                            # Generic handling for other tools or if specific handling didn't occur
                            if not handled_specifically and tool_result_obj:
                                metadata = {"title": f"⚙️ Tool Output: {tool_name}"}
                                # Convert the raw tool_result_obj to a flat string for display
                                formatted_result_str = " ".join(
                                    line.strip() for line in str(tool_result_obj).splitlines()
                                ).strip()
                                metadata["description"] = formatted_result_str
                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content=formatted_result_str,
                                        metadata=metadata,
                                    )
                                )

                            yield chat_history, self.display_file_path, ""

        except Exception as e:
            chat_history.append(
                ChatMessage(
                    role="assistant", content=f"❌ Error: {str(e)}", metadata={"title": "Error"}
                )
            )
            yield chat_history, self.display_file_path


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
            # 🏥 MedRAX
            Medical Reasoning Agent for Chest X-ray
            """
            )

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        [],
                        height=800,
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
                                placeholder="Ask about the uploaded file or type your query...",
                                container=False,
                            )

                with gr.Column(scale=3):
                    image_display = gr.Image(
                        label="Image", type="filepath", height=700, container=True
                    )
                    with gr.Row():
                        upload_button = gr.UploadButton(
                            "📎 Upload X-Ray",
                            file_types=["image"],
                        )
                        dicom_upload = gr.UploadButton(
                            "📄 Upload DICOM",
                            file_types=["file"], # Accepts any file, filtered by .dcm in handle_upload
                        )
                        blood_test_upload = gr.UploadButton( # New button
                            "🩸 Upload Blood Test (.pdf, .csv)",
                            file_types=[".pdf", ".csv"],
                        )
                        mr_ct_series_upload = gr.UploadButton( # New button for MR/CT series
                            "🧲 Upload MR/CT Series (ZIP)",
                            file_types=[".zip"],
                        )
                        ophthalmic_upload = gr.UploadButton( # New button for Ophthalmic images/series
                            "👁️ Upload Eye FFA/OCT",
                            file_types=[".zip", ".dcm", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".mp4", ".avi"],
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

        blood_test_upload.upload(handle_file_upload, inputs=blood_test_upload, outputs=image_display) # Added event handler

        mr_ct_series_upload.upload(handle_file_upload, inputs=mr_ct_series_upload, outputs=image_display) # Added event handler for series

        ophthalmic_upload.upload(handle_file_upload, inputs=ophthalmic_upload, outputs=image_display) # Added event handler for ophthalmic

        clear_btn.click(clear_chat, outputs=[chatbot, image_display])
        new_thread_btn.click(new_thread, outputs=[chatbot, image_display])

    return demo
