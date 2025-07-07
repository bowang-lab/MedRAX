import re
import base64
import gradio as gr
from pathlib import Path
import time
import shutil
from typing import AsyncGenerator, List, Optional, Tuple
from gradio import ChatMessage
import gradio as gr # Ensure gr is imported if not already for gr.State

# Localization support
from medrax.localization import get_string, get_current_language_options, DEFAULT_LANG


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
        self, message: str, display_image: str, history: List[dict], lang_code: str = DEFAULT_LANG
    ) -> Tuple[List[dict], gr.Textbox]:
        """
        Add a new message to the chat history. Now language-aware for notifications.

        Args:
            message (str): Text message to add
            display_image (str): Path to image being displayed
            history (List[dict]): Current chat history
            lang_code (str): Current language code for localization.

        Returns:
            Tuple[List[dict], gr.Textbox]: Updated history and textbox component
        """
        image_path = self.original_file_path or display_image
        if image_path is not None:
            history.append({"role": "user", "content": {"path": image_path}}) # For agent to get path

        if self.last_uploaded_filename_for_chat_display:
            notification_text = get_string(
                lang_code,
                'uploaded_file_notification',
                filename=self.last_uploaded_filename_for_chat_display
            )
            history.append({
                "role": "user", # Or a different role like 'system' if Gradio supports styling it differently
                "content": notification_text
            })
            self.last_uploaded_filename_for_chat_display = None # Reset after displaying

        if message is not None and message.strip() != "": # Add user's actual typed message
            history.append({"role": "user", "content": message})

        # If only a file was uploaded and no text message, the history will contain the path and upload notification.
        # If there's also a text message, it's appended after.

        return history, gr.Textbox(value="", interactive=False) # Clear textbox, set to non-interactive

    async def process_message(
        self, message: str, display_image: Optional[str], chat_history: List[ChatMessage], lang_code: str = DEFAULT_LANG
    ) -> AsyncGenerator[Tuple[List[ChatMessage], Optional[str], str], None]:
        """
        Process a message and generate responses. Now language-aware for tool outputs.

        Args:
            message (str): User message to process
            display_image (Optional[str]): Path to currently displayed image
            chat_history (List[ChatMessage]): Current chat history
            lang_code (str): Current language code for localization.

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
                                        metadata={"title": get_string(lang_code, 'visualized_image_title')}
                                    )
                                )
                                handled_specifically = True

                            elif tool_name == "Medical Imaging Series Analyzer" and isinstance(tool_result_obj, dict):
                                formatted_series_info = get_string(lang_code, 'results_title_prefix', tool_name=tool_name) + "\n"
                                if tool_result_obj.get("status"):
                                    formatted_series_info += get_string(lang_code, 'status_label', status=tool_result_obj['status']) + "\n"
                                if tool_result_obj.get("number_of_slices") is not None:
                                    formatted_series_info += get_string(lang_code, 'slices_found_label', count=tool_result_obj['number_of_slices']) + "\n"

                                series_meta = tool_result_obj.get("series_metadata", {})
                                if series_meta:
                                    formatted_series_info += get_string(lang_code, 'key_series_metadata_label') + "\n"
                                    for k, v_val in series_meta.items():
                                        if v_val and v_val != "N/A" and k in ["Modality", "SeriesDescription", "BodyPartExamined", "StudyDescription", "PatientID"]:
                                            formatted_series_info += f"  - {k}: {v_val}\n" # Keys are not localized here, but values are dynamic.

                                rep_slice_meta = tool_result_obj.get("representative_slice_details", {})
                                if rep_slice_meta and rep_slice_meta.get("error") is None :
                                     formatted_series_info += get_string(lang_code, 'representative_slice_info_label') + "\n"
                                     for k, v_val in rep_slice_meta.items():
                                         if v_val and v_val != "N/A" and k in ["InstanceNumber", "SliceThickness", "ImagePositionPatient"]:
                                             formatted_series_info += f"  - {k}: {v_val}\n"

                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content=formatted_series_info.strip(),
                                        metadata={"title": get_string(lang_code, 'imaging_series_analysis_title')}
                                    )
                                )

                                if tool_result_obj.get("representative_image_path"):
                                    self.display_file_path = tool_result_obj["representative_image_path"]
                                    chat_history.append(
                                        ChatMessage(
                                            role="assistant",
                                            content={"path": self.display_file_path},
                                            metadata={"title": get_string(lang_code, 'representative_slice_title')}
                                        )
                                    )
                                handled_specifically = True

                            elif tool_name == "Ophthalmic Imaging Analyzer" and isinstance(tool_result_obj, dict):
                                formatted_ophth_info = get_string(lang_code, 'results_title_prefix', tool_name=tool_name) + "\n"
                                if tool_result_obj.get("status"):
                                    formatted_ophth_info += get_string(lang_code, 'status_label', status=tool_result_obj['status']) + "\n"
                                if tool_result_obj.get("image_type"):
                                    formatted_ophth_info += get_string(lang_code, 'image_type_label', type=tool_result_obj['image_type']) + "\n"

                                metadata_dict = tool_result_obj.get("metadata", {})
                                series_metadata_dict = tool_result_obj.get("series_metadata", {})
                                rep_slice_details_dict = tool_result_obj.get("representative_slice_details", {})

                                if tool_result_obj.get("number_of_images_in_series") is not None:
                                    formatted_ophth_info += get_string(lang_code, 'images_in_series_label', count=tool_result_obj['number_of_images_in_series']) + "\n"

                                if series_metadata_dict:
                                    formatted_ophth_info += get_string(lang_code, 'key_series_metadata_label') + "\n"
                                    for k, v in series_metadata_dict.items():
                                        if v and v != "N/A" and k in ["Modality", "Laterality", "SeriesDescription", "PatientID"]:
                                            formatted_ophth_info += f"  - {k}: {v}\n"
                                if rep_slice_details_dict and not rep_slice_details_dict.get("error"):
                                    formatted_ophth_info += get_string(lang_code, 'representative_slice_info_label') + "\n" # Re-using key
                                    for k, v in rep_slice_details_dict.items():
                                         if v and v != "N/A" and k in ["Laterality", "ImageType", "ImageComments", "InstanceNumber"]:
                                            formatted_ophth_info += f"  - {k}: {v}\n"
                                if metadata_dict:
                                    formatted_ophth_info += get_string(lang_code, 'file_metadata_label') + "\n"
                                    for k, v in metadata_dict.items():
                                        if v and v != "N/A":
                                            formatted_ophth_info += f"  - {k}: {v}\n"

                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content=formatted_ophth_info.strip(),
                                        metadata={"title": get_string(lang_code, 'ophthalmic_analysis_title')}
                                    )
                                )

                                display_image_path = tool_result_obj.get("representative_image_path") or tool_result_obj.get("image_path")
                                if display_image_path:
                                    self.display_file_path = display_image_path
                                    chat_history.append(
                                        ChatMessage(
                                            role="assistant",
                                            content={"path": self.display_file_path},
                                            metadata={"title": get_string(lang_code, 'ophthalmic_image_frame_title')}
                                        )
                                    )

                                extracted_frames = tool_result_obj.get("extracted_frames_paths", [])
                                if extracted_frames and len(extracted_frames) > 1:
                                    frames_info = get_string(lang_code, 'additional_frames_label', count=len(extracted_frames)-1) + "\n"
                                    for frame_p in extracted_frames[1:3]:
                                        frames_info += f"  - {Path(frame_p).name}\n"
                                    if len(extracted_frames) > 3:
                                        frames_info += "  - ... and more.\n"
                                    chat_history.append(
                                        ChatMessage(
                                            role="assistant",
                                            content=frames_info.strip(),
                                            metadata={"title": get_string(lang_code, 'extracted_video_frames_title')}
                                        )
                                    )
                                handled_specifically = True

                            # Generic handling for other tools
                            if not handled_specifically and tool_result_obj:
                                # Localize the generic tool output title
                                generic_title = get_string(lang_code, 'tool_output_title', tool_name=tool_name)
                                metadata = {"title": generic_title}
                                formatted_result_str = " ".join(
                                    line.strip() for line in str(tool_result_obj).splitlines()
                                ).strip()
                                metadata["description"] = formatted_result_str # Description not directly visible in ChatMessage
                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content=formatted_result_str, # The content itself is tool's raw output, not localized
                                        metadata=metadata,
                                    )
                                )

                            yield chat_history, self.display_file_path, ""

        except Exception as e:
            # Localize the general error message title
            error_title = get_string(lang_code, 'error_message_title')
            chat_history.append(
                ChatMessage(
                    role="assistant", content=f"❌ Error: {str(e)}", metadata={"title": error_title}
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
        current_language_state = gr.State(DEFAULT_LANG)

        with gr.Row():
            lang_selector = gr.Radio(
                choices=get_current_language_options(),
                value=DEFAULT_LANG,
                label="Language / Dil", # Static label for selector itself
                interactive=True,
                elem_id="lang_selector_radio" # For easier targeting if needed
            )

        with gr.Column():
            app_title_md = gr.Markdown(
                value=f"# {get_string(DEFAULT_LANG, 'app_title')}\n{get_string(DEFAULT_LANG, 'app_subtitle')}"
            )

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        [],
                        label=get_string(DEFAULT_LANG, 'agent_label'),
                        height=800,
                        container=True,
                        show_label=True,
                        elem_classes="chat-box",
                        type="messages",
                        avatar_images=(
                            None,
                            "assets/medrax_logo.jpg",
                        ),
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            txt = gr.Textbox(
                                show_label=False,
                                placeholder=get_string(DEFAULT_LANG, 'chat_placeholder'),
                                container=False,
                            )

                with gr.Column(scale=3):
                    image_display = gr.Image(
                        label=get_string(DEFAULT_LANG, 'image_label'),
                        type="filepath",
                        height=700,
                        container=True
                    )
                    with gr.Row():
                        upload_button = gr.UploadButton(
                            get_string(DEFAULT_LANG, 'upload_xray_btn'),
                            file_types=["image"],
                        )
                        dicom_upload = gr.UploadButton(
                            get_string(DEFAULT_LANG, 'upload_dicom_btn'),
                            file_types=["file"],
                        )
                        blood_test_upload = gr.UploadButton(
                            get_string(DEFAULT_LANG, 'upload_blood_test_btn'),
                            file_types=[".pdf", ".csv"],
                        )
                        mr_ct_series_upload = gr.UploadButton(
                            get_string(DEFAULT_LANG, 'upload_mr_ct_btn'),
                            file_types=[".zip"],
                        )
                        ophthalmic_upload = gr.UploadButton(
                            get_string(DEFAULT_LANG, 'upload_ophthalmic_btn'),
                            file_types=[".zip", ".dcm", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".mp4", ".avi"],
                        )
                    with gr.Row():
                        clear_btn = gr.Button(get_string(DEFAULT_LANG, 'clear_chat_btn'))
                        new_thread_btn = gr.Button(get_string(DEFAULT_LANG, 'new_thread_btn'))

        # --- Localization Update Function ---
        def update_ui_language_elements(lang_code):
            # This function returns a list/tuple of gr.update() calls
            # The order must match the 'outputs' list in lang_selector.change()
            return (
                gr.Markdown(value=f"# {get_string(lang_code, 'app_title')}\n{get_string(lang_code, 'app_subtitle')}"),
                gr.Chatbot(label=get_string(lang_code, 'agent_label')),
                gr.Textbox(placeholder=get_string(lang_code, 'chat_placeholder')),
                gr.Image(label=get_string(lang_code, 'image_label')),
                gr.UploadButton(value=get_string(lang_code, 'upload_xray_btn')),
                gr.UploadButton(value=get_string(lang_code, 'upload_dicom_btn')),
                gr.UploadButton(value=get_string(lang_code, 'upload_blood_test_btn')),
                gr.UploadButton(value=get_string(lang_code, 'upload_mr_ct_btn')),
                gr.UploadButton(value=get_string(lang_code, 'upload_ophthalmic_btn')),
                gr.Button(value=get_string(lang_code, 'clear_chat_btn')),
                gr.Button(value=get_string(lang_code, 'new_thread_btn')),
                lang_code # To update current_language_state
            )

        components_to_localize = [
            app_title_md, chatbot, txt, image_display,
            upload_button, dicom_upload, blood_test_upload,
            mr_ct_series_upload, ophthalmic_upload,
            clear_btn, new_thread_btn,
            current_language_state # This state also needs to be an output to be updated
        ]

        lang_selector.change(
            fn=update_ui_language_elements,
            inputs=[lang_selector], # lang_selector value is implicitly the lang_code
            outputs=components_to_localize
        )

        # Event handlers (original structure, will need adaptation later for full i18n)
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
