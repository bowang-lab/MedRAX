from typing import Type, Optional, Literal
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# Assuming ChestXRayReportGeneratorTool is accessible for instantiation
# If it's in a different path, adjust the import
from medrax.tools.report_generation import ChestXRayReportGeneratorTool, ChestXRayInput
from medrax.tools.preliminary_report_utils import (
    create_doctor_preliminary_report,
    generate_patient_focused_report_placeholder, # Using placeholder for now
    parse_generated_report,
    create_patient_simplification_prompt
)

class AutomatedPreliminaryReportInput(BaseModel):
    image_path: str = Field(
        ...,
        description="Path to the radiology image file (JPG or PNG)."
    )
    report_type: Literal["doctor", "patient_prompt", "patient_placeholder"] = Field(
        default="doctor",
        description="Type of report to generate: 'doctor' for clinician-focused, "
                    "'patient_prompt' for LLM prompt for patient explanation, "
                    "or 'patient_placeholder' for a direct simplified patient explanation (uses placeholder logic)."
    )
    # Optional: Pass cache_dir and device if they can vary per call,
    # otherwise, they should be configured during tool initialization.
    # cache_dir: Optional[str] = Field(None, description="Directory for model weights.")
    # device: Optional[str] = Field(None, description="Device to run models on ('cuda' or 'cpu').")


class AutomatedPreliminaryReportTool(BaseTool):
    """
    Tool that generates automated preliminary reports from chest X-ray images.
    It can produce a detailed report for clinicians or a simplified explanation
    for patients (either as an LLM prompt or a direct placeholder explanation).
    """

    name: str = "automated_preliminary_cxr_report_generator"
    description: str = (
        "Generates an automated preliminary report from a chest X-ray image. "
        "Specify 'report_type' as 'doctor' for a clinician-focused report, "
        "'patient_prompt' to get a prompt for an LLM to explain to a patient, "
        "or 'patient_placeholder' for a direct simplified patient explanation."
    )
    args_schema: Type[BaseModel] = AutomatedPreliminaryReportInput

    # Internal tools. These should be initialized when AutomatedPreliminaryReportTool is initialized.
    # This is a simplified way; a more robust setup might pass these as constructor arguments.
    _report_generator_tool: ChestXRayReportGeneratorTool
    # Add _llm if this tool were to directly call an LLM for patient reports.
    # For now, it generates a prompt or uses a placeholder.

    def __init__(self, cache_dir: str = "/model-weights", device: Optional[str] = "cuda", **kwargs):
        super().__init__(**kwargs)
        # Initialize the core report generator tool upon instantiation
        self._report_generator_tool = ChestXRayReportGeneratorTool(cache_dir=cache_dir, device=device)

    def _run(
        self,
        image_path: str,
        report_type: Literal["doctor", "patient_prompt", "patient_placeholder"] = "doctor",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Executes the tool to generate the specified type of preliminary report."""

        # Step 1: Generate the base medical report using ChestXRayReportGeneratorTool
        # The input schema for ChestXRayReportGeneratorTool is ChestXRayInput, which just takes image_path
        base_report_output, metadata = self._report_generator_tool._run(image_path=image_path)

        if metadata.get("analysis_status") == "failed":
            return f"Failed to generate base medical report: {metadata.get('error', 'Unknown error')}"

        # Step 2: Parse the base report
        findings, impression = parse_generated_report(base_report_output)

        if findings == "Not available" and impression == "Not available":
            return "Could not parse findings or impression from the base medical report."

        # Step 3: Generate the requested report type
        if report_type == "doctor":
            return create_doctor_preliminary_report(
                raw_report_text=base_report_output, # Pass the full raw text
                image_identifier=image_path,
                report_generator_tool_name=self._report_generator_tool.name
            )
        elif report_type == "patient_prompt":
            return create_patient_simplification_prompt(
                findings_text=findings,
                impression_text=impression
            )
        elif report_type == "patient_placeholder":
            return generate_patient_focused_report_placeholder(
                findings_text=findings,
                impression_text=impression
            )
        else:
            return "Invalid report_type specified. Choose 'doctor', 'patient_prompt', or 'patient_placeholder'."

    async def _arun(
        self,
        image_path: str,
        report_type: Literal["doctor", "patient_prompt", "patient_placeholder"] = "doctor",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # For simplicity, using the synchronous _run method.
        # A true async implementation would require ChestXRayReportGeneratorTool to also be async
        # and potentially the utility functions if they involved I/O.
        return self._run(image_path=image_path, report_type=report_type, run_manager=run_manager)

if __name__ == '__main__':
    # This basic test requires models to be downloaded to the default cache_dir,
    # or cache_dir and device to be correctly set.
    # For a simple local test, ensure you have a sample image.

    # Create a dummy image file for testing if it doesn't exist
    SAMPLE_IMAGE_PATH = "sample_cxr_image.png"
    try:
        from PIL import Image
        Image.new('RGB', (512, 512), color = 'red').save(SAMPLE_IMAGE_PATH)
        print(f"Created dummy image at {SAMPLE_IMAGE_PATH} for testing.")
    except ImportError:
        print("Pillow not installed, cannot create dummy image. Test might fail if image doesn't exist.")
    except Exception as e:
        print(f"Could not create dummy image: {e}")


    # IMPORTANT: Set your actual model cache directory and device if different from defaults.
    # These environment variables can be used by the from_pretrained HuggingFace calls.
    # os.environ['HF_HOME'] = '/path/to/your/huggingface/cache'

    print("\nInitializing AutomatedPreliminaryReportTool...")
    # Ensure cache_dir points to where your HuggingFace models are downloaded or will be downloaded.
    # Adjust device if not using CUDA or if CUDA is not available.
    try:
        prelim_tool = AutomatedPreliminaryReportTool(cache_dir="hf_cache", device="cpu") # Example: use "cpu" and a local cache folder "hf_cache"
        print("Tool initialized.")

        print("\n--- Testing Doctor Report ---")
        doctor_report = prelim_tool._run(image_path=SAMPLE_IMAGE_PATH, report_type="doctor")
        print(doctor_report)

        print("\n--- Testing Patient Prompt ---")
        patient_prompt = prelim_tool._run(image_path=SAMPLE_IMAGE_PATH, report_type="patient_prompt")
        print(patient_prompt)

        print("\n--- Testing Patient Placeholder Report ---")
        patient_placeholder_report = prelim_tool._run(image_path=SAMPLE_IMAGE_PATH, report_type="patient_placeholder")
        print(patient_placeholder_report)

    except Exception as e:
        print(f"\nAn error occurred during tool testing: {e}")
        print("This might be due to model download issues, incorrect paths, or missing dependencies.")
        print("Ensure you have internet access if models need to be downloaded for the first time.")
        print("Ensure `cache_dir` is correctly set and writable.")

    # Clean up dummy image
    # import os
    # if os.path.exists(SAMPLE_IMAGE_PATH):
    #     os.remove(SAMPLE_IMAGE_PATH)
    #     print(f"Removed dummy image {SAMPLE_IMAGE_PATH}.")
