"""Tool for assisting doctors with diagnosis, examinations, and prescriptions."""

from typing import Type, Optional, List
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field


class AssistantDoctorInput(BaseModel):
    symptoms: Optional[str] = Field(description="Patient's reported symptoms.")
    medical_history: Optional[str] = Field(description="Patient's relevant medical history.")
    image_path: Optional[str] = Field(description="Path to the relevant medical image (e.g., Chest X-ray).")
    current_diagnosis: Optional[str] = Field(description="Current or suspected diagnosis, if any.")
    request_type: str = Field(description="Type of assistance requested: 'diagnosis', 'examination', or 'medication'.")
    # Future fields for more detailed medication prescription:
    # patient_age: Optional[int] = Field(description="Patient's age.")
    # patient_weight: Optional[float] = Field(description="Patient's weight in kg.")
    # allergies: Optional[List[str]] = Field(description="List of known patient allergies.")


class AssistantDoctorTool(BaseTool):
    name: str = "AssistantDoctorTool"
    description: str = (
        "Provides assistance to doctors with diagnosis, suggesting examinations, or recommending medications. "
        "Input should be a JSON object with fields: 'symptoms', 'medical_history', 'image_path' (optional), "
        "'current_diagnosis' (optional), and 'request_type' ('diagnosis', 'examination', or 'medication')."
    )
    args_schema: Type[BaseModel] = AssistantDoctorInput

    # This tool will be a wrapper around the main MedRAX agent's LLM.
    # It won't call other tools directly but will formulate a prompt
    # for the LLM to generate the desired output.
    # The actual LLM call will be handled by the Agent class when this tool is invoked.

    def _run(
        self,
        symptoms: Optional[str] = None,
        medical_history: Optional[str] = None,
        image_path: Optional[str] = None, # Will be used by the agent to provide context to LLM
        current_diagnosis: Optional[str] = None,
        request_type: str = None,
        # patient_age: Optional[int] = None,
        # patient_weight: Optional[float] = None,
        # allergies: Optional[List[str]] = None,
    ) -> str:
        """Uses the input parameters to generate a prompt for the LLM."""
        if not request_type:
            return "Error: 'request_type' is a required field and must be one of 'diagnosis', 'examination', or 'medication'."

        prompt_lines = ["An AI assistant is helping a doctor."]

        if symptoms:
            prompt_lines.append(f"The patient reports the following symptoms: {symptoms}")
        if medical_history:
            prompt_lines.append(f"The patient has the following medical history: {medical_history}")
        if image_path:
            # The agent will handle image loading and provide it to the multimodal LLM.
            # This tool just indicates that an image is relevant.
            prompt_lines.append(f"A medical image at '{image_path}' is provided and should be considered.")

        if request_type == "diagnosis":
            prompt_lines.append("Based on the provided information, please suggest a list of differential diagnoses, ordered from most to least likely. For each, briefly explain your reasoning.")
        elif request_type == "examination":
            if current_diagnosis:
                prompt_lines.append(f"The current working diagnosis is: {current_diagnosis}.")
            prompt_lines.append("Suggest a list of relevant examinations or tests to confirm or refine the diagnosis. For each, explain why it's recommended.")
        elif request_type == "medication":
            if not current_diagnosis:
                return "Error: 'current_diagnosis' is required when 'request_type' is 'medication'."
            prompt_lines.append(f"The confirmed diagnosis is: {current_diagnosis}.")
            # Basic medication prompt for now.
            # Future enhancements could include patient_age, patient_weight, allergies for more specific recommendations.
            prompt_lines.append("Suggest appropriate medications, including dosage and frequency if possible. Also, mention any common side effects or contraindications.")
        else:
            return f"Error: Invalid 'request_type': {request_type}. Must be 'diagnosis', 'examination', or 'medication'."

        # The combined prompt will be returned. The Agent's LLM will process this.
        # The actual analysis of images (if image_path is provided) will be done by the
        # multimodal LLM used by the Agent, based on the image content included in the overall messages.
        return "\n".join(prompt_lines)

    async def _arun(
        self,
        symptoms: Optional[str] = None,
        medical_history: Optional[str] = None,
        image_path: Optional[str] = None,
        current_diagnosis: Optional[str] = None,
        request_type: str = None,
        # patient_age: Optional[int] = None,
        # patient_weight: Optional[float] = None,
        # allergies: Optional[List[str]] = None,
    ) -> str:
        """Asynchronous version of _run."""
        # This tool doesn't perform I/O, so sync and async are the same.
        return self._run(
            symptoms=symptoms,
            medical_history=medical_history,
            image_path=image_path,
            current_diagnosis=current_diagnosis,
            request_type=request_type,
            # patient_age=patient_age,
            # patient_weight=patient_weight,
            # allergies=allergies,
        )

# Example Usage (for testing purposes, not part of the tool itself):
if __name__ == '__main__':
    tool = AssistantDoctorTool()

    # Test diagnosis request
    diag_input = {
        "symptoms": "fever, cough, difficulty breathing",
        "medical_history": "smoker for 20 years",
        "image_path": "demo/chest/pneumonia1.jpg",
        "request_type": "diagnosis"
    }
    print("---- Diagnosis Request ----")
    print(f"Input: {diag_input}")
    output = tool._run(**diag_input)
    print(f"Generated Prompt for LLM:\n{output}\n")

    # Test examination request
    exam_input = {
        "symptoms": "chest pain, shortness of breath",
        "medical_history": "history of hypertension",
        "current_diagnosis": "Suspected Myocardial Infarction",
        "request_type": "examination"
    }
    print("---- Examination Request ----")
    print(f"Input: {exam_input}")
    output = tool._run(**exam_input)
    print(f"Generated Prompt for LLM:\n{output}\n")

    # Test medication request
    med_input = {
        "current_diagnosis": "Community-Acquired Pneumonia",
        "symptoms": "fever, productive cough", # Symptoms can still be relevant for medication choice
        "request_type": "medication"
    }
    print("---- Medication Request ----")
    print(f"Input: {med_input}")
    output = tool._run(**med_input)
    print(f"Generated Prompt for LLM:\n{output}\n")

    # Test missing request_type
    error_input = {
        "symptoms": "headache",
        "request_type": ""
    }
    print("---- Error Request (Missing request_type) ----")
    print(f"Input: {error_input}")
    output = tool._run(**error_input)
    print(f"Output:\n{output}\n")

    # Test invalid request_type
    error_input_invalid = {
        "symptoms": "headache",
        "request_type": "billing"
    }
    print("---- Error Request (Invalid request_type) ----")
    print(f"Input: {error_input_invalid}")
    output = tool._run(**error_input_invalid)
    print(f"Output:\n{output}\n")

    # Test medication request missing diagnosis
    error_med_input = {
        "request_type": "medication"
    }
    print("---- Error Request (Medication missing diagnosis) ----")
    print(f"Input: {error_med_input}")
    output = tool._run(**error_med_input)
    print(f"Output:\n{output}\n")
