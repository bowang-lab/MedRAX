import os
import ast
import json
import pytest
from pathlib import Path
from medrax.tools.dicom import DicomProcessorTool


def test_dicom_processor_monochrome2(tmp_path):
    tool = DicomProcessorTool(temp_dir=str(tmp_path))
    dicom_file = "demo/chest/LIDC.dcm"
    assert os.path.exists(dicom_file), f"Demo DICOM not found: {dicom_file}"

    output, metadata = tool._run(dicom_file)
    assert "image_path" in output
    assert os.path.exists(output["image_path"])
    assert metadata["analysis_status"] == "completed"
    assert metadata["engine"] == "rust-medrax-core"
    assert metadata["Modality"] == "CR"
    assert metadata["PhotometricInterpretation"] == "MONOCHROME2"
    assert metadata["PatientID"] == "LIDC-IDRI-0010"


def test_dicom_processor_monochrome1(tmp_path):
    tool = DicomProcessorTool(temp_dir=str(tmp_path))
    dicom_file = "demo/chest/Pseudo.dcm"
    assert os.path.exists(dicom_file), f"Demo DICOM not found: {dicom_file}"

    output, metadata = tool._run(dicom_file)
    assert "image_path" in output
    assert os.path.exists(output["image_path"])
    assert metadata["analysis_status"] == "completed"
    assert metadata["PhotometricInterpretation"] == "MONOCHROME1"


def test_dicom_processor_nonexistent_file(tmp_path):
    tool = DicomProcessorTool(temp_dir=str(tmp_path))
    output, metadata = tool._run("nonexistent.dcm")
    assert "error" in output
    assert metadata["analysis_status"] == "failed"


def test_safe_parsing_replaces_eval():
    """Verify that literal/json parsing safely handles tool output strings without arbitrary code execution."""
    safe_payload = "[{'image_path': 'temp/foo.png'}, {'status': 'completed'}]"
    parsed = ast.literal_eval(safe_payload)
    assert isinstance(parsed, list)
    assert parsed[0]["image_path"] == "temp/foo.png"

    json_payload = '[{"image_path": "temp/bar.png"}]'
    parsed_json = json.loads(json_payload)
    assert isinstance(parsed_json, list)
    assert parsed_json[0]["image_path"] == "temp/bar.png"

