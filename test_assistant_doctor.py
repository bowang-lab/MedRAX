import sys
import os

# Add the project root to the Python path to allow importing medrax
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from medrax.tools import AssistantDoctorTool

def run_sample_tests():
    tool = AssistantDoctorTool()
    results = []
    passed_all = True

    # Use Case 1: Diagnosis Suggestion (without image for this programmatic test)
    print("---- Test Case 1: Diagnosis Suggestion ----")
    diag_input = {
        "symptoms": "Persistent dry cough for 3 weeks, occasional low-grade fever, fatigue, and mild shortness of breath during exertion.",
        "medical_history": "Non-smoker, no significant past respiratory illnesses. Works in a crowded office environment.",
        "image_path": "demo/chest/normal1.jpg", # Path will be included in prompt
        "request_type": "diagnosis"
    }
    expected_keywords_diag = [
        "patient reports", "symptoms: Persistent dry cough", "medical history: Non-smoker",
        "image at 'demo/chest/normal1.jpg' is provided", "suggest a list of differential diagnoses"
    ]
    output_diag = tool._run(**diag_input)
    print(f"Input: {diag_input}")
    print(f"Generated Prompt for LLM:\n{output_diag}\n")
    test_1_passed = all(keyword in output_diag for keyword in expected_keywords_diag)
    results.append({"Test Case 1 (Diagnosis)": "Passed" if test_1_passed else "Failed"})
    if not test_1_passed: passed_all = False
    print(f"Test 1 Passed: {test_1_passed}\n")

    # Use Case 2: Examination Plan
    print("---- Test Case 2: Examination Plan ----")
    exam_input = {
        "symptoms": "Sudden onset of severe left-sided chest pain, radiating to the left arm, accompanied by nausea and sweating.",
        "medical_history": "60-year-old male, known hypertensive, smoker (1 pack/day for 30 years), family history of heart disease.",
        "current_diagnosis": "Suspected Acute Coronary Syndrome (ACS)",
        "request_type": "examination"
    }
    expected_keywords_exam = [
        "patient reports", "symptoms: Sudden onset", "medical history: 60-year-old male",
        "current working diagnosis is: Suspected Acute Coronary Syndrome (ACS)",
        "Suggest a list of relevant examinations or tests"
    ]
    output_exam = tool._run(**exam_input)
    print(f"Input: {exam_input}")
    print(f"Generated Prompt for LLM:\n{output_exam}\n")
    test_2_passed = all(keyword in output_exam for keyword in expected_keywords_exam)
    results.append({"Test Case 2 (Examination)": "Passed" if test_2_passed else "Failed"})
    if not test_2_passed: passed_all = False
    print(f"Test 2 Passed: {test_2_passed}\n")

    # Use Case 3: Medication Suggestion
    print("---- Test Case 3: Medication Suggestion ----")
    med_input = {
        "current_diagnosis": "Type 2 Diabetes Mellitus, poorly controlled with diet and exercise.",
        "medical_history": "55-year-old female, BMI 32, no known kidney disease or major allergies. Recent HbA1c is 8.5%.",
        "request_type": "medication"
    }
    expected_keywords_med = [
        "medical history: 55-year-old female",
        "confirmed diagnosis is: Type 2 Diabetes Mellitus",
        "Suggest appropriate medications"
    ]
    output_med = tool._run(**med_input)
    print(f"Input: {med_input}")
    print(f"Generated Prompt for LLM:\n{output_med}\n")
    test_3_passed = all(keyword in output_med for keyword in expected_keywords_med)
    results.append({"Test Case 3 (Medication)": "Passed" if test_3_passed else "Failed"})
    if not test_3_passed: passed_all = False
    print(f"Test 3 Passed: {test_3_passed}\n")

    # Use Case 4: Diagnosis with Image (similar to Case 1 for this test, checking image_path)
    print("---- Test Case 4: Diagnosis with Image ----")
    diag_image_input = {
        "symptoms": "Fever, cough, and difficulty breathing for the past 3 days.",
        "medical_history": "Generally healthy 45-year-old.",
        "image_path": "demo/chest/pneumonia1.jpg",
        "request_type": "diagnosis"
    }
    expected_keywords_diag_image = [
        "symptoms: Fever, cough", "medical history: Generally healthy 45-year-old",
        "image at 'demo/chest/pneumonia1.jpg' is provided",
        "suggest a list of differential diagnoses"
    ]
    output_diag_image = tool._run(**diag_image_input)
    print(f"Input: {diag_image_input}")
    print(f"Generated Prompt for LLM:\n{output_diag_image}\n")
    test_4_passed = all(keyword in output_diag_image for keyword in expected_keywords_diag_image)
    results.append({"Test Case 4 (Diagnosis with Image)": "Passed" if test_4_passed else "Failed"})
    if not test_4_passed: passed_all = False
    print(f"Test 4 Passed: {test_4_passed}\n")


    # Test missing request_type
    print("---- Test Case 5: Error - Missing request_type ----")
    error_input = {
        "symptoms": "headache",
        "request_type": "" # Empty string
    }
    expected_output_error_missing = "Error: 'request_type' is a required field"
    output_error_missing = tool._run(**error_input)
    print(f"Input: {error_input}")
    print(f"Output:\n{output_error_missing}\n")
    test_5_passed = expected_output_error_missing in output_error_missing
    results.append({"Test Case 5 (Missing request_type)": "Passed" if test_5_passed else "Failed"})
    if not test_5_passed: passed_all = False
    print(f"Test 5 Passed: {test_5_passed}\n")


    # Test invalid request_type
    print("---- Test Case 6: Error - Invalid request_type ----")
    error_input_invalid = {
        "symptoms": "headache",
        "request_type": "billing"
    }
    expected_output_error_invalid = "Error: Invalid 'request_type': billing"
    output_error_invalid = tool._run(**error_input_invalid)
    print(f"Input: {error_input_invalid}")
    print(f"Output:\n{output_error_invalid}\n")
    test_6_passed = expected_output_error_invalid in output_error_invalid
    results.append({"Test Case 6 (Invalid request_type)": "Passed" if test_6_passed else "Failed"})
    if not test_6_passed: passed_all = False
    print(f"Test 6 Passed: {test_6_passed}\n")


    # Test medication request missing diagnosis
    print("---- Test Case 7: Error - Medication missing diagnosis ----")
    error_med_input = {
        "request_type": "medication"
        # current_diagnosis is missing
    }
    expected_output_error_med = "Error: 'current_diagnosis' is required when 'request_type' is 'medication'."
    output_error_med = tool._run(**error_med_input)
    print(f"Input: {error_med_input}")
    print(f"Output:\n{output_error_med}\n")
    test_7_passed = expected_output_error_med in output_error_med
    results.append({"Test Case 7 (Medication missing diagnosis)": "Passed" if test_7_passed else "Failed"})
    if not test_7_passed: passed_all = False
    print(f"Test 7 Passed: {test_7_passed}\n")

    print("---- Summary of Test Results ----")
    for res in results:
        print(res)

    if passed_all:
        print("\nAll sample tests PASSED.")
    else:
        print("\nSome sample tests FAILED.")

    return passed_all

if __name__ == '__main__':
    run_sample_tests()
