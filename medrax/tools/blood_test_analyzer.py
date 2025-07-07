# Blood Test Analyzer Tool
# Handles uploading, parsing, and basic interpretation of blood test results.

import csv
import re

class BloodTestAnalyzerTool:
    def __init__(self, temp_dir="temp/", device="cpu"):
        self.name = "Blood Test Analyzer"
        self.description = "Analyzes blood test reports (primarily CSV, basic PDF text extraction) and provides interpretations."
        self.temp_dir = temp_dir
        self.device = device  # For consistency, though not used for parsing
        print(f"Initialized BloodTestAnalyzerTool on {self.device}")

    def run(self, file_path: str, file_type: str):
        """
        Processes a blood test file and returns interpretations.

        Args:
            file_path (str): Path to the uploaded blood test file.
            file_type (str): Type of the file ('pdf' or 'csv').

        Returns:
            str: A string containing the interpretation of the blood test.
        """
        file_type = file_type.lower()
        if file_type not in ["pdf", "csv"]:
            return "Error: Invalid file type. Only PDF and CSV are supported for blood tests."

        try:
            raw_data = self._read_file(file_path, file_type)
            if isinstance(raw_data, str) and raw_data.startswith("Error:"):
                return raw_data # Return error message from _read_file

            extracted_data = self._extract_data(raw_data, file_type)
            if isinstance(extracted_data, dict) and "error" in extracted_data and len(extracted_data) == 1:
                return f"Error during data extraction: {extracted_data['error']}"

            interpretation = self._interpret_data(extracted_data)
            return interpretation
        except Exception as e:
            return f"Error processing blood test file: {str(e)}"

    def _read_file(self, file_path: str, file_type: str):
        print(f"Reading {file_type} file from {file_path}...")
        if file_type == "csv":
            rows = []
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f: # utf-8-sig handles BOM
                    reader = csv.reader(f)
                    for row in reader:
                        rows.append(row)
                if not rows:
                    return "Error: CSV file is empty."
                return rows
            except FileNotFoundError:
                return f"Error: File not found at {file_path}"
            except Exception as e:
                return f"Error reading CSV file: {str(e)}"
        elif file_type == "pdf":
            # Placeholder for PDF reading. In a real scenario, use PyPDF2 or pdfplumber
            # For example:
            # try:
            #     import PyPDF2
            #     text_content = ""
            #     with open(file_path, 'rb') as f:
            #         reader = PyPDF2.PdfReader(f)
            #         if not reader.pages:
            #             return "Error: PDF is empty or unreadable."
            #         for page in reader.pages:
            #             page_text = page.extract_text()
            #             if page_text: # Ensure text was extracted
            #                 text_content += page_text + "\n"
            #     if not text_content.strip(): # Check if any text was actually extracted
            #         return "Error: PDF text extraction yielded no content (possibly image-based PDF)."
            #     return text_content
            # except FileNotFoundError:
            #     return f"Error: File not found at {file_path}"
            # except Exception as e:
            #     return f"Error reading PDF (or PyPDF2 not installed/functional): {str(e)}"
            return "PDF file received. PDF text extraction is a placeholder. Full parsing is complex and may require additional libraries."
        return f"Error: Unsupported file type for reading: {file_type}"

    def _extract_data(self, raw_data: any, file_type: str):
        print(f"Extracting data from {file_type} content...")
        extracted = {}

        if file_type == "csv" and isinstance(raw_data, list):
            if not raw_data:
                return {"error": "CSV raw data is empty for extraction."}

            header = [h.strip().lower() for h in raw_data[0]]

            param_col_names = ["parameter", "analyte", "test name", "test", "biomarker"]
            value_col_names = ["value", "result", "reading"]
            unit_col_names = ["units", "unit"]
            ref_col_names = ["referencerange", "reference range", "normal range", "reference"]

            param_idx, value_idx, unit_idx, ref_idx = -1, -1, -1, -1

            for i, h_item in enumerate(header):
                if h_item in param_col_names: param_idx = i
                elif h_item in value_col_names: value_idx = i
                elif h_item in unit_col_names: unit_idx = i
                elif h_item in ref_col_names: ref_idx = i

            if param_idx == -1 or value_idx == -1:
                return {"error": f"CSV header missing required columns (e.g., 'Parameter', 'Value'). Found: {', '.join(header)}"}

            for i, row in enumerate(raw_data[1:]): # Skip header
                if len(row) <= max(param_idx, value_idx): # Ensure row has enough columns
                    print(f"Skipping row {i+1} in CSV due to insufficient columns: {row}")
                    continue

                param_name = row[param_idx].strip()
                value_str = row[value_idx].strip()

                if not param_name or not value_str:
                    print(f"Skipping row {i+1} in CSV due to empty parameter/value: {row}")
                    continue

                units = row[unit_idx].strip() if unit_idx != -1 and len(row) > unit_idx and row[unit_idx] else None
                ref_range = row[ref_idx].strip() if ref_idx != -1 and len(row) > ref_idx and row[ref_idx] else None

                try:
                    value_float = float(re.sub(r'[^\d\.-]', '', value_str)) # Attempt to clean non-numeric chars except . and -
                    extracted[param_name] = {"value": value_float, "units": units, "reference_range": ref_range}
                except ValueError:
                    extracted[param_name] = {"value_str": value_str, "units": units, "reference_range": ref_range, "error": f"Value '{value_str}' is not a number"}

            if not extracted:
                return {"error": "No data rows found or parsed in CSV after header."}
            return extracted

        elif file_type == "pdf":
            if isinstance(raw_data, str) and raw_data.startswith("PDF file received."): # Placeholder handling
                 return {"info": raw_data} # Pass along the placeholder message

            # Actual PDF text processing would go here if _read_file successfully extracted text
            # This is a very basic example assuming text was extracted.
            if isinstance(raw_data, str):
                patterns = {
                    "Hemoglobin": r"(?:hemoglobin|hb)[\s: parietal]*([\d\.]+)\s*(g/dL)?",
                    "Glucose": r"glucose[\s: parietal]*([\d\.]+)\s*(mg/dL)?",
                    "WBC": r"wbc[\s: parietal]*([\d\.]+)\s*(x10\^9/L|10\*3/uL|/uL|K/uL|G/L)?"
                }
                found_any = False
                for key, pattern_str in patterns.items():
                    matches = re.finditer(pattern_str, raw_data, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        found_any = True
                        val_str = match.group(1)
                        unit_str = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
                        try:
                            extracted[key] = {"value": float(val_str), "units": unit_str}
                        except ValueError:
                            extracted[key] = {"value_str": val_str, "units": unit_str, "error": "Value not a number"}
                        break
                if found_any: return extracted
            return {"info": "PDF data extraction is currently a placeholder or did not find common parameters from extracted text."}

        return {"error": f"Data extraction not implemented for file type {file_type} or raw_data format incorrect."}

    def _interpret_data(self, extracted_data: dict):
        print(f"Interpreting extracted data: {extracted_data}")
        interpretation_results = []

        if not isinstance(extracted_data, dict) or not extracted_data:
             return "No data was extracted or data is in incorrect format, so no interpretation can be provided."

        if "error" in extracted_data and len(extracted_data) == 1:
            return f"Cannot interpret data due to extraction error: {extracted_data['error']}"
        if "info" in extracted_data and not any(k not in ["info", "error"] for k in extracted_data):
             return extracted_data["info"]

        normal_ranges = {
            "hemoglobin": (13.5, 17.5, "g/dL"),
            "glucose": (70.0, 100.0, "mg/dL"),
            "wbc": (4.0, 11.0, "x10^9/L")
        }

        for param, data_dict in extracted_data.items():
            if param.lower() in ["error", "info"]: continue
            if not isinstance(data_dict, dict):
                interpretation_results.append(f"{param}: Invalid data format - Expected a dictionary of values.")
                continue

            if "error" in data_dict and "value_str" in data_dict:
                interpretation_results.append(f"{param}: {data_dict['value_str']} {data_dict.get('units','')} - {data_dict['error']}.")
                continue
            elif "value" not in data_dict:
                val_disp = data_dict.get('value_str', '[VALUE NOT FOUND]')
                interpretation_results.append(f"{param}: {val_disp} {data_dict.get('units','')} - Numeric 'value' missing for interpretation.")
                continue

            value = data_dict["value"]
            units = data_dict.get("units", "") if data_dict.get("units") is not None else ""
            result_str = f"{param}: {value} {units}"

            param_lookup_key = param.lower().replace(" ", "").replace("_", "")

            interpretation_found = False
            for r_key, (low, high, expected_unit) in normal_ranges.items():
                if r_key in param_lookup_key:
                    interpretation_found = True
                    if units and expected_unit.lower() not in units.lower().replace(" ", "") and units != "N/A" and units is not None:
                        result_str += f" (Note: Units '{units}' differ from expected '{expected_unit}'. Interpretation assumes value is comparable.)"
                    if value < low: result_str += " - Low"
                    elif value > high: result_str += " - High"
                    else: result_str += " - Normal"
                    break
            if not interpretation_found:
                result_str += " - No interpretation range defined for this parameter."
            interpretation_results.append(result_str)

        if not interpretation_results:
            if "error" in extracted_data: return f"Interpretation incomplete due to error: {extracted_data['error']}"
            if "info" in extracted_data: return extracted_data['info']
            return "No parameters were suitable for interpretation from the extracted data."

        return "\n".join(interpretation_results)

if __name__ == '__main__':
    # Example Usage (for testing the tool independently)
    tool = BloodTestAnalyzerTool(temp_dir="temp_tool_test/")

    # Create dummy temp directory if it doesn't exist
    import os
    if not os.path.exists(tool.temp_dir):
        os.makedirs(tool.temp_dir)

    # --- Test CSV ---
    csv_file_path = os.path.join(tool.temp_dir, "test_blood_data.csv")
    with open(csv_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value", "Units", "ReferenceRange"])
        writer.writerow(["Hemoglobin", "15.0", "g/dL", "13.5-17.5"])
        writer.writerow(["Glucose", "65", "mg/dL", "70-100"])
        writer.writerow(["WBC", "12.5", "x10^9/L", "4.0-11.0"])
        writer.writerow(["Platelets", "250", "x10^9/L", "150-400"]) # No specific range in tool
        writer.writerow(["Sodium", "140mEq/L", "mEq/L", "135-145"]) # Value has text

    print("\n--- Testing CSV File ---")
    interpretation_csv = tool.run(csv_file_path, "csv")
    print("CSV Interpretation:")
    print(interpretation_csv)

    # --- Test PDF (Placeholder) ---
    # This will just return the placeholder message as PDF reading isn't implemented with a library
    pdf_file_path = os.path.join(tool.temp_dir, "test_blood_data.pdf") # Dummy path
    with open(pdf_file_path, "w") as f: # Create an empty file just for path existence
        f.write("This is a dummy PDF file.")

    print("\n--- Testing PDF File (Placeholder Reading) ---")
    interpretation_pdf = tool.run(pdf_file_path, "pdf")
    print("PDF Interpretation:")
    print(interpretation_pdf)

    # --- Test PDF with simulated text extraction (if _read_file were to return text) ---
    print("\n--- Testing PDF with Simulated Text for Extraction ---")
    simulated_pdf_text = """
    Patient Report
    Hemoglobin: 12.1 g/dL
    Glucose: 105 mg/dL
    WBC Count: 3.5 x10^9/L
    Other stuff...
    """
    # Simulate that _read_file returned this text
    extracted_sim_pdf = tool._extract_data(simulated_pdf_text, "pdf")
    interpreted_sim_pdf = tool._interpret_data(extracted_sim_pdf)
    print("Simulated PDF Text Interpretation:")
    print(interpreted_sim_pdf)

    # --- Test error cases ---
    print("\n--- Testing Non-existent file ---")
    interp_err_file = tool.run("non_existent_file.csv", "csv")
    print(interp_err_file)

    print("\n--- Testing Bad CSV header ---")
    bad_csv_path = os.path.join(tool.temp_dir, "bad_header.csv")
    with open(bad_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ColA", "ColB"])
        writer.writerow(["Hemo", "10"])
    interp_bad_csv = tool.run(bad_csv_path, "csv")
    print(interp_bad_csv)

    # Clean up dummy directory and files
    # import shutil
    # shutil.rmtree(tool.temp_dir)
    # print(f"\nCleaned up {tool.temp_dir}")
    print(f"\nTest files are in {tool.temp_dir}. Manual cleanup might be needed.")
