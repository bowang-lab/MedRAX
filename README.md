<h1 align="center">
🤖 MedRAX: Medical Reasoning Agent for Chest X-ray
</h1>
<p align="center"> <a href="https://arxiv.org/abs/2502.02673" target="_blank"><img src="https://img.shields.io/badge/arXiv-ICML 2025-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a> <a href="https://github.com/bowang-lab/MedRAX"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a> <a href="https://huggingface.co/datasets/wanglab/chest-agent-bench"><img src="https://img.shields.io/badge/HuggingFace-Dataset-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Dataset"></a> </p>

![](assets/demo_fast.gif?autoplay=1)

<br>

## Abstract
Chest X-rays (CXRs) play an integral role in driving critical decisions in disease management and patient care. While recent innovations have led to specialized models for various CXR interpretation tasks, these solutions often operate in isolation, limiting their practical utility in clinical practice. We present MedRAX, the first versatile AI agent that seamlessly integrates state-of-the-art CXR analysis tools and multimodal large language models into a unified framework. MedRAX dynamically leverages these models to address complex medical queries without requiring additional training. To rigorously evaluate its capabilities, we introduce ChestAgentBench, a comprehensive benchmark containing 2,500 complex medical queries across 7 diverse categories. Our experiments demonstrate that MedRAX achieves state-of-the-art performance compared to both open-source and proprietary models, representing a significant step toward the practical deployment of automated CXR interpretation systems.
<br><br>


## MedRAX
MedRAX started as a medical reasoning agent for Chest X-rays and has been expanded to support analysis of other medical data types. It is built on a robust technical foundation:
- **Core Architecture**: Built on LangChain and LangGraph frameworks
- **Language Model**: Uses GPT-4o with vision capabilities as the backbone LLM
- **Deployment**: Supports both local and cloud-based deployments
- **Interface**: User-friendly interface built with Gradio, now featuring dedicated upload options for various medical files.
- **Modular Design**: Tool-agnostic architecture allowing easy integration of new capabilities.

### Integrated Tools
The agent integrates a variety of specialized tools. Originally focused on Chest X-ray analysis, it now includes tools for:

**Chest X-ray Analysis:**
- **Visual QA**: Utilizes CheXagent and LLaVA-Med for complex visual understanding and medical reasoning.
- **Segmentation**: Employs MedSAM and PSPNet model trained on ChestX-Det for precise anatomical structure identification.
- **Grounding**: Uses Maira-2 for localizing specific findings in medical images.
- **Report Generation**: Implements SwinV2 Transformer trained on CheXpert Plus for detailed medical reporting.
- **Disease Classification**: Leverages DenseNet-121 from TorchXRayVision for detecting 18 pathology classes.
- **X-ray Generation**: Utilizes RoentGen for synthetic CXR generation.

**Other Medical Data Analysis:**
- **Blood Test Analysis**: Parses and provides basic interpretation (Normal, Low, High based on hardcoded ranges) for blood test results from CSV files. PDF parsing is placeholder.
- **MR and CT Series Analysis**: Processes DICOM series (from ZIP archives), extracts metadata, and displays a representative slice. Does not perform diagnostic AI interpretation.
- **Ophthalmic (Eye) FFA and OCT Analysis**: Handles DICOM series, single ophthalmic images (PNG, JPG), and FFA videos (MP4, AVI). Extracts representative images/frames and relevant metadata. Does not perform diagnostic AI interpretation.

**General Utilities:**
- Includes DICOM processing for single files, image visualization, and custom plotting capabilities.
<br><br>

### Using the New Analysis Features
The Gradio interface now includes specific upload buttons:
- **"🩸 Upload Blood Test (.pdf, .csv)"**: For blood test reports. CSV analysis is functional; PDF is basic.
- **"🧲 Upload MR/CT Series (ZIP)"**: For DICOM series of MRI or CT scans, provided as a ZIP file.
- **"👁️ Upload Eye FFA/OCT"**: For ophthalmic images/series (DICOM, standard image formats) or FFA videos.

After uploading a file using the appropriate button, you can ask the agent to analyze it (e.g., "Analyze this blood test", "Summarize this CT series", "Describe this OCT scan"). The agent will attempt to use the corresponding new tool. For imaging series and ophthalmic images, the "analysis" primarily involves metadata extraction and display of a representative image/frame.

## ChestAgentBench
We introduce ChestAgentBench, a comprehensive evaluation framework with 2,500 complex medical queries across 7 categories, built from 675 expert-curated clinical cases. The benchmark evaluates complex multi-step reasoning in CXR interpretation through:

- Detection
- Classification
- Localization
- Comparison
- Relationship
- Diagnosis
- Characterization

Download the benchmark: [ChestAgentBench on Hugging Face](https://huggingface.co/datasets/wanglab/chest-agent-bench)
```
huggingface-cli download wanglab/chestagentbench --repo-type dataset --local-dir chestagentbench
```

Unzip the Eurorad figures to your local `MedMAX` directory.
```
unzip chestagentbench/figures.zip
```

To evaluate with GPT-4o, set your OpenAI API key and run the quickstart script.
```
export OPENAI_API_KEY="<your-openai-api-key>"
python quickstart.py \
    --model chatgpt-4o-latest \
    --temperature 0.2 \
    --max-cases 2 \
    --log-prefix chatgpt-4o-latest \
    --use-urls
```


<br>

## Installation
### Prerequisites
- Python 3.8+
- CUDA/GPU for best performance (for model inference)
- For Ophthalmic video (FFA) analysis: `opencv-python` (e.g., `pip install opencv-python`)
- Other common Python packages like `pydicom`, `numpy`, `Pillow` are typically installed as dependencies of the core libraries or tools.

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/bowang-lab/MedRAX.git
cd MedRAX

# Install package
pip install -e .
```

### Getting Started
```bash
# Start the Gradio interface
python main.py
```
or if you run into permission issues
```bash
sudo -E env "PATH=$PATH" python main.py
```
You need to setup the `model_dir` inside `main.py` to the directory where you want to download or already have the weights of above tools from Hugging Face.
Comment out the tools that you do not have access to.
Make sure to setup your OpenAI API key in `.env` file!
<br><br><br>


## Tool Selection and Initialization

MedRAX supports selective tool initialization, allowing you to use only the tools you need. Tools can be specified when initializing the agent (look at `main.py`):

```python
selected_tools = [
    "ImageVisualizerTool",
    "ChestXRayClassifierTool",
    "ChestXRaySegmentationTool",
    # Add or remove tools as needed
]

agent, tools_dict = initialize_agent(
    "medrax/docs/system_prompts.txt",
    tools_to_use=selected_tools,
    model_dir="/model-weights"
)
```

<br><br>
## Automatically Downloaded Models

The following tools will automatically download their model weights when initialized:

### Classification Tool
```python
ChestXRayClassifierTool(device=device)
```

### Segmentation Tool
```python
ChestXRaySegmentationTool(device=device)
```

### Grounding Tool
```python
XRayPhraseGroundingTool(
    cache_dir=model_dir, 
    temp_dir=temp_dir, 
    load_in_8bit=True, 
    device=device
)
```
- Maira-2 weights download to specified `cache_dir`
- 8-bit and 4-bit quantization available for reduced memory usage

### LLaVA-Med Tool
```python
LlavaMedTool(
    cache_dir=model_dir, 
    device=device, 
    load_in_8bit=True
)
```
- Automatic weight download to `cache_dir`
- 8-bit and 4-bit quantization available for reduced memory usage

### Report Generation Tool
```python
ChestXRayReportGeneratorTool(
    cache_dir=model_dir, 
    device=device
)
```

### Visual QA Tool
```python
XRayVQATool(
    cache_dir=model_dir, 
    device=device
)
```
- CheXagent weights download automatically

### MedSAM Tool
```
Support for MedSAM segmentation will be added in a future update.
```

### Utility Tools
No additional model weights required:
```python
ImageVisualizerTool()
DicomProcessorTool(temp_dir=temp_dir)
```
<br>

## Manual Setup Required

### Image Generation Tool
```python
ChestXRayGeneratorTool(
    model_path=f"{model_dir}/roentgen", 
    temp_dir=temp_dir, 
    device=device
)
```
- RoentGen weights require manual setup:
  1. Contact authors: https://github.com/StanfordMIMI/RoentGen
  2. Place weights in `{model_dir}/roentgen`
  3. Optional tool, can be excluded if not needed
<br>

## Configuration Notes

### Required Parameters
- `model_dir` or `cache_dir`: Base directory for model weights that Hugging Face uses
- `temp_dir`: Directory for temporary files
- `device`: "cuda" for GPU, "cpu" for CPU-only

### Memory Management
- Consider selective tool initialization for resource constraints
- Use 8-bit quantization where available
- Some tools (LLaVA-Med, Grounding) are more resource-intensive
<br>

## Important Usage Considerations

### Data Privacy and Security
- **Handling Sensitive Data**: This application can process medical images and reports (e.g., DICOM files, blood tests) which may contain Protected Health Information (PHI) or sensitive personal data. Users are responsible for ensuring compliance with all applicable data privacy regulations (e.g., HIPAA, GDPR) in their respective jurisdictions.
- **Temporary Files**: The tools within MedRAX often create temporary files (e.g., processed images, extracted archives) in the directory specified by the `temp_dir` configuration (default is `./temp/`). It is crucial to manage this directory appropriately:
    - Ensure the `temp_dir` is in a secure location with restricted access if handling PHI.
    - Implement regular cleanup procedures for the `temp_dir` to prevent accumulation of sensitive data. The application currently does not automatically delete all temporary files after processing for every tool. Some tools (like `MedicalImagingSeriesTool` and `OphthalmicImagingTool`) attempt to clean up specific temporary data they create (e.g., extracted ZIP archives), but the main `temp_dir` for uploaded files and processed images might persist.
- **Large Language Model (LLM) Data Usage**:
    - If using cloud-based LLMs (e.g., OpenAI's GPT models via API), be aware that data sent in prompts (including image data or content from medical files) will be processed by the third-party LLM provider. Ensure your usage complies with the LLM provider's data usage policies and any relevant data privacy agreements (e.g., Business Associate Agreements if required for HIPAA).
    - For enhanced privacy, consider using locally hosted LLMs (see "Local LLMs" section below).
- **No De-identification**: The application does not currently implement automated de-identification of PHI from uploaded files. Users should use de-identified data if required by their use case or ensure they are operating in a compliant environment.

### Local LLMs
If you are running a local LLM using frameworks like [Ollama](https://ollama.com/) or [LM Studio](https://lmstudio.ai/), you need to configure your environment variables accordingly. For example:
```
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"
```
<br>

## Star History
<div align="center">
  
[![Star History Chart](https://api.star-history.com/svg?repos=bowang-lab/MedRAX&type=Date)](https://star-history.com/#bowang-lab/MedRAX&Date)

</div>
<br>


## Authors
- **Adibvafa Fallahpour**¹²³⁴ * (adibvafa.fallahpour@mail.utoronto.ca)
- ****Jun Ma****²³ *
- **Alif Munim**³⁵ *
- ****Hongwei Lyu****³
- ****Bo Wang****¹²³⁶

¹ Department of Computer Science, University of Toronto, Toronto, Canada <br>
² Vector Institute, Toronto, Canada <br>
³ University Health Network, Toronto, Canada <br>
⁴ Cohere, Toronto, Canada <br>
⁵ Cohere Labs, Toronto, Canada <br>
⁶ Department of Laboratory Medicine and Pathobiology, University of Toronto, Toronto, Canada

<br>
* Equal contribution
<br><br>


## Citation
If you find this work useful, please cite our paper:
```bibtex
@misc{fallahpour2025medraxmedicalreasoningagent,
      title={MedRAX: Medical Reasoning Agent for Chest X-ray}, 
      author={Adibvafa Fallahpour and Jun Ma and Alif Munim and Hongwei Lyu and Bo Wang},
      year={2025},
      eprint={2502.02673},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.02673}, 
}
```

---
<p align="center">
Made with ❤️ at University of Toronto, Vector Institute, and University Health Network
</p>
