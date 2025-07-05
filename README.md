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
MedRAX is built on a robust technical foundation:
- **Core Architecture**: Built on LangChain and LangGraph frameworks
- **Language Models**: Supports multiple LLM providers including OpenAI (GPT-4o) and Google (Gemini) models
- **Deployment**: Supports both local and cloud-based deployments
- **Interface**: Production-ready interface built with Gradio
- **Modular Design**: Tool-agnostic architecture allowing easy integration of new capabilities

### Integrated Tools
- **Visual QA**: Utilizes CheXagent and LLaVA-Med for complex visual understanding and medical reasoning
- **Segmentation**: Employs MedSAM and PSPNet model trained on ChestX-Det for precise anatomical structure identification
- **Grounding**: Uses Maira-2 for localizing specific findings in medical images
- **Report Generation**: Implements SwinV2 Transformer trained on CheXpert Plus for detailed medical reporting
- **Disease Classification**: Leverages DenseNet-121 from TorchXRayVision for detecting 18 pathology classes
- **X-ray Generation**: Utilizes RoentGen for synthetic CXR generation
- **Web Browser**: Provides web search capabilities and URL content retrieval using Google Custom Search API
- **Python Sandbox**: Executes Python code in a secure, stateful sandbox environment using `langchain-sandbox` and Pyodide. Supports custom data analysis, calculations, and dynamic package installations. Pre-configured with medical analysis packages including pandas, numpy, pydicom, SimpleITK, scikit-image, Pillow, scikit-learn, matplotlib, seaborn, and openpyxl. **Requires Deno runtime.**
- **Utilities**: Includes DICOM processing, visualization tools, and custom plotting capabilities
<br><br>


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
- [Deno](https://docs.deno.com/runtime/getting_started/installation/): Required for the Python Sandbox tool. Install using:
  ```bash
  # macOS/Linux
  curl -fsSL https://deno.land/install.sh | sh
  
  # Windows (PowerShell)
  irm https://deno.land/install.ps1 | iex
  ```
- CUDA/GPU for best performance

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
    "TorchXRayVisionClassifierTool",  # Renamed from ChestXRayClassifierTool
    "ArcPlusClassifierTool",          # New ArcPlus classifier
    "ChestXRaySegmentationTool",
    "PythonSandboxTool",              # Python code execution
    "WebBrowserTool",                 # Web search and URL access
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

### Classification Tools
```python
# TorchXRayVision-based classifier (original)
TorchXRayVisionClassifierTool(device=device)

# ArcPlus SwinTransformer-based classifier (new)
ArcPlusClassifierTool(
    model_path="/path/to/Ark6_swinLarge768_ep50.pth.tar",  # Optional
    num_classes=18,  # Default
    device=device
)
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

### Python Sandbox Tool
```python
# Tool name for selection: "PythonSandboxTool" 
# Implementation: create_python_sandbox() -> PyodideSandboxTool
create_python_sandbox()  # Returns configured PyodideSandboxTool instance
```
- **Stateful execution**: Variables, functions, and imports persist between calls
- **Pre-installed packages**: Common medical analysis packages (pandas, numpy, pydicom, SimpleITK, scikit-image, Pillow, scikit-learn, matplotlib, seaborn, openpyxl)
- **Dynamic package installation**: Can install additional packages using `micropip`
- **Network access**: Enabled for package installations from PyPI
- **Secure sandbox**: Runs in isolated Pyodide environment
- **Requires Deno**: Must have Deno runtime installed on host system

### Utility Tools
No additional model weights required:
```python
ImageVisualizerTool()
DicomProcessorTool(temp_dir=temp_dir)
WebBrowserTool()  # Requires Google Search API credentials
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

### Language Model Options
MedRAX supports multiple language model providers:

#### OpenAI Models
Supported prefixes: `gpt-` and `chatgpt-`
```
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional for custom endpoints
```

#### Google Gemini Models
Supported prefix: `gemini-`
```
export GOOGLE_API_KEY="your-google-api-key"
```

#### OpenRouter Models (Open Source & Proprietary)
Supported prefix: `openrouter-`

Access many open source and proprietary models via [OpenRouter](https://openrouter.ai/):
```
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

**Note:** Tool compatibility may vary with open-source models. For best results with tools, we recommend using OpenAI or Google Gemini models.

#### Local LLMs
If you are running a local LLM using frameworks like [Ollama](https://ollama.com/) or [LM Studio](https://lmstudio.ai/), you need to configure your environment variables accordingly. For example:
```
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"
```

#### Tool-Specific Configuration

**WebBrowserTool**: Requires Google Custom Search API credentials:
```bash
export GOOGLE_SEARCH_API_KEY="your-google-search-api-key"
export GOOGLE_SEARCH_ENGINE_ID="your-google-search-engine-id"
```

**PythonSandboxTool**: Requires Deno runtime installation:
```bash
# Verify Deno is installed
deno --version

# If not installed, install using:
curl -fsSL https://deno.land/install.sh | sh  # macOS/Linux
# or
irm https://deno.land/install.ps1 | iex       # Windows PowerShell
```

**Custom Python Sandbox Configuration**:
```python
from medrax.tools import create_python_sandbox

# Create custom sandbox with additional packages
custom_sandbox = create_python_sandbox(
    pip_packages=["your-package", "another-package"],
    stateful=True,  # Maintain state between calls
    allow_net=True,  # Allow network access for package installation
)
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
