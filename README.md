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
MedRAX is built on a high-performance, polyglot architecture combining Python, Rust, and Go:
- **Core Agent Architecture**: Built on LangChain and LangGraph frameworks with concurrent multi-tool execution
- **Language Model**: Uses GPT-4o with vision capabilities as the backbone LLM (with local LLM and OpenAI-compatible provider support)
- **High-Performance Medical Imaging (Rust)**: Native `medrax-core` engine for ultra-fast DICOM parsing, window/level contrast adjustment, MONOCHROME1/2 inversion, and parallel batch ingestion via Rayon
- **High-Concurrency Medical Gateway (Go)**: Production-grade `medrax-server` microservice providing REST APIs, bounded worker pool job scheduling, and Prometheus metrics
- **Deployment**: Supports local, headless server, and cloud-based deployments
- **Interface**: Production-ready interface built with Gradio and REST/Prometheus API gateway
- **Modular Design**: Tool-agnostic architecture allowing selective tool initialization and graceful fallbacks

### Integrated Tools
- **Visual QA**: Utilizes CheXagent and LLaVA-Med for complex visual understanding and medical reasoning
- **Segmentation**: Employs MedSAM and PSPNet model trained on ChestX-Det for precise anatomical structure identification, with Rust-accelerated organ metrics
- **Grounding**: Uses Maira-2 for localizing specific findings in medical images
- **Report Generation**: Implements SwinV2 Transformer trained on CheXpert Plus for detailed medical reporting
- **Disease Classification**: Leverages DenseNet-121 from TorchXRayVision for detecting 18 pathology classes
- **X-ray Generation**: Utilizes RoentGen for synthetic CXR generation
- **Utilities**: High-speed DICOM processor (Rust + Python), visualization tools, and custom plotting capabilities
<br><br>

## Multi-Language Architecture & Engineering Enhancements

MedRAX integrates **Rust** and **Go** where they deliver critical performance, concurrency, and reliability advantages:

```
                            +-----------------------------------------+
                            |           Hospital Clients / PACS       |
                            |       Web UI / Evaluation Scripts       |
                            +-----------------------------------------+
                                                 |
                                                 v
                            +-----------------------------------------+
                            |   MedRAX High-Concurrency Gateway (Go)  |
                            |           services/medrax-server        |
                            |  - REST API & Prometheus Telemetry      |
                            |  - Bounded Worker Pool & Backpressure   |
                            |  - DICOM Multipart Upload & SHA256 Hash |
                            +-----------------------------------------+
                                         /               \
                                        v                 v
                +-------------------------------+    +--------------------------------+
                |    Medical Imaging Engine     |    |       MedRAX Agent Core        |
                |            (Rust)             |    |            (Python)            |
                |       crates/medrax-core      |    |       medrax/agent/agent.py    |
                |  - Fast DICOM Decoder         |    |  - LangGraph State Machine     |
                |  - Window/Level Contrast      |    |  - Parallel Tool Dispatch      |
                |  - MONOCHROME1 Inversion      |    |  - Multimodal GPT-4o Brain     |
                |  - Parallel Batch Runner      |    +--------------------------------+
                |  - 14-Organ Geometric Metrics |                    |
                +-------------------------------+                    v
                                 ^                      +--------------------------+
                                 |                      | Specialized Neural Tools |
                                 +----------------------|  - Classifier / PSPNet   |
                                     (Python Bridge)    |  - SwinV2 Report Gen     |
                                                        |  - CheXagent / LLaVA-Med |
                                                        +--------------------------+
```

### 1. High-Performance Rust Medical Imaging Engine (`crates/medrax-core`)
- **Fast DICOM Processing**: Parses binary DICOM files, extracts clinical tags (`PatientID`, `Modality`, `PixelSpacing`, etc.), applies rescale slope/intercept, and performs contrast window/level adjustment directly to PNG.
- **MONOCHROME1 Correction**: Automatically detects `MONOCHROME1` radiologic polarity and inverts pixel intensities, preventing reversed contrast artifacts where bones appear dark.
- **Parallel Batch Ingestion**: Uses `rayon` for multi-threaded parallel conversion of DICOM datasets (over 15+ high-res images per second).
- **Geometric Organ Metrics**: Computes 14 anatomical structure metrics (area in $\text{cm}^2$, centroid, bounding box, intensity mean/std) in a single cache-coherent pass.
- **Transparent Python Bridge**: `DicomProcessorTool` automatically detects and uses the Rust engine when built, with zero-downtime fallback to pure Python.

**Build and Run Rust Engine:**
```bash
# Build release CLI
cargo build --release --manifest-path crates/medrax-core/Cargo.toml

# Convert single DICOM with window/level
crates/medrax-core/target/release/medrax-dicom process-file \
    --input demo/chest/LIDC.dcm \
    --output temp/output.png \
    --window-center 16383.5 \
    --window-width 32767

# Parallel batch convert an entire folder
crates/medrax-core/target/release/medrax-dicom batch-process \
    --input-dir demo/chest \
    --output-dir temp/batch_out \
    --threads 8

# Run Rust unit tests
cargo test --manifest-path crates/medrax-core/Cargo.toml
```

### 2. High-Concurrency Medical Gateway in Go (`services/medrax-server`)
- **RESTful API**: Standardized endpoints for hospital PACS ingestion, asynchronous reasoning, and status polling:
  - `GET /healthz`: Service status, active workers, queue depth, memory telemetry.
  - `GET /metrics`: Prometheus-compatible runtime metrics.
  - `POST /api/v1/dicom/process`: DICOM ingestion (JSON file path or multipart file upload with SHA-256 deduplication).
  - `POST /api/v1/reason`: Synchronous (`sync=true`) or asynchronous clinical reasoning job submission.
  - `GET /api/v1/jobs/{id}`: Poll job status, execution duration, and tool traces.
- **Resilience & Backpressure**: Bounded worker pool (default 4 workers) with channel queue (default 100 jobs) and graceful shutdown (`SIGINT`/`SIGTERM`).

**Build and Run Go Gateway:**
```bash
cd services/medrax-server

# Build server executable
go build -o bin/medrax-server ./cmd/server

# Start server
./bin/medrax-server -port 8080 -workers 4

# Run Go unit & integration tests
go test -v ./...
```

### 3. Agent Concurrency & Security Hardening (Python)
- **Security Vulnerability Elimination**: Replaced insecure `eval()` execution in `interface.py` with safe AST literal and JSON deserialization, mitigating arbitrary code execution risks.
- **Concurrent Multi-Tool Dispatch**: `Agent.execute_tools` in `medrax/agent/agent.py` supports parallel tool execution via `ThreadPoolExecutor`, significantly cutting latency when LLMs emit multiple tool calls in a single turn.
- **Modular Imports**: Lazy, guarded imports in `medrax/tools/__init__.py` ensure utility and DICOM tools function smoothly even when heavy GPU model checkpoints are not installed.
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

### Local LLMs
If you are running a local LLM using frameworks like [Ollama](https://ollama.com/) or [LM Studio](https://lmstudio.ai/), you need to configure your environment variables accordingly. For example:
```
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"
```
<br>

### Optional: OpenAI-compatible Providers

MedRAX supports OpenAI-compatible APIs, allowing regional or local LLM providers to serve as alternative backends.

For example, to use **Qwen3-VL** via [Alibaba Cloud DashScope](https://bailian.console.aliyun.com/?tab=model#/model-market), set the following environment variables:

```bash
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_API_KEY="<your-dashscope-api-key>"
export OPENAI_MODEL="qwen3-vl-235b-a22b-instruct"
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
