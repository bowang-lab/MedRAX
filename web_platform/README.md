# MedRAX Web Platform

A modern web interface for the MedRAX AI medical imaging analysis system.

## Features

- 🤖 **AI-Powered Analysis** - Agent autonomously selects and runs medical imaging tools
- 🫁 **Chest X-Ray Tools**:
  - Classification (DenseNet-121) - 18 pathologies
  - Segmentation (PSPNet) - 14 anatomical structures
  - Report Generation (ViT-BERT) - Structured medical reports
  - Visual QA (CheXagent) - Answer questions about X-rays
  - DICOM Processing - Convert DICOM to PNG
- 🎨 **Modern UI** - Analysis results front and center, real-time agent logs
- 📊 **Full Observability** - Comprehensive logging at every level

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- OpenAI API key

### 1. Setup Environment
```bash
# In the main MedRAX directory, create .env file
cd /Users/alankritverma/projects/MedRAX
echo "OPENAI_API_KEY=your_key_here" > .env
```

### 2. Install Dependencies

**Backend:**
```bash
cd web_platform
# Backend dependencies are installed via main MedRAX pyproject.toml
# Make sure MedRAX is installed: pip install -e .
```

**Frontend:**
```bash
cd web_platform/frontend
npm install
```

### 3. Run Development Servers

**Terminal 1 - Backend:**
```bash
cd web_platform
./dev-backend.sh
```

**Terminal 2 - Frontend:**
```bash
cd web_platform
./dev-frontend.sh
```

### 4. Open Application
Navigate to: **http://localhost:3000**

## Usage

1. **Upload** - Drag and drop a chest X-ray image
2. **Analyze** - Click "Run AI Analysis"
3. **View Results** - Segmented images, classifications, and metrics appear in the center
4. **Ask Questions** - Use the chat to ask follow-up questions

## Architecture

```
┌─────────────────────────────────────────┐
│ Frontend (Next.js + React + TypeScript) │
│ - Modern UI with real-time updates     │
│ - Analysis results front and center    │
│ - Agent logs and chat interface        │
└────────────┬────────────────────────────┘
             │ HTTP + REST API
┌────────────▼────────────────────────────┐
│ Backend (FastAPI + Python)             │
│ - Session management                   │
│ - File uploads                         │
│ - Agent orchestration                  │
└────────────┬────────────────────────────┘
             │ LangGraph Workflow
┌────────────▼────────────────────────────┐
│ MedRAX Agent (GPT-4o + LangChain)      │
│ - Autonomous tool selection            │
│ - Natural language understanding       │
│ - Multi-step reasoning                 │
└────────────┬────────────────────────────┘
             │ Tool Execution
┌────────────▼────────────────────────────┐
│ Medical Imaging Tools                  │
│ - Classification (DenseNet-121)        │
│ - Segmentation (PSPNet)                │
│ - Report Generation (ViT-BERT)         │
│ - VQA (CheXagent)                      │
│ - DICOM Processing                     │
└─────────────────────────────────────────┘


## Testing

Run comprehensive tests:
```bash
cd web_platform
python tests/test_final.py
```

## Docker (Optional)

Run with Docker Compose:
```bash
cd web_platform
docker-compose up
```


## API Endpoints

- `GET /api/health` - Health check
- `POST /api/sessions` - Create session
- `POST /api/upload/{session_id}` - Upload image
- `POST /api/chat/{session_id}` - Chat with agent
- `POST /api/tools/{tool_name}/run/{session_id}` - Run specific tool
- `GET /api/analysis/{session_id}` - Get results

## Environment Variables

```bash
OPENAI_API_KEY=your_key_here  # Required
OPENAI_BASE_URL=              # Optional (for custom endpoints)
```


## Troubleshooting (on mac)

**Backend won't start:**
```bash
# Check if port 8000 is in use
lsof -i :8000
```

**Frontend won't start:**
```bash
# Check if port 3000 is in use
lsof -i :3000
```

**Tools not loading:**
- Verify MedRAX is installed: `pip install -e .` from main directory
- Check backend logs for specific errors

Last updated: October 1, 2025
