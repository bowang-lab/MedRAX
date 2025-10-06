#!/bin/bash

# MedRAX Web Platform - Backend Development Server
echo "🚀 Starting MedRAX Backend (Development Mode)..."
echo ""

# Check if we're in the right directory
if [ ! -f "backend/main.py" ]; then
    echo "❌ Error: Must run from web_platform directory"
    exit 1
fi

# Check if venv exists in parent directory
if [ ! -d "../.venv" ]; then
    echo "❌ Virtual environment not found!"
    echo ""
    echo "Please create it first:"
    echo "  cd .."
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install fastapi uvicorn[standard] python-multipart"
    echo "  pip install -e ."
    exit 1
fi

# Activate virtual environment
echo "🐍 Activating virtual environment..."
cd ..
source .venv/bin/activate

# Store paths for later use
VENV_PYTHON="$(pwd)/.venv/bin/python"
VENV_UVICORN="$(pwd)/.venv/bin/uvicorn"
VENV_HF_CLI="$(pwd)/.venv/bin/huggingface-cli"

# Ensure Python uses the venv's site-packages
MEDRAX_ROOT="$(pwd)"
export PYTHONPATH="$MEDRAX_ROOT:$PYTHONPATH"
export VIRTUAL_ENV="$MEDRAX_ROOT/.venv"
export PATH="$MEDRAX_ROOT/.venv/bin:$PATH"

cd web_platform

# Load environment variables
if [ -f "../.env" ]; then
    export $(grep -v '^#' ../.env | xargs)
    echo "✅ Environment variables loaded"
else
    echo "⚠️  No .env file found in parent directory"
    echo "   Make sure OPENAI_API_KEY is set"
fi

# Check and setup HuggingFace authentication
echo ""
echo "🤗 Checking HuggingFace authentication..."
if [ ! -z "$HUGGINGFACE_TOKEN" ]; then
    # Token exists in .env, login automatically
    echo "   Found HUGGINGFACE_TOKEN in .env"
    if [ -f "$VENV_HF_CLI" ]; then
        echo "$HUGGINGFACE_TOKEN" | "$VENV_HF_CLI" login --token "$HUGGINGFACE_TOKEN" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "   ✅ Logged into HuggingFace"
        else
            echo "   ⚠️  HuggingFace login failed"
        fi
    else
        echo "   ⚠️  huggingface-cli not found in venv"
    fi
elif [ -f ~/.cache/huggingface/token ]; then
    # Already logged in
    echo "   ✅ Already logged into HuggingFace"
else
    # Not logged in
    echo "   ⚠️  Not logged into HuggingFace"
    echo ""
    echo "   Some tools (Maira-2) require HuggingFace authentication."
    echo "   To enable them, add to your .env file:"
    echo "   HUGGINGFACE_TOKEN=your_token_here"
    echo ""
    echo "   Or run manually: huggingface-cli login"
    echo ""
    echo "   Continuing without HuggingFace auth (core tools will work)..."
    sleep 2
fi

echo ""
echo "📡 Starting FastAPI backend with auto-reload..."
echo ""
echo "Backend will be available at:"
echo "  🔧 API: http://localhost:8000"
echo "  📊 Health: http://localhost:8000/api/health" 
echo "  📝 Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Verify Python environment
echo "🔍 Verifying Python environment..."
"$VENV_PYTHON" -c "import sys; print(f'   Python: {sys.executable}')"
"$VENV_PYTHON" -c "import langchain_cohere; import cohere; print('   ✅ langchain-cohere and cohere available')" 2>&1 | head -1

echo ""

# Check for cached models
echo "📦 Checking for cached models..."
HF_CACHE="${HOME}/.cache/huggingface/hub"
if [ -d "$HF_CACHE" ]; then
    CACHE_SIZE=$(du -sh "$HF_CACHE" 2>/dev/null | cut -f1)
    echo "   HuggingFace cache: $CACHE_SIZE at $HF_CACHE"
    
    # Check for specific large models
    if [ -d "$HF_CACHE/models--StanfordAIMI--CheXagent-2-3b" ]; then
        echo "   ✅ CheXagent VQA model cached"
    fi
    
    if [ -d "$HF_CACHE/models--microsoft--maira-2" ]; then
        echo "   ✅ Phrase Grounding (Maira-2) model cached"
    fi
    
    if [ -d "$HF_CACHE/models--microsoft--llava-med-v1.5-mistral-7b" ]; then
        echo "   ✅ LLaVA-Med model cached"
    fi
else
    echo "   No cached models found (will download on first use)"
fi

echo ""
echo "💡 TIP: Cached models will be auto-loaded by the ToolManager"
echo "   Tools with cached models appear as 'Available' (yellow)"
echo "   Click the download icon in Tools panel to load them"
echo ""

# Start backend with uvicorn auto-reload (using explicit venv paths)
cd backend

# Use Python to run uvicorn directly to ensure correct environment
exec "$VENV_PYTHON" -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
