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
cd web_platform

# Load environment variables
if [ -f "../.env" ]; then
    export $(grep -v '^#' ../.env | xargs)
    echo "✅ Environment variables loaded"
else
    echo "⚠️  No .env file found in parent directory"
    echo "   Make sure OPENAI_API_KEY is set"
fi

echo "📡 Starting FastAPI backend with auto-reload..."
echo ""
echo "Backend will be available at:"
echo "  🔧 API: http://localhost:8000"
echo "  📊 Health: http://localhost:8000/api/health" 
echo "  📝 Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start backend with uvicorn auto-reload
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
