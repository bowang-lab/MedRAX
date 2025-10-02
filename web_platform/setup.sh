#!/bin/bash

echo "🏥 MedRAX Web Platform - Setup"
echo "================================"
echo ""

# Check if we're in the right directory
if [ ! -f "../pyproject.toml" ]; then
    echo "❌ Error: Must run from web_platform directory"
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "../.venv" ]; then
    echo "📦 Creating virtual environment..."
    cd ..
    python3 -m venv .venv
    cd web_platform
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate venv
echo "🐍 Activating virtual environment..."
cd ..
source .venv/bin/activate
cd web_platform

# Install backend dependencies
echo ""
echo "📦 Installing backend dependencies..."
pip install --upgrade pip
pip install fastapi "uvicorn[standard]" python-multipart

echo ""
echo "📦 Installing MedRAX dependencies..."
cd ..
pip install -e .
cd web_platform

# Install frontend dependencies
echo ""
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo ""
echo "================================"
echo "✅ Setup complete!"
echo ""
echo "🚀 To start the platform:"
echo ""
echo "Terminal 1 (Backend):"
echo "  cd web_platform"
echo "  ./dev-backend.sh"
echo ""
echo "Terminal 2 (Frontend):"
echo "  cd web_platform"
echo "  ./dev-frontend.sh"
echo ""
