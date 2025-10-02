#!/bin/bash

# MedRAX Web Platform - Frontend Development Server
echo "🚀 Starting MedRAX Frontend (Development Mode)..."
echo ""

# Check if we're in the right directory
if [ ! -f "frontend/package.json" ]; then
    echo "❌ Error: Must run from web_platform directory"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

echo "⚛️  Starting Next.js with hot reload..."
echo ""
echo "Frontend will be available at:"
echo "  🌐 http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start frontend
cd frontend
npm run dev
