#!/usr/bin/env python3
"""
Quick test script for the MedRAX web backend
"""

import sys
import subprocess
import time
import requests
from pathlib import Path

def test_backend():
    print("🧪 Testing MedRAX Web Backend...")
    
    # Change to backend directory
    backend_dir = Path(__file__).parent / "backend"
    print(f"📁 Backend directory: {backend_dir}")
    
    # Test import
    try:
        sys.path.append(str(backend_dir))
        from medrax_wrapper import check_medrax_availability
        print("✅ MedRAX wrapper import successful")
        print(f"🔍 MedRAX components available: {check_medrax_availability()}")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Start backend process
    print("🚀 Starting backend server...")
    venv_python = Path(__file__).parent.parent / ".venv" / "bin" / "python"
    
    try:
        process = subprocess.Popen([
            str(venv_python), 
            "main.py"
        ], cwd=backend_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        print("⏳ Waiting for server to start...")
        time.sleep(8)
        
        # Test health endpoint
        try:
            response = requests.get("http://localhost:8000/api/health", timeout=5)
            print(f"📊 Health check: {response.status_code}")
            if response.status_code == 200:
                print("✅ Backend is running successfully!")
                print(f"📋 Response: {response.json()}")
                result = True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                result = False
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect to backend: {e}")
            result = False
        
        # Clean up
        process.terminate()
        process.wait(timeout=5)
        
        return result
        
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return False

if __name__ == "__main__":
    success = test_backend()
    sys.exit(0 if success else 1)


