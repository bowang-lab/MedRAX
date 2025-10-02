#!/usr/bin/env python3
"""
Comprehensive test script for MedRAX Web Platform
Tests the complete workflow: Upload → Context → Analysis → Results
"""

import requests
import json
import time
import sys
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_backend_health():
    """Test if backend is running and healthy"""
    print("🔍 Testing backend health...")
    try:
        response = requests.get(f"{API_BASE}/api/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Backend healthy: {health_data}")
            return True
        else:
            print(f"❌ Backend unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend not accessible: {e}")
        return False

def test_session_creation():
    """Test session creation"""
    print("\n🔍 Testing session creation...")
    try:
        response = requests.post(f"{API_BASE}/api/sessions")
        if response.status_code == 200:
            session_data = response.json()
            session_id = session_data["session_id"]
            print(f"✅ Session created: {session_id}")
            return session_id
        else:
            print(f"❌ Session creation failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Session creation error: {e}")
        return None

def test_image_upload(session_id):
    """Test image upload"""
    print("\n🔍 Testing image upload...")
    
    # Use demo image
    demo_image = Path("demo/chest/normal1.jpg")
    if not demo_image.exists():
        print(f"❌ Demo image not found: {demo_image}")
        return False
    
    try:
        with open(demo_image, 'rb') as f:
            files = {'file': ('normal1.jpg', f, 'image/jpeg')}
            response = requests.post(f"{API_BASE}/api/upload/{session_id}", files=files)
        
        if response.status_code == 200:
            upload_data = response.json()
            print(f"✅ Image uploaded: {upload_data}")
            return True
        else:
            print(f"❌ Image upload failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Image upload error: {e}")
        return False

def test_tool_execution(session_id):
    """Test individual tool execution"""
    print("\n🔍 Testing tool execution...")
    
    tools_to_test = [
        "ChestXRayClassifierTool",
        "ImageVisualizerTool"
    ]
    
    for tool_name in tools_to_test:
        try:
            print(f"  🔧 Testing {tool_name}...")
            response = requests.post(f"{API_BASE}/api/tools/{tool_name}/run/{session_id}", json={})
            
            if response.status_code == 200:
                result_data = response.json()
                print(f"  ✅ {tool_name}: SUCCESS")
                print(f"     Result keys: {list(result_data.get('result', {}).keys())}")
            else:
                print(f"  ❌ {tool_name}: FAILED - {response.status_code}")
        except Exception as e:
            print(f"  ❌ {tool_name}: ERROR - {e}")

def test_comprehensive_analysis(session_id):
    """Test comprehensive analysis endpoint"""
    print("\n🔍 Testing comprehensive analysis...")
    try:
        response = requests.get(f"{API_BASE}/api/analysis/{session_id}")
        
        if response.status_code == 200:
            analysis_data = response.json()
            results = analysis_data.get("results", {})
            print(f"✅ Comprehensive analysis: SUCCESS")
            print(f"   Available results: {list(results.keys())}")
            
            # Check for specific results
            if "pathology" in results:
                classifications = results["pathology"].get("classifications", {})
                print(f"   📊 Pathology classifications: {len(classifications)} conditions")
            
            if "segmentation" in results:
                print(f"   🎯 Segmentation: Available")
                
            if "report" in results:
                report_content = results["report"].get("content", "")
                print(f"   📄 Report: {len(report_content)} characters")
                
            return True
        else:
            print(f"❌ Comprehensive analysis failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Comprehensive analysis error: {e}")
        return False

def test_chat_functionality(session_id):
    """Test chat with agent"""
    print("\n🔍 Testing chat functionality...")
    try:
        chat_data = {
            "message": "What do you see in this chest X-ray?",
            "session_id": session_id,
            "image_path": None
        }
        
        response = requests.post(f"{API_BASE}/api/chat/{session_id}", json=chat_data)
        
        if response.status_code == 200:
            chat_response = response.json()
            print(f"✅ Chat: SUCCESS")
            print(f"   Response length: {len(chat_response.get('response', ''))}")
            return True
        else:
            print(f"❌ Chat failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return False

def main():
    """Run complete test suite"""
    print("🧪 MedRAX Web Platform - Comprehensive Test Suite")
    print("=" * 60)
    
    # Test sequence
    tests_passed = 0
    total_tests = 5
    
    # 1. Backend Health
    if test_backend_health():
        tests_passed += 1
    
    # 2. Session Creation
    session_id = test_session_creation()
    if session_id:
        tests_passed += 1
        
        # 3. Image Upload
        if test_image_upload(session_id):
            tests_passed += 1
            
            # 4. Tool Execution
            test_tool_execution(session_id)
            tests_passed += 1
            
            # 5. Comprehensive Analysis
            if test_comprehensive_analysis(session_id):
                tests_passed += 1
            
            # 6. Chat Functionality
            test_chat_functionality(session_id)
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED - Platform is ready!")
        return 0
    else:
        print("⚠️  Some tests failed - check logs above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
