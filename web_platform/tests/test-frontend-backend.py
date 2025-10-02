#!/usr/bin/env python3
"""
Comprehensive Frontend-Backend Integration Test
Tests all connections between the Next.js frontend and FastAPI backend
"""

import requests
import json
from pathlib import Path
from datetime import datetime

API_BASE = 'http://localhost:8000'
FRONTEND_BASE = 'http://localhost:3000'

def print_header(text):
    print(f"\n{'='*80}")
    print(f"  {text}")
    print(f"{'='*80}")

def print_test(name, status, details=""):
    symbol = "✅" if status else "❌"
    print(f"{symbol} {name}")
    if details:
        print(f"   {details}")

def test_backend_health():
    """Test: Frontend Line 89 - checkSystemHealth()"""
    print_header("TEST 1: Backend Health Check")
    try:
        r = requests.get(f'{API_BASE}/api/health', timeout=5)
        success = r.status_code == 200
        data = r.json() if success else {}
        print_test(
            "GET /api/health",
            success,
            f"Tools available: {len(data.get('available_tools', []))}"
        )
        return success, data
    except Exception as e:
        print_test("GET /api/health", False, str(e))
        return False, {}

def test_frontend_accessible():
    """Test: Frontend is accessible"""
    print_header("TEST 2: Frontend Accessibility")
    try:
        r = requests.get(FRONTEND_BASE, timeout=5)
        success = r.status_code == 200
        print_test(
            "GET http://localhost:3000",
            success,
            "Frontend is serving pages"
        )
        return success
    except Exception as e:
        print_test("GET http://localhost:3000", False, str(e))
        return False

def test_session_creation():
    """Test: Frontend Line 202 - createSession()"""
    print_header("TEST 3: Session Management")
    try:
        r = requests.post(f'{API_BASE}/api/sessions', timeout=5)
        success = r.status_code == 200
        session_id = r.json().get('session_id') if success else None
        print_test(
            "POST /api/sessions",
            success,
            f"Session ID: {session_id[:16]}..." if session_id else ""
        )
        return success, session_id
    except Exception as e:
        print_test("POST /api/sessions", False, str(e))
        return False, None

def test_file_upload(session_id):
    """Test: Frontend Line 238 - uploadFile()"""
    print_header("TEST 4: File Upload")
    demo_image = Path('demo/chest/normal1.jpg')
    if not demo_image.exists():
        print_test("File Upload", False, "Demo image not found")
        return False, None
    
    try:
        with open(demo_image, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            r = requests.post(
                f'{API_BASE}/api/upload/{session_id}',
                files=files,
                timeout=10
            )
        success = r.status_code == 200
        data = r.json() if success else {}
        display_path = data.get('display_path')
        print_test(
            "POST /api/upload/{session_id}",
            success,
            f"Display path: {display_path}" if display_path else ""
        )
        return success, display_path
    except Exception as e:
        print_test("POST /api/upload/{session_id}", False, str(e))
        return False, None

def test_image_serving(display_path):
    """Test: Frontend Line 548 - Image src attribute"""
    print_header("TEST 5: Static Image Serving")
    try:
        r = requests.get(f'{API_BASE}/{display_path}', timeout=5)
        success = r.status_code == 200
        size = len(r.content) if success else 0
        print_test(
            f"GET /{display_path}",
            success,
            f"Image size: {size:,} bytes" if size > 0 else ""
        )
        return success
    except Exception as e:
        print_test(f"GET /{display_path}", False, str(e))
        return False

def test_tool_classification(session_id):
    """Test: Frontend Line 794, 941 - runSpecificTool('chest_xray_classifier')"""
    print_header("TEST 6: Classification Tool Execution")
    try:
        r = requests.post(
            f'{API_BASE}/api/tools/chest_xray_classifier/run/{session_id}',
            timeout=15
        )
        success = r.status_code == 200
        data = r.json() if success else {}
        
        if success and 'result' in data and isinstance(data['result'], dict):
            top_findings = sorted(
                data['result'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            findings_str = ", ".join([f"{k}: {v:.3f}" for k, v in top_findings])
            print_test(
                "POST /api/tools/chest_xray_classifier/run/{session_id}",
                True,
                f"Top findings: {findings_str}"
            )
        else:
            print_test(
                "POST /api/tools/chest_xray_classifier/run/{session_id}",
                False,
                r.text[:100] if not success else "No results"
            )
        return success
    except Exception as e:
        print_test("Classification Tool", False, str(e))
        return False

def test_tool_segmentation(session_id):
    """Test: Frontend Line 934 - runSpecificTool('chest_xray_segmentation')"""
    print_header("TEST 7: Segmentation Tool Execution")
    try:
        r = requests.post(
            f'{API_BASE}/api/tools/chest_xray_segmentation/run/{session_id}',
            json={"organs": ["Left Lung", "Right Lung", "Heart"]},
            timeout=20
        )
        success = r.status_code == 200
        data = r.json() if success else {}
        
        viz_path = None
        if success and 'metadata' in data:
            viz_path = data['metadata'].get('segmentation_image_path')
        
        print_test(
            "POST /api/tools/chest_xray_segmentation/run/{session_id}",
            success,
            f"Visualization: {viz_path}" if viz_path else ""
        )
        return success, viz_path
    except Exception as e:
        print_test("Segmentation Tool", False, str(e))
        return False, None

def test_analysis_results(session_id):
    """Test: Frontend Line 107 - fetchAnalysisResults()"""
    print_header("TEST 8: Comprehensive Analysis Results")
    try:
        r = requests.get(f'{API_BASE}/api/analysis/{session_id}', timeout=5)
        success = r.status_code == 200
        data = r.json() if success else {}
        results = data.get('results', {})
        print_test(
            "GET /api/analysis/{session_id}",
            success,
            f"Result keys: {list(results.keys())}" if results else "No results yet"
        )
        return success
    except Exception as e:
        print_test("GET /api/analysis/{session_id}", False, str(e))
        return False

def test_chat(session_id):
    """Test: Frontend Line 280 - sendMessage()"""
    print_header("TEST 9: Chat Functionality")
    try:
        r = requests.post(
            f'{API_BASE}/api/chat/{session_id}',
            json={'message': 'What are the key findings?'},
            timeout=15
        )
        success = r.status_code == 200
        data = r.json() if success else {}
        response = data.get('response', '')
        print_test(
            "POST /api/chat/{session_id}",
            success,
            f"Response: {response[:100]}..." if len(response) > 100 else response
        )
        return success
    except Exception as e:
        print_test("POST /api/chat/{session_id}", False, str(e))
        return False

def test_tools_list():
    """Test: Frontend implied - Available tools"""
    print_header("TEST 10: Tools List")
    try:
        r = requests.get(f'{API_BASE}/api/tools', timeout=5)
        success = r.status_code == 200
        data = r.json() if success else {}
        tools = data.get('tools', [])
        
        print_test(
            "GET /api/tools",
            success,
            f"Tools: {len(tools)}" if tools else ""
        )
        
        if tools:
            print("   Available tools:")
            for tool in tools:
                print(f"      - {tool['name']} ({tool['type']})")
        
        return success
    except Exception as e:
        print_test("GET /api/tools", False, str(e))
        return False

def main():
    print(f"\n{'#'*80}")
    print("  🧪 MEDRAX FRONTEND-BACKEND INTEGRATION TEST")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")
    
    results = []
    
    # Test 1: Backend Health
    success, health_data = test_backend_health()
    results.append(("Backend Health", success))
    if not success:
        print("\n❌ Backend not responding. Please start the backend first.")
        return
    
    # Test 2: Frontend Accessibility
    success = test_frontend_accessible()
    results.append(("Frontend Accessible", success))
    
    # Test 3: Session Creation
    success, session_id = test_session_creation()
    results.append(("Session Creation", success))
    if not session_id:
        print("\n❌ Cannot continue without session")
        return
    
    # Test 4: File Upload
    success, display_path = test_file_upload(session_id)
    results.append(("File Upload", success))
    
    # Test 5: Image Serving
    if display_path:
        success = test_image_serving(display_path)
        results.append(("Image Serving", success))
    
    # Test 6: Classification
    success = test_tool_classification(session_id)
    results.append(("Classification Tool", success))
    
    # Test 7: Segmentation
    success, viz_path = test_tool_segmentation(session_id)
    results.append(("Segmentation Tool", success))
    
    # Test 8: Analysis Results
    success = test_analysis_results(session_id)
    results.append(("Analysis Results", success))
    
    # Test 9: Chat
    success = test_chat(session_id)
    results.append(("Chat", success))
    
    # Test 10: Tools List
    success = test_tools_list()
    results.append(("Tools List", success))
    
    # Summary
    print_header("TEST SUMMARY")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n{'Test Name':<30} {'Status'}")
    print(f"{'-'*40}")
    for name, success in results:
        symbol = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:<30} {symbol}")
    
    print(f"\n{'='*80}")
    print(f"  TOTAL: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print(f"{'='*80}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Frontend and Backend are fully integrated!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - Please review above")
    
    return passed == total

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)


