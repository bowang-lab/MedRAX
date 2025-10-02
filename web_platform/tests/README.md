# MedRAX Web Platform - Tests

## Test Scripts

### Comprehensive Test
**`test_final.py`** - Full system test
- Tests backend health
- Tests session creation
- Tests image upload
- Tests classification tool
- Tests segmentation tool
- Tests agent chat

**Usage:**
```bash
cd /Users/alankritverma/projects/MedRAX
python3 web_platform/tests/test_final.py
```

### Component Tests
- **`test-backend.py`** - Backend API tests
- **`test-complete.py`** - Complete integration test
- **`test-frontend-backend.py`** - Frontend-backend integration
- **`test_tools.py`** - Individual tool tests

## Prerequisites

- Backend and frontend must be running
- Test images must be in `demo/chest/` directory
- `requests` module must be installed: `pip install requests`

## Running Tests

1. Start backend:
   ```bash
   cd web_platform
   ./dev-backend.sh
   ```

2. In another terminal, run tests:
   ```bash
   python3 tests/test_final.py
   ```

## Expected Output

```
🧪 MEDRAX WEB PLATFORM - COMPREHENSIVE TEST
================================================================================

1️⃣ Testing Backend Health...
   ✅ Backend: Healthy
   ✅ Tools available: 6

2️⃣ Creating Session...
   ✅ Session: abc123...

3️⃣ Uploading Test Image...
   ✅ Uploaded: normal1.jpg

4️⃣ Testing Classification Tool...
   ✅ Classification completed in 2.3s
   📊 Top 5 Pathologies: ...

5️⃣ Testing Segmentation Tool...
   ✅ Segmentation completed in 3.2s
   🫁 Segmented Organs: ...

6️⃣ Testing Agent Chat...
   ✅ Agent responded in 5.4s

🎉 ALL TESTS PASSED!
```

## Troubleshooting

**Backend not responding:**
- Check if backend is running on port 8000
- Verify OpenAI API key is set

**Test images not found:**
- Ensure you're running from MedRAX root directory
- Check `demo/chest/` directory exists

**Import errors:**
- Install requests: `pip install requests`



