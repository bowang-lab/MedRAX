#!/usr/bin/env python3
"""Test if MedRAX tools can be loaded"""

import sys
from pathlib import Path

# Add MedRAX to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("Testing MedRAX tool imports...")
print("=" * 60)

# Test 1: ImageVisualizerTool
print("\n1. Testing ImageVisualizerTool...")
try:
    from medrax.tools import ImageVisualizerTool
    tool = ImageVisualizerTool()
    print(f"   ✅ SUCCESS")
    print(f"   Tool name: {tool.name}")
    print(f"   Description: {tool.description[:80]}...")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 2: ChestXRayClassifierTool
print("\n2. Testing ChestXRayClassifierTool...")
try:
    from medrax.tools import ChestXRayClassifierTool
    print(f"   ✅ Import SUCCESS")
    print(f"   Attempting to initialize...")
    tool = ChestXRayClassifierTool(device='cpu')
    print(f"   ✅ Initialization SUCCESS")
    print(f"   Tool name: {tool.name}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 3: Agent
print("\n3. Testing Agent...")
try:
    from medrax.agent import Agent
    from langchain_openai import ChatOpenAI
    print(f"   ✅ Import SUCCESS")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

print("\n" + "=" * 60)
print("Test complete!")


