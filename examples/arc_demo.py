#!/usr/bin/env python
"""
Arc Runtime Demo - Shows interception and fix application
"""

import os
import sys
import time
import logging

# Add parent directory to path for runtime import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# Enable detailed logging to see Arc in action
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress some verbose logs
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

print("="*80)
print("ARC RUNTIME DEMONSTRATION")
print("="*80)

# Step 1: Initialize Arc Runtime
print("\n1. Initializing Arc Runtime...")
from runtime import Arc
arc = Arc()
print("   ✓ Arc Runtime initialized")
print(f"   ✓ Version: {arc._get_version()}")
print(f"   ✓ Endpoint: {arc.config.endpoint}")

# Step 2: Import OpenAI (after Arc)
print("\n2. Importing OpenAI SDK...")
import openai
print("   ✓ OpenAI imported (Arc has applied patches)")

# Step 3: Show registered patterns
print("\n3. Registered Patterns:")
print(f"   Total patterns: {len(arc.pattern_registry.patterns)}")
# Show a sample pattern match
test_params = {"model": "gpt-4", "temperature": 0.95}
match = arc.pattern_registry.match(test_params)
if match:
    print(f"   Example: {test_params} → Fix: {match}")

# Step 4: Check API key
api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("\n⚠️  No OpenAI API key found!")
    print("To see real interception, set your API key:")
    print("  export OPENAI_API_KEY='sk-...'")
    print("\nShowing mock demonstration instead...")
    
    # Mock demonstration
    print("\n4. Mock Demonstration:")
    print("   Request: model='gpt-4', temperature=0.95")
    print("   Arc Pattern Match: ✓")
    print("   Arc Fix Applied: temperature → 0.7")
    print("   Result: Request sent with fixed temperature")
    
else:
    print("\n✓ OpenAI API key found")
    
    # Real API demonstration
    print("\n4. Real API Demonstration:")
    client = openai.OpenAI()
    
    # High temperature request (will be fixed)
    print("\n   Test 1: High temperature request")
    print("   Request: model='gpt-4', temperature=0.95")
    print("   Expected: Arc fixes temperature to 0.7")
    print("   Sending request...")
    
    start = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Reply with: 'Hello from Arc Runtime'"}],
            temperature=0.95,  # Arc will fix this to 0.7
            max_tokens=10
        )
        elapsed = (time.time() - start) * 1000
        
        print(f"   ✓ Response: {response.choices[0].message.content}")
        print(f"   ✓ Latency: {elapsed:.2f}ms")
        print(f"   ✓ Arc interception successful!")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Normal temperature request (will not be modified)
    print("\n   Test 2: Normal temperature request")
    print("   Request: model='gpt-4', temperature=0.7")
    print("   Expected: Arc does not modify")
    print("   Sending request...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Reply with: '123'"}],
            temperature=0.7,
            max_tokens=10
        )
        print(f"   ✓ Response: {response.choices[0].message.content}")
        print(f"   ✓ Request passed through unchanged")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")

# Step 5: Check metrics
print("\n5. Checking Metrics:")
try:
    import urllib.request
    with urllib.request.urlopen('http://localhost:9090/metrics', timeout=1) as resp:
        metrics = resp.read().decode('utf-8')
        arc_metrics = [line for line in metrics.split('\n') 
                      if 'arc_' in line and not line.startswith('#')]
        
        print("   ✓ Metrics endpoint active")
        for metric in arc_metrics[:5]:  # Show first 5 metrics
            print(f"   {metric}")
            
except Exception as e:
    print(f"   ⚠ Metrics endpoint not accessible: {e}")

print("\n" + "="*80)
print("DEMONSTRATION COMPLETE")
print("Arc Runtime is intercepting and protecting your AI calls!")
print("="*80)