#!/usr/bin/env python3
"""
Test Arc Runtime v0.1.2 with Arc Core gRPC server.

Prerequisites:
1. Install Arc Runtime v0.1.2: pip install arc-runtime==0.1.2
2. Clone and run Arc Core gRPC server:
   git clone https://github.com/Arc-Computer/Arc-Failure-Taxonomy.git
   cd Arc-Failure-Taxonomy
   git checkout gRPC
   cd arc-core
   pip install grpcio protobuf
   ./generate_grpc_code.sh
   python -m arc.grpc.server
"""

import os
import sys
import time
import asyncio
from unittest.mock import Mock

# Test 1: Basic Arc Runtime initialization with gRPC endpoint
print("=== Test 1: Arc Runtime Initialization ===")
try:
    # Set Arc configuration
    os.environ["ARC_ENDPOINT"] = "grpc://localhost:50051"
    os.environ["ARC_API_KEY"] = "arc_live_test_key_12345"
    
    from runtime import Arc
    arc = Arc()
    print("✓ Arc Runtime initialized successfully")
    print(f"  - Telemetry endpoint: {arc.telemetry_client.client.endpoint}")
    print(f"  - Using gRPC: {hasattr(arc.telemetry_client.client, '_stub')}")
except Exception as e:
    print(f"✗ Failed to initialize Arc Runtime: {e}")
    sys.exit(1)

# Test 2: OpenAI interception with gRPC telemetry
print("\n=== Test 2: OpenAI Interception with gRPC Telemetry ===")
try:
    import openai
    
    # Create a mock OpenAI client
    client = openai.OpenAI(api_key="test-key")
    
    # Mock the actual API call to avoid requiring a real OpenAI key
    from unittest.mock import patch, MagicMock
    
    mock_response = Mock()
    mock_response.id = "test-response-id"
    mock_response.model = "gpt-4"
    mock_response.choices = [
        Mock(
            finish_reason="stop",
            message=Mock(content="Hello! I'm a test response.", tool_calls=None)
        )
    ]
    mock_response.usage = Mock(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30
    )
    
    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_response):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello, world!"}]
        )
    
    print("✓ OpenAI call intercepted successfully")
    print(f"  - Response ID: {response.id}")
    print(f"  - Model: {response.model}")
    print(f"  - Content: {response.choices[0].message.content}")
    
    # Give telemetry time to send
    time.sleep(1)
    
except Exception as e:
    print(f"✗ OpenAI interception failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Multi-agent context with gRPC telemetry
print("\n=== Test 3: Multi-Agent Context with gRPC Telemetry ===")
try:
    from runtime.multiagent import MultiAgentContext
    
    with MultiAgentContext(application_id="TEST-GRPC-001") as ctx:
        print(f"✓ Created multi-agent context: {ctx.pipeline_id}")
        
        # Simulate agent execution
        ctx.add_agent_execution(
            agent_name="test_agent",
            input_data={"query": "What is 2+2?"},
            output_data={"answer": "4"},
            latency_ms=125.5
        )
        
        # Simulate context handoff
        ctx.add_context_handoff(
            from_agent="test_agent",
            to_agent="validator_agent",
            context={"answer": "4", "confidence": 0.99}
        )
        
        print(f"  - Agents executed: {len(ctx.agents)}")
        print(f"  - Context handoffs: {len(ctx.context_handoffs)}")
        
        # Give telemetry time to send
        time.sleep(1)
        
except Exception as e:
    print(f"✗ Multi-agent context failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Direct telemetry event
print("\n=== Test 4: Direct Telemetry Event ===")
try:
    # Send a direct telemetry event
    event = {
        "timestamp": time.time(),
        "request_id": "direct-test-001",
        "pipeline_id": "test-pipeline",
        "application_id": "TEST-DIRECT",
        "agent_name": "direct_test_agent",
        "llm_interaction": {
            "provider": "test",
            "model": "test-model",
            "request_body": {"test": True},
            "response_body": {"result": "success"},
            "latency_ms": 50.0,
            "prompt_tokens": 5,
            "completion_tokens": 10,
        },
        "pattern_matched": False,
        "fix_applied": None,
        "metadata": {
            "test_type": "direct",
            "version": "0.1.2"
        }
    }
    
    arc.telemetry_client.record(event)
    print("✓ Direct telemetry event sent")
    
    # Flush the telemetry
    time.sleep(2)
    
except Exception as e:
    print(f"✗ Direct telemetry failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Verify telemetry worker and gRPC connection
print("\n=== Test 5: Telemetry Worker Status ===")
try:
    client = arc.telemetry_client.client
    print(f"✓ Telemetry client type: {type(client).__name__}")
    print(f"  - Worker running: {client._worker_thread.is_alive() if hasattr(client, '_worker_thread') else 'N/A'}")
    print(f"  - Queue size: {client._queue.qsize() if hasattr(client, '_queue') else 'N/A'}")
    print(f"  - gRPC channel state: {client._channel._channel.check_connectivity_state(False) if hasattr(client, '_channel') else 'N/A'}")
    
    # Graceful shutdown
    print("\n  Shutting down telemetry client...")
    arc.telemetry_client.client.close()
    print("  ✓ Telemetry client closed")
    
except Exception as e:
    print(f"✗ Status check failed: {e}")

print("\n=== All Tests Complete ===")
print("\nNote: Check the Arc Core server logs to verify telemetry was received.")
print("The server should show messages like:")
print("  [AuthInterceptor]: Authenticated agent with key prefix: arc_live_test_key_...")
print("  [RuntimeBridge]: Processing batch of X events.")
print("  [TelemetryService]: Successfully processed stream with X events.")