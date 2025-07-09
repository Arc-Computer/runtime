#!/usr/bin/env python3
"""
Example demonstrating Kong Konnect integration with Arc Runtime
"""

import os
import time
from runtime.config import TelemetryConfig
from runtime.telemetry.client import TelemetryClient


def example_direct_connection():
    """Example using direct gRPC connection (backward compatibility)"""
    print("=== Direct gRPC Connection Example ===")
    
    client = TelemetryClient(
        endpoint="grpc://localhost:50051",
        api_key="arc_live_test_key"
    )
    
    # Send sample telemetry event
    event = {
        "timestamp": time.time(),
        "request_id": "direct-test-123",
        "llm_interaction": {
            "provider": "openai",
            "model": "gpt-4",
            "latency_ms": 150.0,
            "prompt_tokens": 50,
            "completion_tokens": 100
        },
        "arc_intervention": {
            "pattern_matched": True,
            "fix_applied": {"type": "rate_limit_retry"},
            "interception_latency_ms": 5.0
        }
    }
    
    client.record(event)
    print(f"Sent event: {event['request_id']}")
    
    # Give time for event to be processed
    time.sleep(2)
    
    client.shutdown()
    print("Direct connection example completed\n")


def example_kong_konnect_connection():
    """Example using Kong Konnect gateway"""
    print("=== Kong Konnect Gateway Example ===")
    
    # Create configuration for Kong Konnect
    config = TelemetryConfig(
        endpoint="your-gateway-id.us.cp0.konghq.com:443",
        api_key="arc_live_your_key_here",
        use_kong_gateway=True,
        kong_gateway_url="https://your-gateway-id.us.cp0.konghq.com",
        use_tls=True
    )
    
    client = TelemetryClient(config=config)
    
    # Send sample telemetry event
    event = {
        "timestamp": time.time(),
        "request_id": "kong-test-456",
        "llm_interaction": {
            "provider": "anthropic",
            "model": "claude-3-sonnet",
            "latency_ms": 200.0,
            "prompt_tokens": 75,
            "completion_tokens": 150
        },
        "arc_intervention": {
            "pattern_matched": False,
            "fix_applied": {},
            "interception_latency_ms": 3.0
        }
    }
    
    client.record(event)
    print(f"Sent event: {event['request_id']}")
    
    # Give time for event to be processed
    time.sleep(2)
    
    client.shutdown()
    print("Kong Konnect example completed\n")


def example_environment_variables():
    """Example using environment variables for configuration"""
    print("=== Environment Variables Example ===")
    
    # Set environment variables (in practice, these would be set in your shell)
    os.environ['ARC_TELEMETRY_ENDPOINT'] = 'localhost:50051'
    os.environ['ARC_API_KEY'] = 'arc_live_env_key'
    os.environ['ARC_USE_KONG_GATEWAY'] = 'false'
    os.environ['ARC_USE_TLS'] = 'false'
    
    # Create client from environment
    config = TelemetryConfig.from_env()
    client = TelemetryClient(config=config)
    
    # Send sample telemetry event
    event = {
        "timestamp": time.time(),
        "request_id": "env-test-789",
        "llm_interaction": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "latency_ms": 100.0,
            "prompt_tokens": 30,
            "completion_tokens": 60
        }
    }
    
    client.record(event)
    print(f"Sent event: {event['request_id']}")
    
    # Give time for event to be processed
    time.sleep(2)
    
    client.shutdown()
    print("Environment variables example completed\n")


def main():
    """Run all examples"""
    print("Kong Konnect Integration Examples")
    print("=" * 50)
    
    try:
        example_direct_connection()
        example_kong_konnect_connection()
        example_environment_variables()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Note: These examples require a running Arc Core server or Kong Konnect gateway")


if __name__ == "__main__":
    main()