#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Tracking Example

Demonstrates Arc Runtime's ability to intercept and track MCP server
communications for multi-agent systems.
"""

import os
import sys
import json

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

from runtime import Arc
import httpx

# Initialize Arc Runtime
arc = Arc(endpoint="grpc://localhost:50051")


def simulate_mcp_tool_call(tool_name: str, agent_name: str, params: dict):
    """Simulate an MCP tool call that Arc will intercept."""
    
    print(f"\n{agent_name} calling MCP tool: {tool_name}")
    print(f"Parameters: {json.dumps(params, indent=2)}")
    
    # Create HTTP client (already patched by Arc)
    with httpx.Client() as client:
        try:
            # This call will be intercepted by Arc's MCP interceptor
            response = client.post(
                "https://mcp-server.example.com/tools/call",
                headers={
                    "Content-Type": "application/json",
                    "X-MCP-Protocol": "1.0",
                    "X-Agent-Name": agent_name
                },
                json={
                    "tool": tool_name,
                    "params": params
                },
                timeout=5.0
            )
            
            print(f"Response status: {response.status_code}")
            
        except httpx.RequestError as e:
            # Expected for mock URLs
            print(f"Request failed (expected for demo): {type(e).__name__}")


def simulate_mcp_resource_access(resource: str, agent_name: str):
    """Simulate MCP resource access."""
    
    print(f"\n{agent_name} accessing MCP resource: {resource}")
    
    with httpx.Client() as client:
        try:
            response = client.get(
                f"https://mcp-server.example.com/resources/read/{resource}",
                headers={
                    "X-MCP-Protocol": "1.0",
                    "X-Agent-Name": agent_name
                },
                timeout=5.0
            )
            
            print(f"Response status: {response.status_code}")
            
        except httpx.RequestError as e:
            print(f"Request failed (expected for demo): {type(e).__name__}")


def main():
    """Run MCP tracking demonstration."""
    
    print("ARC RUNTIME MCP TRACKING DEMO")
    print("="*50)
    print("\nThis demo shows how Arc Runtime tracks MCP communications")
    print("between agents and MCP servers.\n")
    
    # Create multi-agent context
    with arc.create_multiagent_context(application_id="MCP-DEMO-001") as ctx:
        
        print(f"Pipeline ID: {ctx.pipeline_id}")
        print("-"*50)
        
        # Simulate document processing workflow with MCP
        
        # Agent 1: Document extractor
        simulate_mcp_tool_call(
            tool_name="extract_text",
            agent_name="document_extractor",
            params={
                "document_id": "DOC-123",
                "format": "pdf"
            }
        )
        
        # Agent 2: Data analyzer accessing resources
        simulate_mcp_resource_access(
            resource="knowledge_base/financial_rules",
            agent_name="data_analyzer"
        )
        
        # Agent 3: Report generator using tool
        simulate_mcp_tool_call(
            tool_name="generate_report",
            agent_name="report_generator",
            params={
                "template": "financial_summary",
                "data_source": "DOC-123"
            }
        )
        
        # Get pipeline summary
        print("\n" + "="*50)
        print("PIPELINE SUMMARY")
        print("="*50)
        
        summary = ctx.get_pipeline_summary()
        print(f"Total agents: {summary['agents_executed']}")
        print(f"Total latency: {summary['total_latency_ms']:.2f}ms")
        
        # Show MCP-specific tracking
        print("\nMCP Operations Tracked:")
        mcp_agents = [a for a in summary['agents'] if a.get('type') == 'mcp_server']
        for agent in mcp_agents:
            print(f"  - {agent['name']}: {agent.get('operation', 'unknown')} "
                  f"({agent['latency_ms']:.2f}ms)")
    
    print("\n" + "="*50)
    print("MCP tracking demonstration complete!")
    print("Arc Runtime intercepted all MCP communications.")
    print("="*50)


if __name__ == "__main__":
    main()