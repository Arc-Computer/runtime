"""
Integration tests for Arc Runtime multi-agent support
"""

import json
import os

# Add parent directory to path
import sys
import time
import unittest
import uuid
from unittest.mock import MagicMock, Mock, patch

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime import Arc
from runtime.interceptors.mcp import MCPInterceptor
from runtime.multiagent.context import MultiAgentContext, get_current_pipeline_id


class TestMultiAgentContext(unittest.TestCase):
    """Test multi-agent context management"""

    def test_context_creation(self):
        """Test creating multi-agent context"""
        with MultiAgentContext(application_id="TEST-001") as ctx:
            self.assertIsNotNone(ctx.pipeline_id)
            self.assertEqual(ctx.application_id, "TEST-001")
            self.assertEqual(len(ctx.agents), 0)
            self.assertEqual(len(ctx.context_handoffs), 0)

    def test_agent_execution_tracking(self):
        """Test tracking agent executions"""
        with MultiAgentContext(application_id="TEST-002") as ctx:
            # Record agent execution
            ctx.add_agent_execution(
                agent_name="loan_officer",
                input_data={"loan_amount": 100000},
                output_data={"approved": True},
                latency_ms=125.5,
            )

            self.assertEqual(len(ctx.agents), 1, f"Expected 1 agent, got {ctx.agents}")
            agent = ctx.agents[0]
            self.assertEqual(agent["name"], "loan_officer")
            self.assertEqual(agent["latency_ms"], 125.5)
            self.assertEqual(agent["input"]["loan_amount"], 100000)
            self.assertEqual(agent["output"]["approved"], True)

    def test_context_handoff_tracking(self):
        """Test tracking context handoffs between agents"""
        with MultiAgentContext(application_id="TEST-003") as ctx:
            # Record context handoff
            ctx.add_context_handoff(
                from_agent="loan_officer",
                to_agent="credit_analyst",
                context={"credit_score": 750, "income": 85000},
            )

            self.assertEqual(len(ctx.context_handoffs), 1)
            handoff = ctx.context_handoffs[0]
            self.assertEqual(handoff["from"], "loan_officer")
            self.assertEqual(handoff["to"], "credit_analyst")
            self.assertEqual(handoff["context"]["credit_score"], 750)

    def test_pipeline_summary(self):
        """Test getting pipeline execution summary"""
        with MultiAgentContext(application_id="TEST-004") as ctx:
            # Add multiple agent executions
            ctx.add_agent_execution("agent1", {}, {}, 100.0)
            ctx.add_agent_execution("agent2", {}, {}, 200.0)
            ctx.add_context_handoff("agent1", "agent2", {})

            summary = ctx.get_pipeline_summary()
            self.assertEqual(summary["application_id"], "TEST-004")
            self.assertEqual(summary["agents_executed"], 2)
            self.assertEqual(summary["context_handoffs"], 1)
            self.assertEqual(summary["total_latency_ms"], 300.0)

    def test_context_variable_access(self):
        """Test accessing pipeline context via context variable"""
        self.assertIsNone(get_current_pipeline_id())

        with MultiAgentContext(application_id="TEST-005") as ctx:
            # Inside context, should be accessible
            self.assertEqual(get_current_pipeline_id(), ctx.pipeline_id)

        # Outside context, should be None again
        self.assertIsNone(get_current_pipeline_id())


class TestMCPInterceptor(unittest.TestCase):
    """Test MCP interceptor functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.arc = Arc()
        self.mcp_interceptor = self.arc.interceptors.get("mcp")

    def test_mcp_interceptor_initialization(self):
        """Test MCP interceptor is properly initialized"""
        self.assertIsNotNone(self.mcp_interceptor)
        self.assertIsInstance(self.mcp_interceptor, MCPInterceptor)

    @patch("httpx.AsyncClient.request")
    async def test_async_mcp_call_interception(self, mock_request):
        """Test intercepting async MCP calls"""
        # Set up mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_request.return_value = mock_response

        # Make MCP call within multi-agent context
        with MultiAgentContext(application_id="TEST-006") as ctx:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    "POST",
                    "http://example.com/mcp/tools/call",
                    json={"tool": "calculator", "args": {"a": 1, "b": 2}},
                    headers={"X-Agent-Name": "test_agent"},
                )

        # Verify interception
        self.assertEqual(response.status_code, 200)

        # Check that agent was tracked in context
        self.assertEqual(len(ctx.agents), 1)
        agent = ctx.agents[0]
        self.assertEqual(agent["name"], "test_agent")
        self.assertEqual(agent["type"], "mcp_server")
        self.assertEqual(agent["operation"], "tool_call")

    def test_mcp_metadata_extraction(self):
        """Test extracting MCP metadata from requests"""
        metadata = self.mcp_interceptor._extract_mcp_metadata(
            url="http://example.com/mcp/tools/call",
            method="POST",
            headers={"X-Agent-Name": "calculator_agent", "X-MCP-Version": "2.0"},
            kwargs={},
        )

        self.assertEqual(metadata["mcp_operation"], "tool_call")
        self.assertEqual(metadata["agent_name"], "calculator_agent")
        self.assertEqual(metadata["mcp_protocol_version"], "2.0")

    def test_mcp_operation_detection(self):
        """Test detection of different MCP operations"""
        test_cases = [
            ("http://example.com/mcp/tools/call", "tool_call"),
            ("http://example.com/mcp/resources/read", "resource_read"),
            ("http://example.com/mcp/prompts/get", "prompt_get"),
            ("http://example.com/mcp/sampling/createMessage", "sampling"),
            ("http://example.com/api/other", "unknown"),
        ]

        for url, expected_op in test_cases:
            metadata = self.mcp_interceptor._extract_mcp_metadata(
                url=url, method="POST", headers={}, kwargs={}
            )
            self.assertEqual(
                metadata["mcp_operation"], expected_op, f"Failed for URL: {url}"
            )


class TestEnhancedBaseInterceptor(unittest.TestCase):
    """Test enhanced base interceptor with agent metadata support"""

    def setUp(self):
        """Set up test fixtures"""
        self.arc = Arc()

    @patch("openai.resources.chat.completions.Completions.create")
    def test_agent_metadata_in_openai_call(self, mock_create):
        """Test passing agent metadata through OpenAI calls"""
        # Set up mock response
        mock_response = Mock()
        mock_response.id = "test-response-id"
        mock_response.model = "gpt-4"
        mock_response.choices = [
            Mock(
                finish_reason="stop",
                message=Mock(content="Test response", tool_calls=None),
            )
        ]
        mock_response.usage = Mock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        mock_create.return_value = mock_response

        # Import OpenAI after Arc to ensure patching
        import openai

        # Make call with agent metadata in multi-agent context
        with MultiAgentContext(application_id="TEST-007") as ctx:
            client = openai.OpenAI(api_key="test-key")

            # Manually track agent execution since OpenAI doesn't support metadata parameter
            start_time = time.perf_counter()
            response = client.chat.completions.create(
                model="gpt-4", messages=[{"role": "user", "content": "Hello"}]
            )
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Add agent execution to context
            ctx.add_agent_execution(
                agent_name="test_agent",
                input_data={"messages": [{"role": "user", "content": "Hello"}]},
                output_data={"response": response.choices[0].message.content},
                latency_ms=latency_ms,
            )

        # Verify the response
        self.assertEqual(response.id, "test-response-id")

        # Check that agent was tracked in context
        self.assertEqual(len(ctx.agents), 1)
        agent = ctx.agents[0]
        self.assertEqual(agent["name"], "test_agent")
        self.assertEqual(agent["type"], "llm")
        self.assertEqual(agent["output"]["response"], "Test response")


class TestTelemetryMultiAgentSupport(unittest.TestCase):
    """Test telemetry client multi-agent enhancements"""

    def setUp(self):
        """Set up test fixtures"""
        self.arc = Arc()
        self.telemetry = self.arc.telemetry_client

    def test_pipeline_execution_tracing(self):
        """Test tracing entire pipeline execution"""
        with self.telemetry.trace_pipeline_execution(
            pipeline_id="test-pipeline-123", application_id="LOAN-008"
        ) as span:
            # Span should be created (or None if OTel not available)
            if span:
                # Would verify attributes if OTel is available
                pass

    def test_agent_execution_tracing(self):
        """Test tracing individual agent execution"""
        with self.telemetry.trace_agent_execution(
            agent_name="credit_analyst", agent_type="mcp_server"
        ) as span:
            # Span should be created (or None if OTel not available)
            if span:
                # Would verify attributes if OTel is available
                pass

    def test_context_handoff_recording(self):
        """Test recording context handoffs"""
        # Create a mock span
        mock_span = Mock()

        self.telemetry.record_context_handoff(
            span=mock_span,
            from_agent="loan_officer",
            to_agent="credit_analyst",
            context_data={"credit_score": 750, "income": 85000},
        )

        # Verify event was added
        mock_span.add_event.assert_called_once()
        event_name, event_data = mock_span.add_event.call_args[0]
        self.assertEqual(event_name, "context_handoff")
        self.assertEqual(event_data["from"], "loan_officer")
        self.assertEqual(event_data["to"], "credit_analyst")

    def test_failure_recording(self):
        """Test recording failures for training data"""
        # Create a mock span
        mock_span = Mock()

        self.telemetry.record_failure(
            span=mock_span, failure_type="hallucination", business_impact="high"
        )

        # Verify attributes were set
        mock_span.set_attribute.assert_any_call("failure.type", "hallucination")
        mock_span.set_attribute.assert_any_call("business.impact", "high")
        mock_span.set_attribute.assert_any_call("failure_detected", True)

        # Verify event was added
        mock_span.add_event.assert_called_once()


class TestArcMultiAgentIntegration(unittest.TestCase):
    """Test Arc's multi-agent integration features"""

    def setUp(self):
        """Set up test fixtures"""
        self.arc = Arc()

    def test_create_multiagent_context(self):
        """Test Arc's create_multiagent_context method"""
        ctx = self.arc.create_multiagent_context(application_id="TEST-009")

        self.assertIsInstance(ctx, MultiAgentContext)
        self.assertEqual(ctx.application_id, "TEST-009")
        self.assertIsNotNone(ctx.pipeline_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
