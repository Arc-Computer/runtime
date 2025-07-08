"""
Test gRPC integration for telemetry streaming
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import time
import grpc
from runtime.telemetry.client import TelemetryClient


class TestGRPCIntegration(unittest.TestCase):
    """Test suite for gRPC telemetry integration"""

    def test_telemetry_event_conversion(self):
        """Test conversion of internal event format to protobuf"""
        client = TelemetryClient(endpoint="grpc://localhost:50051")

        # Create test event matching the expected format
        event = {
            "timestamp": time.time(),
            "request_id": "test-request-123",
            "pipeline_id": "pipeline-456",
            "application_id": "app-789",
            "agent_name": "test-agent",
            "llm_interaction": {
                "provider": "openai",
                "model": "gpt-4",
                "request_body": {"messages": [{"role": "user", "content": "Hello"}]},
                "response_body": {"choices": [{"message": {"content": "Hi!"}}]},
                "latency_ms": 150.5,
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
            "pattern_matched": True,
            "fix_applied": {"temperature": 0.7},
            "interception_latency_ms": 0.011,
            "metadata": {"method": "chat.completions.create", "success": "true"},
        }

        # Test conversion (if proto is available)
        try:
            pb_event = client._convert_to_protobuf(event)
            self.assertEqual(pb_event.request_id, "test-request-123")
            self.assertEqual(pb_event.pipeline_id, "pipeline-456")
            self.assertEqual(pb_event.application_id, "app-789")
            self.assertEqual(pb_event.agent_name, "test-agent")
            self.assertEqual(pb_event.llm_interaction.provider, "openai")
            self.assertEqual(pb_event.llm_interaction.model, "gpt-4")
            self.assertEqual(pb_event.llm_interaction.latency_ms, 150.5)
            self.assertTrue(pb_event.arc_intervention.pattern_matched)
            # Note: total_tokens is not in the proto anymore
        except ImportError:
            # Skip if proto not available
            pass

        # Cleanup
        client.shutdown()

    def test_telemetry_event_with_error(self):
        """Test conversion of event with error info"""
        client = TelemetryClient(endpoint="grpc://localhost:50051")

        # Create test event with error
        event = {
            "timestamp": time.time(),
            "request_id": "test-error-123",
            "llm_interaction": {
                "provider": "openai",
                "model": "gpt-4",
                "latency_ms": 50.0,
            },
            "error_info": {
                "error_type": "ValueError",
                "error_message": "Invalid temperature",
                "stack_trace": "Traceback...",
            },
        }

        # Test conversion
        try:
            pb_event = client._convert_to_protobuf(event)
            self.assertEqual(pb_event.request_id, "test-error-123")
            self.assertTrue(pb_event.HasField("error"))
            self.assertEqual(pb_event.error.error_type, "ValueError")
            self.assertEqual(pb_event.error.error_message, "Invalid temperature")
        except ImportError:
            # Skip if proto not available
            pass

        # Cleanup
        client.shutdown()

    @patch("grpc.insecure_channel")
    @patch("runtime.proto.telemetry_pb2_grpc.TelemetryServiceStub")
    def test_grpc_connection_creation(self, mock_stub_class, mock_channel_func):
        """Test that gRPC connection is created properly"""
        # Mock gRPC channel
        mock_channel = Mock()
        mock_channel_func.return_value = mock_channel
        mock_stub = Mock()
        mock_stub_class.return_value = mock_stub

        # Create telemetry client
        client = TelemetryClient(
            endpoint="grpc://localhost:50051", api_key="arc_live_test_key"
        )

        # Wait for worker thread to initialize
        time.sleep(0.1)

        # Verify gRPC channel was created with correct options
        mock_channel_func.assert_called_with(
            "localhost:50051",
            options=[
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.max_ping_strikes", 0),
            ],
        )

        # Cleanup
        client.shutdown()

    @patch("grpc.insecure_channel")
    @patch("runtime.proto.telemetry_pb2_grpc.TelemetryServiceStub")
    def test_authentication_metadata(self, mock_stub_class, mock_channel_func):
        """Test that API key is included in gRPC metadata"""
        # Mock channel and stub
        mock_channel = Mock()
        mock_channel_func.return_value = mock_channel

        mock_stub = Mock()
        mock_response = Mock(success=True, events_received=1, message="OK")
        mock_stub.StreamTelemetry.return_value = mock_response
        mock_stub_class.return_value = mock_stub

        # Create client with API key
        client = TelemetryClient(
            endpoint="grpc://localhost:50051", api_key="arc_live_test_key"
        )

        # Send an event
        event = {
            "timestamp": time.time(),
            "request_id": "test-123",
            "llm_interaction": {
                "provider": "openai",
                "model": "gpt-4",
                "latency_ms": 10.0,
            },
        }
        client.record(event)

        # Wait for batch processing
        time.sleep(1.5)

        # Verify StreamTelemetry was called
        self.assertTrue(mock_stub.StreamTelemetry.called)

        # Verify metadata included API key
        call_args = mock_stub.StreamTelemetry.call_args
        self.assertIsNotNone(call_args)
        metadata = call_args[1]["metadata"]
        self.assertIn(("x-api-key", "arc_live_test_key"), metadata)

        # Cleanup
        client.shutdown()

    def test_graceful_degradation_without_grpc(self):
        """Test that telemetry works even without gRPC"""
        # Temporarily make gRPC unavailable
        with patch.object(TelemetryClient, "_check_grpc_available", return_value=False):
            # Create client
            client = TelemetryClient(endpoint="grpc://localhost:50051")

            # Record an event
            event = {
                "timestamp": time.time(),
                "request_id": "test-123",
                "pattern_matched": True,
                "fix_applied": {"temperature": 0.7},
                "latency_ms": 5.0,
            }

            # Should not raise an exception
            client.record(event)

            # Verify metrics were updated
            with client.metrics._lock:
                self.assertEqual(
                    client.metrics._counters["arc_requests_intercepted_total"], 1
                )
                self.assertEqual(
                    client.metrics._counters["arc_pattern_matches_total"], 1
                )
                self.assertEqual(client.metrics._counters["arc_fixes_applied_total"], 1)

            # Cleanup
            client.shutdown()

    @patch("grpc.insecure_channel")
    @patch("runtime.proto.telemetry_pb2_grpc.TelemetryServiceStub")
    def test_grpc_error_handling(self, mock_stub_class, mock_channel_func):
        """Test handling of gRPC errors"""
        # Mock channel and stub
        mock_channel = Mock()
        mock_channel_func.return_value = mock_channel

        mock_stub = Mock()
        # Simulate UNAUTHENTICATED error
        mock_error = grpc.RpcError()
        mock_error.code = Mock(return_value=grpc.StatusCode.UNAUTHENTICATED)
        mock_error.details = Mock(return_value="Invalid API key")
        mock_stub.StreamTelemetry.side_effect = mock_error
        mock_stub_class.return_value = mock_stub

        # Create client
        client = TelemetryClient(
            endpoint="grpc://localhost:50051", api_key="invalid_key"
        )

        # Send an event
        event = {
            "timestamp": time.time(),
            "request_id": "test-123",
            "llm_interaction": {
                "provider": "openai",
                "model": "gpt-4",
                "latency_ms": 10.0,
            },
        }
        client.record(event)

        # Wait for batch processing
        time.sleep(1.5)

        # Verify error was handled gracefully (no crash)
        # Verify failed counter was incremented
        with client.metrics._lock:
            self.assertGreater(
                client.metrics._counters.get("arc_telemetry_failed_total", 0), 0
            )

        # Cleanup
        client.shutdown()


if __name__ == "__main__":
    unittest.main()
