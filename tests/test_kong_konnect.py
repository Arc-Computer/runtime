"""
Test Kong Konnect integration features
"""

import os
import time
import unittest
from unittest.mock import Mock, patch

from runtime.config import TelemetryConfig
from runtime.telemetry.client import TelemetryClient


class TestKongKonnectIntegration(unittest.TestCase):
    """Test Kong Konnect configuration and integration"""

    def test_telemetry_config_from_env(self):
        """Test TelemetryConfig creation from environment variables"""
        # Set environment variables
        test_env = {
            'ARC_TELEMETRY_ENDPOINT': 'your-gateway-id.us.cp0.konghq.com:443',
            'ARC_API_KEY': 'arc_live_test_key',
            'ARC_USE_KONG_GATEWAY': 'true',
            'ARC_KONG_GATEWAY_URL': 'https://your-gateway-id.us.cp0.konghq.com',
            'ARC_USE_TLS': 'true'
        }
        
        with patch.dict(os.environ, test_env):
            config = TelemetryConfig.from_env()
            
            self.assertEqual(config.endpoint, 'your-gateway-id.us.cp0.konghq.com:443')
            self.assertEqual(config.api_key, 'arc_live_test_key')
            self.assertTrue(config.use_kong_gateway)
            self.assertEqual(config.kong_gateway_url, 'https://your-gateway-id.us.cp0.konghq.com')
            self.assertTrue(config.use_tls)

    def test_telemetry_config_defaults(self):
        """Test TelemetryConfig with default values"""
        config = TelemetryConfig()
        
        self.assertEqual(config.endpoint, 'localhost:50051')
        self.assertIsNone(config.api_key)
        self.assertFalse(config.use_kong_gateway)
        self.assertIsNone(config.kong_gateway_url)
        self.assertFalse(config.use_tls)

    @patch("grpc.secure_channel")
    @patch("runtime.proto.telemetry_pb2_grpc.TelemetryServiceStub")
    def test_tls_connection_creation(self, mock_stub_class, mock_secure_channel):
        """Test that TLS connection is created for Kong Konnect"""
        # Mock secure channel
        mock_channel = Mock()
        mock_secure_channel.return_value = mock_channel
        mock_stub = Mock()
        mock_stub_class.return_value = mock_stub

        # Create config with TLS enabled
        config = TelemetryConfig(
            kong_gateway_url="https://your-gateway-id.us.cp0.konghq.com:443",
            use_kong_gateway=True,
            api_key="arc_live_test_key"
        )

        # Create telemetry client
        client = TelemetryClient(config=config)
        
        # Wait for worker thread to initialize
        time.sleep(0.1)

        # Verify secure channel was created
        mock_secure_channel.assert_called_once()
        
        # Cleanup
        client.shutdown()

    @patch("grpc.insecure_channel")
    @patch("runtime.proto.telemetry_pb2_grpc.TelemetryServiceStub")
    def test_backward_compatibility(self, mock_stub_class, mock_insecure_channel):
        """Test backward compatibility with existing direct gRPC connections"""
        # Mock insecure channel
        mock_channel = Mock()
        mock_insecure_channel.return_value = mock_channel
        mock_stub = Mock()
        mock_stub_class.return_value = mock_stub

        # Create client with legacy parameters
        client = TelemetryClient(
            endpoint="grpc://localhost:50051",
            api_key="arc_live_test_key"
        )
        
        # Wait for worker thread to initialize
        time.sleep(0.1)

        # Verify insecure channel was created (backward compatibility)
        mock_insecure_channel.assert_called_once()
        
        # Cleanup
        client.shutdown()

    @patch("grpc.secure_channel")
    @patch("runtime.proto.telemetry_pb2_grpc.TelemetryServiceStub")
    def test_kong_gateway_url_parsing(self, mock_stub_class, mock_secure_channel):
        """Test Kong gateway URL parsing with protocol prefix"""
        # Mock secure channel
        mock_channel = Mock()
        mock_secure_channel.return_value = mock_channel
        mock_stub = Mock()
        mock_stub_class.return_value = mock_stub

        # Create config with HTTPS URL
        config = TelemetryConfig(
            kong_gateway_url="https://test-gateway.konghq.com:443",
            use_kong_gateway=True,
            api_key="arc_live_test_key"
        )

        # Create telemetry client
        client = TelemetryClient(config=config)
        
        # Wait for worker thread to initialize
        time.sleep(0.1)

        # Verify secure channel was created with correct endpoint
        mock_secure_channel.assert_called_once()
        args, kwargs = mock_secure_channel.call_args
        self.assertEqual(args[0], "test-gateway.konghq.com:443")
        
        # Cleanup
        client.shutdown()

    @patch("grpc.insecure_channel")
    @patch("runtime.proto.telemetry_pb2_grpc.TelemetryServiceStub")
    def test_enhanced_error_handling(self, mock_stub_class, mock_insecure_channel):
        """Test enhanced error handling for Kong Konnect scenarios"""
        # Mock channel and stub
        mock_channel = Mock()
        mock_insecure_channel.return_value = mock_channel

        mock_stub = Mock()
        # Simulate PERMISSION_DENIED error
        import grpc
        mock_error = grpc.RpcError()
        mock_error.code = Mock(return_value=grpc.StatusCode.PERMISSION_DENIED)
        mock_error.details = Mock(return_value="Permission denied")
        mock_stub.StreamTelemetry.side_effect = mock_error
        mock_stub_class.return_value = mock_stub

        # Create client
        client = TelemetryClient(
            endpoint="grpc://localhost:50051",
            api_key="invalid_key"
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

    def test_telemetry_config_parameter_override(self):
        """Test that explicit parameters override environment variables"""
        # Set environment variables
        test_env = {
            'ARC_TELEMETRY_ENDPOINT': 'env-endpoint:443',
            'ARC_API_KEY': 'env_key',
        }
        
        with patch.dict(os.environ, test_env):
            # Create client with explicit parameters
            client = TelemetryClient(
                endpoint="explicit-endpoint:50051",
                api_key="explicit_key"
            )
            
            # Verify explicit parameters take precedence
            self.assertEqual(client.endpoint, "explicit-endpoint:50051")
            self.assertEqual(client.api_key, "explicit_key")
            
            # Cleanup
            client.shutdown()


if __name__ == "__main__":
    unittest.main()