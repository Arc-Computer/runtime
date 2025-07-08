"""
Telemetry client for streaming events to Arc Core
"""

import json
import logging
import os
import queue
import threading
import time
from typing import Any, Dict, Generator, Optional
from urllib.parse import urlparse

from runtime.telemetry.metrics import Metrics
from runtime.telemetry.otel_client import OTelTelemetryClient

logger = logging.getLogger(__name__)


class TelemetryClient(OTelTelemetryClient):
    """
    Non-blocking telemetry client that streams events to Arc Core

    Features:
    - Async queue with backpressure handling
    - Graceful degradation if Arc Core is unreachable
    - Minimal overhead (<5ms)
    - Local metrics collection
    """

    def __init__(self, endpoint: str, api_key: Optional[str] = None):
        # Initialize OTel parent
        super().__init__(service_name="arc-runtime")

        self.endpoint = endpoint
        self.api_key = api_key
        self.metrics = Metrics()

        # Queue for async telemetry
        self.queue = queue.Queue(maxsize=10000)

        # Worker thread
        self._worker_thread = None
        self._stop_event = threading.Event()

        # Parse endpoint
        self._parse_endpoint()

        # Start worker thread
        self._start_worker_thread()

    def _parse_endpoint(self):
        """Parse gRPC endpoint"""
        parsed = urlparse(self.endpoint)

        # Default to localhost if no host specified
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or 50051

        # Check if gRPC is available
        self.grpc_available = self._check_grpc_available()

    def _check_grpc_available(self):
        """Check if gRPC is available"""
        try:
            import grpc

            from runtime.proto import telemetry_pb2, telemetry_pb2_grpc

            return True
        except ImportError as e:
            logger.warning(
                f"gRPC or proto definitions not available: {e}. "
                "Telemetry will be logged locally only. "
                "Install with: pip install grpcio protobuf"
            )
            return False

    def record(self, event: Dict[str, Any]):
        """
        Record a telemetry event (non-blocking)

        Args:
            event: Event dictionary to record
        """
        # Update metrics
        self.metrics.increment("arc_requests_intercepted_total")

        if event.get("pattern_matched"):
            self.metrics.increment("arc_pattern_matches_total")

        if event.get("fix_applied"):
            self.metrics.increment("arc_fixes_applied_total")

        if "latency_ms" in event:
            self.metrics.record_histogram(
                "arc_interception_latency_ms", event["latency_ms"]
            )

        # Try to enqueue
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            self.metrics.increment("arc_telemetry_dropped_total")
            logger.debug("Telemetry queue full - dropping event")

    def _start_worker_thread(self):
        """Start the telemetry worker thread"""
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="arc-telemetry-worker", daemon=True
        )
        self._worker_thread.start()

    def _worker_loop(self):
        """Main worker loop for processing telemetry"""
        logger.debug(f"Telemetry worker started (endpoint={self.endpoint})")

        # Initialize gRPC channel if available
        channel = None
        stub = None

        if self.grpc_available:
            try:
                channel, stub = self._create_grpc_connection()
            except Exception as e:
                logger.warning(f"Failed to create gRPC connection: {e}")

        # Batch processing
        batch = []
        batch_size = 100
        batch_timeout = 1.0  # seconds
        last_flush = time.time()

        while not self._stop_event.is_set():
            try:
                # Get event with timeout
                timeout = batch_timeout - (time.time() - last_flush)
                if timeout <= 0:
                    timeout = 0.01

                try:
                    event = self.queue.get(timeout=timeout)
                    batch.append(event)
                except queue.Empty:
                    pass

                # Flush batch if needed
                should_flush = (
                    len(batch) >= batch_size
                    or time.time() - last_flush >= batch_timeout
                )

                if should_flush and batch:
                    self._send_batch(batch, stub)
                    batch = []
                    last_flush = time.time()

            except Exception as e:
                logger.error(f"Error in telemetry worker: {e}")
                self.metrics.increment("arc_telemetry_errors_total")

        # Final flush
        if batch:
            self._send_batch(batch, stub)

        # Close gRPC channel
        if channel:
            channel.close()

    def _create_grpc_connection(self):
        """Create gRPC channel and stub"""
        import grpc

        from runtime.proto import telemetry_pb2_grpc

        # Create insecure channel for development
        # TODO: Add TLS support for production
        channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.max_ping_strikes", 0),
            ],
        )

        # Create stub from generated code
        stub = telemetry_pb2_grpc.TelemetryServiceStub(channel)
        return channel, stub

    def _convert_to_protobuf(
        self, event: Dict[str, Any]
    ) -> "telemetry_pb2.TelemetryEvent":
        """Convert internal event format to protobuf TelemetryEvent"""
        from google.protobuf.timestamp_pb2 import Timestamp

        from runtime.proto import telemetry_pb2

        pb_event = telemetry_pb2.TelemetryEvent()

        # Set request metadata
        pb_event.request_id = str(event.get("request_id", ""))
        pb_event.pipeline_id = str(event.get("pipeline_id", ""))
        pb_event.application_id = str(event.get("application_id", ""))
        pb_event.agent_name = str(event.get("agent_name", ""))

        # Set timestamp
        timestamp = Timestamp()
        timestamp.FromSeconds(int(event.get("timestamp", time.time())))
        pb_event.timestamp.CopyFrom(timestamp)

        # Set LLM interaction details
        if "llm_interaction" in event:
            llm = event["llm_interaction"]
            pb_event.llm_interaction.provider = str(llm.get("provider", ""))
            pb_event.llm_interaction.model = str(llm.get("model", ""))
            pb_event.llm_interaction.request_body = json.dumps(
                llm.get("request_body", {})
            )
            pb_event.llm_interaction.response_body = json.dumps(
                llm.get("response_body", {})
            )
            pb_event.llm_interaction.latency_ms = float(llm.get("latency_ms", 0.0))
            pb_event.llm_interaction.prompt_tokens = int(llm.get("prompt_tokens", 0))
            pb_event.llm_interaction.completion_tokens = int(
                llm.get("completion_tokens", 0)
            )

        # Set Arc intervention details
        if event.get("pattern_matched") or event.get("fix_applied"):
            pb_event.arc_intervention.pattern_matched = bool(
                event.get("pattern_matched", False)
            )
            pb_event.arc_intervention.fix_applied = json.dumps(
                event.get("fix_applied", {})
            )
            pb_event.arc_intervention.interception_latency_ms = float(
                event.get("interception_latency_ms", 0.0)
            )

        # Set error info if present
        if "error_info" in event:
            error = event["error_info"]
            pb_event.error.error_type = str(error.get("error_type", ""))
            pb_event.error.error_message = str(error.get("error_message", ""))
            pb_event.error.stack_trace = str(error.get("stack_trace", ""))

        # Set additional metadata
        metadata = event.get("metadata", {})
        for key, value in metadata.items():
            pb_event.metadata[str(key)] = str(value)

        return pb_event

    def _event_generator(self, batch: list) -> Generator:
        """Generate protobuf events for streaming"""
        for event in batch:
            try:
                pb_event = self._convert_to_protobuf(event)
                yield pb_event
            except Exception as e:
                logger.error(f"Error converting event to protobuf: {e}")
                self.metrics.increment("arc_telemetry_conversion_errors_total")

    def _send_batch(self, batch: list, stub: Any):
        """Send a batch of events using streaming RPC"""
        if not batch:
            return

        try:
            if stub and self.grpc_available:
                import grpc

                # Add authentication metadata
                metadata = []
                if self.api_key:
                    metadata.append(("x-api-key", self.api_key))

                # Send events via streaming RPC
                try:
                    response = stub.StreamTelemetry(
                        self._event_generator(batch), metadata=metadata
                    )

                    if response.success:
                        logger.debug(
                            f"Successfully sent {response.events_received} events: {response.message}"
                        )
                        self.metrics.increment("arc_telemetry_sent_total", len(batch))
                    else:
                        logger.warning(f"Failed to send telemetry: {response.message}")
                        self.metrics.increment("arc_telemetry_failed_total", len(batch))

                except grpc.RpcError as e:
                    if e.code() == grpc.StatusCode.UNAUTHENTICATED:
                        logger.error("Authentication failed. Check your API key.")
                    elif e.code() == grpc.StatusCode.UNAVAILABLE:
                        logger.debug("Arc Core server unavailable")
                    else:
                        logger.debug(f"gRPC error: {e.code()}: {e.details()}")
                    self.metrics.increment("arc_telemetry_failed_total", len(batch))
            else:
                # Log locally
                for event in batch:
                    logger.debug(f"Telemetry event (local): {event}")

        except Exception as e:
            logger.debug(f"Failed to send telemetry batch: {e}")
            self.metrics.increment("arc_telemetry_failed_total", len(batch))

    def shutdown(self):
        """Shutdown the telemetry client"""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
