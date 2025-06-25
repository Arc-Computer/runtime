"""
Telemetry client for streaming events to Arc Core
"""

import os
import time
import queue
import logging
import threading
from typing import Any, Dict, Optional
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
            return True
        except ImportError:
            logger.warning(
                "gRPC not available - telemetry will be logged locally only. "
                "Install with: pip install grpcio"
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
            self.metrics.record_histogram("arc_interception_latency_ms", event["latency_ms"])
        
        # Try to enqueue
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            self.metrics.increment("arc_telemetry_dropped_total")
            logger.debug("Telemetry queue full - dropping event")
    
    def _start_worker_thread(self):
        """Start the telemetry worker thread"""
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="arc-telemetry-worker",
            daemon=True
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
                    len(batch) >= batch_size or
                    time.time() - last_flush >= batch_timeout
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
        
        # For MVP, we'll use a simple unary RPC
        # In production, this would use streaming
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        
        # We'll define a minimal stub inline for MVP
        # In production, this would use generated protobuf code
        class TelemetryStub:
            def __init__(self, channel):
                self.channel = channel
                
            def SendBatch(self, events):
                # Simplified - just log for MVP
                logger.debug(f"Would send {len(events)} events to gRPC")
                return True
        
        stub = TelemetryStub(channel)
        return channel, stub
    
    def _send_batch(self, batch: list, stub: Any):
        """Send a batch of events"""
        if not batch:
            return
            
        try:
            if stub:
                # Send to gRPC
                success = stub.SendBatch(batch)
                if success:
                    self.metrics.increment("arc_telemetry_sent_total", len(batch))
                else:
                    self.metrics.increment("arc_telemetry_failed_total", len(batch))
            else:
                # Log locally
                for event in batch:
                    logger.debug(f"Telemetry event: {event}")
                    
        except Exception as e:
            logger.debug(f"Failed to send telemetry batch: {e}")
            self.metrics.increment("arc_telemetry_failed_total", len(batch))
    
    def shutdown(self):
        """Shutdown the telemetry client"""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)