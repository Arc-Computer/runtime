"""
Simple HTTP server for Prometheus metrics endpoint
"""

import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger(__name__)


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP request handler for metrics endpoint"""

    def __init__(self, metrics_provider, *args, **kwargs):
        self.metrics_provider = metrics_provider
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()

            # Get metrics from provider
            metrics_text = self.metrics_provider.get_prometheus_metrics()
            self.wfile.write(metrics_text.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.debug(f"Metrics server: {format % args}")


class MetricsServer:
    """
    Simple HTTP server for exposing Prometheus metrics
    """

    def __init__(self, metrics_provider, port: int = 9090):
        self.metrics_provider = metrics_provider
        self.port = port
        self.server = None
        self.server_thread = None

    def start(self):
        """Start the metrics server in a background thread"""

        # Create handler factory that includes metrics provider
        def handler_factory(*args, **kwargs):
            return MetricsHandler(self.metrics_provider, *args, **kwargs)

        try:
            self.server = HTTPServer(("localhost", self.port), handler_factory)
            self.server_thread = threading.Thread(
                target=self.server.serve_forever, name="arc-metrics-server", daemon=True
            )
            self.server_thread.start()
            logger.info(
                f"Metrics server started on http://localhost:{self.port}/metrics"
            )
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")

    def stop(self):
        """Stop the metrics server"""
        if self.server:
            self.server.shutdown()
            self.server_thread.join(timeout=5.0)
