"""
Arc Runtime - Main entry point
"""

import logging
import os
import sys
import threading
from typing import Optional

from runtime.config import Config
from runtime.interceptors.mcp import MCPInterceptor
from runtime.interceptors.openai import OpenAIInterceptor
from runtime.multiagent.context import MultiAgentContext
from runtime.patterns.registry import PatternRegistry
from runtime.telemetry.client import TelemetryClient
from runtime.telemetry.metrics_server import MetricsServer

logger = logging.getLogger(__name__)


class Arc:
    """
    Arc Runtime - Zero-config AI failure prevention

    Usage:
        from runtime import Arc
        arc = Arc()  # Auto-patches supported libraries
    """

    _instance_lock = threading.Lock()
    _instance = None

    def __new__(cls, *args, **kwargs):
        # Singleton pattern for default instance
        if not args and not kwargs:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                return cls._instance
        return super().__new__(cls)

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        log_level: Optional[str] = None,
    ):
        # Skip re-initialization for singleton
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self.config = Config(
            endpoint=endpoint,
            api_key=api_key,
            cache_dir=cache_dir,
            log_level=log_level,
        )

        # Initialize components
        self.pattern_registry = PatternRegistry()
        self.telemetry_client = TelemetryClient(
            endpoint=self.config.endpoint,
            api_key=self.config.api_key,
        )

        # Initialize interceptors
        self.interceptors = {
            "openai": OpenAIInterceptor(
                pattern_registry=self.pattern_registry,
                telemetry_client=self.telemetry_client,
            ),
            "mcp": MCPInterceptor(
                pattern_registry=self.pattern_registry,
                telemetry_client=self.telemetry_client,
            ),
        }

        # Start metrics server
        self.metrics_server = MetricsServer(
            metrics_provider=self.telemetry_client.metrics, port=9090
        )
        self.metrics_server.start()

        # Apply patches
        self._apply_patches()

        logger.info(
            f"Arc Runtime initialized (endpoint={self.config.endpoint}, "
            f"version={self._get_version()})"
        )

    def _apply_patches(self):
        """Apply all interceptor patches"""
        for name, interceptor in self.interceptors.items():
            try:
                interceptor.patch()
                logger.debug(f"Successfully patched {name}")
            except Exception as e:
                logger.warning(f"Failed to patch {name}: {e}")

    def _get_version(self):
        """Get runtime version"""
        try:
            from runtime import __version__

            return __version__
        except ImportError:
            return "unknown"

    def register_pattern(self, pattern: dict, fix: dict):
        """
        Register a custom pattern and fix

        Args:
            pattern: Pattern to match (e.g., {"model": "gpt-4", "temperature": {">": 0.9}})
            fix: Fix to apply (e.g., {"temperature": 0.7})
        """
        self.pattern_registry.register(pattern, fix)

    def wrap(self, client):
        """
        Explicitly wrap a client (fallback if auto-patching fails)

        Args:
            client: The client instance to wrap

        Returns:
            Wrapped client with Arc protection
        """
        client_type = type(client).__module__.split(".")[0]
        interceptor = self.interceptors.get(client_type)

        if interceptor:
            return interceptor.wrap_client(client)

        logger.warning(f"No interceptor available for {client_type}")
        return client

    def create_multiagent_context(
        self, pipeline_id: Optional[str] = None, application_id: Optional[str] = None
    ) -> MultiAgentContext:
        """
        Create a multi-agent context for tracking pipeline execution

        Args:
            pipeline_id: Optional pipeline identifier
            application_id: Optional business application ID

        Returns:
            MultiAgentContext instance

        Example:
            with arc.create_multiagent_context(application_id="LOAN-001") as ctx:
                # Execute multi-agent pipeline
                pass
        """
        return MultiAgentContext(pipeline_id=pipeline_id, application_id=application_id)
