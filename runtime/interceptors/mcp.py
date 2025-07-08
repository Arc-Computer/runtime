"""MCP (Model Context Protocol) interceptor for Arc Runtime.

This module provides interception capabilities for MCP server calls,
enabling comprehensive tracing of multi-agent interactions that use
the Model Context Protocol.
"""

import json
import logging
import time
from contextlib import suppress
from typing import Any, Callable, Dict, Optional

import httpx

from runtime.interceptors.base import BaseInterceptor
from runtime.multiagent.context import get_current_pipeline_id, pipeline_context

logger = logging.getLogger(__name__)


class MCPInterceptor(BaseInterceptor):
    """Interceptor for MCP (Model Context Protocol) server calls.

    This interceptor patches HTTP clients (httpx, requests) to capture
    MCP server interactions, providing visibility into tool calls,
    resource access, and agent communications.
    """

    def __init__(self, pattern_registry, telemetry_client):
        """Initialize MCP interceptor.

        Args:
            pattern_registry: Pattern registry instance
            telemetry_client: Telemetry client instance
        """
        super().__init__(pattern_registry, telemetry_client)
        self._original_httpx_request = None
        self._original_httpx_async_request = None

    def patch(self):
        """Patch HTTP clients for MCP server interception."""
        try:
            self._patch_httpx()
            logger.info("MCP interceptor successfully patched httpx")
        except Exception as e:
            logger.error(f"Failed to patch MCP interceptor: {e}")

    def _patch_httpx(self):
        """Patch httpx for MCP interception."""
        # Store original methods
        self._original_httpx_request = httpx.Client.request
        self._original_httpx_async_request = httpx.AsyncClient.request

        # Create interceptor reference
        interceptor = self

        # Patch synchronous client
        def patched_sync_request(client_self, method: str, url, **kwargs):
            """Patched synchronous request method."""
            # Check if this is an MCP call
            if interceptor._is_mcp_call(str(url), kwargs.get("headers", {})):
                return interceptor._intercept_sync_mcp_request(
                    original_func=interceptor._original_httpx_request,
                    client_self=client_self,
                    method=method,
                    url=url,
                    kwargs=kwargs,
                )

            # Not an MCP call, proceed normally
            return interceptor._original_httpx_request(
                client_self, method, url, **kwargs
            )

        # Patch asynchronous client
        async def patched_async_request(client_self, method: str, url, **kwargs):
            """Patched asynchronous request method."""
            # Check if this is an MCP call
            if interceptor._is_mcp_call(str(url), kwargs.get("headers", {})):
                return await interceptor._intercept_async_mcp_request(
                    original_func=interceptor._original_httpx_async_request,
                    client_self=client_self,
                    method=method,
                    url=url,
                    kwargs=kwargs,
                )

            # Not an MCP call, proceed normally
            return await interceptor._original_httpx_async_request(
                client_self, method, url, **kwargs
            )

        # Apply patches
        httpx.Client.request = patched_sync_request
        httpx.AsyncClient.request = patched_async_request

    def _is_mcp_call(self, url: str, headers: Dict[str, str]) -> bool:
        """Check if a request is an MCP call.

        Args:
            url: Request URL
            headers: Request headers

        Returns:
            True if this is an MCP call
        """
        # Check for MCP indicators
        mcp_indicators = [
            "/mcp/" in url,
            "/tools/call" in url,
            "/resources/read" in url,
            "/prompts/get" in url,
            headers.get("X-MCP-Protocol") is not None,
            headers.get("Content-Type") == "application/x-mcp+json",
        ]

        return any(mcp_indicators)

    def _extract_mcp_metadata(
        self, url: str, method: str, headers: Dict[str, str], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract MCP-specific metadata from request.

        Args:
            url: Request URL
            method: HTTP method
            headers: Request headers
            kwargs: Request keyword arguments

        Returns:
            Dictionary of MCP metadata
        """
        # Parse URL to extract endpoint info
        url_str = str(url)
        endpoint_parts = url_str.split("/")

        # Determine MCP operation type
        operation = "unknown"
        if "/tools/call" in url_str:
            operation = "tool_call"
        elif "/resources/read" in url_str:
            operation = "resource_read"
        elif "/prompts/get" in url_str:
            operation = "prompt_get"
        elif "/sampling/createMessage" in url_str:
            operation = "sampling"

        return {
            "mcp_endpoint": url_str,
            "mcp_method": method,
            "mcp_operation": operation,
            "agent_name": headers.get("X-Agent-Name"),
            "mcp_protocol_version": headers.get("X-MCP-Version", "1.0"),
            "request_id": headers.get("X-Request-ID"),
        }

    def _intercept_sync_mcp_request(
        self,
        original_func: Callable,
        client_self: httpx.Client,
        method: str,
        url,
        kwargs: Dict[str, Any],
    ) -> httpx.Response:
        """Intercept synchronous MCP server requests.

        Args:
            original_func: Original request function
            client_self: httpx client instance
            method: HTTP method
            url: Request URL
            kwargs: Request keyword arguments

        Returns:
            HTTP response
        """
        start_time = time.perf_counter()

        # Extract metadata
        headers = kwargs.get("headers", {})
        mcp_metadata = self._extract_mcp_metadata(url, method, headers, kwargs)

        # Extract request body
        request_data = None
        if "json" in kwargs:
            request_data = kwargs["json"]
        elif "data" in kwargs:
            with suppress(Exception):
                request_data = json.loads(kwargs["data"])

        # Get pipeline context
        ctx = pipeline_context.get()
        pipeline_id = get_current_pipeline_id()

        # Create telemetry span
        span = None
        if self.telemetry_client:
            span = self.telemetry_client.trace_mcp_call(
                endpoint=str(url),
                method=method,
                agent_name=mcp_metadata.get("agent_name"),
                pipeline_id=pipeline_id,
                request_data=request_data,
            ).__enter__()

        try:
            # Make the actual request
            response = original_func(client_self, method, url, **kwargs)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract response data
            response_data = None
            if response.status_code == 200:
                with suppress(Exception):
                    response_data = response.json()

            # Record telemetry
            if self.telemetry_client and span:
                self.telemetry_client.record_mcp_response(
                    span, response_data, latency_ms, status_code=response.status_code
                )

            # Update pipeline context
            if ctx and mcp_metadata.get("agent_name"):
                agent_record = {
                    "name": mcp_metadata["agent_name"],
                    "type": "mcp_server",
                    "operation": mcp_metadata["mcp_operation"],
                    "endpoint": str(url),
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                }

                # Update context variable
                ctx["agents"].append(agent_record)

                # Update MultiAgentContext instance if available
                if "instance" in ctx and hasattr(ctx["instance"], "agents"):
                    ctx["instance"].agents.append(agent_record)

            # Log the interception
            logger.debug(
                f"MCP request intercepted: {method} {url} "
                f"(operation: {mcp_metadata['mcp_operation']}, "
                f"latency: {latency_ms:.2f}ms)"
            )

            return response

        except Exception as e:
            if span:
                span.record_exception(e)
            raise
        finally:
            if span:
                span.__exit__(None, None, None)

    async def _intercept_async_mcp_request(
        self,
        original_func: Callable,
        client_self: httpx.AsyncClient,
        method: str,
        url,
        kwargs: Dict[str, Any],
    ) -> httpx.Response:
        """Intercept asynchronous MCP server requests.

        Args:
            original_func: Original request function
            client_self: httpx async client instance
            method: HTTP method
            url: Request URL
            kwargs: Request keyword arguments

        Returns:
            HTTP response
        """
        start_time = time.perf_counter()

        # Extract metadata
        headers = kwargs.get("headers", {})
        mcp_metadata = self._extract_mcp_metadata(url, method, headers, kwargs)

        # Extract request body
        request_data = None
        if "json" in kwargs:
            request_data = kwargs["json"]
        elif "data" in kwargs:
            with suppress(Exception):
                request_data = json.loads(kwargs["data"])

        # Get pipeline context
        ctx = pipeline_context.get()
        pipeline_id = get_current_pipeline_id()

        # Create telemetry span
        span = None
        if self.telemetry_client:
            span = self.telemetry_client.trace_mcp_call(
                endpoint=str(url),
                method=method,
                agent_name=mcp_metadata.get("agent_name"),
                pipeline_id=pipeline_id,
                request_data=request_data,
            ).__enter__()

        try:
            # Make the actual request
            response = await original_func(client_self, method, url, **kwargs)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract response data
            response_data = None
            if response.status_code == 200:
                with suppress(Exception):
                    response_data = response.json()

            # Record telemetry
            if self.telemetry_client and span:
                self.telemetry_client.record_mcp_response(
                    span, response_data, latency_ms, status_code=response.status_code
                )

            # Update pipeline context
            if ctx and mcp_metadata.get("agent_name"):
                agent_record = {
                    "name": mcp_metadata["agent_name"],
                    "type": "mcp_server",
                    "operation": mcp_metadata["mcp_operation"],
                    "endpoint": str(url),
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                }

                # Update context variable
                ctx["agents"].append(agent_record)

                # Update MultiAgentContext instance if available
                if "instance" in ctx and hasattr(ctx["instance"], "agents"):
                    ctx["instance"].agents.append(agent_record)

            # Log the interception
            logger.debug(
                f"Async MCP request intercepted: {method} {url} "
                f"(operation: {mcp_metadata['mcp_operation']}, "
                f"latency: {latency_ms:.2f}ms)"
            )

            return response

        except Exception as e:
            if span:
                span.record_exception(e)
            raise
        finally:
            if span:
                span.__exit__(None, None, None)

    def unpatch(self):
        """Remove MCP interception patches."""
        if self._original_httpx_request:
            httpx.Client.request = self._original_httpx_request

        if self._original_httpx_async_request:
            httpx.AsyncClient.request = self._original_httpx_async_request

        logger.info("MCP interceptor unpatched")

    def wrap_client(self, client):
        """Wrap a client instance for explicit protection.

        MCP interceptor works at the HTTP level, so individual client
        wrapping is not necessary. The patch() method handles all interception.

        Args:
            client: The client instance to wrap

        Returns:
            The original client (unchanged)
        """
        logger.debug("MCP interceptor does not require client wrapping")
        return client
