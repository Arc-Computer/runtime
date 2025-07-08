"""
Base interceptor class for all LLM providers
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from runtime.multiagent.context import get_current_pipeline_id, pipeline_context

logger = logging.getLogger(__name__)


class BaseInterceptor(ABC):
    """
    Abstract base class for LLM provider interceptors
    """

    def __init__(self, pattern_registry, telemetry_client):
        self.pattern_registry = pattern_registry
        self.telemetry_client = telemetry_client
        self._original_methods = {}

    @abstractmethod
    def patch(self):
        """Apply monkey patches to intercept API calls"""
        pass

    @abstractmethod
    def wrap_client(self, client):
        """Wrap a client instance for explicit protection"""
        pass

    def _intercept_request(
        self,
        provider: str,
        method: str,
        original_func: Callable,
        args: tuple,
        kwargs: dict,
        agent_metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Common interception logic for all providers

        Args:
            provider: Provider name (e.g., "openai")
            method: Method name (e.g., "chat.completions.create")
            original_func: Original method to call
            args: Positional arguments
            kwargs: Keyword arguments
            agent_metadata: Optional metadata about the calling agent

        Returns:
            Response from the original method (potentially with modified args)
        """
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())

        # Extract request parameters
        request_params = self._extract_params(args, kwargs)

        # Get pipeline context if available
        ctx = pipeline_context.get()
        pipeline_id = get_current_pipeline_id()

        # Extract agent name from metadata or context
        agent_name = None
        if agent_metadata:
            agent_name = agent_metadata.get("agent_name")
        elif ctx:
            # Try to infer from current execution context or kwargs
            agent_name = kwargs.get("metadata", {}).get("agent_name")

        # Start OTel trace with agent metadata
        with self.telemetry_client.trace_llm_request(
            provider,
            method,
            request_params,
            agent_name=agent_name,
            pipeline_id=pipeline_id,
        ) as span:
            # Measure Arc interception overhead
            interception_start = time.perf_counter()

            # Check for pattern match
            pattern_match = self.pattern_registry.match(request_params)
            fix_applied = None

            if pattern_match:
                # Apply fix
                fix_applied = pattern_match
                kwargs = self._apply_fix(kwargs, fix_applied)
                logger.info(
                    f"Arc Runtime: Intercepted {provider} request {request_id} - "
                    f"Applied fix: {fix_applied}"
                )

            # Record Arc intervention
            interception_latency_ms = (time.perf_counter() - interception_start) * 1000
            if span:
                self.telemetry_client.record_arc_intervention(
                    span,
                    pattern_match is not None,
                    fix_applied,
                    interception_latency_ms,
                )

            # Call original method
            try:
                response = original_func(*args, **kwargs)
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Record response details
                if span:
                    self.telemetry_client.record_llm_response(
                        span, response, latency_ms
                    )

                # Update basic metrics
                self.telemetry_client.metrics.increment(
                    "arc_requests_intercepted_total"
                )
                if pattern_match:
                    self.telemetry_client.metrics.increment("arc_pattern_matches_total")
                if fix_applied:
                    self.telemetry_client.metrics.increment("arc_fixes_applied_total")
                self.telemetry_client.metrics.record_histogram(
                    "arc_interception_latency_ms", interception_latency_ms
                )

                # Update pipeline context if in multi-agent execution
                if ctx and agent_name:
                    agent_record = {
                        "name": agent_name,
                        "type": "llm",
                        "provider": provider,
                        "method": method,
                        "pattern_matched": pattern_match is not None,
                        "fix_applied": fix_applied,
                        "latency_ms": latency_ms,
                        "timestamp": time.time(),
                    }

                    # Update context variable
                    ctx["agents"].append(agent_record)

                    # Update MultiAgentContext instance if available
                    if "instance" in ctx and hasattr(ctx["instance"], "agents"):
                        ctx["instance"].agents.append(agent_record)

                # Legacy telemetry for backward compatibility
                self._log_telemetry(
                    request_id=request_id,
                    provider=provider,
                    method=method,
                    request_params=request_params,
                    pattern_matched=pattern_match is not None,
                    fix_applied=fix_applied,
                    latency_ms=latency_ms,
                    success=True,
                )

                return response

            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000

                if span:
                    span.record_exception(e)
                    # Set error status if OTel is available
                    try:
                        from opentelemetry import trace

                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    except ImportError:
                        pass

                # Legacy telemetry for failure
                self._log_telemetry(
                    request_id=request_id,
                    provider=provider,
                    method=method,
                    request_params=request_params,
                    pattern_matched=pattern_match is not None,
                    fix_applied=fix_applied,
                    latency_ms=latency_ms,
                    success=False,
                    error=str(e),
                )

                raise

    def _extract_params(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract relevant parameters from method arguments"""
        # Default implementation - subclasses can override
        return kwargs.copy()

    def _apply_fix(self, kwargs: dict, fix: dict) -> dict:
        """Apply fix to request parameters"""
        # Create a copy to avoid modifying original
        modified_kwargs = kwargs.copy()

        for key, value in fix.items():
            modified_kwargs[key] = value

        return modified_kwargs

    def _log_telemetry(
        self,
        request_id: str,
        provider: str,
        method: str,
        request_params: dict,
        pattern_matched: bool,
        fix_applied: Optional[dict],
        latency_ms: float,
        success: bool,
        error: Optional[str] = None,
    ):
        """Log telemetry event in the format expected by Arc Core gRPC"""
        # Get pipeline context
        ctx = pipeline_context.get()
        pipeline_id = get_current_pipeline_id()

        # Build event matching the protobuf structure
        event = {
            "timestamp": time.time(),
            "request_id": request_id,
            "pipeline_id": pipeline_id or "",
            "application_id": ctx.get("application_id", "") if ctx else "",
            "agent_name": ctx.get("current_agent", "") if ctx else "",
            "llm_interaction": {
                "provider": provider,
                "model": request_params.get("model", ""),
                "request_body": request_params,
                "response_body": {},  # Will be filled by span recording
                "latency_ms": latency_ms,
                "prompt_tokens": 0,  # Will be filled by span recording
                "completion_tokens": 0,  # Will be filled by span recording
                "total_tokens": 0,  # Will be filled by span recording
            },
            "pattern_matched": pattern_matched,
            "fix_applied": fix_applied,
            "interception_latency_ms": 0.0,  # Already recorded in span
            "metadata": {
                "method": method,
                "success": str(success),
            },
        }

        # Arc intervention details are already included in the event structure

        # Add error info if present
        if error:
            event["error_info"] = {
                "error_type": (
                    type(error).__name__ if hasattr(error, "__class__") else "Unknown"
                ),
                "error_message": str(error),
                "stack_trace": "",  # TODO: Add stack trace if needed
            }

        self.telemetry_client.record(event)
