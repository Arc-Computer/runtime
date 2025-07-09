"""
OpenTelemetry-compatible telemetry client for Arc Runtime
Captures comprehensive LLM/agent telemetry including:
- Full request/response traces
- Token usage metrics
- Reasoning traces
- Tool calls (including MCP)
- Agent trajectories
- Latency metrics
"""

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry components
try:
    from opentelemetry import trace
except ImportError:
    trace = None

# OpenTelemetry semantic conventions for Gen AI
# Based on: https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_MESSAGES = "gen_ai.request.messages"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reason"
GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

# Arc-specific attributes
ARC_PATTERN_MATCHED = "arc.pattern_matched"
ARC_FIX_APPLIED = "arc.fix_applied"
ARC_INTERCEPTION_LATENCY_MS = "arc.interception_latency_ms"

# Agent-specific attributes
AGENT_REASONING_TRACE = "agent.reasoning_trace"
AGENT_TOOL_CALLS = "agent.tool_calls"
AGENT_MCP_CALLS = "agent.mcp_calls"
AGENT_TRAJECTORY = "agent.trajectory"
AGENT_USER_INPUT = "agent.user_input"
AGENT_OUTPUT = "agent.output"

# Multi-agent specific attributes
AGENT_NAME = "agent.name"
AGENT_TYPE = "agent.type"  # llm, mcp_server, tool
PIPELINE_ID = "pipeline.id"
APPLICATION_ID = "application.id"
CONTEXT_HANDOFF = "context.handoff"
FAILURE_TYPE = "failure.type"
BUSINESS_IMPACT = "business.impact"


class OTelTelemetryClient:
    """
    OpenTelemetry-compatible telemetry client
    """

    def __init__(self, service_name: str = "arc-runtime"):
        self.service_name = service_name
        self.tracer = None
        self.meter = None
        self._init_otel()

    def _init_otel(self):
        """Initialize OpenTelemetry components if available"""
        try:
            from opentelemetry import metrics, trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            # Create resource
            resource = Resource.create(
                {
                    "service.name": self.service_name,
                    "service.version": "0.1.5",
                }
            )

            # Setup tracing
            trace_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(trace_provider)

            # Add OTLP exporter if available
            try:
                otlp_exporter = OTLPSpanExporter(
                    endpoint="localhost:4317", insecure=True
                )
                span_processor = BatchSpanProcessor(otlp_exporter)
                trace_provider.add_span_processor(span_processor)
            except Exception as e:
                logger.debug(f"OTLP exporter not available: {e}")

            # Get tracer
            self.tracer = trace.get_tracer(__name__, "0.1.5")

            # Setup metrics
            metrics_provider = MeterProvider(resource=resource)
            metrics.set_meter_provider(metrics_provider)
            self.meter = metrics.get_meter(__name__, "0.1.5")

            # Create metrics
            self._create_metrics()

            logger.info("OpenTelemetry initialized successfully")

        except ImportError:
            logger.warning(
                "OpenTelemetry not available - telemetry will use basic logging. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk "
                "opentelemetry-exporter-otlp"
            )

    def _create_metrics(self):
        """Create OTel metrics"""
        if not self.meter:
            return

        # Counters
        self.request_counter = self.meter.create_counter(
            name="arc_requests_total",
            description="Total number of LLM requests intercepted",
            unit="1",
        )

        self.fix_counter = self.meter.create_counter(
            name="arc_fixes_total",
            description="Total number of fixes applied",
            unit="1",
        )

        self.token_counter = self.meter.create_counter(
            name="llm_tokens_total", description="Total tokens used", unit="1"
        )

        # Histograms
        self.latency_histogram = self.meter.create_histogram(
            name="arc_interception_latency",
            description="Latency of Arc interception",
            unit="ms",
        )

        self.request_duration_histogram = self.meter.create_histogram(
            name="llm_request_duration",
            description="Duration of LLM requests",
            unit="s",
        )

    @contextmanager
    def trace_llm_request(
        self,
        provider: str,
        method: str,
        request_params: dict,
        agent_name: Optional[str] = None,
        pipeline_id: Optional[str] = None,
    ):
        """
        Context manager for tracing LLM requests with full OTel support
        """
        if self.tracer:
            # Create span with Gen AI semantic conventions
            span_kind = trace.SpanKind.CLIENT if trace else None
            with self.tracer.start_as_current_span(
                name=f"{provider}.{method}",
                kind=span_kind,
            ) as span:
                # Add Gen AI attributes
                span.set_attribute(GEN_AI_SYSTEM, provider)
                span.set_attribute(
                    GEN_AI_REQUEST_MODEL, request_params.get("model", "unknown")
                )

                if "temperature" in request_params:
                    span.set_attribute(
                        GEN_AI_REQUEST_TEMPERATURE, request_params["temperature"]
                    )
                if "top_p" in request_params:
                    span.set_attribute(GEN_AI_REQUEST_TOP_P, request_params["top_p"])
                if "max_tokens" in request_params:
                    span.set_attribute(
                        GEN_AI_REQUEST_MAX_TOKENS, request_params["max_tokens"]
                    )

                # Add messages (truncated for size)
                messages = request_params.get("messages", [])
                if messages:
                    span.set_attribute(
                        GEN_AI_REQUEST_MESSAGES, json.dumps(messages)[:1000]
                    )

                # Add user input if available
                if messages and messages[-1].get("role") == "user":
                    span.set_attribute(
                        AGENT_USER_INPUT, messages[-1].get("content", "")[:500]
                    )

                # Add multi-agent attributes
                if agent_name:
                    span.set_attribute(AGENT_NAME, agent_name)
                if pipeline_id:
                    span.set_attribute(PIPELINE_ID, pipeline_id)

                yield span
        else:
            # Fallback context manager
            yield None

    def record_llm_response(self, span, response: Any, latency_ms: float):
        """Record LLM response details to span"""
        if not span:
            return

        try:
            # Add response attributes
            if hasattr(response, "id"):
                span.set_attribute(GEN_AI_RESPONSE_ID, response.id)
            if hasattr(response, "model"):
                span.set_attribute(GEN_AI_RESPONSE_MODEL, response.model)

            # Extract completion details
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "finish_reason"):
                    span.set_attribute(
                        GEN_AI_RESPONSE_FINISH_REASON, choice.finish_reason
                    )

                # Capture agent output
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    span.set_attribute(AGENT_OUTPUT, choice.message.content[:500])

                # Capture tool calls
                if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                    tool_calls = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": (
                                {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments[:200],
                                }
                                if hasattr(tc, "function")
                                else None
                            ),
                        }
                        for tc in choice.message.tool_calls
                    ]
                    span.set_attribute(AGENT_TOOL_CALLS, json.dumps(tool_calls))

            # Token usage
            if hasattr(response, "usage"):
                usage = response.usage
                if hasattr(usage, "prompt_tokens"):
                    span.set_attribute(GEN_AI_USAGE_PROMPT_TOKENS, usage.prompt_tokens)
                    if self.token_counter:
                        self.token_counter.add(usage.prompt_tokens, {"type": "prompt"})
                if hasattr(usage, "completion_tokens"):
                    span.set_attribute(
                        GEN_AI_USAGE_COMPLETION_TOKENS, usage.completion_tokens
                    )
                    if self.token_counter:
                        self.token_counter.add(
                            usage.completion_tokens, {"type": "completion"}
                        )
                if hasattr(usage, "total_tokens"):
                    span.set_attribute(GEN_AI_USAGE_TOTAL_TOKENS, usage.total_tokens)

            # Record latency
            span.set_attribute("latency_ms", latency_ms)
            if self.request_duration_histogram:
                self.request_duration_histogram.record(latency_ms / 1000)

        except Exception as e:
            logger.error(f"Error recording LLM response: {e}")

    def record_arc_intervention(
        self,
        span,
        pattern_matched: bool,
        fix_applied: Optional[dict],
        interception_latency_ms: float,
    ):
        """Record Arc-specific intervention details"""
        if not span:
            return

        span.set_attribute(ARC_PATTERN_MATCHED, pattern_matched)
        if fix_applied:
            span.set_attribute(ARC_FIX_APPLIED, json.dumps(fix_applied))
            if self.fix_counter:
                self.fix_counter.add(1)

        span.set_attribute(ARC_INTERCEPTION_LATENCY_MS, interception_latency_ms)
        if self.latency_histogram:
            self.latency_histogram.record(interception_latency_ms)

    def record_agent_trajectory(self, span, trajectory: List[Dict[str, Any]]):
        """Record agent trajectory (sequence of actions/thoughts)"""
        if not span:
            return

        # Truncate for size
        trajectory_summary = json.dumps(trajectory[:10])[:2000]
        span.set_attribute(AGENT_TRAJECTORY, trajectory_summary)

    def record_reasoning_trace(self, span, reasoning: str):
        """Record agent reasoning trace"""
        if not span:
            return

        span.set_attribute(AGENT_REASONING_TRACE, reasoning[:1000])

    def record_mcp_calls(self, span, mcp_calls: List[Dict[str, Any]]):
        """Record MCP (Model Context Protocol) calls"""
        if not span:
            return

        mcp_summary = json.dumps(mcp_calls[:5])[:1000]
        span.set_attribute(AGENT_MCP_CALLS, mcp_summary)

    def create_event(self, name: str, attributes: Dict[str, Any]):
        """Create an OTel event"""
        if self.tracer:
            span = trace.get_current_span()
            if span:
                span.add_event(name, attributes)
        else:
            # Fallback to logging
            logger.info(f"Event: {name} - {attributes}")

    @contextmanager
    def trace_pipeline_execution(
        self, pipeline_id: str, application_id: Optional[str] = None
    ):
        """Trace entire multi-agent pipeline execution"""
        if self.tracer:
            with self.tracer.start_as_current_span(
                name="loan_underwriting_pipeline",
                kind=trace.SpanKind.INTERNAL,
            ) as span:
                span.set_attribute(PIPELINE_ID, pipeline_id)
                if application_id:
                    span.set_attribute(APPLICATION_ID, application_id)
                yield span
        else:
            yield None

    @contextmanager
    def trace_agent_execution(self, agent_name: str, agent_type: str = "llm"):
        """Trace individual agent execution within pipeline"""
        if self.tracer:
            with self.tracer.start_as_current_span(
                name=f"agent.{agent_name}",
                kind=trace.SpanKind.INTERNAL,
            ) as span:
                span.set_attribute(AGENT_NAME, agent_name)
                span.set_attribute(AGENT_TYPE, agent_type)
                yield span
        else:
            yield None

    def record_context_handoff(
        self, span, from_agent: str, to_agent: str, context_data: dict
    ):
        """Record context passed between agents"""
        if not span:
            return

        handoff_event = {
            "from": from_agent,
            "to": to_agent,
            "context_keys": list(context_data.keys()),
            "context_size": len(json.dumps(context_data)),
        }

        span.add_event("context_handoff", handoff_event)
        span.set_attribute(
            f"{CONTEXT_HANDOFF}.{from_agent}_to_{to_agent}", json.dumps(handoff_event)
        )

    def record_failure(self, span, failure_type: str, business_impact: str = "medium"):
        """Record a detected failure for training data"""
        if not span:
            return

        span.set_attribute(FAILURE_TYPE, failure_type)
        span.set_attribute(BUSINESS_IMPACT, business_impact)
        span.set_attribute("failure_detected", True)

        # Add failure event
        span.add_event(
            "failure_detected",
            {"type": failure_type, "impact": business_impact, "timestamp": time.time()},
        )

    @contextmanager
    def trace_mcp_call(
        self,
        endpoint: str,
        method: str,
        agent_name: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        request_data: Optional[dict] = None,
    ):
        """Trace MCP server calls"""
        if self.tracer:
            with self.tracer.start_as_current_span(
                name=f"mcp.{endpoint.split('/')[-1]}",
                kind=trace.SpanKind.CLIENT,
            ) as span:
                span.set_attribute("mcp.endpoint", endpoint)
                span.set_attribute("mcp.method", method)
                if agent_name:
                    span.set_attribute(AGENT_NAME, agent_name)
                if pipeline_id:
                    span.set_attribute(PIPELINE_ID, pipeline_id)
                if request_data:
                    span.set_attribute("mcp.request", json.dumps(request_data)[:1000])
                yield span
        else:
            yield None

    def record_mcp_response(
        self, span, response_data: Optional[dict], latency_ms: float, status_code: int
    ):
        """Record MCP server response"""
        if not span:
            return

        span.set_attribute("mcp.status_code", status_code)
        span.set_attribute("mcp.latency_ms", latency_ms)
        if response_data:
            span.set_attribute("mcp.response", json.dumps(response_data)[:1000])
