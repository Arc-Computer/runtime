"""Multi-agent context management for Arc Runtime.

This module provides context managers and utilities for tracking multi-agent
pipeline execution, including agent handoffs, context propagation, and
comprehensive telemetry capture.
"""

import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Thread-local context for tracking agent pipeline execution
pipeline_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "pipeline_context", default=None
)


class MultiAgentContext:
    """Context manager for multi-agent pipeline execution.

    This context manager tracks the execution of multi-agent pipelines,
    capturing agent interactions, context handoffs, and performance metrics.

    Example:
        with MultiAgentContext(application_id="LOAN-2024-001") as ctx:
            # Execute multi-agent pipeline
            ctx.add_agent_execution("loan_officer", input_data, output_data, 125.5)
            ctx.add_context_handoff("loan_officer", "credit_analyst", handoff_data)
    """

    def __init__(
        self, pipeline_id: Optional[str] = None, application_id: Optional[str] = None
    ):
        """Initialize multi-agent context.

        Args:
            pipeline_id: Unique identifier for this pipeline execution.
                        If not provided, a UUID will be generated.
            application_id: Business application identifier (e.g., loan application ID)
        """
        self.pipeline_id = pipeline_id or str(uuid.uuid4())
        self.application_id = application_id
        self.agents: List[Dict[str, Any]] = []
        self.context_handoffs: List[Dict[str, Any]] = []
        self.start_time = None
        self.end_time = None
        self._token = None

    def __enter__(self):
        """Enter the context manager and set up pipeline context."""
        self.start_time = time.time()

        # Set up context variable with fresh lists and instance reference
        ctx_data = {
            "pipeline_id": self.pipeline_id,
            "application_id": self.application_id,
            "agents": [],  # Use fresh list for context var
            "context_handoffs": [],  # Use fresh list for context var
            "start_time": self.start_time,
            "instance": self,  # Store reference to MultiAgentContext instance
        }
        self._token = pipeline_context.set(ctx_data)

        logger.info(f"Started multi-agent pipeline: {self.pipeline_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and clean up pipeline context."""
        self.end_time = time.time()

        # Calculate total pipeline duration
        duration_ms = (self.end_time - self.start_time) * 1000

        logger.info(
            f"Completed multi-agent pipeline: {self.pipeline_id} "
            f"(duration: {duration_ms:.2f}ms, agents: {len(self.agents)})"
        )

        # Reset context variable
        if self._token:
            pipeline_context.reset(self._token)

        # Don't suppress exceptions
        return False

    def add_agent_execution(
        self,
        agent_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        latency_ms: float,
        agent_type: str = "llm",
    ):
        """Record an agent execution in the pipeline.

        Args:
            agent_name: Name of the agent (e.g., "loan_officer")
            input_data: Input data passed to the agent
            output_data: Output data returned by the agent
            latency_ms: Execution latency in milliseconds
            agent_type: Type of agent (llm, mcp_server, tool)
        """
        execution_record = {
            "name": agent_name,
            "type": agent_type,
            "input": input_data,
            "output": output_data,
            "latency_ms": latency_ms,
            "timestamp": time.time(),
        }

        # Update instance list
        self.agents.append(execution_record)

        # Update context var list
        ctx = pipeline_context.get()
        if ctx:
            ctx["agents"].append(execution_record)

        logger.debug(
            f"Recorded agent execution: {agent_name} "
            f"(type: {agent_type}, latency: {latency_ms:.2f}ms)"
        )

    def add_context_handoff(
        self, from_agent: str, to_agent: str, context: Dict[str, Any]
    ):
        """Record context passed between agents.

        Args:
            from_agent: Name of the source agent
            to_agent: Name of the target agent
            context: Context data being passed
        """
        handoff_record = {
            "from": from_agent,
            "to": to_agent,
            "context": context,
            "timestamp": time.time(),
        }

        # Update instance list
        self.context_handoffs.append(handoff_record)

        # Update context var list
        ctx = pipeline_context.get()
        if ctx:
            ctx["context_handoffs"].append(handoff_record)

        logger.debug(f"Recorded context handoff: {from_agent} -> {to_agent}")

    def trace_agent(self, agent_name: str):
        """Context manager for tracing individual agent execution.

        Args:
            agent_name: Name of the agent being traced

        Returns:
            AgentTracer context manager
        """
        return AgentTracer(self, agent_name)

    def finalize_pipeline(self, result: Any):
        """Finalize the pipeline with the final result.

        Args:
            result: Final result of the pipeline execution
        """
        self.end_time = time.time()
        duration_ms = (self.end_time - self.start_time) * 1000

        logger.info(
            f"Pipeline {self.pipeline_id} finalized - "
            f"Total duration: {duration_ms:.2f}ms, "
            f"Agents executed: {len(self.agents)}, "
            f"Context handoffs: {len(self.context_handoffs)}"
        )

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline execution.

        Returns:
            Dictionary containing pipeline execution summary
        """
        return {
            "pipeline_id": self.pipeline_id,
            "application_id": self.application_id,
            "agents_executed": len(self.agents),
            "context_handoffs": len(self.context_handoffs),
            "total_latency_ms": sum(
                agent.get("latency_ms", 0) for agent in self.agents
            ),
            "agents": self.agents,
            "handoffs": self.context_handoffs,
        }


class AgentTracer:
    """Context manager for tracing individual agent execution."""

    def __init__(self, parent_context: MultiAgentContext, agent_name: str):
        """Initialize agent tracer.

        Args:
            parent_context: Parent MultiAgentContext
            agent_name: Name of the agent being traced
        """
        self.parent_context = parent_context
        self.agent_name = agent_name
        self.start_time = None

    def __enter__(self):
        """Start agent execution timing."""
        self.start_time = time.perf_counter()
        logger.debug(f"Started tracing agent: {self.agent_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End agent execution timing."""
        if self.start_time:
            latency_ms = (time.perf_counter() - self.start_time) * 1000
            logger.debug(
                f"Completed tracing agent: {self.agent_name} (latency: {latency_ms:.2f}ms)"
            )
        return False


def get_current_pipeline_context() -> Optional[Dict[str, Any]]:
    """Get the current pipeline context if one exists.

    Returns:
        Current pipeline context dictionary or None
    """
    return pipeline_context.get()


def get_current_pipeline_id() -> Optional[str]:
    """Get the current pipeline ID if a context exists.

    Returns:
        Current pipeline ID or None
    """
    ctx = pipeline_context.get()
    return ctx.get("pipeline_id") if ctx else None


def get_current_application_id() -> Optional[str]:
    """Get the current application ID if a context exists.

    Returns:
        Current application ID or None
    """
    ctx = pipeline_context.get()
    return ctx.get("application_id") if ctx else None
