"""LangGraph integration for Arc Runtime.

This module provides a custom StateGraph implementation that integrates
with Arc Runtime's multi-agent context tracking, enabling comprehensive
telemetry capture for LangGraph-based agent workflows.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

try:
    from langgraph.graph import StateGraph

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    StateGraph = object  # Fallback for type hints

from runtime.multiagent.context import MultiAgentContext

logger = logging.getLogger(__name__)


class ArcStateGraph(StateGraph if HAS_LANGGRAPH else object):
    """LangGraph StateGraph with Arc Runtime integration.

    This class extends LangGraph's StateGraph to automatically track
    agent executions, context handoffs, and performance metrics through
    Arc Runtime's multi-agent context system.

    Example:
        workflow = ArcStateGraph()

        # Add nodes (agents)
        workflow.add_node("loan_officer", loan_officer_agent)
        workflow.add_node("credit_analyst", credit_analyst_agent)
        workflow.add_node("risk_manager", risk_manager_agent)

        # Add edges (handoffs)
        workflow.add_edge("loan_officer", "credit_analyst")
        workflow.add_edge("credit_analyst", "risk_manager")

        # Run with Arc tracking
        result = workflow.invoke({
            "application_id": "LOAN-2024-001",
            "loan_data": {...}
        })
    """

    def __init__(self, *args, **kwargs):
        """Initialize ArcStateGraph.

        Args:
            *args: Arguments passed to StateGraph
            **kwargs: Keyword arguments passed to StateGraph
        """
        if not HAS_LANGGRAPH:
            raise ImportError(
                "LangGraph is not installed. Install it with: pip install langgraph"
            )

        super().__init__(*args, **kwargs)
        self.arc_context: Optional[MultiAgentContext] = None
        self._node_handlers: Dict[str, Callable] = {}

        logger.info("Initialized ArcStateGraph with Arc Runtime integration")

    def add_node(self, node_name: str, node_func: Callable, **kwargs):
        """Add a node (agent) to the graph with Arc tracking.

        Args:
            node_name: Name of the node/agent
            node_func: Function that implements the agent logic
            **kwargs: Additional arguments passed to StateGraph.add_node
        """
        # Store the original handler
        self._node_handlers[node_name] = node_func

        # Create wrapped handler with Arc tracking
        def wrapped_node_func(state: Dict[str, Any]) -> Dict[str, Any]:
            return self._execute_node_with_tracking(node_name, node_func, state)

        # Add the wrapped node to the graph
        super().add_node(node_name, wrapped_node_func, **kwargs)

    def invoke(
        self, input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Invoke the graph with Arc Runtime tracking.

        Args:
            input_data: Input data for the graph execution
            config: Optional configuration for the execution

        Returns:
            Result of the graph execution
        """
        # Extract application ID if provided
        application_id = input_data.get("application_id")

        # Create multi-agent context
        with MultiAgentContext(application_id=application_id) as arc_ctx:
            self.arc_context = arc_ctx

            try:
                # Run the graph
                result = super().invoke(input_data, config)

                # Capture final pipeline state
                arc_ctx.finalize_pipeline(result)

                return result

            finally:
                self.arc_context = None

    async def ainvoke(
        self, input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Async invoke the graph with Arc Runtime tracking.

        Args:
            input_data: Input data for the graph execution
            config: Optional configuration for the execution

        Returns:
            Result of the graph execution
        """
        # Extract application ID if provided
        application_id = input_data.get("application_id")

        # Create multi-agent context
        with MultiAgentContext(application_id=application_id) as arc_ctx:
            self.arc_context = arc_ctx

            try:
                # Run the graph
                result = await super().ainvoke(input_data, config)

                # Capture final pipeline state
                arc_ctx.finalize_pipeline(result)

                return result

            finally:
                self.arc_context = None

    def _execute_node_with_tracking(
        self, node_name: str, node_func: Callable, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a node with Arc Runtime tracking.

        Args:
            node_name: Name of the node being executed
            node_func: Original node function
            state: Current state of the graph

        Returns:
            Updated state after node execution
        """
        start_time = time.perf_counter()

        # Create a copy of input state for tracking
        input_state = state.copy()

        try:
            # Execute with agent tracing if context is available
            if self.arc_context:
                with self.arc_context.trace_agent(node_name):
                    result = node_func(state)
            else:
                result = node_func(state)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Record agent execution if context is available
            if self.arc_context:
                self.arc_context.add_agent_execution(
                    agent_name=node_name,
                    input_data=input_state,
                    output_data=result,
                    latency_ms=latency_ms,
                    agent_type="langgraph_node",
                )

                # Check for context handoffs
                self._track_context_handoffs(node_name, input_state, result)

            logger.debug(f"Executed node '{node_name}' in {latency_ms:.2f}ms")

            return result

        except Exception as e:
            logger.error(f"Error in node '{node_name}': {e}")
            raise

    def _track_context_handoffs(
        self,
        current_node: str,
        input_state: Dict[str, Any],
        output_state: Dict[str, Any],
    ):
        """Track context handoffs between agents.

        Args:
            current_node: Name of the current node
            input_state: State before node execution
            output_state: State after node execution
        """
        if not self.arc_context:
            return

        # Detect what context was added/modified
        context_changes = {}

        for key, value in output_state.items():
            if key not in input_state:
                # New context added
                context_changes[key] = {"action": "added", "value": value}
            elif input_state[key] != value:
                # Context modified
                context_changes[key] = {
                    "action": "modified",
                    "old_value": input_state[key],
                    "new_value": value,
                }

        # If there are context changes, record them
        if context_changes:
            # Try to determine the next node (this is simplified)
            # In a real implementation, you'd track the graph edges
            next_node = "next_agent"  # Placeholder

            self.arc_context.add_context_handoff(
                from_agent=current_node, to_agent=next_node, context=context_changes
            )

    def add_conditional_edges(
        self,
        source: str,
        path_func: Callable,
        path_map: Optional[Dict[str, str]] = None,
    ):
        """Add conditional edges with context tracking.

        Args:
            source: Source node name
            path_func: Function that determines the path
            path_map: Optional mapping of path results to node names
        """

        # Wrap the path function to track decisions
        def wrapped_path_func(state: Dict[str, Any]) -> str:
            decision = path_func(state)

            # Log the routing decision if context is available
            if self.arc_context:
                logger.debug(f"Routing decision from '{source}': {decision}")

            return decision

        super().add_conditional_edges(source, wrapped_path_func, path_map)

    def get_graph_summary(self) -> Dict[str, Any]:
        """Get a summary of the graph structure.

        Returns:
            Dictionary containing graph metadata
        """
        return {
            "nodes": list(self._node_handlers.keys()),
            "node_count": len(self._node_handlers),
            "has_arc_context": self.arc_context is not None,
            "langgraph_version": getattr(StateGraph, "__version__", "unknown"),
        }
