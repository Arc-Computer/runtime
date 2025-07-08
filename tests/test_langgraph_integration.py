"""
Integration tests for Arc Runtime LangGraph support
"""

import os

# Add parent directory to path
import sys
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime import Arc
from runtime.multiagent.context import MultiAgentContext


class TestLangGraphIntegration(unittest.TestCase):
    """Test LangGraph integration functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.arc = Arc()

    def test_import_without_langgraph(self):
        """Test that ArcStateGraph handles missing LangGraph gracefully"""
        # Mock the import to simulate LangGraph not being installed
        with patch.dict("sys.modules", {"langgraph": None, "langgraph.graph": None}):
            # Clear the module cache
            if "runtime.integrations.langgraph" in sys.modules:
                del sys.modules["runtime.integrations.langgraph"]

            try:
                from runtime.integrations.langgraph import ArcStateGraph

                # Try to create instance - should raise ImportError
                with self.assertRaises(ImportError) as cm:
                    graph = ArcStateGraph()
                self.assertIn("LangGraph is not installed", str(cm.exception))
            except ImportError:
                # This is expected if LangGraph isn't installed
                pass

    @patch("runtime.integrations.langgraph.HAS_LANGGRAPH", True)
    @patch("runtime.integrations.langgraph.StateGraph")
    def test_arc_state_graph_initialization(self, mock_state_graph):
        """Test ArcStateGraph initialization"""
        from runtime.integrations.langgraph import ArcStateGraph

        # Create instance
        graph = ArcStateGraph()

        # Verify initialization
        self.assertIsNone(graph.arc_context)
        self.assertEqual(graph._node_handlers, {})

    @patch("runtime.integrations.langgraph.HAS_LANGGRAPH", True)
    @patch("runtime.integrations.langgraph.StateGraph")
    def test_add_node_with_tracking(self, mock_state_graph):
        """Test adding nodes with Arc tracking"""
        from runtime.integrations.langgraph import ArcStateGraph

        # Create instance
        graph = ArcStateGraph()

        # Create a mock node function
        def mock_agent(state):
            return {"result": "processed"}

        # Add node
        graph.add_node("test_agent", mock_agent)

        # Verify node handler was stored
        self.assertIn("test_agent", graph._node_handlers)
        self.assertEqual(graph._node_handlers["test_agent"], mock_agent)

        # Verify super().add_node was called with wrapped function
        mock_state_graph.add_node.assert_called_once()
        node_name, wrapped_func = mock_state_graph.add_node.call_args[0]
        self.assertEqual(node_name, "test_agent")
        self.assertNotEqual(wrapped_func, mock_agent)  # Should be wrapped

    @patch("runtime.integrations.langgraph.HAS_LANGGRAPH", True)
    @patch("runtime.integrations.langgraph.StateGraph")
    def test_invoke_with_arc_tracking(self, mock_state_graph):
        """Test invoke method with Arc tracking"""
        from runtime.integrations.langgraph import ArcStateGraph

        # Set up mock parent class behavior
        mock_state_graph.invoke = Mock(return_value={"final_result": "success"})

        # Create instance
        graph = ArcStateGraph()

        # Mock the parent's invoke method
        with patch.object(
            graph.__class__.__bases__[0],
            "invoke",
            return_value={"final_result": "success"},
        ):
            # Invoke with application ID
            result = graph.invoke(
                {"application_id": "LOAN-TEST-001", "loan_amount": 100000}
            )

        # Verify result
        self.assertEqual(result, {"final_result": "success"})

    @patch("runtime.integrations.langgraph.HAS_LANGGRAPH", True)
    @patch("runtime.integrations.langgraph.StateGraph")
    def test_node_execution_tracking(self, mock_state_graph):
        """Test that node execution is tracked properly"""
        from runtime.integrations.langgraph import ArcStateGraph

        # Create instance
        graph = ArcStateGraph()

        # Create multi-agent context
        with MultiAgentContext(application_id="TEST-LANG-001") as ctx:
            graph.arc_context = ctx

            # Create a mock node function
            def mock_agent(state):
                time.sleep(0.01)  # Simulate some work
                return {"processed": True, "score": 750}

            # Execute node with tracking
            input_state = {"loan_amount": 100000}
            result = graph._execute_node_with_tracking(
                "credit_analyst", mock_agent, input_state
            )

            # Verify result
            self.assertEqual(result["processed"], True)
            self.assertEqual(result["score"], 750)

            # Verify agent execution was tracked
            self.assertEqual(len(ctx.agents), 1)
            agent = ctx.agents[0]
            self.assertEqual(agent["name"], "credit_analyst")
            self.assertEqual(agent["type"], "langgraph_node")
            self.assertGreater(agent["latency_ms"], 0)
            self.assertEqual(agent["input"]["loan_amount"], 100000)
            self.assertEqual(agent["output"]["score"], 750)

    @patch("runtime.integrations.langgraph.HAS_LANGGRAPH", True)
    @patch("runtime.integrations.langgraph.StateGraph")
    def test_context_handoff_detection(self, mock_state_graph):
        """Test detection of context handoffs between agents"""
        from runtime.integrations.langgraph import ArcStateGraph

        # Create instance
        graph = ArcStateGraph()

        # Create multi-agent context
        with MultiAgentContext(application_id="TEST-LANG-002") as ctx:
            graph.arc_context = ctx

            # Track context handoff
            input_state = {"loan_amount": 100000}
            output_state = {
                "loan_amount": 100000,
                "credit_score": 750,  # New context added
                "risk_level": "low",  # New context added
            }

            graph._track_context_handoffs("loan_officer", input_state, output_state)

            # Verify context handoff was tracked
            self.assertEqual(len(ctx.context_handoffs), 1)
            handoff = ctx.context_handoffs[0]
            self.assertEqual(handoff["from"], "loan_officer")
            self.assertIn("credit_score", handoff["context"])
            self.assertIn("risk_level", handoff["context"])

    @patch("runtime.integrations.langgraph.HAS_LANGGRAPH", True)
    @patch("runtime.integrations.langgraph.StateGraph")
    def test_conditional_edges_with_tracking(self, mock_state_graph):
        """Test conditional edges tracking routing decisions"""
        from runtime.integrations.langgraph import ArcStateGraph

        # Create instance
        graph = ArcStateGraph()

        # Create multi-agent context
        with MultiAgentContext(application_id="TEST-LANG-003") as ctx:
            graph.arc_context = ctx

            # Create path function
            def router(state):
                if state.get("credit_score", 0) > 700:
                    return "approve"
                else:
                    return "deny"

            # Mock parent's add_conditional_edges
            with patch.object(graph.__class__.__bases__[0], "add_conditional_edges"):
                # Add conditional edges
                graph.add_conditional_edges(
                    "credit_analyst",
                    router,
                    {"approve": "loan_approver", "deny": "loan_denier"},
                )

                # Verify parent was called
                graph.__class__.__bases__[0].add_conditional_edges.assert_called_once()

                # Get the wrapped function
                args = graph.__class__.__bases__[0].add_conditional_edges.call_args[0]
                wrapped_func = args[1]

                # Test the wrapped function logs decisions
                with patch("runtime.integrations.langgraph.logger") as mock_logger:
                    decision = wrapped_func({"credit_score": 750})
                    self.assertEqual(decision, "approve")
                    mock_logger.debug.assert_called_with(
                        "Routing decision from 'credit_analyst': approve"
                    )

    @patch("runtime.integrations.langgraph.HAS_LANGGRAPH", True)
    @patch("runtime.integrations.langgraph.StateGraph")
    def test_graph_summary(self, mock_state_graph):
        """Test getting graph summary"""
        from runtime.integrations.langgraph import ArcStateGraph

        # Create instance and add nodes
        graph = ArcStateGraph()
        graph._node_handlers = {
            "loan_officer": Mock(),
            "credit_analyst": Mock(),
            "risk_manager": Mock(),
        }

        summary = graph.get_graph_summary()

        self.assertEqual(summary["node_count"], 3)
        self.assertIn("loan_officer", summary["nodes"])
        self.assertIn("credit_analyst", summary["nodes"])
        self.assertIn("risk_manager", summary["nodes"])
        self.assertFalse(summary["has_arc_context"])


class TestEndToEndMultiAgent(unittest.TestCase):
    """Test end-to-end multi-agent scenarios"""

    def setUp(self):
        """Set up test fixtures"""
        self.arc = Arc()

    @patch("runtime.integrations.langgraph.HAS_LANGGRAPH", True)
    @patch("runtime.integrations.langgraph.StateGraph")
    def test_loan_underwriting_pipeline_simulation(self, mock_state_graph):
        """Simulate a loan underwriting pipeline"""
        from runtime.integrations.langgraph import ArcStateGraph

        # Create workflow
        workflow = ArcStateGraph()

        # Define mock agents
        def loan_officer_agent(state):
            return {**state, "initial_review": "complete", "documents_verified": True}

        def credit_analyst_agent(state):
            return {**state, "credit_score": 750, "credit_history": "excellent"}

        def risk_manager_agent(state):
            risk_level = "low" if state.get("credit_score", 0) > 700 else "high"
            return {
                **state,
                "risk_assessment": risk_level,
                "final_decision": "approved" if risk_level == "low" else "denied",
            }

        # Add nodes
        workflow.add_node("loan_officer", loan_officer_agent)
        workflow.add_node("credit_analyst", credit_analyst_agent)
        workflow.add_node("risk_manager", risk_manager_agent)

        # Mock parent's invoke to execute nodes in sequence
        def mock_invoke(input_data, config=None):
            # Create context for tracking
            state = input_data.copy()

            # Execute nodes in sequence
            for node_name, node_func in workflow._node_handlers.items():
                state = workflow._execute_node_with_tracking(
                    node_name, node_func, state
                )

            return state

        with patch.object(
            workflow.__class__.__bases__[0], "invoke", side_effect=mock_invoke
        ):
            # Run pipeline
            result = workflow.invoke(
                {
                    "application_id": "LOAN-2024-TEST",
                    "loan_amount": 250000,
                    "applicant_name": "John Doe",
                }
            )

        # Verify final result
        self.assertEqual(result["application_id"], "LOAN-2024-TEST")
        self.assertEqual(result["final_decision"], "approved")
        self.assertEqual(result["risk_assessment"], "low")
        self.assertEqual(result["credit_score"], 750)

        # Verify pipeline context captured all agents
        if workflow.arc_context:
            summary = workflow.arc_context.get_pipeline_summary()
            self.assertEqual(summary["agents_executed"], 3)
            self.assertGreater(summary["total_latency_ms"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
