#!/usr/bin/env python3
"""
LangGraph Integration Example

Demonstrates Arc Runtime's automatic tracking of LangGraph workflows
with multi-agent orchestration.
"""

import os
import sys
from typing import TypedDict, Literal

# Add parent directory to path for runtime import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

from runtime import Arc, ArcStateGraph

# Initialize Arc Runtime
arc = Arc(endpoint="grpc://localhost:50051")

# Check if LangGraph is available
try:
    from langgraph.graph import END
    HAS_LANGGRAPH = True
except ImportError:
    print("Warning: LangGraph not installed. Install with: pip install langgraph")
    HAS_LANGGRAPH = False
    END = "END"  # Mock for demo purposes


# Define the state for our workflow
class LoanApplicationState(TypedDict):
    application_id: str
    status: str
    credit_check_result: str
    document_verification: str
    risk_assessment: str
    final_decision: str
    errors: list


def check_credit(state: LoanApplicationState) -> LoanApplicationState:
    """Credit checking agent."""
    print("  [Credit Agent] Checking credit score...")
    
    # Simulate credit check
    state["credit_check_result"] = "PASS - Score: 750"
    state["status"] = "credit_checked"
    
    return state


def verify_documents(state: LoanApplicationState) -> LoanApplicationState:
    """Document verification agent."""
    print("  [Document Agent] Verifying documents...")
    
    # Simulate document verification
    state["document_verification"] = "VERIFIED - All documents present"
    state["status"] = "documents_verified"
    
    return state


def assess_risk(state: LoanApplicationState) -> LoanApplicationState:
    """Risk assessment agent."""
    print("  [Risk Agent] Assessing loan risk...")
    
    # Simulate risk assessment based on previous checks
    if "PASS" in state.get("credit_check_result", "") and "VERIFIED" in state.get("document_verification", ""):
        state["risk_assessment"] = "LOW RISK"
    else:
        state["risk_assessment"] = "HIGH RISK"
    
    state["status"] = "risk_assessed"
    
    return state


def make_decision(state: LoanApplicationState) -> LoanApplicationState:
    """Final decision agent."""
    print("  [Decision Agent] Making final decision...")
    
    # Make decision based on risk assessment
    if state.get("risk_assessment") == "LOW RISK":
        state["final_decision"] = "APPROVED"
    else:
        state["final_decision"] = "DENIED"
    
    state["status"] = "completed"
    
    return state


def should_approve(state: LoanApplicationState) -> Literal["approve", "deny"]:
    """Routing function to determine next step.
    
    Note: This function is included as an example of conditional routing
    but is not used in the current linear workflow.
    """
    if state.get("risk_assessment") == "LOW RISK":
        return "approve"
    return "deny"


def create_loan_workflow():
    """Create the loan processing workflow with Arc tracking."""
    
    if not HAS_LANGGRAPH:
        return None
    
    # Use ArcStateGraph instead of StateGraph for automatic tracking
    workflow = ArcStateGraph(LoanApplicationState)
    
    # Add nodes - each node is automatically tracked as an agent
    workflow.add_node("check_credit", check_credit)
    workflow.add_node("verify_documents", verify_documents)
    workflow.add_node("assess_risk", assess_risk)
    workflow.add_node("make_decision", make_decision)
    
    # Define the workflow edges
    workflow.set_entry_point("check_credit")
    
    # Parallel execution of credit check and document verification
    workflow.add_edge("check_credit", "verify_documents")
    workflow.add_edge("verify_documents", "assess_risk")
    workflow.add_edge("assess_risk", "make_decision")
    workflow.add_edge("make_decision", END)
    
    # Compile the workflow
    return workflow.compile()


def process_loan_application(application_id: str):
    """Process a loan application through the workflow."""
    
    print(f"\nProcessing loan application: {application_id}")
    print("-" * 50)
    
    # Create multi-agent context for tracking
    with arc.create_multiagent_context(application_id=application_id) as ctx:
        
        # Create and run the workflow
        app = create_loan_workflow()
        
        # Initial state
        initial_state = {
            "application_id": application_id,
            "status": "new",
            "errors": []
        }
        
        print("\nExecuting workflow...")
        
        # Run the workflow - Arc automatically tracks all agent executions
        if HAS_LANGGRAPH and app:
            result = app.invoke(initial_state)
        else:
            # Simulate workflow execution for demo
            result = initial_state
            for node_func in [check_credit, verify_documents, assess_risk, make_decision]:
                result = node_func(result)
        
        # Get pipeline summary
        print("\n" + "=" * 50)
        print("WORKFLOW EXECUTION SUMMARY")
        print("=" * 50)
        
        summary = ctx.get_pipeline_summary()
        print(f"Pipeline ID: {summary['pipeline_id']}")
        print(f"Application ID: {summary['application_id']}")
        print(f"Agents executed: {summary['agents_executed']}")
        print(f"Total latency: {summary['total_latency_ms']:.2f}ms")
        
        # Show execution timeline
        print("\nAgent Execution Timeline:")
        for agent in summary['agents']:
            print(f"  - {agent['name']}: {agent['latency_ms']:.2f}ms")
        
        # Final result
        print(f"\nFinal Decision: {result.get('final_decision', 'PENDING')}")
        print(f"Status: {result.get('status', 'unknown')}")
        
        return result


def main():
    """Run the LangGraph workflow example."""
    
    print("ARC RUNTIME LANGGRAPH INTEGRATION DEMO")
    print("======================================")
    
    if not HAS_LANGGRAPH:
        print("\nNOTE: Running in simulation mode. Install LangGraph for full functionality:")
        print("  pip install langgraph")
    
    # Process multiple applications to show pipeline tracking
    applications = [
        "LOAN-2024-001",
        "LOAN-2024-002",
        "LOAN-2024-003"
    ]
    
    for app_id in applications:
        result = process_loan_application(app_id)
        print("\n" + "="*70 + "\n")
    
    print("All applications processed!")
    print("View detailed telemetry in your Arc Core dashboard")


if __name__ == "__main__":
    main()