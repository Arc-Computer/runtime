#!/usr/bin/env python3
"""
Multi-Agent Pipeline Example

Demonstrates Arc Runtime's multi-agent tracking capabilities with a loan
underwriting pipeline that includes multiple specialized agents.
"""

import os
import sys

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

from runtime import Arc
import openai

# Initialize Arc Runtime (singleton will handle duplicate calls)
arc = Arc(endpoint="grpc://localhost:50051")

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def loan_underwriting_pipeline(application_data: dict):
    """
    Execute a multi-agent loan underwriting pipeline.
    
    This pipeline demonstrates:
    - Multiple agent execution tracking
    - Context handoffs between agents
    - Pipeline telemetry and summarization
    """
    
    # Start multi-agent context for the loan application
    with arc.create_multiagent_context(
        application_id=application_data["application_id"]
    ) as ctx:
        
        print(f"Starting loan underwriting pipeline: {ctx.pipeline_id}")
        print(f"Application ID: {application_data['application_id']}")
        print("-" * 50)
        
        # Step 1: Loan Officer reviews initial application
        print("\n1. Loan Officer reviewing application...")
        loan_officer_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a loan officer reviewing applications."
                },
                {
                    "role": "user",
                    "content": f"Review this loan application: {application_data}"
                }
            ],
            max_tokens=200,
            extra_headers={"X-Agent-Name": "loan_officer"}
        )
        
        initial_assessment = loan_officer_response.choices[0].message.content
        print(f"   Initial assessment: {initial_assessment[:100]}...")
        
        # Context handoff to credit analyst
        ctx.add_context_handoff(
            from_agent="loan_officer",
            to_agent="credit_analyst",
            context={
                "loan_amount": application_data["loan_amount"],
                "initial_assessment": initial_assessment,
                "applicant_id": application_data["applicant_id"]
            }
        )
        
        # Step 2: Credit Analyst reviews credit history
        print("\n2. Credit Analyst reviewing credit history...")
        credit_analyst_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a credit analyst reviewing credit scores."
                },
                {
                    "role": "user",
                    "content": f"Analyze credit score {application_data['credit_score']} "
                             f"for loan amount ${application_data['loan_amount']}"
                }
            ],
            max_tokens=150,
            extra_headers={"X-Agent-Name": "credit_analyst"}
        )
        
        credit_assessment = credit_analyst_response.choices[0].message.content
        print(f"   Credit assessment: {credit_assessment[:100]}...")
        
        # Context handoff to risk manager
        ctx.add_context_handoff(
            from_agent="credit_analyst",
            to_agent="risk_manager",
            context={
                "credit_score": application_data["credit_score"],
                "credit_assessment": credit_assessment,
                "loan_amount": application_data["loan_amount"],
                "debt_to_income": application_data.get("debt_to_income", 0.3)
            }
        )
        
        # Step 3: Risk Manager performs final assessment
        print("\n3. Risk Manager assessing overall risk...")
        risk_manager_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a risk manager making final loan decisions."
                },
                {
                    "role": "user",
                    "content": f"Assess risk for: Credit Score: {application_data['credit_score']}, "
                             f"DTI: {application_data.get('debt_to_income', 0.3)}, "
                             f"Loan: ${application_data['loan_amount']}"
                }
            ],
            max_tokens=150,
            extra_headers={"X-Agent-Name": "risk_manager"}
        )
        
        risk_assessment = risk_manager_response.choices[0].message.content
        print(f"   Risk assessment: {risk_assessment[:100]}...")
        
        # Get pipeline summary
        print("\n" + "=" * 50)
        print("PIPELINE SUMMARY")
        print("=" * 50)
        
        summary = ctx.get_pipeline_summary()
        print(f"Pipeline ID: {summary['pipeline_id']}")
        print(f"Application ID: {summary['application_id']}")
        print(f"Agents executed: {summary['agents_executed']}")
        print(f"Context handoffs: {summary['context_handoffs']}")
        print(f"Total latency: {summary['total_latency_ms']:.2f}ms")
        
        # Show agent execution details
        print("\nAgent Execution Timeline:")
        for agent in summary['agents']:
            print(f"  - {agent['name']}: {agent['latency_ms']:.2f}ms "
                  f"(Pattern matched: {agent.get('pattern_matched', False)})")
        
        # Show context handoffs
        print("\nContext Handoffs:")
        for handoff in summary['handoffs']:
            print(f"  - {handoff['from']} → {handoff['to']}")
        
        return {
            "approved": "approved" in risk_assessment.lower(),
            "pipeline_id": ctx.pipeline_id,
            "summary": summary
        }


def main():
    """Run example loan underwriting pipeline."""
    
    # Example loan application
    application = {
        "application_id": "LOAN-2024-12345",
        "applicant_id": "APP-98765",
        "loan_amount": 250000,
        "credit_score": 750,
        "debt_to_income": 0.28,
        "employment_years": 5,
        "down_payment": 50000
    }
    
    print("ARC RUNTIME MULTI-AGENT PIPELINE DEMO")
    print("=====================================\n")
    
    # Execute pipeline
    result = loan_underwriting_pipeline(application)
    
    print(f"\nFinal Decision: {'APPROVED' if result['approved'] else 'DENIED'}")
    print(f"\nView full telemetry at your Arc Core dashboard")
    print(f"Pipeline ID: {result['pipeline_id']}")


if __name__ == "__main__":
    main()