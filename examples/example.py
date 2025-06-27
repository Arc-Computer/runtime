"""
Arc Runtime - Hello World Example
Demonstrates automatic AI failure prevention and multi-agent tracking
"""

import os
import sys
import logging

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

# Enable debug logging to see Arc in action
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Import Arc Runtime BEFORE OpenAI
# This automatically patches the OpenAI client
from runtime import Arc

# Initialize Arc
arc = Arc()

# Now import and use OpenAI normally
import openai

# Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    print("Arc Runtime - Hello World Example")
    print("=" * 50)
    
    # Create OpenAI client (already protected by Arc)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nNote: OPENAI_API_KEY not set. Using test mode.")
        print("Set your OpenAI API key to make real API calls.")
        return
    
    client = openai.OpenAI(api_key=api_key)
    
    print("\nMaking API call with high temperature (0.95)...")
    print("Arc Runtime will automatically fix this to 0.7")
    print("-" * 50)
    
    try:
        # This call will be intercepted by Arc
        response = client.chat.completions.create(
            model="gpt-4.1", 
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Write a haiku about Python programming"
                }
            ],
            temperature=0.95,  # Arc will fix this to 0.7
            max_tokens=100
        )
        
        print("\nResponse received:")
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"\nNote: To run with real API calls, set OPENAI_API_KEY environment variable")
        print(f"Error: {e}")
    
    # Demonstrate multi-agent tracking (optional)
    print("\n" + "=" * 50)
    print("\nBonus: Multi-Agent Tracking Example")
    print("-" * 50)
    
    try:
        with arc.create_multiagent_context(application_id="EXAMPLE-001") as ctx:
            # Make a call with agent metadata
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi!"}],
                max_tokens=10,
                extra_headers={"X-Agent-Name": "greeting_agent"}
            )
            
            summary = ctx.get_pipeline_summary()
            print(f"Pipeline tracked {summary['agents_executed']} agent(s)")
    except Exception as e:
        print(f"Multi-agent demo skipped: {type(e).__name__}")
    
    print("\n" + "=" * 50)
    print("Check the logs above to see Arc Runtime in action!")
    print("Metrics available at: http://localhost:9090/metrics")


if __name__ == "__main__":
    main()