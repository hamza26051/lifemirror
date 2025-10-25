#!/usr/bin/env python3
"""
Simple test script to verify posture agent functionality
"""

import os
import sys
sys.path.append('.')

from src.agents.posture_agent import PostureAgent
from src.schemas.agents import AgentInput

def test_posture_agent():
    """Test posture agent with a simple test case"""
    print("Testing Posture Agent...")
    
    # Set to mock mode for testing
    os.environ['LIFEMIRROR_MODE'] = 'mock'
    
    # Create test input
    test_input = AgentInput(
        media_id="test-123",
        url="https://example.com/test.jpg"
    )
    
    # Initialize and run posture agent
    posture_agent = PostureAgent()
    result = posture_agent.run(test_input)
    
    print(f"Posture Agent Result:")
    print(f"Success: {result.success}")
    print(f"Data: {result.data}")
    print(f"Error: {result.error}")
    
    return result.success

if __name__ == "__main__":
    success = test_posture_agent()
    if success:
        print("\n✅ Posture Agent test PASSED")
    else:
        print("\n❌ Posture Agent test FAILED")
        sys.exit(1)
