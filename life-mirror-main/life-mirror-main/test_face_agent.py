#!/usr/bin/env python3
"""
Simple test script to verify face agent functionality
"""

import os
import sys
sys.path.append('.')

from src.agents.face_agent import FaceAgent
from src.schemas.agents import AgentInput

def test_face_agent():
    """Test face agent with a simple test case"""
    print("Testing Face Agent...")
    
    # Set to mock mode for testing
    os.environ['LIFEMIRROR_MODE'] = 'mock'
    
    # Create test input
    test_input = AgentInput(
        media_id="test-123",
        url="https://example.com/test.jpg"
    )
    
    # Initialize and run face agent
    face_agent = FaceAgent()
    result = face_agent.run(test_input)
    
    print(f"Face Agent Result:")
    print(f"Success: {result.success}")
    print(f"Data: {result.data}")
    print(f"Error: {result.error}")
    
    return result.success

if __name__ == "__main__":
    success = test_face_agent()
    if success:
        print("\n✅ Face Agent test PASSED")
    else:
        print("\n❌ Face Agent test FAILED")
        sys.exit(1)
