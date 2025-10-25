#!/usr/bin/env python3
"""
Simple test script to verify fashion agent functionality
"""

import os
import sys
sys.path.append('.')

from src.agents.fashion_agent import FashionAgent
from src.schemas.agents import AgentInput

def test_fashion_agent():
    """Test fashion agent with a simple test case"""
    print("Testing Fashion Agent...")
    
    # Set to mock mode for testing
    os.environ['LIFEMIRROR_MODE'] = 'mock'
    
    # Create test input
    test_input = AgentInput(
        media_id="test-123",
        url="https://example.com/test.jpg"
    )
    
    # Initialize and run fashion agent
    fashion_agent = FashionAgent()
    result = fashion_agent.run(test_input)
    
    print(f"Fashion Agent Result:")
    print(f"Success: {result.success}")
    print(f"Data: {result.data}")
    print(f"Error: {result.error}")
    
    return result.success

def test_fashion_agent_prod():
    """Test fashion agent in production mode to see if it hangs"""
    print("\nTesting Fashion Agent in PROD mode...")
    
    # Set to prod mode for testing
    os.environ['LIFEMIRROR_MODE'] = 'prod'
    
    # Create test input
    test_input = AgentInput(
        media_id="test-123",
        url="https://example.com/test.jpg"
    )
    
    # Initialize and run fashion agent
    fashion_agent = FashionAgent()
    
    try:
        result = fashion_agent.run(test_input)
        print(f"Fashion Agent PROD Result:")
        print(f"Success: {result.success}")
        print(f"Data: {result.data}")
        print(f"Error: {result.error}")
        return result.success
    except Exception as e:
        print(f"Fashion Agent PROD Error: {str(e)}")
        return False

if __name__ == "__main__":
    # Test mock mode first
    success_mock = test_fashion_agent()
    if success_mock:
        print("\n✅ Fashion Agent MOCK test PASSED")
    else:
        print("\n❌ Fashion Agent MOCK test FAILED")
    
    # Test prod mode
    success_prod = test_fashion_agent_prod()
    if success_prod:
        print("\n✅ Fashion Agent PROD test PASSED")
    else:
        print("\n❌ Fashion Agent PROD test FAILED")
    
    if success_mock and success_prod:
        print("\n🎉 All Fashion Agent tests PASSED")
    else:
        print("\n⚠️ Some Fashion Agent tests FAILED")
        sys.exit(1)
