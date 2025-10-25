import os
import sys
sys.path.append('src')

from src.agents.fashion_agent import FashionAgent
from src.schemas.agents import AgentInput

# Set to production mode
os.environ["LIFEMIRROR_MODE"] = "prod"

def test_fashion_agent_production():
    print("Testing Fashion Agent in PRODUCTION mode...")
    
    agent = FashionAgent()
    
    # Use a real image URL that should work
    test_input = AgentInput(
        media_id="test_fashion_prod",
        url="https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=800&h=600&fit=crop",  # Fashion image from Unsplash
        user_id="test_user",
        metadata={}
    )
    
    try:
        result = agent.run(test_input)
        print(f"Production mode result: {result}")
        
        if result.success:
            print("✅ Fashion Agent working in PRODUCTION mode!")
            print(f"Detected items: {len(result.data.get('items', []))}")
            print(f"Style: {result.data.get('style', 'unknown')}")
        else:
            print(f"❌ Fashion Agent failed: {result.error}")
            
    except Exception as e:
        print(f"❌ Exception in production mode: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fashion_agent_production()