import os
import sys
sys.path.append('src')

from src.agents.fashion_agent import FashionAgent
from src.schemas.agents import AgentInput

# Set to production mode
os.environ["LIFEMIRROR_MODE"] = "prod"

def test_fashion_agent_with_real_image():
    print("Testing Fashion Agent with real image in PRODUCTION mode...")
    
    agent = FashionAgent()
    
    # Use direct file path (not file:// URL)
    import os
    current_dir = os.getcwd()
    test_image_path = os.path.join(current_dir, "test_person.jpg")
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    test_input = AgentInput(
        media_id="test_fashion_direct",
        url=test_image_path,  # Direct file path
        user_id="test_user",
        metadata={}
    )
    
    try:
        print(f"Testing with image: {test_image_path}")
        result = agent.run(test_input)
        print(f"\nResult: {result}")
        
        if result.success:
            print("\n✅ Fashion Agent working in PRODUCTION mode!")
            data = result.data
            print(f"Style: {data.get('style', 'unknown')}")
            print(f"Items detected: {len(data.get('items', []))}")
            print(f"Overall rating: {data.get('overall_rating', 0)}")
            
            for i, item in enumerate(data.get('items', [])):
                print(f"  Item {i+1}: {item}")
        else:
            print(f"❌ Fashion Agent failed: {result.error}")
            
    except Exception as e:
        print(f"❌ Exception in production mode: {e}")
        import traceback
        traceback.print_exc()

# Also test with a simple HTTP image URL
def test_with_http_url():
    print("\n" + "="*50)
    print("Testing Fashion Agent with HTTP image URL...")
    
    agent = FashionAgent()
    
    # Use a reliable image URL from Unsplash
    test_input = AgentInput(
        media_id="test_fashion_http",
        url="https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=400&h=600&fit=crop",
        user_id="test_user",
        metadata={}
    )
    
    try:
        result = agent.run(test_input)
        print(f"\nHTTP URL Result: {result}")
        
        if result.success:
            print("\n✅ Fashion Agent working with HTTP URLs!")
            data = result.data
            print(f"Style: {data.get('style', 'unknown')}")
            print(f"Items detected: {len(data.get('items', []))}")
        else:
            print(f"❌ Fashion Agent failed with HTTP URL: {result.error}")
            
    except Exception as e:
        print(f"❌ Exception with HTTP URL: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fashion_agent_with_real_image()
    test_with_http_url()