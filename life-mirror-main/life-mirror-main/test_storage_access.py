#!/usr/bin/env python3
"""
Test storage access and agent image processing
"""

import os
import requests
import time
from pathlib import Path

def test_storage_endpoint():
    """Test if storage endpoint is accessible"""
    print("🔍 Testing Storage Endpoint...")
    
    # Test if we can access a test image through storage endpoint
    test_urls = [
        "http://localhost:8000/storage/media/test/test_person.jpg",
        "http://localhost:8000/docs",  # API docs should be accessible
        "http://localhost:8000/health"  # Health check
    ]
    
    for url in test_urls:
        try:
            print(f"   - Testing: {url}")
            response = requests.get(url, timeout=5)
            print(f"     Status: {response.status_code}")
            if response.status_code == 200:
                print(f"     ✅ Accessible")
            else:
                print(f"     ⚠️  Status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"     ❌ Failed: {e}")
    
    return True

def test_agent_with_http_url():
    """Test if agents can process HTTP URLs vs local files"""
    print("\n🔍 Testing Agent HTTP URL Processing...")
    
    os.environ["LIFEMIRROR_MODE"] = "mock"
    
    from src.agents.fashion_agent import FashionAgent
    from src.agents.base_agent import AgentInput
    
    fashion_agent = FashionAgent()
    
    # Test 1: Local file path
    print("\n1️⃣ Testing with local file path:")
    try:
        start_time = time.time()
        result1 = fashion_agent.run(AgentInput(
            media_id="test_local",
            url="test_person.jpg",  # Local file
            context={"user_id": "test"}
        ))
        end_time = time.time()
        print(f"   ✅ Local file: {end_time - start_time:.2f}s")
        print(f"   - Success: {result1.success}")
    except Exception as e:
        print(f"   ❌ Local file failed: {e}")
    
    # Test 2: HTTP URL (non-existent)
    print("\n2️⃣ Testing with HTTP URL (non-existent):")
    try:
        start_time = time.time()
        result2 = fashion_agent.run(AgentInput(
            media_id="test_http",
            url="http://localhost:8000/storage/media/nonexistent/test.jpg",
            context={"user_id": "test"}
        ))
        end_time = time.time()
        print(f"   ⚠️  HTTP URL (non-existent): {end_time - start_time:.2f}s")
        print(f"   - Success: {result2.success}")
        if end_time - start_time > 10:
            print(f"   ⚠️  SLOW RESPONSE - This could cause hanging!")
    except Exception as e:
        print(f"   ❌ HTTP URL failed: {e}")
    
    # Test 3: HTTP URL with timeout
    print("\n3️⃣ Testing with timeout detection:")
    import threading
    
    def test_with_timeout():
        try:
            result = fashion_agent.run(AgentInput(
                media_id="test_timeout",
                url="http://localhost:8000/storage/media/timeout/test.jpg",
                context={"user_id": "test"}
            ))
            return result
        except Exception as e:
            print(f"     Exception in timeout test: {e}")
            return None
    
    result = [None]
    def target():
        result[0] = test_with_timeout()
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=15)  # 15 second timeout
    
    if thread.is_alive():
        print("   ❌ AGENT HANGING DETECTED with HTTP URL!")
        print("   - This is likely the cause of Streamlit hanging")
        print("   - Agents hang when trying to access non-existent HTTP URLs")
        return False
    else:
        print("   ✅ Agent completed within timeout")
        return True

def test_file_upload_simulation():
    """Simulate the file upload process"""
    print("\n🔍 Testing File Upload Simulation...")
    
    # Check if uploads directory exists
    uploads_dir = Path("uploads")
    print(f"   - Uploads directory exists: {uploads_dir.exists()}")
    
    if uploads_dir.exists():
        media_dirs = list(uploads_dir.glob("media/*"))
        print(f"   - Found {len(media_dirs)} media directories")
        
        for media_dir in media_dirs[:3]:  # Check first 3
            files = list(media_dir.glob("*"))
            print(f"   - {media_dir.name}: {len(files)} files")
            
            for file_path in files[:2]:  # Check first 2 files
                storage_url = f"http://localhost:8000/storage/media/{media_dir.name}/{file_path.name}"
                print(f"     - Testing: {storage_url}")
                
                try:
                    response = requests.get(storage_url, timeout=3)
                    print(f"       Status: {response.status_code}")
                except Exception as e:
                    print(f"       ❌ Failed: {e}")
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Storage Access Test")
    print("=" * 50)
    
    # Test 1: Storage endpoint
    test_storage_endpoint()
    
    # Test 2: Agent HTTP processing
    agent_success = test_agent_with_http_url()
    
    # Test 3: File upload simulation
    test_file_upload_simulation()
    
    print("\n" + "=" * 50)
    print("🏁 Storage Access Test Complete")
    
    if not agent_success:
        print("\n❌ HANGING ISSUE IDENTIFIED:")
        print("   - Agents hang when accessing non-existent HTTP URLs")
        print("   - This explains Streamlit analysis hanging")
        print("   - Solution: Fix URL generation or add timeouts")
    else:
        print("\n✅ No hanging issues detected in storage access")