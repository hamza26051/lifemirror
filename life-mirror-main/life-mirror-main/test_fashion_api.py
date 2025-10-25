import requests
import json
import os
from pathlib import Path

def test_fashion_analysis_api():
    """Test fashion analysis through the actual API endpoint"""
    
    # API base URL
    base_url = "http://localhost:8000"
    
    print("Testing Fashion Analysis API...")
    
    # First, test health endpoint
    try:
        health_response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Health check: {health_response.status_code} - {health_response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test with a local image file
    test_image_path = "test_person.jpg"
    if not os.path.exists(test_image_path):
        print(f"❌ Test image {test_image_path} not found")
        return
    
    # Upload image to media endpoint
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            upload_response = requests.post(
                f"{base_url}/media/upload", 
                files=files,
                timeout=30
            )
        
        if upload_response.status_code != 200:
            print(f"❌ Image upload failed: {upload_response.status_code}")
            print(f"Response: {upload_response.text}")
            return
            
        upload_data = upload_response.json()
        media_id = upload_data.get('media_id')
        media_url = upload_data.get('url')
        
        print(f"✅ Image uploaded successfully: {media_id}")
        print(f"Media URL: {media_url}")
        
    except Exception as e:
        print(f"❌ Image upload error: {e}")
        return
    
    # Now test fashion analysis
    try:
        analysis_payload = {
            "media_id": media_id,
            "user_consent": {
                "face_analysis": True,
                "posture_analysis": True,
                "fashion_analysis": True
            },
            "options": {}
        }
        
        analysis_response = requests.post(
            f"{base_url}/analysis/analyze",
            json=analysis_payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if analysis_response.status_code == 200:
            result = analysis_response.json()
            print(f"✅ Analysis completed successfully!")
            
            # Check fashion analysis results
            fashion_data = result.get('fashion', {})
            if fashion_data:
                print(f"Fashion Analysis Results:")
                print(f"  - Style: {fashion_data.get('style', 'unknown')}")
                print(f"  - Items detected: {len(fashion_data.get('items', []))}")
                print(f"  - Overall rating: {fashion_data.get('overall_rating', 0)}")
                
                for i, item in enumerate(fashion_data.get('items', [])):
                    print(f"  - Item {i+1}: {item.get('label', 'unknown')} (confidence: {item.get('confidence', 0):.2f})")
            else:
                print("❌ No fashion analysis data in response")
                
        else:
            print(f"❌ Analysis failed: {analysis_response.status_code}")
            print(f"Response: {analysis_response.text}")
            
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fashion_analysis_api()