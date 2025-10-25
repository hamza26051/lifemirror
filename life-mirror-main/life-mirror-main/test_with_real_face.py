import os
import sys
sys.path.append('.')

from src.tools.face_tool import FaceTool
from src.schemas.tool_schemas import FaceToolInput
import requests
from PIL import Image
import io

# Set environment variables
os.environ['PROCESSING_MODE'] = 'prod'
os.environ['USE_DEEPFACE'] = 'true'

# Download a test image with a clear face
test_url = "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop&crop=face"
print(f"Downloading test image from: {test_url}")

try:
    response = requests.get(test_url)
    if response.status_code == 200:
        # Save the image locally
        with open("clear_face_test.jpg", "wb") as f:
            f.write(response.content)
        print("Downloaded clear face test image successfully")
        
        # Verify the image
        img = Image.open("clear_face_test.jpg")
        print(f"Image size: {img.size}")
        print(f"Image mode: {img.mode}")
        
        # Test with FaceTool
        face_tool = FaceTool()
        input_data = FaceToolInput(url="clear_face_test.jpg")
        result = face_tool.run(input_data)
        
        print(f"Tool result success: {result.success}")
        if result.success and result.data:
            faces = result.data.get('faces', [])
            print(f"Faces found: {len(faces)}")
            
            for i, face in enumerate(faces):
                print(f"\nFace {i+1}:")
                print(f"  Bounding box: {face.get('bbox')}")
                print(f"  Gender: {face.get('attributes', {}).get('gender')}")
                print(f"  Age: {face.get('attributes', {}).get('age')}")
                print(f"  Expression: {face.get('attributes', {}).get('expression')}")
                print(f"  Crop URL: {face.get('crop_url')}")
        else:
            print("No faces detected or tool failed")
            
except Exception as e:
    print(f"Error: {e}")