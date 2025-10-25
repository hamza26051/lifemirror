import cv2
import numpy as np
from PIL import Image
import os

# Check if image exists and load it
image_path = "test_person.jpg"
print(f"Image exists: {os.path.exists(image_path)}")
print(f"Image size: {os.path.getsize(image_path)} bytes")

# Try loading with OpenCV
try:
    img = cv2.imread(image_path)
    if img is not None:
        print(f"OpenCV loaded image successfully")
        print(f"Image shape: {img.shape}")
        print(f"Image dtype: {img.dtype}")
        print(f"Image min/max values: {img.min()}/{img.max()}")
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"RGB image shape: {img_rgb.shape}")
        print(f"RGB image dtype: {img_rgb.dtype}")
        print(f"RGB image min/max values: {img_rgb.min()}/{img_rgb.max()}")
        
        # Check if image is not corrupted
        if img.shape[0] > 0 and img.shape[1] > 0:
            print("Image appears to be valid")
        else:
            print("Image has invalid dimensions")
    else:
        print("OpenCV failed to load image")
except Exception as e:
    print(f"Error loading with OpenCV: {e}")

# Try loading with PIL
try:
    pil_img = Image.open(image_path)
    print(f"PIL loaded image successfully")
    print(f"PIL image size: {pil_img.size}")
    print(f"PIL image mode: {pil_img.mode}")
except Exception as e:
    print(f"Error loading with PIL: {e}")

# Try a simple MediaPipe test
try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.1) as face_detection:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)
        print(f"MediaPipe FaceDetection with 0.1 confidence: {results.detections is not None}")
        if results.detections:
            print(f"Detections: {len(results.detections)}")
except Exception as e:
    print(f"MediaPipe test failed: {e}")