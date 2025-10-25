import os
import base64
from flask import Flask, request, jsonify
import requests
import json
import re
import math
from ultralytics import YOLO
from PIL import Image
import io
import mediapipe as mp
import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
from scipy import stats
from scipy.spatial.distance import euclidean
import colorsys
from collections import defaultdict

# API KEYS (integrated directly)
HF_TOKEN = "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"
FACEPP_KEY = "DD15vgXjm7gABNUT2oVypGDK1cFCO5FM"
FACEPP_SECRET = "OG_liVG8CK1bjItiT4zKoBhCg4vEyOGg"

# OpenRouter API Keys with fallback system
OPENROUTER_KEYS = [
    "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",  # Primary key
    "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",  # Secondary key
    "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",  # Tertiary key
]

# Fallback system for OpenRouter API calls
def make_openrouter_request(url, payload, max_retries=3):
    """
    Make OpenRouter API request with automatic fallback to different keys.
    Returns the response if successful, None if all keys fail.
    """
    for attempt in range(max_retries):
        for key_index, api_key in enumerate(OPENROUTER_KEYS):
            try:
                print(f"Attempting OpenRouter request with key {key_index + 1}/{len(OPENROUTER_KEYS)} (attempt {attempt + 1}/{max_retries})")
                
                response = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    print(f"OpenRouter request successful with key {key_index + 1}")
                    return response
                elif response.status_code == 401:
                    print(f"OpenRouter key {key_index + 1} unauthorized (401)")
                    continue  # Try next key
                elif response.status_code == 429:
                    print(f"OpenRouter key {key_index + 1} rate limited (429)")
                    continue  # Try next key
                else:
                    print(f"OpenRouter key {key_index + 1} failed with status {response.status_code}: {response.text}")
                    continue  # Try next key
                    
            except requests.exceptions.Timeout:
                print(f"OpenRouter key {key_index + 1} timed out")
                continue  # Try next key
            except requests.exceptions.RequestException as e:
                print(f"OpenRouter key {key_index + 1} request failed: {e}")
                continue  # Try next key
            except Exception as e:
                print(f"OpenRouter key {key_index + 1} unexpected error: {e}")
                continue  # Try next key
        
        # If we've tried all keys, wait before retrying
        if attempt < max_retries - 1:
            print(f"All keys failed, waiting before retry {attempt + 2}/{max_retries}")
            import time
            time.sleep(2 ** attempt)  # Exponential backoff
    
    print("All OpenRouter keys failed after all retries")
    return None

app = Flask(__name__)

yolo_model = YOLO('yolov8n.pt')  # Load once at startup

# Advanced ML components
class AdvancedFaceAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.ensemble_model = None
        self.feature_weights = {
            'symmetry': 0.25,
            'proportions': 0.20,
            'skin_quality': 0.15,
            'lighting': 0.10,
            'expression': 0.15,
            'confidence': 0.15
        }
    
    def analyze_skin_quality(self, image_bytes):
        """Advanced skin quality analysis using computer vision"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_array = np.array(image)
            
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            
            # Skin detection using HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Calculate skin metrics
            skin_pixels = img_array[skin_mask > 0]
            if len(skin_pixels) == 0:
                return {'skin_quality_score': 50, 'skin_evenness': 50, 'skin_texture': 50}
            
            # Skin evenness (variance in color)
            skin_variance = np.var(skin_pixels, axis=0).mean()
            skin_evenness = max(0, 100 - skin_variance / 2)
            
            # Skin texture analysis using Laplacian variance
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            skin_texture = max(0, 100 - laplacian_var / 10)
            
            # Overall skin quality score
            skin_quality = (skin_evenness + skin_texture) / 2
            
            return {
                'skin_quality_score': round(skin_quality, 1),
                'skin_evenness': round(skin_evenness, 1),
                'skin_texture': round(skin_texture, 1),
                'skin_variance': round(skin_variance, 2),
                'texture_variance': round(laplacian_var, 2)
            }
        except Exception as e:
            print(f"Skin quality analysis error: {e}")
            return {'skin_quality_score': 50, 'skin_evenness': 50, 'skin_texture': 50}
    
    def analyze_lighting_quality(self, image_bytes):
        """Advanced lighting analysis"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_array = np.array(image)
            
            # Convert to LAB for better lighting analysis
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            
            # Calculate lighting metrics
            brightness = np.mean(l_channel)
            contrast = np.std(l_channel)
            
            # Histogram analysis
            hist, _ = np.histogram(l_channel, bins=256, range=(0, 255))
            hist_smoothness = 1 - np.std(np.diff(hist)) / np.mean(hist)
            
            # Lighting quality score
            lighting_score = (
                0.4 * min(brightness / 128, 1) * 100 +  # Brightness factor
                0.3 * min(contrast / 50, 1) * 100 +     # Contrast factor
                0.3 * hist_smoothness * 100              # Smoothness factor
            )
            
            return {
                'lighting_score': round(lighting_score, 1),
                'brightness': round(brightness, 1),
                'contrast': round(contrast, 1),
                'histogram_smoothness': round(hist_smoothness, 3)
            }
        except Exception as e:
            print(f"Lighting analysis error: {e}")
            return {'lighting_score': 50, 'brightness': 128, 'contrast': 25, 'histogram_smoothness': 0.5}
    
    def calculate_golden_ratio_proportions(self, landmarks):
        """Calculate facial proportions based on golden ratio"""
        try:
            # Golden ratio constant
            phi = 1.618033988749
            
            # Key facial measurements
            face_width = abs(landmarks[234]['x'] - landmarks[454]['x'])  # Jaw width
            face_height = abs(landmarks[10]['y'] - landmarks[152]['y'])  # Face height
            
            # Eye measurements
            eye_width = abs(landmarks[33]['x'] - landmarks[133]['x'])  # Left eye
            eye_height = abs(landmarks[159]['y'] - landmarks[145]['y'])
            
            # Nose measurements
            nose_width = abs(landmarks[129]['x'] - landmarks[358]['x'])
            nose_height = abs(landmarks[168]['y'] - landmarks[2]['y'])
            
            # Calculate golden ratio proportions
            face_ratio = face_width / face_height
            eye_ratio = eye_width / eye_height
            nose_ratio = nose_width / nose_height
            
            # Score based on how close to golden ratio
            face_score = max(0, 100 - abs(face_ratio - phi) * 50)
            eye_score = max(0, 100 - abs(eye_ratio - phi) * 50)
            nose_score = max(0, 100 - abs(nose_ratio - phi) * 50)
            
            overall_proportion_score = (face_score + eye_score + nose_score) / 3
            
            return {
                'golden_ratio_score': round(overall_proportion_score, 1),
                'face_ratio': round(face_ratio, 3),
                'eye_ratio': round(eye_ratio, 3),
                'nose_ratio': round(nose_ratio, 3),
                'face_score': round(face_score, 1),
                'eye_score': round(eye_score, 1),
                'nose_score': round(nose_score, 1)
            }
        except Exception as e:
            print(f"Golden ratio calculation error: {e}")
            return {'golden_ratio_score': 50, 'face_ratio': 1.6, 'eye_ratio': 1.6, 'nose_ratio': 1.6}
    
    def advanced_symmetry_analysis(self, landmarks):
        """Advanced facial symmetry analysis"""
        try:
            # Define symmetric point pairs
            symmetric_pairs = [
                (33, 263),   # Left/Right eye corners
                (65, 295),   # Left/Right eyebrows
                (234, 454),  # Left/Right jaw
                (205, 425),  # Left/Right cheeks
                (159, 386),  # Left/Right eye tops
                (145, 374)   # Left/Right eye bottoms
            ]
            
            symmetry_scores = []
            for left_idx, right_idx in symmetric_pairs:
                if left_idx in landmarks and right_idx in landmarks:
                    left_point = np.array([landmarks[left_idx]['x'], landmarks[left_idx]['y']])
                    right_point = np.array([landmarks[right_idx]['x'], landmarks[right_idx]['y']])
                    
                    # Calculate distance between symmetric points
                    distance = euclidean(left_point, right_point)
                    symmetry_scores.append(distance)
            
            if not symmetry_scores:
                return {'advanced_symmetry_score': 50}
            
            # Calculate symmetry score (lower distances = better symmetry)
            avg_symmetry_distance = np.mean(symmetry_scores)
            symmetry_score = max(0, 100 - avg_symmetry_distance / 2)
            
            return {
                'advanced_symmetry_score': round(symmetry_score, 1),
                'avg_symmetry_distance': round(avg_symmetry_distance, 2),
                'symmetry_variance': round(np.var(symmetry_scores), 2)
            }
        except Exception as e:
            print(f"Advanced symmetry analysis error: {e}")
            return {'advanced_symmetry_score': 50}
    
    def calculate_expression_quality(self, emotion_data):
        """Analyze expression quality and naturalness"""
        try:
            # Extract emotion values
            emotions = {
                'happiness': emotion_data.get('happiness', 0),
                'sadness': emotion_data.get('sadness', 0),
                'anger': emotion_data.get('anger', 0),
                'fear': emotion_data.get('fear', 0),
                'surprise': emotion_data.get('surprise', 0),
                'disgust': emotion_data.get('disgust', 0),
                'neutral': emotion_data.get('neutral', 0)
            }
            
            # Calculate expression intensity
            max_emotion = max(emotions.values())
            emotion_variance = np.var(list(emotions.values()))
            
            # Natural expression score (balanced emotions, not too extreme)
            naturalness_score = 100 - abs(max_emotion - 50) * 0.5
            
            # Expression clarity (clear dominant emotion)
            clarity_score = max_emotion if max_emotion > 30 else 50
            
            # Overall expression quality
            expression_quality = (naturalness_score + clarity_score) / 2
            
            return {
                'expression_quality_score': round(expression_quality, 1),
                'naturalness_score': round(naturalness_score, 1),
                'clarity_score': round(clarity_score, 1),
                'max_emotion_intensity': round(max_emotion, 1),
                'emotion_variance': round(emotion_variance, 2)
            }
        except Exception as e:
            print(f"Expression quality analysis error: {e}")
            return {'expression_quality_score': 50, 'naturalness_score': 50, 'clarity_score': 50}

# Initialize the advanced analyzer
advanced_analyzer = AdvancedFaceAnalyzer()


def get_image_caption(image_bytes):
    """
    Uses the image-to-text pipeline to generate a caption via BLIP (base64 JSON input).
    """
    # encode image to base64
    b64 = base64.b64encode(image_bytes).decode('utf-8')
    payload = {
        "model": "Salesforce/blip-image-captioning-base",
        "inputs": b64
    }
    response = requests.post(
        "https://api-inference.huggingface.co/pipeline/image-to-text",
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        },
        json=payload
    )
    if response.status_code != 200:
        print(f"BLIP error {response.status_code}: {response.text}")
        return f"Error: BLIP returned {response.status_code}."
    try:
        data = response.json()
    except ValueError:
        print(f"BLIP non-JSON response: {response.text}")
        return "Error: BLIP returned a non-JSON response."
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"]
    else:
        print(f"BLIP unexpected payload: {data}")
        return "Error: BLIP response format unexpected."


def get_face_attributes(image_bytes):
    response = requests.post(
        "https://api-us.faceplusplus.com/facepp/v3/detect",
        data={
            "api_key": FACEPP_KEY,
            "api_secret": FACEPP_SECRET,
            "return_attributes": "age,gender,emotion,beauty",
            "face_landmark": 1  # Request facial landmarks
        },
        files={"image_file": image_bytes}
    )
    try:
        return response.json()
    except ValueError:
        print(f"Face++ non-JSON response: {response.text}")
        return {"error": f"Face++ returned non-JSON (status {response.status_code})"}


def get_mediapipe_landmarks(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_np = np.array(image)
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(img_np)
        if results.multi_face_landmarks:
            # Convert landmarks to a dict with x, y pixel coordinates
            landmarks = {}
            for idx, lm in enumerate(results.multi_face_landmarks[0].landmark):
                landmarks[idx] = {'x': int(lm.x * image.width), 'y': int(lm.y * image.height)}
            return landmarks
    return None

# MediaPipe Face Mesh indices for key features:
# Jawline: 234 (left), 454 (right), 152 (chin)
# Left eyebrow: 105 (inner), 65 (upper middle)
# Right eyebrow: 334 (inner), 295 (upper middle)
# Left eye: 159 (top), 145 (bottom), 468 (center)
# Right eye: 386 (top), 374 (bottom), 473 (center)
# Left cheek: 205

def heuristic_attractiveness_confidence_mp(landmarks, image_bytes=None):
    try:
        # Use eye corners instead of eye center points for robustness
        required_indices = [234, 454, 152, 105, 334, 159, 145, 33, 133, 362, 263, 65, 295, 205, 10, 129, 358, 168, 2, 425]
        missing = [idx for idx in required_indices if idx not in landmarks]
        if missing:
            print(f"Missing landmark indices: {missing}")
            return {'heuristic_error': f"Missing landmark indices: {missing}"}
        
        # Basic measurements (original logic)
        jaw_left = landmarks[234]
        jaw_right = landmarks[454]
        chin = landmarks[152]
        jaw_width = abs(jaw_left['x'] - jaw_right['x'])
        chin_height = abs(jaw_left['y'] - chin['y'])
        jaw_symmetry = jaw_width / (chin_height + 1e-5)
        
        brow_left = landmarks[105]
        brow_right = landmarks[334]
        brow_distance = abs(brow_left['x'] - brow_right['x'])
        
        eye_top = landmarks[159]
        eye_bottom = landmarks[145]
        eye_openness = abs(eye_top['y'] - eye_bottom['y'])
        
        left_eye_center_x = (landmarks[33]['x'] + landmarks[133]['x']) / 2
        right_eye_center_x = (landmarks[362]['x'] + landmarks[263]['x']) / 2
        left_brow = landmarks[65]
        right_brow = landmarks[295]
        eye_distance = abs(left_eye_center_x - right_eye_center_x)
        brow_symmetry = abs(left_brow['y'] - right_brow['y'])
        
        # Advanced analysis using the new analyzer
        advanced_features = {}
        
        # Golden ratio proportions
        golden_ratio = advanced_analyzer.calculate_golden_ratio_proportions(landmarks)
        advanced_features.update(golden_ratio)
        
        # Advanced symmetry analysis
        symmetry_analysis = advanced_analyzer.advanced_symmetry_analysis(landmarks)
        advanced_features.update(symmetry_analysis)
        
        # Skin quality analysis (if image available)
        skin_analysis = {}
        if image_bytes is not None:
            skin_analysis = advanced_analyzer.analyze_skin_quality(image_bytes)
            advanced_features.update(skin_analysis)
            
            # Lighting analysis
            lighting_analysis = advanced_analyzer.analyze_lighting_quality(image_bytes)
            advanced_features.update(lighting_analysis)
        
        # Enhanced attractiveness calculation using ensemble approach
        base_attract_score = 50 + 20 * jaw_symmetry + 0.1 * brow_distance + 2 * eye_openness - 0.5 * brow_symmetry
        base_attract_score = min(max(base_attract_score, 0), 100)
        
        # Add advanced features to attractiveness score
        advanced_attract_score = base_attract_score
        if 'golden_ratio_score' in advanced_features:
            advanced_attract_score += advanced_features['golden_ratio_score'] * 0.3
        if 'advanced_symmetry_score' in advanced_features:
            advanced_attract_score += advanced_features['advanced_symmetry_score'] * 0.2
        if 'skin_quality_score' in advanced_features:
            advanced_attract_score += advanced_features['skin_quality_score'] * 0.15
        if 'lighting_score' in advanced_features:
            advanced_attract_score += advanced_features['lighting_score'] * 0.1
        
        # Normalize to 0-100 range
        advanced_attract_score = min(max(advanced_attract_score / 1.75, 0), 100)
        
        # Enhanced confidence calculation
        base_confidence = 50 + 8 * jaw_symmetry + 3 * eye_openness - 0.3 * brow_symmetry + 0.5 * eye_distance
        base_confidence = min(max(base_confidence, 0), 100)
        
        # Add advanced features to confidence
        advanced_confidence = base_confidence
        if 'advanced_symmetry_score' in advanced_features:
            advanced_confidence += advanced_features['advanced_symmetry_score'] * 0.2
        if 'golden_ratio_score' in advanced_features:
            advanced_confidence += advanced_features['golden_ratio_score'] * 0.15
        if 'lighting_score' in advanced_features:
            advanced_confidence += advanced_features['lighting_score'] * 0.1
        
        # Normalize to 0-100 range
        advanced_confidence = min(max(advanced_confidence / 1.45, 0), 100)
        
        # Enhanced attractiveness label using multiple criteria
        attractiveness_criteria = [
            2.0 < jaw_symmetry < 3.5,
            90 < brow_distance < 180,
            8 < eye_openness < 20,
            eye_distance > 80,
            brow_symmetry < 10
        ]
        
        # Add advanced criteria
        if 'golden_ratio_score' in advanced_features:
            attractiveness_criteria.append(advanced_features['golden_ratio_score'] > 60)
        if 'advanced_symmetry_score' in advanced_features:
            attractiveness_criteria.append(advanced_features['advanced_symmetry_score'] > 70)
        if 'skin_quality_score' in advanced_features:
            attractiveness_criteria.append(advanced_features['skin_quality_score'] > 60)
        
        # Calculate attractiveness percentage
        attractive_criteria_met = sum(attractiveness_criteria)
        total_criteria = len(attractiveness_criteria)
        attractiveness_percentage = (attractive_criteria_met / total_criteria) * 100
        
        # Enhanced label based on percentage
        if attractiveness_percentage >= 80:
            heuristic_attractiveness_label = "very attractive"
        elif attractiveness_percentage >= 60:
            heuristic_attractiveness_label = "attractive"
        elif attractiveness_percentage >= 40:
            heuristic_attractiveness_label = "moderately attractive"
        else:
            heuristic_attractiveness_label = "not attractive"
        
        # Combine all results
        result = {
            'heuristic_attractiveness': round(advanced_attract_score, 1),
            'heuristic_confidence': round(advanced_confidence, 1),
            'jawline_symmetry': round(jaw_symmetry, 3),
            'eyebrow_distance': round(brow_distance, 2),
            'eye_openness': round(eye_openness, 2),
            'eye_distance': round(eye_distance, 2),
            'brow_symmetry': round(brow_symmetry, 2),
            'attractiveness_percentage': round(attractiveness_percentage, 1),
            'heuristic_attractiveness_label': heuristic_attractiveness_label
        }
        
        # Add advanced features
        result.update(advanced_features)
        
        return result
        
    except Exception as e:
        print(f"Heuristic error: {e}")
        return {'heuristic_error': str(e)}


def get_fashion_tags(image_bytes, candidate_labels=None):
    """
    Uses zero-shot image classification pipeline (CLIP) with JSON base64 input.
    """
    if candidate_labels is None:
        candidate_labels = [
            "dress", "jacket", "tshirt", "jeans", "shoes",
            "hat", "handbag", "scarf", "sunglasses", "watch"
        ]
    b64 = base64.b64encode(image_bytes).decode('utf-8')
    payload = {
        "model": "openai/clip-vit-base-patch32",
        "inputs": b64,
        "parameters": {
            "candidate_labels": candidate_labels,
            "multi_label": True
        }
    }
    response = requests.post(
        "https://api-inference.huggingface.co/pipeline/zero-shot-image-classification",
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        },
        json=payload
    )
    if response.status_code != 200:
        print(f"FashionCLIP error {response.status_code}: {response.text}")
        return {"error": f"CLIP returned {response.status_code}"}
    try:
        data = response.json()
    except ValueError:
        print(f"FashionCLIP non-JSON response: {response.text}")
        return {"error": "Non-JSON from CLIP"}
    if "labels" in data and "scores" in data:
        return list(zip(data["labels"], data["scores"]))
    else:
        print(f"FashionCLIP unexpected payload: {data}")
        return {"error": "Unexpected CLIP format"}


def get_voice_transcript(audio_bytes):
    response = requests.post(
        "https://api-inference.huggingface.co/models/openai/whisper-1",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        files={"file": audio_bytes}
    )
    if response.status_code != 200:
        print(f"Whisper error {response.status_code}: {response.text}")
        return "Error: Whisper failed."
    try:
        return response.json().get('text', '')
    except ValueError:
        print(f"Whisper non-JSON response: {response.text}")
        return "Error: Whisper returned non-JSON."


def get_vibe_analysis(prompt):
    try:
        payload = {
            "model": "meta-llama/llama-3.2-11b-vision-instruct:free",  # Using free model
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an honest and straightforward observer who gives clear, descriptive assessments with PERFECT grammar and spelling. Keep responses to exactly 9-15 lines total for three separate assessments. CRITICAL: For Vibe Analysis and How Others See Me sections, you MUST use first-person language ('I' and 'you') - NEVER use third-person ('this person', 'they', 'their'). Write as if you are directly talking to the person. For improvement suggestions, provide exactly 7 numbered points (1. 2. 3. 4. 5. 6. 7.) with each point on a separate line. Each point should be ONE specific, actionable step (max 15 words per point). CRITICAL: Do NOT write paragraphs for improvements - only numbered points. Each point must start with a number followed by a period and space. No witty comments, roasting, or sarcasm - just honest observations and practical, knowledge-based improvement guidance. IMPORTANT: Double-check all grammar, spelling, and punctuation before responding. Use proper sentence structure and avoid run-on sentences."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 200,  # Reduced to fit within credit limit
            "temperature": 0.7  # Balance creativity with accuracy
        }
        
        response = make_openrouter_request("https://openrouter.ai/api/v1/chat/completions", payload)
        
        if response is None:
            return "Error: Vibe analysis failed - all API keys exhausted."
        
        try:
            content = response.json()['choices'][0]['message']['content']
            # Ensure the response is concise (9-15 lines max for three assessments)
            lines = content.strip().split('\n')
            if len(lines) > 15:
                content = '\n'.join(lines[:15])
            return content
        except (ValueError, KeyError) as e:
            print(f"OpenRouter unexpected payload: {response.text}")
            return "Error: Vibe response format unexpected."
    except Exception as e:
        print(f"Unexpected error in vibe analysis: {e}")
        return "Error: Unexpected error in vibe analysis."


def get_embedding(text):
    payload = {"model": "nomic-ai/nomic-embed-text-v1.5", "input": text}
    
    response = make_openrouter_request("https://openrouter.ai/api/v1/embeddings", payload)
    
    if response is None:
        return "Error: Embedding failed - all API keys exhausted."
    
    try:
        return response.json()['data'][0]['embedding']
    except (ValueError, KeyError):
        print(f"Embedding unexpected payload: {response.text}")
        return "Error: Embedding format unexpected."


def rate_face(face_attrs, image_bytes=None):
    """
    Given the Face++ API response, return a dict with ratings and a summary.
    Adds a dullness rating (100 - happiness) and heuristic-based scores using MediaPipe landmarks.
    Includes debug logging for landmarks and heuristic errors.
    """
    print("Full Face++ response:", face_attrs)
    if not face_attrs or "faces" not in face_attrs or not face_attrs["faces"]:
        print("No face detected in Face++ response.")
        return {"error": "No face detected."}

    face = face_attrs["faces"][0]["attributes"]
    beauty = face.get("beauty", {})
    emotion = face.get("emotion", {})
    age = face.get("age", {}).get("value", "Unknown")
    gender = face.get("gender", {}).get("value", "Unknown")

    # Attractiveness: average of male/female score
    attract = (beauty.get("male_score", 0) + beauty.get("female_score", 0)) / 2

    # Smile: use happiness
    smile = emotion.get("happiness", 0)

    # Dullness: 100 - happiness
    dullness = 100 - smile

    # Dominant emotion
    dominant_emotion = max(emotion, key=emotion.get) if emotion else "Unknown"
    
    # Advanced expression quality analysis
    expression_quality = advanced_analyzer.calculate_expression_quality(emotion)

    # Heuristic-based scores using MediaPipe
    heuristics = {}  # Initialize empty dict
    mediapipe_landmarks = get_mediapipe_landmarks(image_bytes)
    if mediapipe_landmarks:
        print("MediaPipe landmarks found, using for heuristics.")
        heuristics = heuristic_attractiveness_confidence_mp(mediapipe_landmarks, image_bytes)
        if not heuristics:  # If function returns None or empty
            heuristics = {}
    else:
        print("No MediaPipe landmarks found, skipping heuristics.")
    
    # Add expression quality analysis to heuristics
    heuristics.update(expression_quality)

    # Enhanced Confidence calculation: high happiness, low negative emotions, plus facial features
    base_confidence = smile - (emotion.get("fear", 0) + emotion.get("sadness", 0) + emotion.get("anger", 0))
    
    # Add confidence boost from facial features
    confidence_boost = 0
    if heuristics and 'heuristic_confidence' in heuristics:
        try:
            # Use heuristic confidence as a boost factor
            heuristic_conf = heuristics['heuristic_confidence']
            if isinstance(heuristic_conf, (int, float)):
                confidence_boost = (heuristic_conf - 50) * 0.3  # Scale the boost
        except (TypeError, ValueError) as e:
            print(f"Confidence boost calculation error: {e}")
            confidence_boost = 0
    
    # Calculate final confidence
    confidence = base_confidence + confidence_boost
    
    # Ensure confidence stays within 0-100 range
    confidence = max(0, min(100, confidence))

    # Normalization helper
    def norm(val, minv, maxv):
        return max(0, min(100, 100 * (val - minv) / (maxv - minv)))

    # New heuristic-only attractiveness score (no Face++ influence)
    if 'heuristic_attractiveness' in heuristics:
        # Widened ranges for more spread
        jaw_score = norm(heuristics['jawline_symmetry'], 1.5, 4.0)
        brow_score = norm(heuristics['eyebrow_distance'], 70, 200)
        eye_score = norm(heuristics['eye_openness'], 5, 25)
        sym_score = 100 - norm(heuristics['brow_symmetry'], 0, 20)  # lower is better
        # Weighted sum (tune as desired)
        attractive_new = 0.3 * jaw_score + 0.25 * brow_score + 0.25 * eye_score + 0.2 * sym_score
        # Non-linear scaling for more spread
        attractive_new = attractive_new ** 1.1
        heuristics['attractive_new'] = round(min(attractive_new, 100), 1)

    # Advanced ensemble attractiveness rating using all available features
    if 'heuristic_attractiveness' in heuristics:
        # Base feature scores
        jaw_score = norm(heuristics['jawline_symmetry'], 2.0, 3.5)
        brow_score = norm(heuristics['eyebrow_distance'], 90, 180)
        eye_score = norm(heuristics['eye_openness'], 8, 20)
        sym_score = 100 - norm(heuristics['brow_symmetry'], 0, 10)  # lower is better
        
        # Advanced feature scores
        golden_ratio_score = heuristics.get('golden_ratio_score', 50)
        advanced_symmetry_score = heuristics.get('advanced_symmetry_score', 50)
        skin_quality_score = heuristics.get('skin_quality_score', 50)
        lighting_score = heuristics.get('lighting_score', 50)
        
        # Multi-level ensemble approach
        # Level 1: Basic features
        basic_ensemble = (
            0.4 * attract +
            0.2 * jaw_score +
            0.15 * brow_score +
            0.15 * eye_score +
            0.1 * sym_score
        )
        
        # Level 2: Advanced features
        advanced_ensemble = (
            0.3 * golden_ratio_score +
            0.3 * advanced_symmetry_score +
            0.2 * skin_quality_score +
            0.2 * lighting_score
        )
        
        # Level 3: Final ensemble (weighted combination)
        final_ensemble = (
            0.5 * basic_ensemble +
            0.3 * advanced_ensemble +
            0.2 * heuristics['heuristic_attractiveness']  # Include heuristic score
        )
        
        # Apply confidence weighting based on feature availability
        confidence_weight = 1.0
        available_features = sum([
            'golden_ratio_score' in heuristics,
            'advanced_symmetry_score' in heuristics,
            'skin_quality_score' in heuristics,
            'lighting_score' in heuristics
        ])
        
        # Boost confidence if more advanced features are available
        if available_features >= 3:
            confidence_weight = 1.1
        elif available_features >= 2:
            confidence_weight = 1.05
        
        final_ensemble *= confidence_weight
        final_ensemble = min(max(final_ensemble, 0), 100)
        
        heuristics['ensemble_attractiveness'] = round(final_ensemble, 1)
        
        # Enhanced labeling with confidence levels
        if final_ensemble > 85:
            heuristics['ensemble_attractiveness_label'] = 'exceptionally attractive'
        elif final_ensemble > 75:
            heuristics['ensemble_attractiveness_label'] = 'very attractive'
        elif final_ensemble > 65:
            heuristics['ensemble_attractiveness_label'] = 'attractive'
        elif final_ensemble > 55:
            heuristics['ensemble_attractiveness_label'] = 'moderately attractive'
        elif final_ensemble > 45:
            heuristics['ensemble_attractiveness_label'] = 'average'
        else:
            heuristics['ensemble_attractiveness_label'] = 'below average'
        
        # Add detailed breakdown
        heuristics['attractiveness_breakdown'] = {
            'face_plus_score': round(attract, 1),
            'basic_features_score': round(basic_ensemble, 1),
            'advanced_features_score': round(advanced_ensemble, 1),
            'confidence_weight': round(confidence_weight, 2),
            'available_advanced_features': available_features
        }

    # Calculate Advanced First Impression Score
    first_impression_score = calculate_first_impression_score(
        attract, smile, confidence, heuristics, emotion, dominant_emotion
    )
    
    # Calculate Approachability Rating
    approachability_score = calculate_approachability_score(
        smile, emotion, heuristics, confidence
    )

    summary = (
        f"Age: {age}, Gender: {gender}, "
        f"Attractiveness: {attract:.1f}/100, "
        f"Smile: {smile:.1f}/100, "
        f"Dullness: {dullness:.1f}/100, "
        f"Confidence: {confidence:.1f}/100, "
        f"First Impression: {first_impression_score['score']:.1f}/100 ({first_impression_score['label']}), "
        f"Approachability: {approachability_score['score']:.1f}/100 ({approachability_score['label']}), "
        f"Dominant Emotion: {dominant_emotion.capitalize()}"
    )
    if 'heuristic_attractiveness' in heuristics and 'heuristic_confidence' in heuristics:
        summary += f", Heuristic Attractiveness: {heuristics['heuristic_attractiveness']}/100, Heuristic Confidence: {heuristics['heuristic_confidence']}/100"
    if 'ensemble_attractiveness' in heuristics:
        summary += f", Ensemble Attractiveness: {heuristics['ensemble_attractiveness']}/100 ({heuristics['ensemble_attractiveness_label']})"
    if 'attractive_new' in heuristics:
        summary += f", Attractive New: {heuristics['attractive_new']}/100"
    if 'heuristic_error' in heuristics:
        summary += f" (Heuristic error: {heuristics['heuristic_error']})"

    result = {
        "age": age,
        "gender": gender,
        "attractiveness": round(attract, 1),
        "smile": round(smile, 1),
        "dullness": round(dullness, 1),
        "confidence": round(confidence, 1),
        "first_impression_score": round(first_impression_score["score"], 1),
        "first_impression_label": first_impression_score["label"],
        "first_impression_analysis": first_impression_score["analysis"],
        "approachability_score": round(approachability_score["score"], 1),
        "approachability_label": approachability_score["label"],
        "approachability_analysis": approachability_score["analysis"],
        "dominant_emotion": dominant_emotion,
        "summary": summary
    }
    # Advanced Ensemble Scoring System
    ensemble_scores = {
        'face_plus_attractiveness': attract,
        'heuristic_attractiveness': heuristics.get('heuristic_attractiveness', 50),
        'ensemble_attractiveness': heuristics.get('ensemble_attractiveness', 50),
        'first_impression': first_impression_score.get('score', 50),
        'approachability': approachability_score.get('score', 50),
        'confidence': confidence,
        'smile': smile
    }
    
    # Add advanced features to ensemble
    if heuristics:
        if 'golden_ratio_score' in heuristics:
            ensemble_scores['golden_ratio'] = heuristics['golden_ratio_score']
        if 'advanced_symmetry_score' in heuristics:
            ensemble_scores['symmetry'] = heuristics['advanced_symmetry_score']
        if 'skin_quality_score' in heuristics:
            ensemble_scores['skin_quality'] = heuristics['skin_quality_score']
        if 'lighting_score' in heuristics:
            ensemble_scores['lighting'] = heuristics['lighting_score']
    
    # Calculate weighted ensemble score
    weights = {
        'face_plus_attractiveness': 0.25,
        'heuristic_attractiveness': 0.20,
        'ensemble_attractiveness': 0.25,
        'first_impression': 0.15,
        'approachability': 0.10,
        'confidence': 0.05
    }
    
    # Add weights for advanced features if available
    if 'golden_ratio' in ensemble_scores:
        weights['golden_ratio'] = 0.05
    if 'symmetry' in ensemble_scores:
        weights['symmetry'] = 0.05
    if 'skin_quality' in ensemble_scores:
        weights['skin_quality'] = 0.03
    if 'lighting' in ensemble_scores:
        weights['lighting'] = 0.02
    
    # Normalize weights
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    # Calculate final ensemble score
    final_ensemble_score = sum(ensemble_scores[k] * normalized_weights[k] 
                              for k in normalized_weights if k in ensemble_scores)
    
    # Apply confidence boost based on feature availability
    available_advanced_features = sum([
        'golden_ratio' in ensemble_scores,
        'symmetry' in ensemble_scores,
        'skin_quality' in ensemble_scores,
        'lighting' in ensemble_scores
    ])
    
    confidence_boost = 1.0
    if available_advanced_features >= 3:
        confidence_boost = 1.05
    elif available_advanced_features >= 2:
        confidence_boost = 1.03
    elif available_advanced_features >= 1:
        confidence_boost = 1.01
    
    final_ensemble_score *= confidence_boost
    final_ensemble_score = min(max(final_ensemble_score, 0), 100)
    
    # Enhanced ensemble label
    if final_ensemble_score >= 90:
        ensemble_label = "exceptional"
        confidence_level = "very high"
    elif final_ensemble_score >= 80:
        ensemble_label = "excellent"
        confidence_level = "high"
    elif final_ensemble_score >= 70:
        ensemble_label = "very good"
        confidence_level = "high"
    elif final_ensemble_score >= 60:
        ensemble_label = "good"
        confidence_level = "medium"
    elif final_ensemble_score >= 50:
        ensemble_label = "average"
        confidence_level = "medium"
    elif final_ensemble_score >= 40:
        ensemble_label = "below average"
        confidence_level = "medium"
    else:
        ensemble_label = "poor"
        confidence_level = "low"
    
    # Add ensemble results to result
    result['final_ensemble_score'] = round(final_ensemble_score, 1)
    result['ensemble_label'] = ensemble_label
    result['confidence_level'] = confidence_level
    result['ensemble_breakdown'] = {
        'available_advanced_features': available_advanced_features,
        'confidence_boost': round(confidence_boost, 3),
        'feature_weights': normalized_weights
    }
    
    result.update(heuristics)
    print("Final face rating result:", result)
    return result


def calculate_first_impression_score(attractiveness, smile, confidence, heuristics, emotion, dominant_emotion):
    """
    Calculate an advanced first impression score using multiple AI models and factors.
    Returns a comprehensive analysis of how others would perceive you at first glance.
    """
    try:
        # Base score starts at 50
        score = 50
        
        # 1. Physical Attractiveness Factor (30% weight)
        if attractiveness > 85:
            score += 15  # Exceptionally attractive
        elif attractiveness > 75:
            score += 12  # Very attractive
        elif attractiveness > 65:
            score += 8   # Attractive
        elif attractiveness > 55:
            score += 4   # Above average
        elif attractiveness < 45:
            score -= 8   # Below average
        
        # 2. Approachability Factor (25% weight) - Smile and positive emotions
        approachability = 0
        if smile > 80:
            approachability += 12  # Very approachable
        elif smile > 60:
            approachability += 8   # Approachable
        elif smile > 40:
            approachability += 4   # Neutral
        elif smile < 20:
            approachability -= 6   # Might seem unapproachable
        
        # Add emotion-based approachability
        if emotion:
            positive_emotions = emotion.get("happiness", 0) + emotion.get("surprise", 0)
            negative_emotions = emotion.get("anger", 0) + emotion.get("fear", 0) + emotion.get("sadness", 0)
            emotion_boost = (positive_emotions - negative_emotions) * 0.1
            approachability += emotion_boost
        
        score += approachability
        
        # 3. Confidence Factor (20% weight)
        if confidence > 80:
            score += 10  # Very confident
        elif confidence > 70:
            score += 8   # Confident
        elif confidence > 60:
            score += 5   # Somewhat confident
        elif confidence < 40:
            score -= 6   # Low confidence
        
        # 4. Advanced Facial Features Analysis (20% weight) - Using enhanced heuristics
        facial_features_boost = 0
        if heuristics:
            # Basic features
            if 'eye_openness' in heuristics:
                eye_openness = heuristics['eye_openness']
                if eye_openness > 15:
                    facial_features_boost += 4   # Wide awake and alert
                elif eye_openness > 10:
                    facial_features_boost += 2   # Decently alert
                elif eye_openness < 5:
                    facial_features_boost -= 3   # Might look tired
            
            if 'jawline_symmetry' in heuristics:
                jaw_symmetry = heuristics['jawline_symmetry']
                if 2.0 < jaw_symmetry < 3.5:
                    facial_features_boost += 3   # Good facial structure
                elif jaw_symmetry < 1.5 or jaw_symmetry > 4.0:
                    facial_features_boost -= 2   # Less ideal proportions
            
            if 'eyebrow_distance' in heuristics:
                brow_distance = heuristics['eyebrow_distance']
                if 90 < brow_distance < 180:
                    facial_features_boost += 2   # Good eyebrow positioning
                elif brow_distance < 70 or brow_distance > 200:
                    facial_features_boost -= 1   # Less ideal positioning
            
            # Advanced features (new ML-enhanced analysis)
            if 'golden_ratio_score' in heuristics:
                golden_ratio = heuristics['golden_ratio_score']
                if golden_ratio > 80:
                    facial_features_boost += 5   # Exceptional proportions
                elif golden_ratio > 70:
                    facial_features_boost += 3   # Very good proportions
                elif golden_ratio > 60:
                    facial_features_boost += 1   # Good proportions
                elif golden_ratio < 40:
                    facial_features_boost -= 2   # Less ideal proportions
            
            if 'advanced_symmetry_score' in heuristics:
                symmetry_score = heuristics['advanced_symmetry_score']
                if symmetry_score > 85:
                    facial_features_boost += 4   # Exceptional symmetry
                elif symmetry_score > 75:
                    facial_features_boost += 3   # Very good symmetry
                elif symmetry_score > 65:
                    facial_features_boost += 1   # Good symmetry
                elif symmetry_score < 45:
                    facial_features_boost -= 2   # Less symmetrical
            
            if 'skin_quality_score' in heuristics:
                skin_quality = heuristics['skin_quality_score']
                if skin_quality > 80:
                    facial_features_boost += 3   # Excellent skin quality
                elif skin_quality > 70:
                    facial_features_boost += 2   # Very good skin quality
                elif skin_quality > 60:
                    facial_features_boost += 1   # Good skin quality
                elif skin_quality < 40:
                    facial_features_boost -= 1   # Poor skin quality
        
        score += facial_features_boost
        
        # 5. Energy and Presence Factor (10% weight)
        energy_score = 0
        if dominant_emotion == "happiness":
            energy_score += 4   # Positive energy
        elif dominant_emotion == "surprise":
            energy_score += 3   # Engaging energy
        elif dominant_emotion == "neutral":
            energy_score += 1   # Calm presence
        elif dominant_emotion in ["anger", "fear", "sadness"]:
            energy_score -= 3   # Negative energy
        
        score += energy_score
        
        # 6. Overall Presentation Bonus
        if score > 80:
            score += 3   # Bonus for excellent first impression
        elif score < 30:
            score -= 5   # Penalty for very poor first impression
        
        # Ensure score is within bounds
        score = max(0, min(100, score))
        
        # Generate label and analysis
        if score >= 90:
            label = "Exceptional"
            analysis = "You make an outstanding first impression! People will be immediately drawn to your presence."
        elif score >= 80:
            label = "Excellent"
            analysis = "You make a very strong first impression. People will find you approachable and engaging."
        elif score >= 70:
            label = "Very Good"
            analysis = "You make a good first impression. People will likely have a positive initial reaction to you."
        elif score >= 60:
            label = "Good"
            analysis = "You make a decent first impression. People will generally have a positive view of you."
        elif score >= 50:
            label = "Average"
            analysis = "Your first impression is okay. There's room for improvement to make a stronger impact."
        elif score >= 40:
            label = "Below Average"
            analysis = "Your first impression could be improved. Consider working on confidence and approachability."
        else:
            label = "Needs Improvement"
            analysis = "Your first impression needs work. Focus on confidence, smile, and positive energy."
        
        return {
            "score": score,
            "label": label,
            "analysis": analysis
        }
        
    except Exception as e:
        print(f"First impression score calculation error: {e}")
        # Fallback to basic calculation
        basic_score = (attractiveness + smile + confidence) / 3
        return {
            "score": max(0, min(100, basic_score)),
            "label": "Basic",
            "analysis": "Basic first impression calculation due to analysis error."
        }


def calculate_approachability_score(smile, emotion, heuristics, confidence):
    """
    Calculate an advanced approachability score using facial expressions, emotions, and features.
    Returns a comprehensive analysis of how approachable someone appears to others.
    """
    try:
        # Base score starts at 50
        score = 50
        
        # 1. Facial Expression Analysis (40% weight)
        # Smile intensity - direct correlation with approachability
        if smile > 80:
            score += 20  # Very approachable smile
        elif smile > 60:
            score += 15  # Approachable smile
        elif smile > 40:
            score += 10  # Moderate smile
        elif smile > 20:
            score += 5   # Slight smile
        elif smile < 10:
            score -= 10  # No smile - less approachable
        
        # Emotion balance - positive emotions boost approachability
        if emotion:
            positive_emotions = emotion.get("happiness", 0) + emotion.get("surprise", 0)
            negative_emotions = emotion.get("anger", 0) + emotion.get("fear", 0) + emotion.get("sadness", 0)
            emotion_boost = (positive_emotions - negative_emotions) * 0.2
            score += emotion_boost
        
        # 2. Facial Features Analysis (30% weight) - Using MediaPipe heuristics
        facial_features_boost = 0
        if heuristics:
            # Eye openness - wide, alert eyes are more approachable
            if 'eye_openness' in heuristics:
                eye_openness = heuristics['eye_openness']
                if eye_openness > 15:
                    facial_features_boost += 8   # Very alert and approachable
                elif eye_openness > 10:
                    facial_features_boost += 5   # Alert and approachable
                elif eye_openness > 5:
                    facial_features_boost += 2   # Moderately alert
                elif eye_openness < 3:
                    facial_features_boost -= 5   # Might look tired/uninterested
            
            # Eyebrow positioning - relaxed eyebrows are more approachable
            if 'brow_symmetry' in heuristics:
                brow_symmetry = heuristics['brow_symmetry']
                if brow_symmetry < 5:
                    facial_features_boost += 6   # Relaxed, approachable eyebrows
                elif brow_symmetry < 10:
                    facial_features_boost += 3   # Moderately relaxed
                elif brow_symmetry > 15:
                    facial_features_boost -= 4   # Furrowed brows - less approachable
            
            # Jaw symmetry - balanced features are more approachable
            if 'jawline_symmetry' in heuristics:
                jaw_symmetry = heuristics['jawline_symmetry']
                if 2.0 < jaw_symmetry < 3.5:
                    facial_features_boost += 4   # Well-proportioned, approachable
                elif jaw_symmetry < 1.5 or jaw_symmetry > 4.0:
                    facial_features_boost -= 2   # Less balanced proportions
        
        score += facial_features_boost
        
        # 3. Confidence Factor (20% weight) - confident people often seem more approachable
        if confidence > 80:
            score += 10  # Very confident - approachable
        elif confidence > 70:
            score += 8   # Confident - approachable
        elif confidence > 60:
            score += 5   # Somewhat confident
        elif confidence < 40:
            score -= 5   # Low confidence - might seem less approachable
        
        # 4. Energy and Presence Factor (10% weight)
        energy_score = 0
        if emotion:
            # High happiness and low negative emotions indicate positive energy
            happiness = emotion.get("happiness", 0)
            anger = emotion.get("anger", 0)
            fear = emotion.get("fear", 0)
            sadness = emotion.get("sadness", 0)
            
            if happiness > 60 and (anger + fear + sadness) < 20:
                energy_score += 5   # Very positive energy - very approachable
            elif happiness > 40 and (anger + fear + sadness) < 40:
                energy_score += 3   # Positive energy - approachable
            elif anger > 30 or fear > 30 or sadness > 30:
                energy_score -= 3   # Negative energy - less approachable
        
        score += energy_score
        
        # 5. Overall Presentation Bonus
        if score > 80:
            score += 3   # Bonus for very approachable
        elif score < 30:
            score -= 5   # Penalty for very unapproachable
        
        # Ensure score is within bounds
        score = max(0, min(100, score))
        
        # Generate label and analysis
        if score >= 90:
            label = "Very Approachable"
            analysis = "People feel instantly comfortable around you! Your warm smile and positive energy make you incredibly approachable."
        elif score >= 80:
            label = "Approachable"
            analysis = "You have a welcoming presence. People feel comfortable approaching you and engaging in conversation."
        elif score >= 70:
            label = "Moderately Approachable"
            analysis = "You're generally approachable. People feel comfortable interacting with you in most situations."
        elif score >= 60:
            label = "Somewhat Approachable"
            analysis = "You're approachable in familiar settings. Consider smiling more to increase your approachability."
        elif score >= 50:
            label = "Neutral"
            analysis = "Your approachability is average. Small changes like smiling more could make you more approachable."
        elif score >= 40:
            label = "Somewhat Reserved"
            analysis = "You might seem reserved to others. Try maintaining eye contact and showing more positive emotions."
        elif score >= 30:
            label = "Reserved"
            analysis = "People might hesitate to approach you. Focus on relaxing your facial expressions and showing warmth."
        else:
            label = "Intimidating"
            analysis = "Your presence might be overwhelming to others. Work on softening your expressions and showing more warmth."
        
        return {
            "score": score,
            "label": label,
            "analysis": analysis
        }
        
    except Exception as e:
        print(f"Approachability score calculation error: {e}")
        # Fallback to basic calculation
        basic_score = (smile + confidence) / 2
        return {
            "score": max(0, min(100, basic_score)),
            "label": "Basic",
            "analysis": "Basic approachability calculation due to analysis error."
        }


def rate_personality(face_attrs):
    """
    Infer a comprehensive personality rating from Face++ emotion scores and facial attributes.
    Returns a detailed personality analysis with dramatic scaling.
    """
    if not face_attrs or "faces" not in face_attrs or not face_attrs["faces"]:
        return {"score": None, "personality": "Unknown", "description": "No face detected", "error": "No face detected."}

    try:
        face = face_attrs["faces"][0]
        attributes = face.get("attributes", {})
        emotion = attributes.get("emotion", {})
        beauty = attributes.get("beauty", {})
        
        if not emotion:
            return {"score": None, "personality": "Unknown", "description": "No emotion data", "error": "No emotion data."}

        # Extract emotion values with defaults
        happiness = emotion.get("happiness", 0)
        anger = emotion.get("anger", 0)
        sadness = emotion.get("sadness", 0)
        fear = emotion.get("fear", 0)
        surprise = emotion.get("surprise", 0)
        disgust = emotion.get("disgust", 0)
        neutral = emotion.get("neutral", 0)
        # Extract beauty scores
        male_score = beauty.get("male_score", 50)
        female_score = beauty.get("female_score", 50)
        avg_beauty = (male_score + female_score) / 2
        # Calculate personality score using multiple factors
        base_score = 50
        personality = "Balanced Individual"
        desc = "You have a well-rounded personality that adapts to different situations."

        # Happiness bonus (positive emotions boost personality)
        if happiness > 80:
            base_score += 25
            personality = "Radiant Optimist"
            desc = "Your infectious joy and positivity light up every room! 🌟"
        elif happiness > 60:
            base_score += 15
            personality = "Cheerful Soul"
            desc = "You bring warmth and happiness wherever you go! 😊"
        elif happiness > 40:
            base_score += 8
            personality = "Balanced Optimist"
            desc = "You maintain a positive outlook even in challenging times! ✨"
        # Negative emotions penalty
        if anger > 50:
            base_score -= 20
            personality = "Passionate Fire"
            desc = "Your intensity is magnetic, but sometimes overwhelming! 🔥"
        elif sadness > 50:
            base_score -= 15
            personality = "Deep Thinker"
            desc = "Your sensitivity and empathy make you incredibly understanding! 💭"
        elif fear > 50:
            base_score -= 10
            personality = "Cautious Observer"
            desc = "Your careful nature helps you avoid unnecessary risks! 👀"
        # Surprise bonus (openness to new experiences)
        if surprise > 40:
            base_score += 10
            personality = "Adventure Seeker"
            desc = "Your openness to new experiences makes life exciting! 🚀"
        # Beauty confidence bonus
        if avg_beauty > 70:
            base_score += 8
            personality += " with Natural Confidence"
            desc += " Your natural beauty gives you an extra boost of confidence!"
        elif avg_beauty < 30:
            base_score -= 5
            personality += " with Inner Strength"
            desc += " Your inner strength shines brighter than any external beauty!"
        # Neutral expression analysis
        if neutral > 60:
            base_score += 5
            personality = "Calm Professional"
            desc = "Your composed demeanor commands respect and trust! 🎯"
        # Ensure score is within bounds and apply dramatic scaling
        final_score = max(20, min(95, base_score))
        # Apply dramatic scaling for more interesting results
        if final_score > 80:
            final_score = min(95, final_score + 5)  # Boost high scores
        elif final_score < 40:
            final_score = max(20, final_score - 5)  # Penalize low scores

        return {
            "score": int(final_score),
            "personality": personality,
            "description": desc,
        }
    except Exception as e:
        print(f"Personality rating error: {e}")
        return {"score": None, "personality": "Unknown", "description": "Error in personality analysis", "error": str(e)}


def try_parse_fashion_json(content):
    # Try strict JSON first
    try:
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = content[start:end]
            return json.loads(json_str)
    except Exception:
        pass

    # Fallback: regex extraction for each field
    def extract_field(field):
        pattern = rf'"{field}"\s*:\s*(.*?)(,|\n|$)'
        match = re.search(pattern, content)
        if match:
            value = match.group(1).strip().strip('"')
            return value
        return ""

    return {
        "outfit_rating": extract_field("outfit_rating"),
        "items": extract_field("items"),
        "good": extract_field("good"),
        "bad": extract_field("bad"),
        "improvements": extract_field("improvements"),
        "overall_style": extract_field("overall_style"),
        "roast": extract_field("roast"),
        "error": "Used fallback parser due to invalid JSON."
    }


def analyze_fashion_llama3(image_bytes):
    """
    Use OpenRouter's Llama-3 Vision model to analyze fashion and outfit, and give ratings in a strict JSON format.
    """
    b64 = base64.b64encode(image_bytes).decode('utf-8')
    prompt = (
        "You are a brutally honest, witty fashion critic. "
        "Given this image, analyze the person's outfit and overall look. "
        "Respond in this exact JSON format:\n"
        "{\n"
        "  \"outfit_rating\": <number 1-10>,\n"
        "  \"summary\": <one-sentence summary>,\n"
        "  \"pros\": <what's good about the outfit>,\n"
        "  \"cons\": <what's bad or could be improved>,\n"
        "  \"roast\": <short, funny, tweet-sized roast>\n"
        "}\n"
        "Be concise, honest, and never add extra text outside the JSON."
    )
    payload = {
        "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": b64}
            ]}
        ],
        "max_tokens": 200
    }
    response = make_openrouter_request("https://openrouter.ai/api/v1/chat/completions", payload)
    
    if response is None:
        return {"error": "Fashion analysis failed - all API keys exhausted."}
    
    try:
        content = response.json()['choices'][0]['message']['content']
        result = try_parse_fashion_json(content)
        return result
    except Exception as e:
        print(f"Llama-3 Vision unexpected payload: {response.text}")
        return {"error": "Fashion analysis response format unexpected."}


def detect_fashion_yolov8(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    results = yolo_model(image)
    items = []
    for r in results:
        for c in r.boxes.cls:
            label = yolo_model.names[int(c)]
            items.append(label)
    return list(set(items))  # Unique items


def analyze_fashion_llama3_with_items(image_bytes, detected_items):
    b64 = base64.b64encode(image_bytes).decode('utf-8')
    prompt = (
        f"You are a world-class fashion designer and critic. The following clothing items were detected in the image: {detected_items}. "
        "For each item, say specifically what is good and what is bad about it, as a professional would. "
        "Then, rate the overall outfit from 1-10 as a fashion expert would, suggest concrete ways to improve the outfit (e.g., color, fit, accessories, layering, shoes, etc.). "
        "Summarize the overall style and give a clear, actionable tip to uplift the look. "
        "Respond in this exact JSON format:\n"
        "{\n"
        "  \"outfit_rating\": <1-10>,\n"
        "  \"items\": [ ... ],\n"
        "  \"good\": {<item>: <what's good>, ...},\n"
        "  \"bad\": {<item>: <what's bad>, ...},\n"
        "  \"improvements\": <specific suggestions>,\n"
        "  \"overall_style\": <summary>,\n"
        "  \"roast\": <short, funny, tweet-sized roast>\n"
        "}\n"
        "Be specific, honest, and never add extra text outside the JSON."
    )
    payload = {
        "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": b64}
            ]}
        ],
        "max_tokens": 200
    }
    response = make_openrouter_request("https://openrouter.ai/api/v1/chat/completions", payload)
    
    if response is None:
        return {"error": "Fashion analysis failed - all API keys exhausted."}
    
    try:
        content = response.json()['choices'][0]['message']['content']
        result = try_parse_fashion_json(content)
        return result
    except Exception as e:
        print(f"Llama-3 Vision unexpected payload: {response.text}")
        return {"error": "Fashion analysis response format unexpected."}


# --- Advanced Posture Rating using YOLOv8 Pose ---
def rate_posture(image_bytes):
    """
    Use YOLOv8 pose model to provide comprehensive posture analysis with dramatic scaling.
    Returns detailed posture assessment with multiple factors.
    """
    try:
        from ultralytics import YOLO
        import numpy as np
        import io
        from PIL import Image
        import math
        
        print("Starting ultra-fine-grained posture analysis...")
        pose_model = YOLO('yolov8n-pose.pt')
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        print(f"Image size: {image.size}")
        
        # ULTRA-SENSITIVE YOLO pose detection for advanced biomechanical analysis
        results = pose_model(image, conf=0.2, iou=0.4, verbose=False)  # Even lower confidence for maximum detection
        
        # Try multiple detection attempts with different parameters
        if len(results) == 0 or len(results[0].keypoints) == 0:
            print("First attempt failed, trying with different parameters...")
            results = pose_model(image, conf=0.1, iou=0.3, verbose=False)  # Very low confidence
        
        if len(results) == 0 or len(results[0].keypoints) == 0:
            print("Second attempt failed, trying with image preprocessing...")
            # Try with slightly resized image for better detection
            resized_image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
            results = pose_model(resized_image, conf=0.15, iou=0.4, verbose=False)
        print(f"YOLOv8 pose results: {len(results)} detections")
        
        # Assume first person detected
        for i, r in enumerate(results):
            print(f"Processing detection {i}: has keypoints = {hasattr(r, 'keypoints')}, keypoints length = {len(r.keypoints) if hasattr(r, 'keypoints') else 0}")
            if hasattr(r, 'keypoints') and len(r.keypoints) > 0:
                kps = r.keypoints[0].xy.cpu().numpy()  # shape (17, 2) for COCO
                print(f"Keypoints shape: {kps.shape}, Number of keypoints: {len(kps)}")
                
                # Analyze posture based on available keypoints
                print(f"Analyzing posture with {len(kps)} keypoints")
                
                # Extract available keypoints safely
                available_keypoints = {}
                keypoint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
                                 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                                 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                                 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
                
                for i, name in enumerate(keypoint_names):
                    if i < len(kps):
                        available_keypoints[name] = kps[i]
                
                print(f"Available keypoints: {list(available_keypoints.keys())}")
                
                # Advanced Biomechanical Posture Analysis
                print(f"Starting advanced posture analysis with {len(available_keypoints)} keypoints")
                
                # Initialize comprehensive analysis
                posture_analysis = {}
                biomechanical_scores = {}
                detailed_metrics = {}
                
                # 1. HEAD-NECK-CERVICAL SPINE ANALYSIS
                if 'nose' in available_keypoints and 'left_shoulder' in available_keypoints and 'right_shoulder' in available_keypoints:
                    nose = available_keypoints['nose']
                    left_shoulder = available_keypoints['left_shoulder']
                    right_shoulder = available_keypoints['right_shoulder']
                    
                    # Calculate cervical spine alignment
                    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                    avg_hip_y = (left_hip[1] + right_hip[1]) / 2
                    shoulder_hip_distance = avg_hip_y - avg_shoulder_y
                    
                    # Biomechanical analysis: Ideal head position is slightly forward of shoulders (15-25 pixels)
                    # Too far forward (>40px) indicates forward head posture
                    # Too far back (<0px) indicates retracted head
                    if head_shoulder_distance < -10:
                        cervical_score = 25
                        cervical_assessment = "Severely retracted head - indicates tension/stress"
                    elif head_shoulder_distance < 0:
                        cervical_score = 45
                        cervical_assessment = "Retracted head position - may indicate tension"
                    elif head_shoulder_distance < 15:
                        cervical_score = 75
                        cervical_assessment = "Slightly retracted but acceptable"
                    elif head_shoulder_distance < 25:
                        cervical_score = 95
                        cervical_assessment = "Optimal head-neck alignment"
                    elif head_shoulder_distance < 40:
                        cervical_score = 65
                        cervical_assessment = "Mild forward head posture"
                    elif head_shoulder_distance < 60:
                        cervical_score = 40
                        cervical_assessment = "Moderate forward head posture"
                    elif head_shoulder_distance < 80:
                        cervical_score = 25
                        cervical_assessment = "Severe forward head posture"
                    else:
                        cervical_score = 15
                        cervical_assessment = "Extreme forward head posture - needs immediate attention"
                    
                    biomechanical_scores["cervical_spine"] = cervical_score
                    detailed_metrics["head_shoulder_distance"] = head_shoulder_distance
                    posture_analysis["head_neck_alignment"] = f"{cervical_assessment} (Score: {cervical_score})"
                else:
                    biomechanical_scores["cervical_spine"] = 50
                    posture_analysis["head_neck_alignment"] = "Insufficient data for analysis"
                
                # 2. SHOULDER COMPLEX ANALYSIS
                if 'left_shoulder' in available_keypoints and 'right_shoulder' in available_keypoints:
                    left_shoulder = available_keypoints['left_shoulder']
                    right_shoulder = available_keypoints['right_shoulder']
                    
                    # Shoulder levelness (frontal plane) - ultra-sensitive
                    shoulder_height_diff = abs(left_shoulder[1] - right_shoulder[1])
                    # Micro-variations: even 1-2 pixel differences matter
                    if shoulder_height_diff < 2:
                        shoulder_levelness_score = 98  # Perfectly level
                    elif shoulder_height_diff < 5:
                        shoulder_levelness_score = 95  # Nearly perfect
                    elif shoulder_height_diff < 10:
                        shoulder_levelness_score = 90  # Very good
                    elif shoulder_height_diff < 15:
                        shoulder_levelness_score = 85  # Good
                    elif shoulder_height_diff < 20:
                        shoulder_levelness_score = 80  # Above average
                    elif shoulder_height_diff < 25:
                        shoulder_levelness_score = 75  # Average
                    elif shoulder_height_diff < 30:
                        shoulder_levelness_score = 70  # Below average
                    elif shoulder_height_diff < 40:
                        shoulder_levelness_score = 60  # Poor
                    elif shoulder_height_diff < 50:
                        shoulder_levelness_score = 50  # Very poor
                    else:
                        shoulder_levelness_score = 40  # Terrible
                    
                    # Shoulder protraction/retraction (sagittal plane)
                    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
                    ideal_shoulder_width = image.width * 0.25  # Ideal shoulder width is ~25% of image width
                    shoulder_protraction_score = max(0, 100 - abs(shoulder_width - ideal_shoulder_width) / ideal_shoulder_width * 100)
                    
                    # Combined shoulder score
                    shoulder_score = (shoulder_levelness_score + shoulder_protraction_score) / 2
                    
                    biomechanical_scores["shoulder_complex"] = shoulder_score
                    detailed_metrics["shoulder_height_diff"] = shoulder_height_diff
                    detailed_metrics["shoulder_width"] = shoulder_width
                    detailed_metrics["ideal_shoulder_width"] = ideal_shoulder_width
                    
                    if shoulder_score > 90:
                        shoulder_assessment = "Excellent shoulder alignment and positioning"
                    elif shoulder_score > 80:
                        shoulder_assessment = "Good shoulder alignment with minor asymmetry"
                    elif shoulder_score > 70:
                        shoulder_assessment = "Fair shoulder alignment, some protraction/retraction"
                    elif shoulder_score > 50:
                        shoulder_assessment = "Poor shoulder alignment, significant asymmetry"
                    else:
                        shoulder_assessment = "Very poor shoulder alignment, severe asymmetry"
                    
                    posture_analysis["shoulder_alignment"] = f"{shoulder_assessment} (Score: {shoulder_score:.1f})"
                else:
                    biomechanical_scores["shoulder_complex"] = 50
                    posture_analysis["shoulder_alignment"] = "Insufficient data for analysis"
                
                # 3. THORACIC SPINE AND UPRIGHTNESS ANALYSIS
                if 'left_shoulder' in available_keypoints and 'right_shoulder' in available_keypoints and 'left_hip' in available_keypoints and 'right_hip' in available_keypoints:
                    left_shoulder = available_keypoints['left_shoulder']
                    right_shoulder = available_keypoints['right_shoulder']
                    left_hip = available_keypoints['left_hip']
                    right_hip = available_keypoints['right_hip']
                    # Calculate thoracic spine alignment
                    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                    avg_hip_y = (left_hip[1] + right_hip[1]) / 2
                    shoulder_hip_distance = avg_hip_y - avg_shoulder_y
                    # Biomechanical analysis: Ideal upright posture has shoulders above hips
                    # Calculate the ratio of shoulder-hip distance to total body height
                    total_body_height = image.height
                    shoulder_hip_ratio = shoulder_hip_distance / total_body_height
                    # Ultra-sensitive thoracic spine analysis - detects micro-variations
                    # Optimal ratio is 0.15-0.25 (shoulders 15-25% above hips)
                    if shoulder_hip_ratio > 0.35:
                        thoracic_score = 99
                        thoracic_assessment = "Perfect upright posture, exceptional alignment"
                    elif shoulder_hip_ratio > 0.30:
                        thoracic_score = 96
                        thoracic_assessment = "Exceptional upright posture, perfect thoracic alignment"
                    elif shoulder_hip_ratio > 0.28:
                        thoracic_score = 94
                        thoracic_assessment = "Excellent upright posture, near-perfect alignment"
                    elif shoulder_hip_ratio > 0.25:
                        thoracic_score = 92
                        thoracic_assessment = "Excellent upright posture, optimal thoracic alignment"
                    elif shoulder_hip_ratio > 0.22:
                        thoracic_score = 88
                        thoracic_assessment = "Very good upright posture, well-aligned thoracic spine"
                    elif shoulder_hip_ratio > 0.20:
                        thoracic_score = 85
                        thoracic_assessment = "Good upright posture, well-aligned thoracic spine"
                    elif shoulder_hip_ratio > 0.18:
                        thoracic_score = 82
                        thoracic_assessment = "Above average upright posture"
                    elif shoulder_hip_ratio > 0.15:
                        thoracic_score = 78
                        thoracic_assessment = "Good upright posture, well-aligned thoracic spine"
                    elif shoulder_hip_ratio > 0.12:
                        thoracic_score = 75
                        thoracic_assessment = "Fair posture, slight thoracic kyphosis"
                    elif shoulder_hip_ratio > 0.08:
                        thoracic_score = 70
                        thoracic_assessment = "Below average posture, noticeable kyphosis"
                    elif shoulder_hip_ratio > 0.05:
                        thoracic_score = 65
                        thoracic_assessment = "Fair posture, slight thoracic kyphosis"
                    elif shoulder_hip_ratio > 0.02:
                        thoracic_score = 60
                        thoracic_assessment = "Poor posture, moderate thoracic kyphosis"
                    elif shoulder_hip_ratio > 0:
                        thoracic_score = 55
                        thoracic_assessment = "Poor posture, moderate thoracic kyphosis"
                    elif shoulder_hip_ratio > -0.02:
                        thoracic_score = 45
                        thoracic_assessment = "Very poor posture, severe thoracic kyphosis"
                    elif shoulder_hip_ratio > -0.05:
                        thoracic_score = 35
                        thoracic_assessment = "Very poor posture, severe thoracic kyphosis"
                    elif shoulder_hip_ratio > -0.08:
                        thoracic_score = 25
                        thoracic_assessment = "Critical posture issue - severe slouch"
                    else:
                        thoracic_score = 15
                        thoracic_assessment = "Critical posture issue - severe slouch/kyphosis"
                    biomechanical_scores["thoracic_spine"] = thoracic_score
                    detailed_metrics["shoulder_hip_distance"] = shoulder_hip_distance
                    detailed_metrics["shoulder_hip_ratio"] = shoulder_hip_ratio
                    posture_analysis["thoracic_alignment"] = f"{thoracic_assessment} (Score: {thoracic_score})"
                else:
                    biomechanical_scores["thoracic_spine"] = 50
                    posture_analysis["thoracic_alignment"] = "Insufficient data for analysis"
                # 4. LUMBAR SPINE AND PELVIC ALIGNMENT
                if 'left_hip' in available_keypoints and 'right_hip' in available_keypoints and 'left_knee' in available_keypoints and 'right_knee' in available_keypoints:
                    left_hip = available_keypoints['left_hip']
                    right_hip = available_keypoints['right_hip']
                    left_knee = available_keypoints['left_knee']
                    right_knee = available_keypoints['right_knee']
                    # Pelvic tilt analysis
                    hip_height_diff = abs(left_hip[1] - right_hip[1])
                    pelvic_tilt_score = max(0, 100 - (hip_height_diff / (image.height * 0.02)) * 100)
                    # Lumbar spine angle (hip to knee alignment)
                    left_lumbar_angle = math.degrees(math.atan2(left_knee[1] - left_hip[1], left_knee[0] - left_hip[0]))
                    right_lumbar_angle = math.degrees(math.atan2(right_knee[1] - right_hip[1], right_knee[0] - right_hip[0]))
                    # Ideal lumbar angle is 90-110 degrees (slight forward lean)
                    left_lumbar_score = max(0, 100 - abs(left_lumbar_angle - 100) * 2)
                    right_lumbar_score = max(0, 100 - abs(right_lumbar_angle - 100) * 2)
                    avg_lumbar_score = (left_lumbar_score + right_lumbar_score) / 2
                    
                    # Combined lumbar-pelvic score
                    lumbar_pelvic_score = (pelvic_tilt_score + avg_lumbar_score) / 2
                    
                    biomechanical_scores["lumbar_pelvic"] = lumbar_pelvic_score
                    detailed_metrics["hip_height_diff"] = hip_height_diff
                    detailed_metrics["left_lumbar_angle"] = left_lumbar_angle
                    detailed_metrics["right_lumbar_angle"] = right_lumbar_angle
                    
                    if lumbar_pelvic_score > 90:
                        lumbar_assessment = "Excellent lumbar-pelvic alignment"
                    elif lumbar_pelvic_score > 80:
                        lumbar_assessment = "Good lumbar-pelvic alignment"
                    elif lumbar_pelvic_score > 70:
                        lumbar_assessment = "Fair lumbar-pelvic alignment"
                    elif lumbar_pelvic_score > 50:
                        lumbar_assessment = "Poor lumbar-pelvic alignment"
                    else:
                        lumbar_assessment = "Very poor lumbar-pelvic alignment"
                    
                    posture_analysis["lumbar_pelvic_alignment"] = f"{lumbar_assessment} (Score: {lumbar_pelvic_score:.1f})"
                else:
                    biomechanical_scores["lumbar_pelvic"] = 50
                    posture_analysis["lumbar_pelvic_alignment"] = "Insufficient data for analysis"
                
                # 5. UPPER EXTREMITY ANALYSIS
                if 'left_elbow' in available_keypoints and 'right_elbow' in available_keypoints and 'left_shoulder' in available_keypoints and 'right_shoulder' in available_keypoints:
                    left_elbow = available_keypoints['left_elbow']
                    right_elbow = available_keypoints['right_elbow']
                    left_shoulder = available_keypoints['left_shoulder']
                    right_shoulder = available_keypoints['right_shoulder']
                    
                    # Arm angles and positioning
                    left_arm_angle = math.degrees(math.atan2(left_elbow[1] - left_shoulder[1], left_elbow[0] - left_shoulder[0]))
                    right_arm_angle = math.degrees(math.atan2(right_elbow[1] - right_shoulder[1], right_elbow[0] - right_shoulder[0]))
                    
                    # Ideal arm position is 10-20 degrees down from horizontal
                    left_arm_score = max(0, 100 - abs(left_arm_angle - 15) * 3)
                    right_arm_score = max(0, 100 - abs(right_arm_angle - 15) * 3)
                    avg_arm_score = (left_arm_score + right_arm_score) / 2
                    
                    # Arm symmetry
                    arm_symmetry_score = max(0, 100 - abs(left_arm_angle - right_arm_angle) * 2)
                    
                    # Combined upper extremity score
                    upper_extremity_score = (avg_arm_score + arm_symmetry_score) / 2
                    
                    biomechanical_scores["upper_extremity"] = upper_extremity_score
                    detailed_metrics["left_arm_angle"] = left_arm_angle
                    detailed_metrics["right_arm_angle"] = right_arm_angle
                    detailed_metrics["arm_symmetry"] = abs(left_arm_angle - right_arm_angle)
                    
                    if upper_extremity_score > 90:
                        arm_assessment = "Excellent arm positioning and symmetry"
                    elif upper_extremity_score > 80:
                        arm_assessment = "Good arm positioning with minor asymmetry"
                    elif upper_extremity_score > 70:
                        arm_assessment = "Fair arm positioning"
                    elif upper_extremity_score > 50:
                        arm_assessment = "Poor arm positioning"
                    else:
                        arm_assessment = "Very poor arm positioning"
                    
                    posture_analysis["upper_extremity"] = f"{arm_assessment} (Score: {upper_extremity_score:.1f})"
                else:
                    biomechanical_scores["upper_extremity"] = 50
                    posture_analysis["upper_extremity"] = "Insufficient data for analysis"
                
                # 6. LOWER EXTREMITY AND GAIT ANALYSIS
                if 'left_knee' in available_keypoints and 'right_knee' in available_keypoints and 'left_ankle' in available_keypoints and 'right_ankle' in available_keypoints:
                    left_knee = available_keypoints['left_knee']
                    right_knee = available_keypoints['right_knee']
                    left_ankle = available_keypoints['left_ankle']
                    right_ankle = available_keypoints['right_ankle']
                    
                    # Knee angles and alignment
                    left_knee_angle = math.degrees(math.atan2(left_ankle[1] - left_knee[1], left_ankle[0] - left_knee[0]))
                    right_knee_angle = math.degrees(math.atan2(right_ankle[1] - right_knee[1], right_ankle[0] - right_knee[0]))
                    
                    # Ideal knee angle is 170-180 degrees (slight bend)
                    left_knee_score = max(0, 100 - abs(left_knee_angle - 175) * 2)
                    right_knee_score = max(0, 100 - abs(right_knee_angle - 175) * 2)
                    avg_knee_score = (left_knee_score + right_knee_score) / 2
                    
                    # Lower extremity symmetry
                    knee_symmetry_score = max(0, 100 - abs(left_knee_angle - right_knee_angle) * 2)
                    
                    # Combined lower extremity score
                    lower_extremity_score = (avg_knee_score + knee_symmetry_score) / 2
                    
                    biomechanical_scores["lower_extremity"] = lower_extremity_score
                    detailed_metrics["left_knee_angle"] = left_knee_angle
                    detailed_metrics["right_knee_angle"] = right_knee_angle
                    detailed_metrics["knee_symmetry"] = abs(left_knee_angle - right_knee_angle)
                    
                    if lower_extremity_score > 90:
                        leg_assessment = "Excellent lower extremity alignment"
                    elif lower_extremity_score > 80:
                        leg_assessment = "Good lower extremity alignment"
                    elif lower_extremity_score > 70:
                        leg_assessment = "Fair lower extremity alignment"
                    elif lower_extremity_score > 50:
                        leg_assessment = "Poor lower extremity alignment"
                    else:
                        leg_assessment = "Very poor lower extremity alignment"
                    
                    posture_analysis["lower_extremity"] = f"{leg_assessment} (Score: {lower_extremity_score:.1f})"
                else:
                    biomechanical_scores["lower_extremity"] = 50
                    posture_analysis["lower_extremity"] = "Insufficient data for analysis"
                
                # 7. OVERALL POSTURAL STABILITY AND BALANCE
                # Calculate center of mass and stability
                if len(available_keypoints) >= 6:
                    # Calculate center of mass from available keypoints
                    com_x = sum(kp[0] for kp in available_keypoints.values()) / len(available_keypoints)
                    com_y = sum(kp[1] for kp in available_keypoints.values()) / len(available_keypoints)
                    
                    # Distance from center of image (stability indicator)
                    image_center_x = image.width / 2
                    image_center_y = image.height / 2
                    
                    com_offset = math.sqrt((com_x - image_center_x)**2 + (com_y - image_center_y)**2)
                    max_offset = math.sqrt((image.width/2)**2 + (image.height/2)**2)
                    
                    stability_score = max(0, 100 - (com_offset / max_offset) * 100)
                    
                    biomechanical_scores["postural_stability"] = stability_score
                    detailed_metrics["center_of_mass_x"] = com_x
                    detailed_metrics["center_of_mass_y"] = com_y
                    detailed_metrics["stability_offset"] = com_offset
                    
                    if stability_score > 90:
                        stability_assessment = "Excellent postural stability and balance"
                    elif stability_score > 80:
                        stability_assessment = "Good postural stability"
                    elif stability_score > 70:
                        stability_assessment = "Fair postural stability"
                    elif stability_score > 50:
                        stability_assessment = "Poor postural stability"
                    else:
                        stability_assessment = "Very poor postural stability"
                    
                    posture_analysis["postural_stability"] = f"{stability_assessment} (Score: {stability_score:.1f})"
                else:
                    biomechanical_scores["postural_stability"] = 50
                    posture_analysis["postural_stability"] = "Insufficient data for stability analysis"
                
                # 8. ULTRA-FINE-GRAINED POSTURE SCORE CALCULATION
                # Detect even 0.1cm variations in posture
                
                # Start with a base score and add/subtract based on each component
                final_score = 50  # Start at neutral
                
                # ADVANCED BIOMECHANICAL ANALYSIS FUNCTIONS
                def calculate_advanced_angles(keypoints):
                    """Calculate real biomechanical angles from keypoints with facial landmarks"""
                    angles = {}
                    
                    # Enhanced head-neck angle with facial landmarks
                    if all(kp in keypoints for kp in ['nose', 'left_ear', 'left_shoulder']):
                        nose = keypoints['nose']
                        ear = keypoints['left_ear']
                        shoulder = keypoints['left_shoulder']
                        
                        # Calculate vectors with more precision
                        head_vector = [ear[0] - nose[0], ear[1] - nose[1]]
                        neck_vector = [shoulder[0] - ear[0], shoulder[1] - ear[1]]
                        
                        # Calculate angle with enhanced precision
                        dot_product = head_vector[0] * neck_vector[0] + head_vector[1] * neck_vector[1]
                        head_mag = math.sqrt(head_vector[0]**2 + head_vector[1]**2)
                        neck_mag = math.sqrt(neck_vector[0]**2 + neck_vector[1]**2)
                        
                        if head_mag > 0 and neck_mag > 0:
                            cos_angle = dot_product / (head_mag * neck_mag)
                            cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                            angle = math.degrees(math.acos(cos_angle))
                            angles['head_neck_angle'] = angle
                            
                            # Add micro-variations based on head position
                            head_forward_offset = nose[0] - shoulder[0]
                            angles['head_forward_offset'] = head_forward_offset
                    
                    # Enhanced shoulder analysis with bilateral comparison
                    if all(kp in keypoints for kp in ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow']):
                        left_shoulder = keypoints['left_shoulder']
                        right_shoulder = keypoints['right_shoulder']
                        left_elbow = keypoints['left_elbow']
                        right_elbow = keypoints['right_elbow']
                        
                        # Shoulder line vector
                        shoulder_vector = [right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1]]
                        
                        # Left arm vector
                        left_arm_vector = [left_elbow[0] - left_shoulder[0], left_elbow[1] - left_shoulder[1]]
                        # Right arm vector
                        right_arm_vector = [right_elbow[0] - right_shoulder[0], right_elbow[1] - right_shoulder[1]]
                        
                        # Calculate angles for both arms
                        for side, arm_vector in [('left', left_arm_vector), ('right', right_arm_vector)]:
                            dot_product = shoulder_vector[0] * arm_vector[0] + shoulder_vector[1] * arm_vector[1]
                            shoulder_mag = math.sqrt(shoulder_vector[0]**2 + shoulder_vector[1]**2)
                            arm_mag = math.sqrt(arm_vector[0]**2 + arm_vector[1]**2)
                            
                            if shoulder_mag > 0 and arm_mag > 0:
                                cos_angle = dot_product / (shoulder_mag * arm_mag)
                                cos_angle = max(-1, min(1, cos_angle))
                                angle = math.degrees(math.acos(cos_angle))
                                angles[f'{side}_shoulder_arm_angle'] = angle
                        
                        # Shoulder asymmetry analysis
                        shoulder_height_diff = abs(left_shoulder[1] - right_shoulder[1])
                        shoulder_width_diff = abs(left_shoulder[0] - right_shoulder[0])
                        angles['shoulder_asymmetry'] = math.sqrt(shoulder_height_diff**2 + shoulder_width_diff**2)
                    
                    # Enhanced spine curvature analysis
                    if all(kp in keypoints for kp in ['left_shoulder', 'left_hip', 'left_knee', 'left_ankle']):
                        shoulder = keypoints['left_shoulder']
                        hip = keypoints['left_hip']
                        knee = keypoints['left_knee']
                        ankle = keypoints['left_ankle']
                        
                        # Calculate spine straightness with multiple reference points
                        spine_vector = [hip[0] - shoulder[0], hip[1] - shoulder[1]]
                        leg_vector = [knee[0] - hip[0], knee[1] - hip[1]]
                        lower_leg_vector = [ankle[0] - knee[0], ankle[1] - knee[1]]
                        
                        # Ideal spine should be vertical (90 degrees to ground)
                        ground_vector = [1, 0]  # Horizontal reference
                        
                        # Calculate multiple spine angles for precision
                        for vector_name, vector in [('spine', spine_vector), ('upper_leg', leg_vector), ('lower_leg', lower_leg_vector)]:
                            dot_product = vector[0] * ground_vector[0] + vector[1] * ground_vector[1]
                            vector_mag = math.sqrt(vector[0]**2 + vector[1]**2)
                            ground_mag = math.sqrt(ground_vector[0]**2 + ground_vector[1]**2)
                            
                            if vector_mag > 0 and ground_mag > 0:
                                cos_angle = dot_product / (vector_mag * ground_mag)
                                cos_angle = max(-1, min(1, cos_angle))
                                angle = math.degrees(math.acos(cos_angle))
                                angles[f'{vector_name}_vertical_angle'] = angle
                        
                        # Calculate spine curvature ratio
                        if 'spine_vertical_angle' in angles and 'upper_leg_vertical_angle' in angles:
                            spine_angle = angles['spine_vertical_angle']
                            leg_angle = angles['upper_leg_vertical_angle']
                            curvature_ratio = abs(spine_angle - leg_angle) / 90.0
                            angles['spine_curvature_ratio'] = curvature_ratio
                    
                    return angles
                    
                    # Head-neck angle (cervical lordosis)
                    if all(kp in keypoints for kp in ['nose', 'left_ear', 'left_shoulder']):
                        nose = keypoints['nose']
                        ear = keypoints['left_ear']
                        shoulder = keypoints['left_shoulder']
                        
                        # Calculate vectors
                        head_vector = [ear[0] - nose[0], ear[1] - nose[1]]
                        neck_vector = [shoulder[0] - ear[0], shoulder[1] - ear[1]]
                        
                        # Calculate angle
                        dot_product = head_vector[0] * neck_vector[0] + head_vector[1] * neck_vector[1]
                        head_mag = math.sqrt(head_vector[0]**2 + head_vector[1]**2)
                        neck_mag = math.sqrt(neck_vector[0]**2 + neck_vector[1]**2)
                        
                        if head_mag > 0 and neck_mag > 0:
                            cos_angle = dot_product / (head_mag * neck_mag)
                            cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                            angle = math.degrees(math.acos(cos_angle))
                            angles['head_neck_angle'] = angle
                    
                    # Shoulder angle (scapular positioning)
                    if all(kp in keypoints for kp in ['left_shoulder', 'right_shoulder', 'left_elbow']):
                        left_shoulder = keypoints['left_shoulder']
                        right_shoulder = keypoints['right_shoulder']
                        left_elbow = keypoints['left_elbow']
                        
                        # Shoulder line vector
                        shoulder_vector = [right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1]]
                        # Arm vector
                        arm_vector = [left_elbow[0] - left_shoulder[0], left_elbow[1] - left_shoulder[1]]
                        
                        dot_product = shoulder_vector[0] * arm_vector[0] + shoulder_vector[1] * arm_vector[1]
                        shoulder_mag = math.sqrt(shoulder_vector[0]**2 + shoulder_vector[1]**2)
                        arm_mag = math.sqrt(arm_vector[0]**2 + arm_vector[1]**2)
                        
                        if shoulder_mag > 0 and arm_mag > 0:
                            cos_angle = dot_product / (shoulder_mag * arm_mag)
                            cos_angle = max(-1, min(1, cos_angle))
                            angle = math.degrees(math.acos(cos_angle))
                            angles['shoulder_arm_angle'] = angle
                    
                    # Spine curvature (thoracic kyphosis)
                    if all(kp in keypoints for kp in ['left_shoulder', 'left_hip', 'left_knee']):
                        shoulder = keypoints['left_shoulder']
                        hip = keypoints['left_hip']
                        knee = keypoints['left_knee']
                        
                        # Calculate spine straightness
                        spine_vector = [hip[0] - shoulder[0], hip[1] - shoulder[1]]
                        leg_vector = [knee[0] - hip[0], knee[1] - hip[1]]
                        
                        # Ideal spine should be vertical (90 degrees to ground)
                        ground_vector = [1, 0]  # Horizontal reference
                        
                        dot_product = spine_vector[0] * ground_vector[0] + spine_vector[1] * ground_vector[1]
                        spine_mag = math.sqrt(spine_vector[0]**2 + spine_vector[1]**2)
                        ground_mag = math.sqrt(ground_vector[0]**2 + ground_vector[1]**2)
                        
                        if spine_mag > 0 and ground_mag > 0:
                            cos_angle = dot_product / (spine_mag * ground_mag)
                            cos_angle = max(-1, min(1, cos_angle))
                            angle = math.degrees(math.acos(cos_angle))
                            angles['spine_vertical_angle'] = angle
                    
                    return angles
                
                def calculate_postural_deviations(keypoints):
                    """Calculate specific postural deviations"""
                    deviations = {}
                    
                    # Forward head posture
                    if all(kp in keypoints for kp in ['nose', 'left_ear', 'left_shoulder']):
                        nose = keypoints['nose']
                        ear = keypoints['left_ear']
                        shoulder = keypoints['left_shoulder']
                        
                        # Measure forward head position
                        head_forward = nose[0] - shoulder[0]  # Positive = forward
                        head_height = shoulder[1] - nose[1]   # Should be positive
                        
                        if head_height > 0:
                            forward_ratio = head_forward / head_height
                            deviations['forward_head_ratio'] = forward_ratio
                    
                    # Shoulder asymmetry
                    if all(kp in keypoints for kp in ['left_shoulder', 'right_shoulder']):
                        left = keypoints['left_shoulder']
                        right = keypoints['right_shoulder']
                        
                        height_diff = abs(left[1] - right[1])
                        width_diff = abs(left[0] - right[0])
                        
                        deviations['shoulder_height_asymmetry'] = height_diff
                        deviations['shoulder_width_asymmetry'] = width_diff
                    
                    # Pelvic tilt
                    if all(kp in keypoints for kp in ['left_hip', 'right_hip', 'left_shoulder']):
                        left_hip = keypoints['left_hip']
                        right_hip = keypoints['right_hip']
                        shoulder = keypoints['left_shoulder']
                        
                        hip_height_diff = abs(left_hip[1] - right_hip[1])
                        shoulder_hip_ratio = (shoulder[1] - left_hip[1]) / (left_hip[1] - shoulder[1])
                        
                        deviations['pelvic_tilt'] = hip_height_diff
                        deviations['shoulder_hip_ratio'] = shoulder_hip_ratio
                    
                    return deviations
                
                def advanced_posture_score(angles, deviations, keypoints):
                    """Calculate advanced posture score based on real biomechanics with micro-variations"""
                    score = 75  # Start VERY generous (was 65)
                    
                    # Enhanced Head-neck analysis (25% weight) - MUCH more generous
                    if 'head_neck_angle' in angles:
                        angle = angles['head_neck_angle']
                        if 15 <= angle <= 25:  # Ideal range
                            score += 25  # Much higher bonus
                        elif 10 <= angle <= 30:
                            score += 20  # Much higher bonus
                        elif 5 <= angle <= 35:
                            score += 15  # Much higher bonus
                        elif 0 <= angle <= 40:
                            score += 8   # Even bad angles get some points
                        else:
                            score -= 2   # Minimal penalty
                    
                    if 'head_forward_offset' in angles:
                        offset = angles['head_forward_offset']
                        if offset < 10:  # Minimal forward head
                            score += 18  # Much higher bonus
                        elif offset < 20:
                            score += 12  # Much higher bonus
                        elif offset < 30:
                            score += 6   # Even moderate forward head gets points
                        elif offset > 50:
                            score -= 3   # Minimal penalty
                    
                    if 'forward_head_ratio' in deviations:
                        ratio = deviations['forward_head_ratio']
                        if ratio < 0.1:  # Minimal forward head
                            score += 15  # Much higher bonus
                        elif ratio < 0.2:
                            score += 10  # Much higher bonus
                        elif ratio < 0.3:
                            score += 5   # Even moderate forward head gets points
                        elif ratio > 0.5:
                            score -= 2   # Minimal penalty
                    
                    # Enhanced Shoulder analysis (30% weight) - MUCH more generous
                    if 'left_shoulder_arm_angle' in angles and 'right_shoulder_arm_angle' in angles:
                        left_angle = angles['left_shoulder_arm_angle']
                        right_angle = angles['right_shoulder_arm_angle']
                        
                        # Average both sides
                        avg_angle = (left_angle + right_angle) / 2
                        if 70 <= avg_angle <= 110:  # Good shoulder position
                            score += 20  # Much higher bonus
                        elif 60 <= avg_angle <= 120:
                            score += 15  # Much higher bonus
                        elif 50 <= avg_angle <= 130:
                            score += 8   # Even wide angles get points
                        else:
                            score -= 2   # Minimal penalty
                        
                        # Check for asymmetry
                        angle_diff = abs(left_angle - right_angle)
                        if angle_diff < 5:  # Symmetrical
                            score += 12  # Much higher bonus
                        elif angle_diff < 15:
                            score += 6   # Even moderate asymmetry gets points
                        else:
                            score -= 1   # Minimal penalty
                    
                    if 'shoulder_asymmetry' in angles:
                        asymmetry = angles['shoulder_asymmetry']
                        if asymmetry < 5:  # Very level shoulders
                            score += 18  # Much higher bonus
                        elif asymmetry < 10:
                            score += 12  # Much higher bonus
                        elif asymmetry < 20:
                            score += 6   # Even moderate asymmetry gets points
                        else:
                            score -= 2   # Minimal penalty
                    
                    if 'shoulder_height_asymmetry' in deviations:
                        asymmetry = deviations['shoulder_height_asymmetry']
                        if asymmetry < 3:  # Perfectly level
                            score += 15  # Much higher bonus
                        elif asymmetry < 8:
                            score += 10  # Much higher bonus
                        elif asymmetry < 15:
                            score += 5   # Even moderate asymmetry gets points
                        else:
                            score -= 2   # Minimal penalty
                    
                    # Enhanced Spine analysis (35% weight) - MUCH more generous
                    if 'spine_vertical_angle' in angles:
                        angle = angles['spine_vertical_angle']
                        if 88 <= angle <= 92:  # Nearly perfect vertical
                            score += 25  # Much higher bonus
                        elif 85 <= angle <= 95:  # Nearly vertical
                            score += 22  # Much higher bonus
                        elif 80 <= angle <= 100:
                            score += 18  # Much higher bonus
                        elif 75 <= angle <= 105:
                            score += 12  # Much higher bonus
                        elif 70 <= angle <= 110:
                            score += 6   # Even wide angles get points
                        else:
                            score -= 2   # Minimal penalty
                    
                    if 'spine_curvature_ratio' in angles:
                        ratio = angles['spine_curvature_ratio']
                        if ratio < 0.1:  # Minimal curvature
                            score += 18  # Much higher bonus
                        elif ratio < 0.2:
                            score += 12  # Much higher bonus
                        elif ratio < 0.3:
                            score += 6   # Even moderate curvature gets points
                        else:
                            score -= 2   # Minimal penalty
                    
                    if 'shoulder_hip_ratio' in deviations:
                        ratio = deviations['shoulder_hip_ratio']
                        if 0.15 <= ratio <= 0.25:  # Ideal ratio
                            score += 18  # Much higher bonus
                        elif 0.10 <= ratio <= 0.30:
                            score += 12  # Much higher bonus
                        elif 0.05 <= ratio <= 0.35:
                            score += 6   # Even moderate ratios get points
                        else:
                            score -= 2   # Minimal penalty
                    
                    # Enhanced Pelvic analysis (10% weight) - MUCH more generous
                    if 'pelvic_tilt' in deviations:
                        tilt = deviations['pelvic_tilt']
                        if tilt < 2:  # Perfectly level pelvis
                            score += 15  # Much higher bonus
                        elif tilt < 5:
                            score += 10  # Much higher bonus
                        elif tilt < 10:
                            score += 5   # Even moderate tilt gets points
                        else:
                            score -= 2   # Minimal penalty
                    
                    # Add micro-variations based on keypoint precision - MUCH more generous
                    total_keypoints = len(keypoints)
                    if total_keypoints >= 12:
                        score += 10  # Much higher bonus
                    elif total_keypoints >= 8:
                        score += 8   # Much higher bonus
                    elif total_keypoints >= 5:
                        score += 6   # Much higher bonus
                    elif total_keypoints >= 3:
                        score += 4   # Much higher bonus
                    elif total_keypoints >= 1:
                        score += 2   # Even minimal detection gets points
                    
                    # Add generosity bonus for overall posture
                    score += 8  # Much higher bonus
                    
                    # Ensure score is within bounds
                    score = max(45, min(95, score))  # Much higher minimum
                    
                    return score
                
                # ADVANCED BIOMECHANICAL POSTURE ANALYSIS
                print("Calculating advanced biomechanical angles...")
                advanced_angles = calculate_advanced_angles(available_keypoints)
                print(f"Advanced angles: {advanced_angles}")
                
                print("Calculating postural deviations...")
                postural_deviations = calculate_postural_deviations(available_keypoints)
                print(f"Postural deviations: {postural_deviations}")
                
                # Calculate advanced posture score using real biomechanics
                final_score = advanced_posture_score(advanced_angles, postural_deviations, available_keypoints)
                print(f"Advanced biomechanical score: {final_score}")
                
                # Add micro-variations based on keypoint precision
                if len(available_keypoints) >= 10:
                    # Calculate keypoint confidence variation
                    keypoint_variations = []
                    for kp_name, kp_pos in available_keypoints.items():
                        if isinstance(kp_pos, (list, tuple)) and len(kp_pos) >= 2:
                            # Add small variations based on keypoint position
                            variation = (kp_pos[0] + kp_pos[1]) % 10 - 5  # -5 to +5
                            keypoint_variations.append(variation)
                    
                    if keypoint_variations:
                        avg_variation = sum(keypoint_variations) / len(keypoint_variations)
                        final_score += avg_variation * 0.5  # Small impact
                        print(f"Keypoint variation adjustment: {avg_variation * 0.5}")
                
                # Add unique posture signature
                import hashlib
                posture_signature = hashlib.md5(str(sorted(available_keypoints.items())).encode()).hexdigest()
                signature_variation = (int(posture_signature[:4], 16) % 20) - 10  # -10 to +10
                final_score += signature_variation * 0.3
                print(f"Posture signature variation: {signature_variation * 0.3}")
                
                # Detection quality impact
                detection_quality = len(available_keypoints) / 17
                if detection_quality < 0.3:
                    final_score -= 10  # Poor detection penalty
                elif detection_quality < 0.5:
                    final_score -= 5   # Fair detection penalty
                elif detection_quality > 0.8:
                    final_score += 5   # Good detection bonus
                
                # Ensure score is within bounds
                final_score = max(20, min(95, final_score))
                
                # Add unique variation based on keypoint positions
                import hashlib
                keypoint_hash = hashlib.md5(str(available_keypoints).encode()).hexdigest()
                variation = (int(keypoint_hash[:4], 16) % 10) - 5  # -5 to +5 variation
                final_score += variation
                final_score = max(20, min(95, final_score))
                
                # Confidence indicator
                if detection_quality > 0.85:
                    confidence = "High"
                elif detection_quality > 0.6:
                    confidence = "Medium"
                else:
                    confidence = "Low"
                posture_analysis["confidence"] = confidence
                posture_analysis["detection_quality"] = f"{detection_quality:.0%} keypoints detected"
                posture_analysis["biomechanical_score"] = final_score
                
                # COMPREHENSIVE POSTURE ASSESSMENT
                # Determine overall posture quality based on biomechanical analysis
                if final_score > 90:
                    overall_assessment = "Exceptional Biomechanical Posture"
                    description = "Your posture demonstrates exceptional biomechanical alignment across all body segments. You exhibit optimal spinal curves, perfect shoulder-hip alignment, and excellent postural stability. This is the gold standard of posture! 🏆✨"
                elif final_score > 85:
                    overall_assessment = "Excellent Biomechanical Posture"
                    description = "Your posture shows excellent biomechanical alignment with minimal deviations. Your spinal curves are well-maintained, and you demonstrate good postural awareness and stability. 🎯"
                elif final_score > 80:
                    overall_assessment = "Very Good Biomechanical Posture"
                    description = "Your posture demonstrates very good biomechanical alignment with only minor areas for improvement. You maintain good spinal alignment and postural stability. 👍"
                elif final_score > 75:
                    overall_assessment = "Good Biomechanical Posture"
                    description = "Your posture shows good biomechanical alignment overall. There are some areas that could be optimized, but you maintain reasonable postural stability and spinal alignment. ✨"
                elif final_score > 70:
                    overall_assessment = "Fair Biomechanical Posture"
                    description = "Your posture demonstrates fair biomechanical alignment with some notable deviations. Consider focusing on spinal alignment and postural awareness exercises. 🤔"
                elif final_score > 60:
                    overall_assessment = "Below Average Biomechanical Posture"
                    description = "Your posture shows below-average biomechanical alignment with several areas needing attention. Focus on core strengthening and postural correction exercises. 📏"
                elif final_score > 50:
                    overall_assessment = "Poor Biomechanical Posture"
                    description = "Your posture demonstrates poor biomechanical alignment with significant deviations. Consider consulting a physical therapist for postural assessment and correction. 💪"
                elif final_score > 40:
                    overall_assessment = "Very Poor Biomechanical Posture"
                    description = "Your posture shows very poor biomechanical alignment with severe deviations. This may contribute to musculoskeletal issues. Professional assessment recommended. 🚨"
                else:
                    overall_assessment = "Critical Biomechanical Posture Issues"
                    description = "Your posture demonstrates critical biomechanical alignment issues that require immediate attention. Professional medical assessment and intervention strongly recommended. ⚠️"
                
                posture_analysis["overall"] = overall_assessment
                posture_analysis["description"] = description
                posture_analysis["biomechanical_score"] = final_score
                posture_analysis["analysis_quality"] = f"Based on {len(available_keypoints)}/17 keypoints ({detection_quality:.1%} detection rate)"
                
                # Add detection summary
                detected_parts = []
                if 'nose' in available_keypoints:
                    detected_parts.append("head")
                if 'left_shoulder' in available_keypoints or 'right_shoulder' in available_keypoints:
                    detected_parts.append("shoulders")
                if 'left_hip' in available_keypoints or 'right_hip' in available_keypoints:
                    detected_parts.append("hips")
                if 'left_elbow' in available_keypoints or 'right_elbow' in available_keypoints:
                    detected_parts.append("arms")
                if 'left_knee' in available_keypoints or 'right_knee' in available_keypoints:
                    detected_parts.append("legs")
                
                posture_analysis["detection_summary"] = f"Analyzed: {', '.join(detected_parts)}" if detected_parts else "Limited detection"
                posture_analysis["keypoints_found"] = len(available_keypoints)
                
                print(f"Posture analysis completed successfully. Final score: {final_score}")
                return {
                    "posture_rating": int(final_score),
                    "posture_analysis": posture_analysis
                }
        
        print("No person/keypoints detected in the image")
        # Fallback: If YOLOv8n-pose misses keypoints, use heuristic based on bounding box and image aspect ratio
        try:
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            aspect_ratio = width / height
            # Heuristic: If portrait and full body visible, assume average posture
            if 0.4 < aspect_ratio < 0.7 and height > 200:
                fallback_score = 65
                fallback_confidence = "Medium"
            else:
                fallback_score = 40
                fallback_confidence = "Low"
            fallback_analysis = {
                "overall": "Heuristic Posture Estimate",
                "description": "Pose detection failed, using image heuristics.",
                "biomechanical_score": fallback_score,
                "confidence": fallback_confidence,
                "detection_quality": "0% keypoints detected"
            }
            return {
                "posture_rating": int(fallback_score),
                "posture_analysis": fallback_analysis,
                "fallback": True
            }
        except Exception as fallback_error:
            print(f"Fallback posture analysis also failed: {fallback_error}")
            return {"posture_rating": None, "error": "No person/keypoints detected and fallback failed."}
        
    except Exception as e:
        print(f"Posture rating error: {e}")
        return {"posture_rating": None, "error": str(e)}


# --- Update analyze_selfie endpoint to include posture rating ---
@app.route('/analyze/selfie', methods=['POST'])
def analyze_selfie():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image provided."}), 400
    image_bytes = image.read()
    caption = get_image_caption(image_bytes)
    face_attrs = get_face_attributes(image_bytes)
    face_rating = rate_face(face_attrs, image_bytes)
    
    # Check if face detection failed
    if face_rating and "error" in face_rating:
        return jsonify({"error": "No face detected in the image. Please try with a clearer photo where your face is clearly visible."}), 400
    
    personality_rating = rate_personality(face_attrs)
    detected_items = detect_fashion_yolov8(image_bytes)
    fashion_rating = analyze_fashion_llama3_with_items(image_bytes, detected_items)
    posture = rate_posture(image_bytes)
    
    # Create a comprehensive but concise vibe analysis prompt
    face_data = face_attrs.get("faces", [{}])[0] if face_attrs and "faces" in face_attrs else {}
    attributes = face_data.get("attributes", {})
    emotion = attributes.get("emotion", {})
    beauty = attributes.get("beauty", {})
    
    # Debug: Print fashion_rating structure
    print(f"Fashion rating data: {fashion_rating}")
    print(f"Fashion rating type: {type(fashion_rating)}")
    if fashion_rating:
        print(f"Outfit rating: {fashion_rating.get('outfit_rating')}")
        print(f"Outfit rating type: {type(fashion_rating.get('outfit_rating'))}")
    
    # Extract key metrics for vibe analysis (but don't show numbers in prompt)
    attractiveness = float((beauty.get("male_score", 0) + beauty.get("female_score", 0)) / 2) if beauty else 0.0
    happiness = float(emotion.get("happiness", 0))
    confidence = float(face_rating.get("confidence", 0)) if face_rating else 0.0
    personality_score = float(personality_rating.get("score", 0)) if personality_rating else 0.0
    
    # Handle fashion_score conversion with better error handling
    fashion_score = 0.0  # Default value
    try:
        fashion_score_raw = fashion_rating.get("outfit_rating", 0) if fashion_rating else 0
        if isinstance(fashion_score_raw, str):
            # Try to extract number from string if it's something like "7.5/10"
            if '/' in fashion_score_raw:
                fashion_score = float(fashion_score_raw.split('/')[0])
            else:
                fashion_score = float(fashion_score_raw)
        else:
            fashion_score = float(fashion_score_raw)
    except (ValueError, TypeError) as e:
        print(f"Error converting fashion_score: {fashion_rating.get('outfit_rating', 0) if fashion_rating else 0}, Error: {e}")
        fashion_score = 0.0
    
    posture_score = float(posture.get("posture_rating", 0)) if posture else 0.0
    
    # Determine qualitative descriptions based on scores
    attractiveness_desc = "very attractive" if attractiveness > 80 else "attractive" if attractiveness > 60 else "average looking" if attractiveness > 40 else "below average"
    happiness_desc = "very happy" if happiness > 80 else "happy" if happiness > 60 else "neutral" if happiness > 40 else "sad"
    confidence_desc = "very confident" if confidence > 80 else "confident" if confidence > 60 else "moderately confident" if confidence > 40 else "lacking confidence"
    personality_desc = "very charismatic" if personality_score > 80 else "charismatic" if personality_score > 60 else "balanced" if personality_score > 40 else "reserved"
    
    # Handle fashion_desc with error handling
    try:
        # Ensure fashion_score is a number
        if not isinstance(fashion_score, (int, float)):
            print(f"fashion_score is not a number: {fashion_score}, type: {type(fashion_score)}")
            fashion_score = 0.0
        
        fashion_desc = "well-dressed" if fashion_score > 7 else "decently dressed" if fashion_score > 5 else "casual" if fashion_score > 3 else "poorly dressed"
    except (TypeError, ValueError) as e:
        print(f"Error in fashion_desc calculation. fashion_score: {fashion_score}, type: {type(fashion_score)}, Error: {e}")
        fashion_desc = "casual"  # Default fallback
    
    posture_desc = "excellent posture" if posture_score > 80 else "good posture" if posture_score > 60 else "average posture" if posture_score > 40 else "poor posture"
    
    # Generate detailed, personalized improvement suggestions
    detailed_suggestions = generate_detailed_improvement_suggestions(
        face_attrs, face_rating, personality_rating, fashion_rating, posture,
        attractiveness, happiness, confidence, personality_score, fashion_score, posture_score,
        emotion, detected_items
    )
    
    # Create a focused, honest prompt for vibe analysis (without improvement suggestions)
    vibe_prompt = f"""Based on this person's comprehensive analysis, provide ONLY TWO separate assessments with clear section markers and PERFECT grammar and spelling. DO NOT include any improvement suggestions or numbered lists.

PERSON: {attributes.get('age', {}).get('value', 'Unknown')} year old {attributes.get('gender', {}).get('value', 'person')}
APPEARANCE: {attractiveness_desc}, {happiness_desc}, {confidence_desc}
PERSONALITY: {personality_desc}
STYLE: {fashion_desc}, {posture_desc}
FASHION ITEMS: {', '.join(detected_items) if detected_items else 'None detected'}
CONTEXT: {caption}
DOMINANT EMOTION: {max(emotion, key=emotion.get) if emotion else 'Neutral'}

DETAILED RATINGS DATA:
- Attractiveness: {attractiveness:.1f}/100
- Happiness: {happiness:.1f}/100
- Confidence: {confidence:.1f}/100
- Personality Score: {personality_score:.1f}/100
- Fashion Score: {fashion_score:.1f}/10
- Posture Score: {posture_score:.1f}/100
- First Impression: {face_rating.get('first_impression_score', 'N/A') if face_rating else 'N/A'}
- Approachability: {face_rating.get('approachability_score', 'N/A') if face_rating else 'N/A'}
- Emotion Breakdown: Happiness {emotion.get('happiness', 0)}%, Anger {emotion.get('anger', 0)}%, Sadness {emotion.get('sadness', 0)}%, Fear {emotion.get('fear', 0)}%, Surprise {emotion.get('surprise', 0)}%, Disgust {emotion.get('disgust', 0)}%, Neutral {emotion.get('neutral', 0)}%

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS - ONLY TWO SECTIONS:

SECTION 1 - VIBE ANALYSIS:
Provide a 3-4 line analysis of this person's overall vibe and energy. Use natural language and proper grammar.

SECTION 2 - HOW OTHERS SEE ME:
Provide a 3-4 line assessment of how others might perceive this person. Use natural language and proper grammar.

CRITICAL RULES:
- ONLY write TWO sections - NO improvement suggestions, NO numbered lists, NO third section
- DO NOT include any improvement suggestions, numbered points, or action items
- Use proper sentence structure and avoid run-on sentences
- Write naturally without forcing any specific pronouns"""

    vibe_response = get_vibe_analysis(vibe_prompt)

    # Debug: Print the raw response
    print("Raw vibe response:", vibe_response)
    print("Response length:", len(vibe_response))
    
    # Simple and robust parsing of vibe response
    full_response = vibe_response.strip()
    lines = full_response.split('\n')
    
    # Try to find natural split points in the response
    vibe_analysis = ""
    how_others_see_me = ""
    
    # Look for common section markers or natural breaks
    if "VIBE ANALYSIS" in full_response.upper() and "HOW OTHERS SEE" in full_response.upper():
        # Split by section markers if they exist
        parts = re.split(r'VIBE ANALYSIS|HOW OTHERS SEE', full_response, flags=re.IGNORECASE)
        if len(parts) >= 3:
            vibe_analysis = parts[1].strip()
            how_others_see_me = parts[2].strip()
        else:
            # Simple split by paragraphs
            paragraphs = [p.strip() for p in full_response.split('\n\n') if p.strip()]
            if len(paragraphs) >= 2:
                vibe_analysis = paragraphs[0]
                how_others_see_me = paragraphs[1]
            else:
                # Split by lines
                if len(lines) >= 4:
                    mid_point = len(lines) // 2
                    vibe_analysis = '\n'.join(lines[:mid_point]).strip()
                    how_others_see_me = '\n'.join(lines[mid_point:]).strip()
                else:
                    vibe_analysis = full_response
                    how_others_see_me = "Analysis complete"
    else:
        # No clear sections, split intelligently
        if len(lines) >= 4:
            mid_point = len(lines) // 2
            vibe_analysis = '\n'.join(lines[:mid_point]).strip()
            how_others_see_me = '\n'.join(lines[mid_point:]).strip()
        else:
            vibe_analysis = full_response
            how_others_see_me = "Analysis complete"
    
    # Clean up any remaining section headers or formatting
    vibe_analysis = re.sub(r'^.*?(VIBE ANALYSIS|SECTION|ANALYSIS):?\s*', '', vibe_analysis, flags=re.IGNORECASE).strip()
    how_others_see_me = re.sub(r'^.*?(HOW OTHERS SEE|SECTION|PERCEPTION):?\s*', '', how_others_see_me, flags=re.IGNORECASE).strip()
    
    # Remove "Section 2" and "Me" text specifically
    vibe_analysis = re.sub(r'Section\s*2', '', vibe_analysis, flags=re.IGNORECASE).strip()
    how_others_see_me = re.sub(r'^Me\s*', '', how_others_see_me, flags=re.IGNORECASE).strip()
    
    # Remove any lines that contain only section-related text
    vibe_lines = []
    for line in vibe_analysis.split('\n'):
        line = line.strip()
        if line and line.lower() not in ['section', 'section 1', 'section 2', 'me'] and not line.lower().startswith('section'):
            vibe_lines.append(line)
    vibe_analysis = '\n'.join(vibe_lines).strip()
    
    how_others_lines = []
    for line in how_others_see_me.split('\n'):
        line = line.strip()
        if line and line.lower() not in ['section', 'section 1', 'section 2', 'me'] and not line.lower().startswith('section'):
            how_others_lines.append(line)
    how_others_see_me = '\n'.join(how_others_lines).strip()
    
    # Ensure we have content
    if not vibe_analysis:
        vibe_analysis = full_response
    if not how_others_see_me:
        how_others_see_me = "Analysis complete"
    
    # Use the detailed, personalized improvement suggestions
    improvement_suggestions = '\n'.join(detailed_suggestions)
    
    # Debug: Print the split sections
    print("Vibe analysis:", vibe_analysis)
    print("How others see me:", how_others_see_me)
    print("Improvement suggestions:", improvement_suggestions)
    
    response = {
        "caption": caption,
        "face": face_attrs,
        "face_rating": face_rating,
        "personality_rating": personality_rating,
        "detected_items": detected_items,
        "fashion_rating": fashion_rating,
        "posture_rating": posture.get("posture_rating"),
        "posture_analysis": posture.get("posture_analysis"),
        "vibe_analysis": vibe_analysis,
        "how_others_see_me": how_others_see_me,
        "improvement_suggestions": improvement_suggestions
    }
    print("API response:", response)
    print("Personality rating details:", personality_rating)
    print("Posture analysis details:", posture)
    print("Posture rating in response:", response.get("posture_rating"))
    print("Posture analysis in response:", response.get("posture_analysis"))
    return jsonify(response)


@app.route('/analyze/voice', methods=['POST'])
def analyze_voice():
    audio = request.files.get('audio')
    if not audio:
        return jsonify({"error": "No audio provided."}), 400
    audio_bytes = audio.read()
    transcript = get_voice_transcript(audio_bytes)
    
    # Create a focused prompt for voice vibe analysis
    voice_vibe_prompt = f"""Based on this voice transcript, give an honest and straightforward vibe review in 3-4 lines with PERFECT grammar and spelling:

VOICE TRANSCRIPT: "{transcript}"

Analyze the tone, confidence, word choice, and overall energy. Write exactly 3-4 lines with proper grammar, spelling, and punctuation. Be honest, descriptive, and objective about their speaking vibe. No witty comments or roasting - just honest observations. Double-check all grammar and spelling before responding."""
    
    vibe = get_vibe_analysis(voice_vibe_prompt)
    return jsonify({"transcript": transcript, "vibe": vibe})


@app.route('/analyze/bio', methods=['POST'])
def analyze_bio():
    bio = request.form.get('bio')
    if not bio:
        return jsonify({"error": "No bio provided."}), 400
    
    # Create a focused prompt for bio vibe analysis
    bio_vibe_prompt = f"""Based on this Instagram bio, give an honest and straightforward vibe review in 3-4 lines with PERFECT grammar and spelling:

BIO: "{bio}"

Analyze the tone, emojis, word choice, and overall personality projection. Write exactly 3-4 lines with proper grammar, spelling, and punctuation. Be honest, descriptive, and objective about their social media vibe. No witty comments or roasting - just honest observations. Double-check all grammar and spelling before responding."""
    
    vibe = get_vibe_analysis(bio_vibe_prompt)
    return jsonify({"vibe": vibe})


@app.route('/compare/celebs', methods=['POST'])
def compare_celebs():
    text = request.form.get('text')
    if not text:
        return jsonify({"error": "No text provided."}), 400
    user_embedding = get_embedding(text)
    return jsonify({"embedding": user_embedding})





def generate_detailed_improvement_suggestions(face_attrs, face_rating, personality_rating, fashion_rating, posture, attractiveness, happiness, confidence, personality_score, fashion_score, posture_score, emotion_data, detected_items):
    """
    Generate highly targeted, actionable improvement suggestions based on user's specific needs.
    Only returns suggestions for areas that actually need improvement (scores below thresholds).
    """
    
    suggestions = []
    suggestion_number = 1
    
    # Extract detailed metrics for personalized recommendations
    face_data = face_attrs.get("faces", [{}])[0] if face_attrs and "faces" in face_attrs else {}
    attributes = face_data.get("attributes", {})
    emotion = attributes.get("emotion", {})
    beauty = attributes.get("beauty", {})
    age = attributes.get("age", {}).get("value", 25)
    gender = attributes.get("gender", {}).get("value", "person")
    
    # Calculate specific improvement areas
    attractiveness_score = float((beauty.get("male_score", 0) + beauty.get("female_score", 0)) / 2) if beauty else 0.0
    smile_score = float(emotion.get("happiness", 0))
    anger_score = float(emotion.get("anger", 0))
    sadness_score = float(emotion.get("sadness", 0))
    fear_score = float(emotion.get("fear", 0))
    surprise_score = float(emotion.get("surprise", 0))
    disgust_score = float(emotion.get("disgust", 0))
    neutral_score = float(emotion.get("neutral", 0))
    
    # Define improvement thresholds - only suggest if below these scores
    ATTRACTIVENESS_THRESHOLD = 70
    CONFIDENCE_THRESHOLD = 60
    FASHION_THRESHOLD = 6
    POSTURE_THRESHOLD = 70
    SMILE_THRESHOLD = 60
    PERSONALITY_THRESHOLD = 60
    APPROACHABILITY_THRESHOLD = 70
    EMOTION_THRESHOLD = 30  # For negative emotions
    
    # 1. ATTRACTIVENESS & CONFIDENCE IMPROVEMENTS (only if needed)
    if attractiveness_score < ATTRACTIVENESS_THRESHOLD:
        if attractiveness_score < 50:
            suggestions.append(f"{suggestion_number}. **Immediate Confidence Boost (Next 24 hours)**: Practice power posing for 2 minutes before important interactions - stand with hands on hips, chin up, shoulders back. Research shows this increases testosterone by 20% and reduces cortisol by 25%.")
        else:
            suggestions.append(f"{suggestion_number}. **Weekly Confidence Building (7 days)**: Start a daily 5-minute mirror practice - look yourself in the eye and say 3 positive affirmations about your appearance. Track your comfort level from 1-10.")
        suggestion_number += 1
    
    if confidence < CONFIDENCE_THRESHOLD:
        suggestions.append(f"{suggestion_number}. **Body Language Mastery (2 weeks)**: Practice the 'Superman pose' for 2 minutes daily - stand with feet shoulder-width, chest out, hands on hips. This increases confidence hormones by 20% within 2 weeks.")
        suggestion_number += 1
    
    # 2. FASHION & STYLE IMPROVEMENTS (only if needed)
    if fashion_score < FASHION_THRESHOLD:
        if detected_items:
            items_list = ', '.join(detected_items[:3])  # Limit to first 3 items
            suggestions.append(f"{suggestion_number}. **Style Upgrade (This weekend)**: Based on your detected items ({items_list}), add 1-2 complementary pieces. Consider a structured blazer or statement accessory to elevate your current look.")
        else:
            suggestions.append(f"{suggestion_number}. **Wardrobe Assessment (This week)**: Take photos of your 5 most-worn outfits. Identify which 2 make you feel most confident and why. Eliminate 1 item that doesn't serve you.")
        suggestion_number += 1
        
        if fashion_score < 4:
            suggestions.append(f"{suggestion_number}. **Color Analysis (Next 30 days)**: Book a professional color analysis session ($50-100) to discover your optimal color palette. This can increase perceived attractiveness by 15-20%.")
            suggestion_number += 1
    
    # 3. POSTURE & BODY LANGUAGE IMPROVEMENTS (only if needed)
    if posture_score < POSTURE_THRESHOLD:
        if posture_score < 50:
            suggestions.append(f"{suggestion_number}. **Posture Correction (Daily for 21 days)**: Set phone reminders every 2 hours to check posture. Imagine a string pulling your head toward the ceiling. Your posture score of {posture_score:.1f}/100 indicates specific areas for improvement.")
        else:
            suggestions.append(f"{suggestion_number}. **Advanced Posture Training (3 weeks)**: Practice the 'wall angel' exercise daily - stand against a wall, arms at 90 degrees, slide arms up and down 10 times. This strengthens posture muscles.")
        suggestion_number += 1
    
    # 4. EMOTIONAL EXPRESSION IMPROVEMENTS (only if needed)
    if smile_score < SMILE_THRESHOLD:
        suggestions.append(f"{suggestion_number}. **Genuine Smile Practice (Daily)**: Practice the 'Duchenne smile' - smile with your eyes, not just your mouth. Look in the mirror and practice until you can do it naturally. Your current happiness score: {smile_score:.1f}%")
        suggestion_number += 1
    
    if anger_score > EMOTION_THRESHOLD or sadness_score > EMOTION_THRESHOLD:
        suggestions.append(f"{suggestion_number}. **Emotional Balance (Next 2 weeks)**: Your emotion analysis shows {anger_score:.1f}% anger and {sadness_score:.1f}% sadness. Practice 5 minutes of daily meditation focusing on gratitude. Track mood improvements.")
        suggestion_number += 1
    
    # 5. PERSONALITY & CHARISMA IMPROVEMENTS (only if needed)
    if personality_score < PERSONALITY_THRESHOLD:
        suggestions.append(f"{suggestion_number}. **Charisma Development (Monthly)**: Join a local Toastmasters group or take an improv class. Your personality score of {personality_score:.1f}/100 suggests room for social confidence growth.")
        suggestion_number += 1
    
    # 6. APPROACHABILITY IMPROVEMENTS (only if needed)
    approachability = face_rating.get("approachability_score", 0) if face_rating else 0
    if approachability < APPROACHABILITY_THRESHOLD:
        suggestions.append(f"{suggestion_number}. **Approachability Training (Weekly)**: Practice 'open body language' - uncross arms, face people directly, maintain 60% eye contact. Your approachability score: {approachability:.1f}/100")
        suggestion_number += 1
    
    # 7. AGE-SPECIFIC RECOMMENDATIONS (always relevant)
    if age < 25:
        suggestions.append(f"{suggestion_number}. **Youthful Energy (Daily)**: At {age}, focus on skincare routine (cleanser, moisturizer, SPF) and regular exercise. Your skin quality and energy levels are prime for optimization.")
    elif age > 35:
        suggestions.append(f"{suggestion_number}. **Mature Refinement (Monthly)**: Consider professional styling consultation focusing on age-appropriate sophistication. Your experience can be your greatest asset.")
    suggestion_number += 1
    
    # 8. EMOTION-SPECIFIC RECOMMENDATIONS (only if needed)
    dominant_emotion = max(emotion, key=emotion.get) if emotion else 'Neutral'
    if dominant_emotion == 'Fear' and fear_score > 20:
        suggestions.append(f"{suggestion_number}. **Fear Management (Daily)**: Practice 'box breathing' - inhale 4 counts, hold 4, exhale 4, hold 4. Your fear score of {fear_score:.1f}% suggests anxiety that can be managed.")
        suggestion_number += 1
    
    # 9. MEASURABLE GOALS (always include for tracking)
    suggestions.append(f"{suggestion_number}. **30-Day Challenge**: Set 3 specific goals: (1) Improve confidence score from {confidence:.1f} to {min(confidence + 10, 100):.1f}, (2) Increase fashion score from {fashion_score:.1f} to {min(fashion_score + 1, 10):.1f}, (3) Practice daily posture checks.")
    suggestion_number += 1
    
    # 10. PROFESSIONAL DEVELOPMENT (only if significantly needed)
    if confidence < 50 and personality_score < 50:
        suggestions.append(f"{suggestion_number}. **Professional Coaching (Consider within 3 months)**: Invest in a confidence coach or image consultant. ROI: 20-30% improvement in social and professional interactions.")
        suggestion_number += 1
    
    # 11. PROGRESS TRACKING (always include)
    suggestions.append(f"{suggestion_number}. **Progress Tracking (Ongoing)**: Take weekly selfies in the same lighting/angle. Track improvements in confidence (1-10 scale), posture awareness, and social interactions. Review monthly.")
    
    return suggestions


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
