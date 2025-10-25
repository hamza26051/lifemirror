import os
import cv2
import numpy as np
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentInput, AgentOutput
import requests
from PIL import Image
import io

class FaceAgent(BaseAgent):
    """Advanced face analysis agent with DeepFace integration"""
    
    def __init__(self):
        super().__init__()
        self.deepface_available = self._check_deepface()
        
    def _check_deepface(self) -> bool:
        """Check if DeepFace is available"""
        try:
            import deepface
            return True
        except ImportError:
            self.logger.warning("DeepFace not available, using fallback analysis")
            return False
    
    def run(self, input: AgentInput) -> AgentOutput:
        """Analyze faces in the image with advanced attributes"""
        try:
            # Download and process image
            image_data = self._download_image(input.url)
            if image_data is None:
                return self._create_output(
                    success=False,
                    data={},
                    error="Failed to download image",
                    confidence=0.0
                )
            
            # Perform face analysis
            if self.deepface_available:
                analysis_result = self._analyze_with_deepface(image_data)
            else:
                analysis_result = self._analyze_with_opencv(image_data)
            
            return self._create_output(
                success=True,
                data=analysis_result,
                confidence=analysis_result.get('confidence', 0.5)
            )
            
        except Exception as e:
            return self._handle_error(e, input)
    
    def _download_image(self, url: str) -> np.ndarray:
        """Download image from URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Convert to PIL Image then to numpy array
            pil_image = Image.open(io.BytesIO(response.content))
            image_array = np.array(pil_image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            return image_array
            
        except Exception as e:
            self.logger.error(f"Failed to download image: {e}")
            return None
    
    def _analyze_with_deepface(self, image: np.ndarray) -> Dict[str, Any]:
        """Advanced analysis using DeepFace"""
        try:
            from deepface import DeepFace
            
            # Perform comprehensive analysis
            result = DeepFace.analyze(
                img_path=image,
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=False,
                silent=True
            )
            
            # Handle both single face and multiple faces
            if isinstance(result, list):
                faces_data = result
            else:
                faces_data = [result]
            
            faces = []
            total_confidence = 0
            
            for i, face_data in enumerate(faces_data):
                face_info = {
                    'face_id': i,
                    'age': face_data.get('age', 25),
                    'gender': face_data.get('dominant_gender', 'unknown'),
                    'gender_confidence': face_data.get('gender', {}).get(face_data.get('dominant_gender', ''), 0),
                    'emotion': face_data.get('dominant_emotion', 'neutral'),
                    'emotion_confidence': face_data.get('emotion', {}).get(face_data.get('dominant_emotion', ''), 0),
                    'race': face_data.get('dominant_race', 'unknown'),
                    'race_confidence': face_data.get('race', {}).get(face_data.get('dominant_race', ''), 0),
                    'region': face_data.get('region', {})
                }
                
                # Calculate face-specific confidence
                face_confidence = self._calculate_face_confidence(face_info)
                face_info['confidence'] = face_confidence
                total_confidence += face_confidence
                
                faces.append(face_info)
            
            # Calculate overall confidence
            overall_confidence = total_confidence / len(faces) if faces else 0.0
            
            # Generate attractiveness score (simplified algorithm)
            attractiveness_score = self._calculate_attractiveness(faces)
            
            return {
                'num_faces': len(faces),
                'faces': faces,
                'confidence': overall_confidence,
                'attractiveness_score': attractiveness_score,
                'analysis_method': 'deepface',
                'features_detected': ['age', 'gender', 'emotion', 'race']
            }
            
        except Exception as e:
            self.logger.error(f"DeepFace analysis failed: {e}")
            return self._analyze_with_opencv(image)
    
    def _analyze_with_opencv(self, image: np.ndarray) -> Dict[str, Any]:
        """Fallback analysis using OpenCV"""
        try:
            # Load OpenCV face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces_rects = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            faces = []
            for i, (x, y, w, h) in enumerate(faces_rects):
                face_info = {
                    'face_id': i,
                    'age': 25,  # Default estimate
                    'gender': 'unknown',
                    'gender_confidence': 0.5,
                    'emotion': 'neutral',
                    'emotion_confidence': 0.5,
                    'race': 'unknown',
                    'race_confidence': 0.5,
                    'region': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                    'confidence': 0.6  # Basic detection confidence
                }
                faces.append(face_info)
            
            overall_confidence = 0.6 if faces else 0.0
            attractiveness_score = 0.5  # Neutral score for basic detection
            
            return {
                'num_faces': len(faces),
                'faces': faces,
                'confidence': overall_confidence,
                'attractiveness_score': attractiveness_score,
                'analysis_method': 'opencv_fallback',
                'features_detected': ['basic_detection']
            }
            
        except Exception as e:
            self.logger.error(f"OpenCV analysis failed: {e}")
            return {
                'num_faces': 0,
                'faces': [],
                'confidence': 0.0,
                'attractiveness_score': 0.0,
                'analysis_method': 'failed',
                'features_detected': [],
                'error': str(e)
            }
    
    def _calculate_face_confidence(self, face_info: Dict[str, Any]) -> float:
        """Calculate confidence score for a single face"""
        confidence_factors = []
        
        # Gender confidence
        if face_info.get('gender_confidence', 0) > 0.7:
            confidence_factors.append(0.9)
        elif face_info.get('gender_confidence', 0) > 0.5:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Emotion confidence
        if face_info.get('emotion_confidence', 0) > 0.7:
            confidence_factors.append(0.9)
        elif face_info.get('emotion_confidence', 0) > 0.5:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Age reasonableness (ages between 10-80 are more confident)
        age = face_info.get('age', 25)
        if 15 <= age <= 70:
            confidence_factors.append(0.9)
        elif 10 <= age <= 80:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _calculate_attractiveness(self, faces: List[Dict[str, Any]]) -> float:
        """Calculate attractiveness score based on facial features"""
        if not faces:
            return 0.0
        
        # Simplified attractiveness algorithm
        # In a real implementation, this would use more sophisticated models
        
        total_score = 0
        for face in faces:
            score = 0.5  # Base score
            
            # Positive emotions boost attractiveness
            emotion = face.get('emotion', 'neutral')
            if emotion in ['happy', 'surprise']:
                score += 0.2
            elif emotion == 'neutral':
                score += 0.1
            
            # Age factor (peak attractiveness in certain age ranges)
            age = face.get('age', 25)
            if 20 <= age <= 35:
                score += 0.2
            elif 18 <= age <= 45:
                score += 0.1
            
            # Confidence in detection affects score
            confidence = face.get('confidence', 0.5)
            score = score * confidence
            
            total_score += min(score, 1.0)  # Cap at 1.0
        
        return total_score / len(faces)
