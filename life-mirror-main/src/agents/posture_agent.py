import os
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_agent import BaseAgent, AgentInput, AgentOutput
import requests
from PIL import Image
import io
import math

class PostureAgent(BaseAgent):
    """Advanced posture and body alignment analysis agent"""
    
    def __init__(self):
        super().__init__()
        self.mediapipe_available = self._check_mediapipe()
        self.pose_landmarks = self._get_pose_landmarks()
        
    def _check_mediapipe(self) -> bool:
        """Check if MediaPipe is available"""
        try:
            import mediapipe as mp
            return True
        except ImportError:
            self.logger.warning("MediaPipe not available, using basic posture analysis")
            return False
    
    def _get_pose_landmarks(self) -> Dict[str, int]:
        """Get MediaPipe pose landmark indices"""
        return {
            'nose': 0,
            'left_eye': 1,
            'right_eye': 2,
            'left_ear': 7,
            'right_ear': 8,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
    
    def run(self, input: AgentInput) -> AgentOutput:
        """Analyze posture and body alignment in the image"""
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
            
            # Perform posture analysis
            if self.mediapipe_available:
                analysis_result = self._analyze_with_mediapipe(image_data)
            else:
                analysis_result = self._analyze_basic_posture(image_data)
            
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
            
            # Convert RGB to BGR for OpenCV if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # MediaPipe expects RGB, so we'll keep it as RGB
                pass
            
            return image_array
            
        except Exception as e:
            self.logger.error(f"Failed to download image: {e}")
            return None
    
    def _analyze_with_mediapipe(self, image: np.ndarray) -> Dict[str, Any]:
        """Advanced posture analysis using MediaPipe"""
        try:
            import mediapipe as mp
            
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            
            # Process image
            results = pose.process(image)
            
            if not results.pose_landmarks:
                return {
                    'pose_detected': False,
                    'landmarks': [],
                    'posture_score': 0.0,
                    'confidence': 0.0,
                    'analysis_method': 'mediapipe',
                    'issues': ['No pose detected'],
                    'recommendations': ['Ensure full body is visible in the image']
                }
            
            # Extract landmarks
            landmarks = self._extract_landmarks(results.pose_landmarks, image.shape)
            
            # Analyze posture
            posture_analysis = self._analyze_posture_alignment(landmarks)
            
            # Calculate overall posture score
            posture_score = self._calculate_posture_score(posture_analysis)
            
            # Generate recommendations
            recommendations = self._generate_posture_recommendations(posture_analysis)
            
            # Calculate confidence
            confidence = self._calculate_pose_confidence(landmarks)
            
            return {
                'pose_detected': True,
                'landmarks': landmarks,
                'posture_score': posture_score,
                'confidence': confidence,
                'analysis_method': 'mediapipe',
                'alignment_analysis': posture_analysis,
                'recommendations': recommendations,
                'overall_rating': self._get_posture_rating(posture_score)
            }
            
        except Exception as e:
            self.logger.error(f"MediaPipe posture analysis failed: {e}")
            return self._analyze_basic_posture(image)
    
    def _analyze_basic_posture(self, image: np.ndarray) -> Dict[str, Any]:
        """Basic posture analysis without MediaPipe"""
        try:
            # Basic analysis using image properties
            height, width = image.shape[:2]
            
            # Mock posture analysis for demonstration
            return {
                'pose_detected': True,
                'landmarks': [],
                'posture_score': 0.6,  # Neutral score
                'confidence': 0.4,  # Lower confidence for basic analysis
                'analysis_method': 'basic',
                'alignment_analysis': {
                    'head_alignment': 0.6,
                    'shoulder_alignment': 0.6,
                    'spine_alignment': 0.6,
                    'hip_alignment': 0.6
                },
                'recommendations': ['Use MediaPipe for detailed posture analysis'],
                'overall_rating': 'Fair'
            }
            
        except Exception as e:
            self.logger.error(f"Basic posture analysis failed: {e}")
            return {
                'pose_detected': False,
                'landmarks': [],
                'posture_score': 0.0,
                'confidence': 0.0,
                'analysis_method': 'failed',
                'alignment_analysis': {},
                'recommendations': [],
                'overall_rating': 'Unknown',
                'error': str(e)
            }
    
    def _extract_landmarks(self, pose_landmarks, image_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """Extract landmark coordinates from MediaPipe results"""
        landmarks = []
        height, width = image_shape[:2]
        
        for idx, landmark in enumerate(pose_landmarks.landmark):
            landmarks.append({
                'id': idx,
                'x': landmark.x * width,
                'y': landmark.y * height,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        return landmarks
    
    def _analyze_posture_alignment(self, landmarks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze various aspects of posture alignment"""
        if not landmarks:
            return {}
        
        analysis = {}
        
        # Head alignment (ear to shoulder alignment)
        analysis['head_alignment'] = self._analyze_head_alignment(landmarks)
        
        # Shoulder alignment (left vs right shoulder height)
        analysis['shoulder_alignment'] = self._analyze_shoulder_alignment(landmarks)
        
        # Spine alignment (overall spinal curve)
        analysis['spine_alignment'] = self._analyze_spine_alignment(landmarks)
        
        # Hip alignment (left vs right hip height)
        analysis['hip_alignment'] = self._analyze_hip_alignment(landmarks)
        
        # Forward head posture
        analysis['forward_head_posture'] = self._analyze_forward_head_posture(landmarks)
        
        # Shoulder position (rounded shoulders)
        analysis['shoulder_position'] = self._analyze_shoulder_position(landmarks)
        
        return analysis
    
    def _analyze_head_alignment(self, landmarks: List[Dict[str, Any]]) -> float:
        """Analyze head alignment relative to shoulders"""
        try:
            # Get relevant landmarks
            nose = self._get_landmark(landmarks, self.pose_landmarks['nose'])
            left_shoulder = self._get_landmark(landmarks, self.pose_landmarks['left_shoulder'])
            right_shoulder = self._get_landmark(landmarks, self.pose_landmarks['right_shoulder'])
            
            if not all([nose, left_shoulder, right_shoulder]):
                return 0.5
            
            # Calculate shoulder midpoint
            shoulder_mid_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            
            # Calculate head deviation from center
            head_deviation = abs(nose['x'] - shoulder_mid_x)
            
            # Normalize deviation (assuming max acceptable deviation is 10% of shoulder width)
            shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
            if shoulder_width > 0:
                normalized_deviation = head_deviation / (shoulder_width * 0.1)
                alignment_score = max(0, 1 - normalized_deviation)
            else:
                alignment_score = 0.5
            
            return min(alignment_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Head alignment analysis failed: {e}")
            return 0.5
    
    def _analyze_shoulder_alignment(self, landmarks: List[Dict[str, Any]]) -> float:
        """Analyze shoulder height alignment"""
        try:
            left_shoulder = self._get_landmark(landmarks, self.pose_landmarks['left_shoulder'])
            right_shoulder = self._get_landmark(landmarks, self.pose_landmarks['right_shoulder'])
            
            if not all([left_shoulder, right_shoulder]):
                return 0.5
            
            # Calculate height difference
            height_diff = abs(left_shoulder['y'] - right_shoulder['y'])
            
            # Normalize based on shoulder width
            shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
            if shoulder_width > 0:
                normalized_diff = height_diff / (shoulder_width * 0.1)  # 10% of width is acceptable
                alignment_score = max(0, 1 - normalized_diff)
            else:
                alignment_score = 0.5
            
            return min(alignment_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Shoulder alignment analysis failed: {e}")
            return 0.5
    
    def _analyze_spine_alignment(self, landmarks: List[Dict[str, Any]]) -> float:
        """Analyze overall spine alignment"""
        try:
            # Get key spine points
            nose = self._get_landmark(landmarks, self.pose_landmarks['nose'])
            left_shoulder = self._get_landmark(landmarks, self.pose_landmarks['left_shoulder'])
            right_shoulder = self._get_landmark(landmarks, self.pose_landmarks['right_shoulder'])
            left_hip = self._get_landmark(landmarks, self.pose_landmarks['left_hip'])
            right_hip = self._get_landmark(landmarks, self.pose_landmarks['right_hip'])
            
            if not all([nose, left_shoulder, right_shoulder, left_hip, right_hip]):
                return 0.5
            
            # Calculate midpoints
            shoulder_mid_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            shoulder_mid_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            hip_mid_x = (left_hip['x'] + right_hip['x']) / 2
            hip_mid_y = (left_hip['y'] + right_hip['y']) / 2
            
            # Calculate spine angle deviation from vertical
            if hip_mid_y != shoulder_mid_y:
                spine_angle = math.atan2(hip_mid_x - shoulder_mid_x, hip_mid_y - shoulder_mid_y)
                angle_degrees = abs(math.degrees(spine_angle))
                
                # Good posture should have spine close to vertical (0 degrees)
                # Allow up to 10 degrees deviation
                alignment_score = max(0, 1 - (angle_degrees / 10))
            else:
                alignment_score = 0.5
            
            return min(alignment_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Spine alignment analysis failed: {e}")
            return 0.5
    
    def _analyze_hip_alignment(self, landmarks: List[Dict[str, Any]]) -> float:
        """Analyze hip height alignment"""
        try:
            left_hip = self._get_landmark(landmarks, self.pose_landmarks['left_hip'])
            right_hip = self._get_landmark(landmarks, self.pose_landmarks['right_hip'])
            
            if not all([left_hip, right_hip]):
                return 0.5
            
            # Calculate height difference
            height_diff = abs(left_hip['y'] - right_hip['y'])
            
            # Normalize based on hip width
            hip_width = abs(left_hip['x'] - right_hip['x'])
            if hip_width > 0:
                normalized_diff = height_diff / (hip_width * 0.1)  # 10% of width is acceptable
                alignment_score = max(0, 1 - normalized_diff)
            else:
                alignment_score = 0.5
            
            return min(alignment_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Hip alignment analysis failed: {e}")
            return 0.5
    
    def _analyze_forward_head_posture(self, landmarks: List[Dict[str, Any]]) -> float:
        """Analyze forward head posture"""
        try:
            nose = self._get_landmark(landmarks, self.pose_landmarks['nose'])
            left_shoulder = self._get_landmark(landmarks, self.pose_landmarks['left_shoulder'])
            right_shoulder = self._get_landmark(landmarks, self.pose_landmarks['right_shoulder'])
            
            if not all([nose, left_shoulder, right_shoulder]):
                return 0.5
            
            # Calculate shoulder line
            shoulder_mid_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            
            # Check if head is forward of shoulder line
            head_forward_distance = nose['x'] - shoulder_mid_x
            
            # Normalize based on shoulder width
            shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
            if shoulder_width > 0:
                normalized_distance = abs(head_forward_distance) / (shoulder_width * 0.2)
                posture_score = max(0, 1 - normalized_distance)
            else:
                posture_score = 0.5
            
            return min(posture_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Forward head posture analysis failed: {e}")
            return 0.5
    
    def _analyze_shoulder_position(self, landmarks: List[Dict[str, Any]]) -> float:
        """Analyze shoulder position (rounded shoulders)"""
        try:
            left_shoulder = self._get_landmark(landmarks, self.pose_landmarks['left_shoulder'])
            right_shoulder = self._get_landmark(landmarks, self.pose_landmarks['right_shoulder'])
            left_elbow = self._get_landmark(landmarks, self.pose_landmarks['left_elbow'])
            right_elbow = self._get_landmark(landmarks, self.pose_landmarks['right_elbow'])
            
            if not all([left_shoulder, right_shoulder, left_elbow, right_elbow]):
                return 0.5
            
            # Calculate shoulder-elbow angles
            left_angle = self._calculate_angle(
                [left_shoulder['x'], left_shoulder['y']],
                [left_elbow['x'], left_elbow['y']],
                [left_shoulder['x'], left_shoulder['y'] + 100]  # Reference point below shoulder
            )
            
            right_angle = self._calculate_angle(
                [right_shoulder['x'], right_shoulder['y']],
                [right_elbow['x'], right_elbow['y']],
                [right_shoulder['x'], right_shoulder['y'] + 100]  # Reference point below shoulder
            )
            
            # Good shoulder position should have arms hanging naturally
            # Ideal angle is around 90 degrees
            avg_angle = (left_angle + right_angle) / 2
            angle_deviation = abs(avg_angle - 90)
            
            # Allow up to 30 degrees deviation
            position_score = max(0, 1 - (angle_deviation / 30))
            
            return min(position_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Shoulder position analysis failed: {e}")
            return 0.5
    
    def _get_landmark(self, landmarks: List[Dict[str, Any]], landmark_id: int) -> Dict[str, Any]:
        """Get specific landmark by ID"""
        for landmark in landmarks:
            if landmark['id'] == landmark_id:
                return landmark
        return None
    
    def _calculate_angle(self, point1: List[float], point2: List[float], point3: List[float]) -> float:
        """Calculate angle between three points"""
        try:
            # Calculate vectors
            vector1 = [point1[0] - point2[0], point1[1] - point2[1]]
            vector2 = [point3[0] - point2[0], point3[1] - point2[1]]
            
            # Calculate dot product and magnitudes
            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
            magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 90.0
            
            # Calculate angle
            cos_angle = dot_product / (magnitude1 * magnitude2)
            cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
            angle_radians = math.acos(cos_angle)
            angle_degrees = math.degrees(angle_radians)
            
            return angle_degrees
            
        except Exception:
            return 90.0  # Default angle
    
    def _calculate_posture_score(self, alignment_analysis: Dict[str, float]) -> float:
        """Calculate overall posture score"""
        if not alignment_analysis:
            return 0.0
        
        # Weight different aspects of posture
        weights = {
            'head_alignment': 0.2,
            'shoulder_alignment': 0.2,
            'spine_alignment': 0.25,
            'hip_alignment': 0.15,
            'forward_head_posture': 0.1,
            'shoulder_position': 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for aspect, score in alignment_analysis.items():
            if aspect in weights:
                weighted_score += score * weights[aspect]
                total_weight += weights[aspect]
        
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return 0.5
    
    def _calculate_pose_confidence(self, landmarks: List[Dict[str, Any]]) -> float:
        """Calculate confidence in pose detection"""
        if not landmarks:
            return 0.0
        
        # Calculate average visibility of key landmarks
        key_landmarks = [
            self.pose_landmarks['nose'],
            self.pose_landmarks['left_shoulder'],
            self.pose_landmarks['right_shoulder'],
            self.pose_landmarks['left_hip'],
            self.pose_landmarks['right_hip']
        ]
        
        total_visibility = 0.0
        count = 0
        
        for landmark in landmarks:
            if landmark['id'] in key_landmarks:
                total_visibility += landmark.get('visibility', 0.5)
                count += 1
        
        if count > 0:
            avg_visibility = total_visibility / count
            # Boost confidence if all key landmarks are detected
            completeness_bonus = min(count / len(key_landmarks), 1.0) * 0.2
            return min(avg_visibility + completeness_bonus, 1.0)
        else:
            return 0.0
    
    def _generate_posture_recommendations(self, alignment_analysis: Dict[str, float]) -> List[str]:
        """Generate posture improvement recommendations"""
        recommendations = []
        
        for aspect, score in alignment_analysis.items():
            if score < 0.6:
                if aspect == 'head_alignment':
                    recommendations.append("Keep your head centered over your shoulders")
                elif aspect == 'shoulder_alignment':
                    recommendations.append("Level your shoulders - one appears higher than the other")
                elif aspect == 'spine_alignment':
                    recommendations.append("Straighten your spine and avoid leaning to one side")
                elif aspect == 'hip_alignment':
                    recommendations.append("Balance your weight evenly on both hips")
                elif aspect == 'forward_head_posture':
                    recommendations.append("Pull your head back to align with your shoulders")
                elif aspect == 'shoulder_position':
                    recommendations.append("Roll your shoulders back and down to avoid rounding")
        
        if not recommendations:
            recommendations.append("Your posture looks good! Keep maintaining proper alignment.")
        
        return recommendations
    
    def _get_posture_rating(self, posture_score: float) -> str:
        """Convert posture score to rating"""
        if posture_score >= 0.8:
            return 'Excellent'
        elif posture_score >= 0.6:
            return 'Good'
        elif posture_score >= 0.4:
            return 'Fair'
        else:
            return 'Needs Improvement'
