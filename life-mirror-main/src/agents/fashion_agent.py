import os
import cv2
import numpy as np
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentInput, AgentOutput
import requests
from PIL import Image
import io

class FashionAgent(BaseAgent):
    """Advanced fashion and clothing analysis agent"""
    
    def __init__(self):
        super().__init__()
        self.yolo_available = self._check_yolo()
        self.clothing_categories = {
            'tops': ['shirt', 'blouse', 'sweater', 'jacket', 'coat', 't-shirt', 'tank top'],
            'bottoms': ['pants', 'jeans', 'shorts', 'skirt', 'dress'],
            'footwear': ['shoes', 'boots', 'sneakers', 'sandals', 'heels'],
            'accessories': ['hat', 'cap', 'glasses', 'watch', 'jewelry', 'bag', 'purse']
        }
        
    def _check_yolo(self) -> bool:
        """Check if YOLO is available"""
        try:
            from ultralytics import YOLO
            return True
        except ImportError:
            self.logger.warning("YOLO not available, using basic fashion analysis")
            return False
    
    def run(self, input: AgentInput) -> AgentOutput:
        """Analyze fashion and clothing in the image"""
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
            
            # Perform fashion analysis
            if self.yolo_available:
                analysis_result = self._analyze_with_yolo(image_data)
            else:
                analysis_result = self._analyze_basic_fashion(image_data)
            
            # Add style assessment
            style_analysis = self._assess_style(analysis_result)
            analysis_result.update(style_analysis)
            
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
    
    def _analyze_with_yolo(self, image: np.ndarray) -> Dict[str, Any]:
        """Advanced fashion analysis using YOLO"""
        try:
            from ultralytics import YOLO
            
            # Load YOLO model (you might want to use a fashion-specific model)
            model = YOLO('yolov8n.pt')  # Using general model, can be replaced with fashion-specific
            
            # Run inference
            results = model(image, verbose=False)
            
            detected_items = []
            clothing_items = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        item_info = {
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'category': self._categorize_item(class_name)
                        }
                        
                        detected_items.append(item_info)
                        
                        # Filter for clothing items
                        if self._is_clothing_item(class_name):
                            clothing_items.append(item_info)
            
            # Analyze color scheme
            color_analysis = self._analyze_colors(image, clothing_items)
            
            # Calculate fashion score
            fashion_score = self._calculate_fashion_score(clothing_items, color_analysis)
            
            return {
                'detected_items': detected_items,
                'clothing_items': clothing_items,
                'color_analysis': color_analysis,
                'fashion_score': fashion_score,
                'confidence': self._calculate_overall_confidence(clothing_items),
                'analysis_method': 'yolo',
                'outfit_completeness': self._assess_outfit_completeness(clothing_items)
            }
            
        except Exception as e:
            self.logger.error(f"YOLO fashion analysis failed: {e}")
            return self._analyze_basic_fashion(image)
    
    def _analyze_basic_fashion(self, image: np.ndarray) -> Dict[str, Any]:
        """Basic fashion analysis without YOLO"""
        try:
            # Basic color analysis
            color_analysis = self._analyze_colors_basic(image)
            
            # Mock clothing detection for demonstration
            clothing_items = [
                {
                    'class': 'clothing',
                    'confidence': 0.6,
                    'bbox': [0, 0, image.shape[1], image.shape[0]],
                    'category': 'general'
                }
            ]
            
            return {
                'detected_items': clothing_items,
                'clothing_items': clothing_items,
                'color_analysis': color_analysis,
                'fashion_score': 0.5,  # Neutral score
                'confidence': 0.4,  # Lower confidence for basic analysis
                'analysis_method': 'basic',
                'outfit_completeness': 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Basic fashion analysis failed: {e}")
            return {
                'detected_items': [],
                'clothing_items': [],
                'color_analysis': {'dominant_colors': [], 'color_harmony': 0.5},
                'fashion_score': 0.0,
                'confidence': 0.0,
                'analysis_method': 'failed',
                'outfit_completeness': 0.0,
                'error': str(e)
            }
    
    def _categorize_item(self, class_name: str) -> str:
        """Categorize detected item into fashion category"""
        class_name_lower = class_name.lower()
        
        for category, items in self.clothing_categories.items():
            if any(item in class_name_lower for item in items):
                return category
        
        return 'other'
    
    def _is_clothing_item(self, class_name: str) -> bool:
        """Check if detected item is clothing"""
        clothing_keywords = ['shirt', 'pants', 'dress', 'jacket', 'shoes', 'hat', 'bag']
        return any(keyword in class_name.lower() for keyword in clothing_keywords)
    
    def _analyze_colors(self, image: np.ndarray, clothing_items: List[Dict]) -> Dict[str, Any]:
        """Analyze color scheme of clothing items"""
        try:
            # Extract colors from clothing regions
            colors = []
            
            for item in clothing_items:
                bbox = item['bbox']
                x1, y1, x2, y2 = bbox
                
                # Extract region
                region = image[y1:y2, x1:x2]
                if region.size > 0:
                    # Get dominant color
                    dominant_color = self._get_dominant_color(region)
                    colors.append(dominant_color)
            
            # If no clothing items, analyze whole image
            if not colors:
                dominant_color = self._get_dominant_color(image)
                colors = [dominant_color]
            
            # Analyze color harmony
            color_harmony = self._calculate_color_harmony(colors)
            
            return {
                'dominant_colors': colors,
                'color_harmony': color_harmony,
                'color_count': len(set(tuple(c) for c in colors))
            }
            
        except Exception as e:
            self.logger.error(f"Color analysis failed: {e}")
            return {'dominant_colors': [], 'color_harmony': 0.5, 'color_count': 0}
    
    def _analyze_colors_basic(self, image: np.ndarray) -> Dict[str, Any]:
        """Basic color analysis for entire image"""
        try:
            dominant_color = self._get_dominant_color(image)
            return {
                'dominant_colors': [dominant_color],
                'color_harmony': 0.5,
                'color_count': 1
            }
        except Exception as e:
            return {'dominant_colors': [], 'color_harmony': 0.5, 'color_count': 0}
    
    def _get_dominant_color(self, image_region: np.ndarray) -> List[int]:
        """Extract dominant color from image region"""
        try:
            # Reshape image to be a list of pixels
            pixels = image_region.reshape(-1, 3)
            
            # Use k-means to find dominant color
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            dominant_color = kmeans.cluster_centers_[0]
            return [int(c) for c in dominant_color]
            
        except Exception:
            # Fallback to mean color
            mean_color = np.mean(image_region, axis=(0, 1))
            return [int(c) for c in mean_color]
    
    def _calculate_color_harmony(self, colors: List[List[int]]) -> float:
        """Calculate color harmony score"""
        if len(colors) < 2:
            return 0.8  # Single color is harmonious
        
        # Simple harmony calculation based on color distance
        total_distance = 0
        comparisons = 0
        
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                # Calculate Euclidean distance in RGB space
                distance = np.sqrt(sum((colors[i][k] - colors[j][k]) ** 2 for k in range(3)))
                total_distance += distance
                comparisons += 1
        
        if comparisons == 0:
            return 0.8
        
        avg_distance = total_distance / comparisons
        
        # Normalize to 0-1 scale (lower distance = higher harmony)
        # Max possible distance in RGB is ~441 (sqrt(255^2 * 3))
        harmony_score = 1 - min(avg_distance / 441, 1)
        
        return max(0.1, harmony_score)  # Minimum score of 0.1
    
    def _calculate_fashion_score(self, clothing_items: List[Dict], color_analysis: Dict) -> float:
        """Calculate overall fashion score"""
        if not clothing_items:
            return 0.0
        
        score_factors = []
        
        # Item detection confidence
        avg_confidence = sum(item['confidence'] for item in clothing_items) / len(clothing_items)
        score_factors.append(avg_confidence)
        
        # Color harmony
        color_harmony = color_analysis.get('color_harmony', 0.5)
        score_factors.append(color_harmony)
        
        # Outfit completeness
        completeness = self._assess_outfit_completeness(clothing_items)
        score_factors.append(completeness)
        
        # Variety bonus (having different types of items)
        categories = set(item['category'] for item in clothing_items)
        variety_bonus = min(len(categories) / 4, 1.0)  # Max bonus for 4+ categories
        score_factors.append(variety_bonus)
        
        return sum(score_factors) / len(score_factors)
    
    def _assess_outfit_completeness(self, clothing_items: List[Dict]) -> float:
        """Assess how complete the outfit is"""
        if not clothing_items:
            return 0.0
        
        categories_present = set(item['category'] for item in clothing_items)
        
        # Basic outfit should have tops and bottoms
        completeness_score = 0.0
        
        if 'tops' in categories_present:
            completeness_score += 0.4
        if 'bottoms' in categories_present:
            completeness_score += 0.4
        if 'footwear' in categories_present:
            completeness_score += 0.15
        if 'accessories' in categories_present:
            completeness_score += 0.05
        
        return min(completeness_score, 1.0)
    
    def _calculate_overall_confidence(self, clothing_items: List[Dict]) -> float:
        """Calculate overall confidence in fashion analysis"""
        if not clothing_items:
            return 0.0
        
        # Average detection confidence
        avg_confidence = sum(item['confidence'] for item in clothing_items) / len(clothing_items)
        
        # Boost confidence if multiple items detected
        item_count_bonus = min(len(clothing_items) / 5, 0.2)  # Max 0.2 bonus
        
        return min(avg_confidence + item_count_bonus, 1.0)
    
    def _assess_style(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall style and provide recommendations"""
        try:
            clothing_items = analysis_result.get('clothing_items', [])
            color_analysis = analysis_result.get('color_analysis', {})
            fashion_score = analysis_result.get('fashion_score', 0.5)
            
            # Determine style category
            style_category = self._determine_style_category(clothing_items)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(fashion_score, color_analysis, clothing_items)
            
            # Calculate style confidence
            style_confidence = self._calculate_style_confidence(clothing_items, color_analysis)
            
            return {
                'style_category': style_category,
                'style_confidence': style_confidence,
                'recommendations': recommendations,
                'overall_rating': self._get_overall_rating(fashion_score)
            }
            
        except Exception as e:
            self.logger.error(f"Style assessment failed: {e}")
            return {
                'style_category': 'casual',
                'style_confidence': 0.5,
                'recommendations': ['Consider adding accessories to enhance your look'],
                'overall_rating': 'Good'
            }
    
    def _determine_style_category(self, clothing_items: List[Dict]) -> str:
        """Determine the style category based on detected items"""
        if not clothing_items:
            return 'casual'
        
        # Simple style categorization logic
        categories = [item['category'] for item in clothing_items]
        
        if 'accessories' in categories and len(set(categories)) >= 3:
            return 'formal'
        elif 'footwear' in categories and 'tops' in categories:
            return 'smart-casual'
        else:
            return 'casual'
    
    def _generate_recommendations(self, fashion_score: float, color_analysis: Dict, clothing_items: List[Dict]) -> List[str]:
        """Generate fashion recommendations"""
        recommendations = []
        
        if fashion_score < 0.4:
            recommendations.append("Consider coordinating your outfit colors better")
        
        color_harmony = color_analysis.get('color_harmony', 0.5)
        if color_harmony < 0.5:
            recommendations.append("Try using complementary colors for better harmony")
        
        categories = set(item['category'] for item in clothing_items)
        if 'accessories' not in categories:
            recommendations.append("Adding accessories could enhance your overall look")
        
        if len(recommendations) == 0:
            recommendations.append("Your outfit looks great! Keep up the good style.")
        
        return recommendations
    
    def _calculate_style_confidence(self, clothing_items: List[Dict], color_analysis: Dict) -> float:
        """Calculate confidence in style assessment"""
        confidence_factors = []
        
        # Item detection confidence
        if clothing_items:
            avg_confidence = sum(item['confidence'] for item in clothing_items) / len(clothing_items)
            confidence_factors.append(avg_confidence)
        
        # Color analysis confidence
        color_count = color_analysis.get('color_count', 0)
        if color_count > 0:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _get_overall_rating(self, fashion_score: float) -> str:
        """Convert fashion score to rating"""
        if fashion_score >= 0.8:
            return 'Excellent'
        elif fashion_score >= 0.6:
            return 'Good'
        elif fashion_score >= 0.4:
            return 'Fair'
        else:
            return 'Needs Improvement'
