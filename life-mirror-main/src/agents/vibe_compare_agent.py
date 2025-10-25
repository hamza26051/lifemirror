import json
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter
from .base_agent import BaseAgent, AgentInput, AgentOutput

class VibeCompareAgent(BaseAgent):
    """Agent for comparing personality vibes and characteristics between profiles"""
    
    def __init__(self):
        super().__init__()
        
        # Vibe dimensions for comparison
        self.vibe_dimensions = {
            'energy_level': {
                'high_energy': ['energetic', 'dynamic', 'vibrant', 'enthusiastic', 'passionate', 'driven'],
                'low_energy': ['calm', 'peaceful', 'relaxed', 'steady', 'composed', 'tranquil'],
                'weight': 0.15
            },
            'social_orientation': {
                'extroverted': ['outgoing', 'social', 'talkative', 'expressive', 'gregarious', 'people-person'],
                'introverted': ['quiet', 'reserved', 'thoughtful', 'introspective', 'private', 'independent'],
                'weight': 0.15
            },
            'emotional_expression': {
                'expressive': ['emotional', 'expressive', 'open', 'warm', 'empathetic', 'sensitive'],
                'reserved': ['controlled', 'professional', 'measured', 'diplomatic', 'composed', 'rational'],
                'weight': 0.12
            },
            'communication_style': {
                'direct': ['direct', 'straightforward', 'honest', 'blunt', 'clear', 'transparent'],
                'diplomatic': ['diplomatic', 'tactful', 'considerate', 'gentle', 'polite', 'respectful'],
                'weight': 0.12
            },
            'approach_to_life': {
                'optimistic': ['positive', 'optimistic', 'hopeful', 'upbeat', 'cheerful', 'bright'],
                'realistic': ['realistic', 'practical', 'grounded', 'pragmatic', 'sensible', 'logical'],
                'weight': 0.12
            },
            'work_style': {
                'collaborative': ['collaborative', 'team-player', 'cooperative', 'supportive', 'inclusive'],
                'independent': ['independent', 'self-reliant', 'autonomous', 'self-directed', 'solo'],
                'weight': 0.10
            },
            'innovation_tendency': {
                'innovative': ['innovative', 'creative', 'original', 'inventive', 'pioneering', 'visionary'],
                'traditional': ['traditional', 'conventional', 'established', 'proven', 'reliable', 'stable'],
                'weight': 0.10
            },
            'risk_tolerance': {
                'risk_taking': ['adventurous', 'bold', 'daring', 'fearless', 'brave', 'courageous'],
                'risk_averse': ['cautious', 'careful', 'prudent', 'conservative', 'safe', 'secure'],
                'weight': 0.08
            },
            'leadership_style': {
                'authoritative': ['leader', 'decisive', 'commanding', 'authoritative', 'strong', 'confident'],
                'collaborative': ['facilitator', 'supportive', 'inclusive', 'empowering', 'democratic'],
                'weight': 0.06
            }
        }
        
        # Personality traits mapping
        self.personality_traits = {
            'openness': ['creative', 'imaginative', 'curious', 'open-minded', 'artistic', 'innovative'],
            'conscientiousness': ['organized', 'responsible', 'disciplined', 'reliable', 'thorough', 'systematic'],
            'extraversion': ['outgoing', 'social', 'talkative', 'assertive', 'energetic', 'enthusiastic'],
            'agreeableness': ['friendly', 'cooperative', 'trusting', 'helpful', 'compassionate', 'kind'],
            'neuroticism': ['anxious', 'stressed', 'emotional', 'sensitive', 'worried', 'tense']
        }
        
        # Vibe compatibility factors
        self.compatibility_factors = {
            'complementary': {
                'energy_level': 0.7,  # Different energy levels can complement
                'social_orientation': 0.6,  # Intro/extro can balance
                'communication_style': 0.8,  # Different styles can work well
                'work_style': 0.7
            },
            'similar': {
                'approach_to_life': 0.9,  # Similar outlooks work well
                'innovation_tendency': 0.8,  # Similar innovation levels align
                'risk_tolerance': 0.8,  # Similar risk preferences align
                'emotional_expression': 0.7
            }
        }
        
        # Comparison types
        self.comparison_types = {
            'similarity': 'How similar are the vibes',
            'compatibility': 'How compatible are the vibes',
            'contrast': 'How different are the vibes',
            'evolution': 'How the vibe has evolved over time',
            'influence': 'How one vibe influences another'
        }
        
        # Vibe evolution indicators
        self.evolution_indicators = {
            'maturity': ['mature', 'experienced', 'seasoned', 'wise', 'developed'],
            'confidence': ['confident', 'assured', 'self-assured', 'certain', 'bold'],
            'focus': ['focused', 'targeted', 'specialized', 'concentrated', 'dedicated'],
            'balance': ['balanced', 'well-rounded', 'harmonious', 'stable', 'centered']
        }
        
    def run(self, input: AgentInput) -> AgentOutput:
        """Perform vibe comparison analysis"""
        try:
            # Get input data
            profile_a = input.context.get('profile_a', '')
            profile_b = input.context.get('profile_b', '')
            comparison_type = input.context.get('comparison_type', 'similarity')  # 'similarity', 'compatibility', 'contrast', 'evolution', 'influence'
            focus_dimensions = input.context.get('focus_dimensions', None)  # Specific dimensions to focus on
            historical_profiles = input.context.get('historical_profiles', [])  # For evolution analysis
            
            if not profile_a or not profile_b:
                return self._create_output(
                    success=False,
                    data={},
                    error="Two profiles are required for vibe comparison",
                    confidence=0.0
                )
            
            # Perform vibe comparison based on type
            if comparison_type == 'similarity':
                comparison_result = self._perform_similarity_comparison(profile_a, profile_b, focus_dimensions)
            elif comparison_type == 'compatibility':
                comparison_result = self._perform_compatibility_comparison(profile_a, profile_b, focus_dimensions)
            elif comparison_type == 'contrast':
                comparison_result = self._perform_contrast_comparison(profile_a, profile_b, focus_dimensions)
            elif comparison_type == 'evolution':
                comparison_result = self._perform_evolution_comparison(profile_a, profile_b, historical_profiles)
            elif comparison_type == 'influence':
                comparison_result = self._perform_influence_comparison(profile_a, profile_b, focus_dimensions)
            else:
                comparison_result = self._perform_similarity_comparison(profile_a, profile_b, focus_dimensions)
            
            # Generate vibe insights
            vibe_insights = self._generate_vibe_insights(comparison_result, comparison_type)
            
            # Create vibe recommendations
            vibe_recommendations = self._create_vibe_recommendations(comparison_result, comparison_type)
            
            # Calculate overall vibe score
            vibe_score = self._calculate_vibe_score(comparison_result)
            
            return self._create_output(
                success=True,
                data={
                    'vibe_comparison': comparison_result,
                    'vibe_insights': vibe_insights,
                    'vibe_recommendations': vibe_recommendations,
                    'vibe_score': vibe_score,
                    'comparison_type': comparison_type,
                    'focus_dimensions': focus_dimensions,
                    'timestamp': datetime.now().isoformat()
                },
                confidence=self._calculate_vibe_confidence(profile_a, profile_b)
            )
            
        except Exception as e:
            return self._handle_error(e, input)
    
    def _perform_similarity_comparison(self, profile_a: str, profile_b: str, focus_dimensions: Optional[List[str]]) -> Dict[str, Any]:
        """Perform similarity-based vibe comparison"""
        try:
            # Extract vibe profiles for both
            vibe_a = self._extract_vibe_profile(profile_a)
            vibe_b = self._extract_vibe_profile(profile_b)
            
            # Calculate similarity for each dimension
            dimension_similarities = {}
            dimensions_to_compare = focus_dimensions if focus_dimensions else list(self.vibe_dimensions.keys())
            
            for dimension in dimensions_to_compare:
                if dimension in self.vibe_dimensions:
                    similarity = self._calculate_dimension_similarity(vibe_a, vibe_b, dimension)
                    dimension_similarities[dimension] = similarity
            
            # Calculate overall similarity
            overall_similarity = self._calculate_overall_similarity(dimension_similarities)
            
            # Identify similar traits
            similar_traits = self._identify_similar_traits(vibe_a, vibe_b)
            
            # Identify different traits
            different_traits = self._identify_different_traits(vibe_a, vibe_b)
            
            return {
                'vibe_profile_a': vibe_a,
                'vibe_profile_b': vibe_b,
                'dimension_similarities': dimension_similarities,
                'overall_similarity': overall_similarity,
                'similar_traits': similar_traits,
                'different_traits': different_traits,
                'similarity_ranking': self._rank_similarities(dimension_similarities)
            }
            
        except Exception as e:
            self.logger.error(f"Similarity comparison failed: {e}")
            return {}
    
    def _perform_compatibility_comparison(self, profile_a: str, profile_b: str, focus_dimensions: Optional[List[str]]) -> Dict[str, Any]:
        """Perform compatibility-based vibe comparison"""
        try:
            # Extract vibe profiles
            vibe_a = self._extract_vibe_profile(profile_a)
            vibe_b = self._extract_vibe_profile(profile_b)
            
            # Calculate compatibility for each dimension
            dimension_compatibility = {}
            dimensions_to_compare = focus_dimensions if focus_dimensions else list(self.vibe_dimensions.keys())
            
            for dimension in dimensions_to_compare:
                if dimension in self.vibe_dimensions:
                    compatibility = self._calculate_dimension_compatibility(vibe_a, vibe_b, dimension)
                    dimension_compatibility[dimension] = compatibility
            
            # Calculate overall compatibility
            overall_compatibility = self._calculate_overall_compatibility(dimension_compatibility)
            
            # Identify complementary aspects
            complementary_aspects = self._identify_complementary_aspects(vibe_a, vibe_b)
            
            # Identify potential conflicts
            potential_conflicts = self._identify_potential_conflicts(vibe_a, vibe_b)
            
            # Generate compatibility insights
            compatibility_insights = self._generate_compatibility_insights(vibe_a, vibe_b, dimension_compatibility)
            
            return {
                'vibe_profile_a': vibe_a,
                'vibe_profile_b': vibe_b,
                'dimension_compatibility': dimension_compatibility,
                'overall_compatibility': overall_compatibility,
                'complementary_aspects': complementary_aspects,
                'potential_conflicts': potential_conflicts,
                'compatibility_insights': compatibility_insights,
                'compatibility_ranking': self._rank_compatibility(dimension_compatibility)
            }
            
        except Exception as e:
            self.logger.error(f"Compatibility comparison failed: {e}")
            return {}
    
    def _perform_contrast_comparison(self, profile_a: str, profile_b: str, focus_dimensions: Optional[List[str]]) -> Dict[str, Any]:
        """Perform contrast-based vibe comparison"""
        try:
            # Extract vibe profiles
            vibe_a = self._extract_vibe_profile(profile_a)
            vibe_b = self._extract_vibe_profile(profile_b)
            
            # Calculate contrasts for each dimension
            dimension_contrasts = {}
            dimensions_to_compare = focus_dimensions if focus_dimensions else list(self.vibe_dimensions.keys())
            
            for dimension in dimensions_to_compare:
                if dimension in self.vibe_dimensions:
                    contrast = self._calculate_dimension_contrast(vibe_a, vibe_b, dimension)
                    dimension_contrasts[dimension] = contrast
            
            # Calculate overall contrast
            overall_contrast = self._calculate_overall_contrast(dimension_contrasts)
            
            # Identify key differences
            key_differences = self._identify_key_differences(vibe_a, vibe_b)
            
            # Identify opposing traits
            opposing_traits = self._identify_opposing_traits(vibe_a, vibe_b)
            
            # Generate contrast insights
            contrast_insights = self._generate_contrast_insights(vibe_a, vibe_b, dimension_contrasts)
            
            return {
                'vibe_profile_a': vibe_a,
                'vibe_profile_b': vibe_b,
                'dimension_contrasts': dimension_contrasts,
                'overall_contrast': overall_contrast,
                'key_differences': key_differences,
                'opposing_traits': opposing_traits,
                'contrast_insights': contrast_insights,
                'contrast_ranking': self._rank_contrasts(dimension_contrasts)
            }
            
        except Exception as e:
            self.logger.error(f"Contrast comparison failed: {e}")
            return {}
    
    def _perform_evolution_comparison(self, profile_a: str, profile_b: str, historical_profiles: List[str]) -> Dict[str, Any]:
        """Perform evolution-based vibe comparison"""
        try:
            # Extract current vibe profiles
            vibe_a = self._extract_vibe_profile(profile_a)
            vibe_b = self._extract_vibe_profile(profile_b)
            
            # Extract historical vibe profiles
            historical_vibes = [self._extract_vibe_profile(profile) for profile in historical_profiles]
            
            # Analyze evolution patterns
            evolution_patterns_a = self._analyze_evolution_patterns(vibe_a, historical_vibes)
            evolution_patterns_b = self._analyze_evolution_patterns(vibe_b, historical_vibes)
            
            # Calculate evolution similarity
            evolution_similarity = self._calculate_evolution_similarity(evolution_patterns_a, evolution_patterns_b)
            
            # Identify evolution trends
            evolution_trends = self._identify_evolution_trends(vibe_a, vibe_b, historical_vibes)
            
            # Predict future evolution
            future_predictions = self._predict_future_evolution(vibe_a, vibe_b, evolution_patterns_a, evolution_patterns_b)
            
            return {
                'current_vibe_a': vibe_a,
                'current_vibe_b': vibe_b,
                'historical_vibes': historical_vibes,
                'evolution_patterns_a': evolution_patterns_a,
                'evolution_patterns_b': evolution_patterns_b,
                'evolution_similarity': evolution_similarity,
                'evolution_trends': evolution_trends,
                'future_predictions': future_predictions
            }
            
        except Exception as e:
            self.logger.error(f"Evolution comparison failed: {e}")
            return {}
    
    def _perform_influence_comparison(self, profile_a: str, profile_b: str, focus_dimensions: Optional[List[str]]) -> Dict[str, Any]:
        """Perform influence-based vibe comparison"""
        try:
            # Extract vibe profiles
            vibe_a = self._extract_vibe_profile(profile_a)
            vibe_b = self._extract_vibe_profile(profile_b)
            
            # Calculate influence potential
            influence_a_on_b = self._calculate_influence_potential(vibe_a, vibe_b)
            influence_b_on_a = self._calculate_influence_potential(vibe_b, vibe_a)
            
            # Identify influence areas
            influence_areas_a = self._identify_influence_areas(vibe_a, vibe_b)
            influence_areas_b = self._identify_influence_areas(vibe_b, vibe_a)
            
            # Calculate mutual influence
            mutual_influence = self._calculate_mutual_influence(vibe_a, vibe_b)
            
            # Predict influence outcomes
            influence_outcomes = self._predict_influence_outcomes(vibe_a, vibe_b, influence_a_on_b, influence_b_on_a)
            
            return {
                'vibe_profile_a': vibe_a,
                'vibe_profile_b': vibe_b,
                'influence_a_on_b': influence_a_on_b,
                'influence_b_on_a': influence_b_on_a,
                'influence_areas_a': influence_areas_a,
                'influence_areas_b': influence_areas_b,
                'mutual_influence': mutual_influence,
                'influence_outcomes': influence_outcomes,
                'dominant_influencer': 'a' if influence_a_on_b > influence_b_on_a else 'b'
            }
            
        except Exception as e:
            self.logger.error(f"Influence comparison failed: {e}")
            return {}
    
    def _extract_vibe_profile(self, profile: str) -> Dict[str, Any]:
        """Extract vibe profile from text"""
        try:
            profile_lower = profile.lower()
            vibe_profile = {
                'dimensions': {},
                'personality_traits': {},
                'overall_vibe': '',
                'vibe_strength': 0.0,
                'vibe_keywords': []
            }
            
            # Analyze each vibe dimension
            for dimension, dimension_data in self.vibe_dimensions.items():
                dimension_scores = {}
                
                for pole, keywords in dimension_data.items():
                    if pole == 'weight':
                        continue
                    
                    matches = [keyword for keyword in keywords if keyword in profile_lower]
                    score = len(matches) / len(keywords) if keywords else 0
                    dimension_scores[pole] = {
                        'score': score,
                        'matches': matches
                    }
                
                vibe_profile['dimensions'][dimension] = dimension_scores
            
            # Analyze personality traits
            for trait, keywords in self.personality_traits.items():
                matches = [keyword for keyword in keywords if keyword in profile_lower]
                score = len(matches) / len(keywords) if keywords else 0
                vibe_profile['personality_traits'][trait] = {
                    'score': score,
                    'matches': matches
                }
            
            # Determine overall vibe
            vibe_profile['overall_vibe'] = self._determine_overall_vibe(vibe_profile)
            
            # Calculate vibe strength
            vibe_profile['vibe_strength'] = self._calculate_vibe_strength(vibe_profile)
            
            # Extract vibe keywords
            vibe_profile['vibe_keywords'] = self._extract_vibe_keywords(profile_lower)
            
            return vibe_profile
            
        except Exception as e:
            self.logger.error(f"Vibe profile extraction failed: {e}")
            return {}
    
    def _calculate_dimension_similarity(self, vibe_a: Dict[str, Any], vibe_b: Dict[str, Any], dimension: str) -> Dict[str, Any]:
        """Calculate similarity for a specific dimension"""
        try:
            if dimension not in vibe_a['dimensions'] or dimension not in vibe_b['dimensions']:
                return {'similarity': 0.0, 'details': 'Dimension not found'}
            
            dim_a = vibe_a['dimensions'][dimension]
            dim_b = vibe_b['dimensions'][dimension]
            
            similarities = []
            details = {}
            
            for pole in dim_a.keys():
                if pole in dim_b:
                    score_a = dim_a[pole]['score']
                    score_b = dim_b[pole]['score']
                    
                    # Calculate similarity (1 - absolute difference)
                    pole_similarity = 1 - abs(score_a - score_b)
                    similarities.append(pole_similarity)
                    
                    details[pole] = {
                        'score_a': score_a,
                        'score_b': score_b,
                        'similarity': pole_similarity
                    }
            
            overall_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            
            return {
                'similarity': overall_similarity,
                'details': details,
                'dimension': dimension
            }
            
        except Exception as e:
            self.logger.error(f"Dimension similarity calculation failed: {e}")
            return {'similarity': 0.0, 'details': 'Calculation failed'}
    
    def _calculate_dimension_compatibility(self, vibe_a: Dict[str, Any], vibe_b: Dict[str, Any], dimension: str) -> Dict[str, Any]:
        """Calculate compatibility for a specific dimension"""
        try:
            if dimension not in vibe_a['dimensions'] or dimension not in vibe_b['dimensions']:
                return {'compatibility': 0.0, 'details': 'Dimension not found'}
            
            dim_a = vibe_a['dimensions'][dimension]
            dim_b = vibe_b['dimensions'][dimension]
            
            # Determine if this dimension benefits from similarity or complementarity
            compatibility_type = 'similar' if dimension in self.compatibility_factors['similar'] else 'complementary'
            compatibility_weight = self.compatibility_factors[compatibility_type].get(dimension, 0.5)
            
            compatibilities = []
            details = {}
            
            for pole in dim_a.keys():
                if pole in dim_b:
                    score_a = dim_a[pole]['score']
                    score_b = dim_b[pole]['score']
                    
                    if compatibility_type == 'similar':
                        # Higher compatibility when scores are similar
                        pole_compatibility = 1 - abs(score_a - score_b)
                    else:
                        # Higher compatibility when scores are different (complementary)
                        pole_compatibility = abs(score_a - score_b)
                    
                    weighted_compatibility = pole_compatibility * compatibility_weight
                    compatibilities.append(weighted_compatibility)
                    
                    details[pole] = {
                        'score_a': score_a,
                        'score_b': score_b,
                        'compatibility': pole_compatibility,
                        'weighted_compatibility': weighted_compatibility,
                        'type': compatibility_type
                    }
            
            overall_compatibility = sum(compatibilities) / len(compatibilities) if compatibilities else 0.0
            
            return {
                'compatibility': overall_compatibility,
                'details': details,
                'dimension': dimension,
                'compatibility_type': compatibility_type
            }
            
        except Exception as e:
            self.logger.error(f"Dimension compatibility calculation failed: {e}")
            return {'compatibility': 0.0, 'details': 'Calculation failed'}
    
    def _calculate_dimension_contrast(self, vibe_a: Dict[str, Any], vibe_b: Dict[str, Any], dimension: str) -> Dict[str, Any]:
        """Calculate contrast for a specific dimension"""
        try:
            if dimension not in vibe_a['dimensions'] or dimension not in vibe_b['dimensions']:
                return {'contrast': 0.0, 'details': 'Dimension not found'}
            
            dim_a = vibe_a['dimensions'][dimension]
            dim_b = vibe_b['dimensions'][dimension]
            
            contrasts = []
            details = {}
            
            for pole in dim_a.keys():
                if pole in dim_b:
                    score_a = dim_a[pole]['score']
                    score_b = dim_b[pole]['score']
                    
                    # Calculate contrast (absolute difference)
                    pole_contrast = abs(score_a - score_b)
                    contrasts.append(pole_contrast)
                    
                    details[pole] = {
                        'score_a': score_a,
                        'score_b': score_b,
                        'contrast': pole_contrast,
                        'direction': 'a_higher' if score_a > score_b else 'b_higher' if score_b > score_a else 'equal'
                    }
            
            overall_contrast = sum(contrasts) / len(contrasts) if contrasts else 0.0
            
            return {
                'contrast': overall_contrast,
                'details': details,
                'dimension': dimension
            }
            
        except Exception as e:
            self.logger.error(f"Dimension contrast calculation failed: {e}")
            return {'contrast': 0.0, 'details': 'Calculation failed'}
    
    def _calculate_overall_similarity(self, dimension_similarities: Dict[str, Any]) -> float:
        """Calculate overall similarity score"""
        try:
            if not dimension_similarities:
                return 0.0
            
            weighted_similarities = []
            total_weight = 0.0
            
            for dimension, similarity_data in dimension_similarities.items():
                if dimension in self.vibe_dimensions:
                    weight = self.vibe_dimensions[dimension]['weight']
                    similarity = similarity_data['similarity']
                    
                    weighted_similarities.append(similarity * weight)
                    total_weight += weight
            
            if total_weight > 0:
                return sum(weighted_similarities) / total_weight
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Overall similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_overall_compatibility(self, dimension_compatibility: Dict[str, Any]) -> float:
        """Calculate overall compatibility score"""
        try:
            if not dimension_compatibility:
                return 0.0
            
            weighted_compatibilities = []
            total_weight = 0.0
            
            for dimension, compatibility_data in dimension_compatibility.items():
                if dimension in self.vibe_dimensions:
                    weight = self.vibe_dimensions[dimension]['weight']
                    compatibility = compatibility_data['compatibility']
                    
                    weighted_compatibilities.append(compatibility * weight)
                    total_weight += weight
            
            if total_weight > 0:
                return sum(weighted_compatibilities) / total_weight
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Overall compatibility calculation failed: {e}")
            return 0.0
    
    def _calculate_overall_contrast(self, dimension_contrasts: Dict[str, Any]) -> float:
        """Calculate overall contrast score"""
        try:
            if not dimension_contrasts:
                return 0.0
            
            weighted_contrasts = []
            total_weight = 0.0
            
            for dimension, contrast_data in dimension_contrasts.items():
                if dimension in self.vibe_dimensions:
                    weight = self.vibe_dimensions[dimension]['weight']
                    contrast = contrast_data['contrast']
                    
                    weighted_contrasts.append(contrast * weight)
                    total_weight += weight
            
            if total_weight > 0:
                return sum(weighted_contrasts) / total_weight
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Overall contrast calculation failed: {e}")
            return 0.0
    
    def _identify_similar_traits(self, vibe_a: Dict[str, Any], vibe_b: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify similar traits between two vibe profiles"""
        try:
            similar_traits = []
            
            # Compare personality traits
            for trait in self.personality_traits.keys():
                if trait in vibe_a['personality_traits'] and trait in vibe_b['personality_traits']:
                    score_a = vibe_a['personality_traits'][trait]['score']
                    score_b = vibe_b['personality_traits'][trait]['score']
                    
                    # Consider similar if difference is less than 0.3
                    if abs(score_a - score_b) < 0.3 and (score_a > 0.3 or score_b > 0.3):
                        similar_traits.append({
                            'trait': trait,
                            'score_a': score_a,
                            'score_b': score_b,
                            'similarity': 1 - abs(score_a - score_b),
                            'type': 'personality_trait'
                        })
            
            # Compare vibe dimensions
            for dimension in self.vibe_dimensions.keys():
                if dimension in vibe_a['dimensions'] and dimension in vibe_b['dimensions']:
                    dim_a = vibe_a['dimensions'][dimension]
                    dim_b = vibe_b['dimensions'][dimension]
                    
                    for pole in dim_a.keys():
                        if pole in dim_b:
                            score_a = dim_a[pole]['score']
                            score_b = dim_b[pole]['score']
                            
                            if abs(score_a - score_b) < 0.3 and (score_a > 0.3 or score_b > 0.3):
                                similar_traits.append({
                                    'trait': f"{dimension}_{pole}",
                                    'score_a': score_a,
                                    'score_b': score_b,
                                    'similarity': 1 - abs(score_a - score_b),
                                    'type': 'vibe_dimension'
                                })
            
            # Sort by similarity
            similar_traits.sort(key=lambda x: x['similarity'], reverse=True)
            
            return similar_traits[:5]  # Return top 5 similar traits
            
        except Exception as e:
            self.logger.error(f"Similar traits identification failed: {e}")
            return []
    
    def _identify_different_traits(self, vibe_a: Dict[str, Any], vibe_b: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify different traits between two vibe profiles"""
        try:
            different_traits = []
            
            # Compare personality traits
            for trait in self.personality_traits.keys():
                if trait in vibe_a['personality_traits'] and trait in vibe_b['personality_traits']:
                    score_a = vibe_a['personality_traits'][trait]['score']
                    score_b = vibe_b['personality_traits'][trait]['score']
                    
                    # Consider different if difference is greater than 0.4
                    if abs(score_a - score_b) > 0.4:
                        different_traits.append({
                            'trait': trait,
                            'score_a': score_a,
                            'score_b': score_b,
                            'difference': abs(score_a - score_b),
                            'higher_in': 'a' if score_a > score_b else 'b',
                            'type': 'personality_trait'
                        })
            
            # Compare vibe dimensions
            for dimension in self.vibe_dimensions.keys():
                if dimension in vibe_a['dimensions'] and dimension in vibe_b['dimensions']:
                    dim_a = vibe_a['dimensions'][dimension]
                    dim_b = vibe_b['dimensions'][dimension]
                    
                    for pole in dim_a.keys():
                        if pole in dim_b:
                            score_a = dim_a[pole]['score']
                            score_b = dim_b[pole]['score']
                            
                            if abs(score_a - score_b) > 0.4:
                                different_traits.append({
                                    'trait': f"{dimension}_{pole}",
                                    'score_a': score_a,
                                    'score_b': score_b,
                                    'difference': abs(score_a - score_b),
                                    'higher_in': 'a' if score_a > score_b else 'b',
                                    'type': 'vibe_dimension'
                                })
            
            # Sort by difference
            different_traits.sort(key=lambda x: x['difference'], reverse=True)
            
            return different_traits[:5]  # Return top 5 different traits
            
        except Exception as e:
            self.logger.error(f"Different traits identification failed: {e}")
            return []
    
    def _determine_overall_vibe(self, vibe_profile: Dict[str, Any]) -> str:
        """Determine overall vibe description"""
        try:
            vibe_descriptors = []
            
            # Analyze dominant dimensions
            for dimension, dimension_data in vibe_profile['dimensions'].items():
                max_score = 0.0
                dominant_pole = None
                
                for pole, pole_data in dimension_data.items():
                    if pole_data['score'] > max_score and pole_data['score'] > 0.3:
                        max_score = pole_data['score']
                        dominant_pole = pole
                
                if dominant_pole:
                    vibe_descriptors.append(dominant_pole.replace('_', ' '))
            
            # Analyze dominant personality traits
            for trait, trait_data in vibe_profile['personality_traits'].items():
                if trait_data['score'] > 0.4:
                    vibe_descriptors.append(trait)
            
            # Create overall vibe description
            if vibe_descriptors:
                return ', '.join(vibe_descriptors[:3])  # Top 3 descriptors
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.error(f"Overall vibe determination failed: {e}")
            return 'unknown'
    
    def _calculate_vibe_strength(self, vibe_profile: Dict[str, Any]) -> float:
        """Calculate the strength/intensity of the vibe"""
        try:
            strength_scores = []
            
            # Calculate strength from dimensions
            for dimension_data in vibe_profile['dimensions'].values():
                max_score = max(pole_data['score'] for pole_data in dimension_data.values())
                strength_scores.append(max_score)
            
            # Calculate strength from personality traits
            for trait_data in vibe_profile['personality_traits'].values():
                strength_scores.append(trait_data['score'])
            
            if strength_scores:
                return sum(strength_scores) / len(strength_scores)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Vibe strength calculation failed: {e}")
            return 0.0
    
    def _extract_vibe_keywords(self, profile_text: str) -> List[str]:
        """Extract vibe-related keywords from profile text"""
        try:
            vibe_keywords = []
            
            # Collect all vibe-related keywords
            all_keywords = set()
            
            # Add dimension keywords
            for dimension_data in self.vibe_dimensions.values():
                for pole_keywords in dimension_data.values():
                    if isinstance(pole_keywords, list):
                        all_keywords.update(pole_keywords)
            
            # Add personality trait keywords
            for trait_keywords in self.personality_traits.values():
                all_keywords.update(trait_keywords)
            
            # Find keywords in profile
            for keyword in all_keywords:
                if keyword in profile_text:
                    vibe_keywords.append(keyword)
            
            return vibe_keywords[:10]  # Limit to top 10 keywords
            
        except Exception as e:
            self.logger.error(f"Vibe keywords extraction failed: {e}")
            return []
    
    def _generate_vibe_insights(self, comparison_result: Dict[str, Any], comparison_type: str) -> List[str]:
        """Generate insights from vibe comparison"""
        try:
            insights = []
            
            if comparison_type == 'similarity':
                overall_similarity = comparison_result.get('overall_similarity', 0.0)
                insights.append(f"Overall vibe similarity: {overall_similarity:.1%}")
                
                similar_traits = comparison_result.get('similar_traits', [])
                if similar_traits:
                    top_similar = similar_traits[0]
                    insights.append(f"Most similar trait: {top_similar['trait'].replace('_', ' ').title()}")
                
                different_traits = comparison_result.get('different_traits', [])
                if different_traits:
                    top_different = different_traits[0]
                    insights.append(f"Most different trait: {top_different['trait'].replace('_', ' ').title()}")
            
            elif comparison_type == 'compatibility':
                overall_compatibility = comparison_result.get('overall_compatibility', 0.0)
                insights.append(f"Overall vibe compatibility: {overall_compatibility:.1%}")
                
                complementary_aspects = comparison_result.get('complementary_aspects', [])
                if complementary_aspects:
                    insights.append(f"Key complementary aspect: {complementary_aspects[0].replace('_', ' ').title()}")
            
            elif comparison_type == 'contrast':
                overall_contrast = comparison_result.get('overall_contrast', 0.0)
                insights.append(f"Overall vibe contrast: {overall_contrast:.1%}")
                
                key_differences = comparison_result.get('key_differences', [])
                if key_differences:
                    insights.append(f"Key difference: {key_differences[0].replace('_', ' ').title()}")
            
            return insights[:5]  # Limit to top 5 insights
            
        except Exception as e:
            self.logger.error(f"Vibe insights generation failed: {e}")
            return []
    
    def _create_vibe_recommendations(self, comparison_result: Dict[str, Any], comparison_type: str) -> List[str]:
        """Create recommendations based on vibe comparison"""
        try:
            recommendations = []
            
            if comparison_type == 'similarity':
                similar_traits = comparison_result.get('similar_traits', [])
                different_traits = comparison_result.get('different_traits', [])
                
                if similar_traits:
                    recommendations.append(f"Leverage shared {similar_traits[0]['trait'].replace('_', ' ')} for collaboration")
                
                if different_traits:
                    recommendations.append(f"Use different {different_traits[0]['trait'].replace('_', ' ')} approaches for diverse perspectives")
            
            elif comparison_type == 'compatibility':
                complementary_aspects = comparison_result.get('complementary_aspects', [])
                potential_conflicts = comparison_result.get('potential_conflicts', [])
                
                if complementary_aspects:
                    recommendations.append(f"Capitalize on complementary {complementary_aspects[0].replace('_', ' ')} styles")
                
                if potential_conflicts:
                    recommendations.append(f"Address potential conflict in {potential_conflicts[0].replace('_', ' ')} approaches")
            
            elif comparison_type == 'contrast':
                key_differences = comparison_result.get('key_differences', [])
                
                if key_differences:
                    recommendations.append(f"Bridge the gap in {key_differences[0].replace('_', ' ')} perspectives")
                    recommendations.append("Use contrasting styles to create dynamic interactions")
            
            # Add general recommendations
            recommendations.extend([
                "Focus on understanding each other's communication preferences",
                "Create opportunities for mutual learning and growth",
                "Establish clear expectations and boundaries"
            ])
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            self.logger.error(f"Vibe recommendations creation failed: {e}")
            return []
    
    def _calculate_vibe_score(self, comparison_result: Dict[str, Any]) -> float:
        """Calculate overall vibe comparison score"""
        try:
            score_components = []
            
            # Similarity component
            if 'overall_similarity' in comparison_result:
                score_components.append(comparison_result['overall_similarity'] * 0.3)
            
            # Compatibility component
            if 'overall_compatibility' in comparison_result:
                score_components.append(comparison_result['overall_compatibility'] * 0.4)
            
            # Contrast component (inverted for harmony)
            if 'overall_contrast' in comparison_result:
                score_components.append((1 - comparison_result['overall_contrast']) * 0.3)
            
            # Evolution component
            if 'evolution_similarity' in comparison_result:
                score_components.append(comparison_result['evolution_similarity'] * 0.2)
            
            # Influence component
            if 'mutual_influence' in comparison_result:
                score_components.append(comparison_result['mutual_influence'] * 0.2)
            
            if score_components:
                return sum(score_components) / len(score_components)
            else:
                return 0.5  # Default neutral score
                
        except Exception as e:
            self.logger.error(f"Vibe score calculation failed: {e}")
            return 0.5
    
    def _calculate_vibe_confidence(self, profile_a: str, profile_b: str) -> float:
        """Calculate confidence in vibe comparison"""
        try:
            confidence_factors = []
            
            # Profile length factor
            avg_length = (len(profile_a) + len(profile_b)) / 2
            if avg_length > 200:
                confidence_factors.append(0.9)
            elif avg_length > 100:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
            
            # Profile quality factor (presence of vibe keywords)
            vibe_keywords_a = self._extract_vibe_keywords(profile_a.lower())
            vibe_keywords_b = self._extract_vibe_keywords(profile_b.lower())
            
            avg_keywords = (len(vibe_keywords_a) + len(vibe_keywords_b)) / 2
            if avg_keywords > 5:
                confidence_factors.append(0.8)
            elif avg_keywords > 2:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
            
            # Content richness factor
            combined_text = profile_a + " " + profile_b
            word_count = len(combined_text.split())
            if word_count > 100:
                confidence_factors.append(0.8)
            elif word_count > 50:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
            
            return sum(confidence_factors) / len(confidence_factors)
            
        except Exception as e:
            self.logger.error(f"Vibe confidence calculation failed: {e}")
            return 0.6
    
    # Additional helper methods for specific comparison types
    
    def _rank_similarities(self, dimension_similarities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank dimensions by similarity"""
        try:
            rankings = []
            for dimension, similarity_data in dimension_similarities.items():
                rankings.append({
                    'dimension': dimension,
                    'similarity': similarity_data['similarity'],
                    'rank': 0  # Will be set after sorting
                })
            
            rankings.sort(key=lambda x: x['similarity'], reverse=True)
            
            for i, ranking in enumerate(rankings):
                ranking['rank'] = i + 1
            
            return rankings
            
        except Exception as e:
            self.logger.error(f"Similarities ranking failed: {e}")
            return []
    
    def _rank_compatibility(self, dimension_compatibility: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank dimensions by compatibility"""
        try:
            rankings = []
            for dimension, compatibility_data in dimension_compatibility.items():
                rankings.append({
                    'dimension': dimension,
                    'compatibility': compatibility_data['compatibility'],
                    'rank': 0  # Will be set after sorting
                })
            
            rankings.sort(key=lambda x: x['compatibility'], reverse=True)
            
            for i, ranking in enumerate(rankings):
                ranking['rank'] = i + 1
            
            return rankings
            
        except Exception as e:
            self.logger.error(f"Compatibility ranking failed: {e}")
            return []
    
    def _rank_contrasts(self, dimension_contrasts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank dimensions by contrast"""
        try:
            rankings = []
            for dimension, contrast_data in dimension_contrasts.items():
                rankings.append({
                    'dimension': dimension,
                    'contrast': contrast_data['contrast'],
                    'rank': 0  # Will be set after sorting
                })
            
            rankings.sort(key=lambda x: x['contrast'], reverse=True)
            
            for i, ranking in enumerate(rankings):
                ranking['rank'] = i + 1
            
            return rankings
            
        except Exception as e:
            self.logger.error(f"Contrasts ranking failed: {e}")
            return []