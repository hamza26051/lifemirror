import json
import math
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from .base_agent import BaseAgent, AgentInput, AgentOutput

class CompareAgent(BaseAgent):
    """Agent for comparing analysis results and profiles"""
    
    def __init__(self):
        super().__init__()
        
        # Comparison types
        self.comparison_types = {
            'celebrity': self._compare_celebrity,
            'past_self': self._compare_past_self,
            'peer': self._compare_peer,
            'ideal': self._compare_ideal,
            'demographic': self._compare_demographic,
            'trend': self._compare_trend,
            'similarity': self._calculate_similarity,
            'improvement': self._track_improvement
        }
        
        # Celebrity database (mock data for demonstration)
        self.celebrity_profiles = {
            'brad_pitt': {
                'name': 'Brad Pitt',
                'category': 'actor',
                'scores': {
                    'attractiveness': 0.92,
                    'fashion': 0.85,
                    'posture': 0.88,
                    'charisma': 0.95
                },
                'attributes': {
                    'age_range': '50-60',
                    'style': 'classic_masculine',
                    'personality': 'confident_charismatic'
                }
            },
            'angelina_jolie': {
                'name': 'Angelina Jolie',
                'category': 'actress',
                'scores': {
                    'attractiveness': 0.94,
                    'fashion': 0.90,
                    'posture': 0.92,
                    'charisma': 0.93
                },
                'attributes': {
                    'age_range': '40-50',
                    'style': 'elegant_sophisticated',
                    'personality': 'strong_independent'
                }
            },
            'ryan_gosling': {
                'name': 'Ryan Gosling',
                'category': 'actor',
                'scores': {
                    'attractiveness': 0.89,
                    'fashion': 0.82,
                    'posture': 0.85,
                    'charisma': 0.87
                },
                'attributes': {
                    'age_range': '40-50',
                    'style': 'casual_cool',
                    'personality': 'mysterious_charming'
                }
            },
            'emma_stone': {
                'name': 'Emma Stone',
                'category': 'actress',
                'scores': {
                    'attractiveness': 0.86,
                    'fashion': 0.88,
                    'posture': 0.84,
                    'charisma': 0.91
                },
                'attributes': {
                    'age_range': '30-40',
                    'style': 'modern_chic',
                    'personality': 'witty_approachable'
                }
            }
        }
        
        # Demographic averages (mock data)
        self.demographic_averages = {
            'male_20_30': {
                'attractiveness': 0.65,
                'fashion': 0.60,
                'posture': 0.58,
                'confidence': 0.62
            },
            'female_20_30': {
                'attractiveness': 0.68,
                'fashion': 0.72,
                'posture': 0.65,
                'confidence': 0.66
            },
            'male_30_40': {
                'attractiveness': 0.70,
                'fashion': 0.65,
                'posture': 0.62,
                'confidence': 0.68
            },
            'female_30_40': {
                'attractiveness': 0.72,
                'fashion': 0.75,
                'posture': 0.68,
                'confidence': 0.70
            }
        }
        
    def run(self, input: AgentInput) -> AgentOutput:
        """Execute comparison operation"""
        try:
            comparison_type = input.context.get('comparison_type', 'similarity')
            current_data = input.context.get('current_data', {})
            
            if comparison_type not in self.comparison_types:
                return self._create_output(
                    success=False,
                    data={},
                    error=f"Unknown comparison type: {comparison_type}",
                    confidence=0.0
                )
            
            if not current_data:
                return self._create_output(
                    success=False,
                    data={},
                    error="Current data is required for comparison",
                    confidence=0.0
                )
            
            # Execute the comparison
            result = self.comparison_types[comparison_type](input)
            
            return self._create_output(
                success=True,
                data={
                    'comparison_type': comparison_type,
                    'result': result,
                    'timestamp': datetime.now().isoformat(),
                    'metadata': {
                        'comparison_method': comparison_type,
                        'data_quality': self._assess_data_quality(current_data)
                    }
                },
                confidence=result.get('confidence', 0.8)
            )
            
        except Exception as e:
            return self._handle_error(e, input)
    
    def _compare_celebrity(self, input: AgentInput) -> Dict[str, Any]:
        """Compare with celebrity profiles"""
        try:
            current_data = input.context.get('current_data', {})
            target_celebrity = input.context.get('target_celebrity')
            top_n = input.context.get('top_n', 5)
            
            current_scores = self._extract_scores(current_data)
            
            if target_celebrity:
                # Compare with specific celebrity
                if target_celebrity not in self.celebrity_profiles:
                    return {
                        'error': f"Celebrity '{target_celebrity}' not found",
                        'available_celebrities': list(self.celebrity_profiles.keys())
                    }
                
                celebrity_data = self.celebrity_profiles[target_celebrity]
                similarity = self._calculate_score_similarity(current_scores, celebrity_data['scores'])
                
                return {
                    'target_celebrity': celebrity_data['name'],
                    'similarity_score': similarity,
                    'score_comparison': self._compare_score_breakdown(current_scores, celebrity_data['scores']),
                    'insights': self._generate_celebrity_insights(current_data, celebrity_data, similarity),
                    'recommendations': self._generate_celebrity_recommendations(current_data, celebrity_data)
                }
            else:
                # Find most similar celebrities
                similarities = []
                
                for celeb_id, celeb_data in self.celebrity_profiles.items():
                    similarity = self._calculate_score_similarity(current_scores, celeb_data['scores'])
                    similarities.append({
                        'celebrity_id': celeb_id,
                        'name': celeb_data['name'],
                        'category': celeb_data['category'],
                        'similarity_score': similarity,
                        'attributes': celeb_data['attributes']
                    })
                
                # Sort by similarity
                similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
                
                return {
                    'top_matches': similarities[:top_n],
                    'best_match': similarities[0] if similarities else None,
                    'average_similarity': sum(s['similarity_score'] for s in similarities) / len(similarities) if similarities else 0,
                    'insights': self._generate_top_matches_insights(similarities[:top_n]),
                    'confidence': 0.85
                }
                
        except Exception as e:
            self.logger.error(f"Celebrity comparison failed: {e}")
            return {'error': str(e)}
    
    def _compare_past_self(self, input: AgentInput) -> Dict[str, Any]:
        """Compare with past analysis results"""
        try:
            current_data = input.context.get('current_data', {})
            past_data = input.context.get('past_data', [])
            time_period = input.context.get('time_period', 'all')  # 'week', 'month', 'year', 'all'
            
            if not past_data:
                return {
                    'error': 'No past data available for comparison',
                    'recommendation': 'Continue using the app to build comparison history'
                }
            
            current_scores = self._extract_scores(current_data)
            
            # Filter past data by time period
            filtered_past_data = self._filter_by_time_period(past_data, time_period)
            
            if not filtered_past_data:
                return {
                    'error': f'No past data available for time period: {time_period}',
                    'available_periods': self._get_available_periods(past_data)
                }
            
            # Calculate trends and improvements
            trends = self._calculate_trends(filtered_past_data, current_scores)
            improvements = self._calculate_improvements(filtered_past_data, current_scores)
            
            # Find best and worst periods
            best_period = self._find_best_period(filtered_past_data)
            worst_period = self._find_worst_period(filtered_past_data)
            
            return {
                'time_period': time_period,
                'data_points': len(filtered_past_data),
                'trends': trends,
                'improvements': improvements,
                'best_period': best_period,
                'worst_period': worst_period,
                'overall_progress': self._calculate_overall_progress(filtered_past_data, current_scores),
                'insights': self._generate_progress_insights(trends, improvements),
                'recommendations': self._generate_progress_recommendations(trends, improvements),
                'confidence': 0.9
            }
            
        except Exception as e:
            self.logger.error(f"Past self comparison failed: {e}")
            return {'error': str(e)}
    
    def _compare_peer(self, input: AgentInput) -> Dict[str, Any]:
        """Compare with peer group"""
        try:
            current_data = input.context.get('current_data', {})
            peer_data = input.context.get('peer_data', [])
            demographic_info = input.context.get('demographic_info', {})
            
            current_scores = self._extract_scores(current_data)
            
            if peer_data:
                # Compare with actual peer data
                peer_scores = [self._extract_scores(peer) for peer in peer_data]
                peer_averages = self._calculate_peer_averages(peer_scores)
                
                comparison = self._compare_with_averages(current_scores, peer_averages)
                percentiles = self._calculate_percentiles(current_scores, peer_scores)
                
                return {
                    'peer_group_size': len(peer_data),
                    'peer_averages': peer_averages,
                    'your_scores': current_scores,
                    'comparison': comparison,
                    'percentiles': percentiles,
                    'ranking': self._calculate_ranking(current_scores, peer_scores),
                    'insights': self._generate_peer_insights(comparison, percentiles),
                    'confidence': 0.88
                }
            else:
                # Use demographic averages
                demographic_key = self._get_demographic_key(demographic_info)
                
                if demographic_key not in self.demographic_averages:
                    return {
                        'error': 'No demographic data available for comparison',
                        'available_demographics': list(self.demographic_averages.keys())
                    }
                
                demographic_scores = self.demographic_averages[demographic_key]
                comparison = self._compare_with_averages(current_scores, demographic_scores)
                
                return {
                    'demographic_group': demographic_key,
                    'demographic_averages': demographic_scores,
                    'your_scores': current_scores,
                    'comparison': comparison,
                    'insights': self._generate_demographic_insights(comparison, demographic_key),
                    'recommendations': self._generate_demographic_recommendations(comparison),
                    'confidence': 0.75
                }
                
        except Exception as e:
            self.logger.error(f"Peer comparison failed: {e}")
            return {'error': str(e)}
    
    def _compare_ideal(self, input: AgentInput) -> Dict[str, Any]:
        """Compare with ideal/target profile"""
        try:
            current_data = input.context.get('current_data', {})
            ideal_profile = input.context.get('ideal_profile', {})
            
            if not ideal_profile:
                # Generate default ideal profile
                ideal_profile = self._generate_default_ideal_profile(current_data)
            
            current_scores = self._extract_scores(current_data)
            ideal_scores = self._extract_scores(ideal_profile)
            
            # Calculate gaps and achievements
            gaps = self._calculate_gaps(current_scores, ideal_scores)
            achievements = self._calculate_achievements(current_scores, ideal_scores)
            
            # Calculate overall progress toward ideal
            progress = self._calculate_ideal_progress(current_scores, ideal_scores)
            
            return {
                'ideal_profile': ideal_scores,
                'current_scores': current_scores,
                'gaps': gaps,
                'achievements': achievements,
                'overall_progress': progress,
                'priority_areas': self._identify_priority_areas(gaps),
                'action_plan': self._generate_action_plan(gaps),
                'estimated_timeline': self._estimate_improvement_timeline(gaps),
                'insights': self._generate_ideal_insights(gaps, achievements, progress),
                'confidence': 0.82
            }
            
        except Exception as e:
            self.logger.error(f"Ideal comparison failed: {e}")
            return {'error': str(e)}
    
    def _compare_demographic(self, input: AgentInput) -> Dict[str, Any]:
        """Compare with demographic groups"""
        try:
            current_data = input.context.get('current_data', {})
            target_demographics = input.context.get('target_demographics', [])
            
            current_scores = self._extract_scores(current_data)
            
            if not target_demographics:
                # Compare with all available demographics
                target_demographics = list(self.demographic_averages.keys())
            
            comparisons = {}
            
            for demo in target_demographics:
                if demo in self.demographic_averages:
                    demo_scores = self.demographic_averages[demo]
                    comparison = self._compare_with_averages(current_scores, demo_scores)
                    
                    comparisons[demo] = {
                        'demographic_scores': demo_scores,
                        'comparison': comparison,
                        'overall_similarity': self._calculate_score_similarity(current_scores, demo_scores)
                    }
            
            # Find best matching demographic
            best_match = max(comparisons.items(), 
                           key=lambda x: x[1]['overall_similarity']) if comparisons else None
            
            return {
                'comparisons': comparisons,
                'best_demographic_match': best_match[0] if best_match else None,
                'best_match_similarity': best_match[1]['overall_similarity'] if best_match else 0,
                'insights': self._generate_multi_demographic_insights(comparisons),
                'confidence': 0.78
            }
            
        except Exception as e:
            self.logger.error(f"Demographic comparison failed: {e}")
            return {'error': str(e)}
    
    def _compare_trend(self, input: AgentInput) -> Dict[str, Any]:
        """Compare with trending profiles or standards"""
        try:
            current_data = input.context.get('current_data', {})
            trend_period = input.context.get('trend_period', 'current')  # 'current', 'emerging', 'classic'
            
            current_scores = self._extract_scores(current_data)
            
            # Mock trending data (in real implementation, this would come from a trends database)
            trending_profiles = {
                'current': {
                    'name': 'Current Trends 2024',
                    'scores': {
                        'attractiveness': 0.78,
                        'fashion': 0.82,
                        'posture': 0.75,
                        'authenticity': 0.85
                    },
                    'characteristics': ['natural_beauty', 'sustainable_fashion', 'wellness_focused']
                },
                'emerging': {
                    'name': 'Emerging Trends',
                    'scores': {
                        'attractiveness': 0.80,
                        'fashion': 0.88,
                        'posture': 0.82,
                        'innovation': 0.90
                    },
                    'characteristics': ['tech_savvy', 'experimental_style', 'bold_choices']
                },
                'classic': {
                    'name': 'Timeless Classic',
                    'scores': {
                        'attractiveness': 0.85,
                        'fashion': 0.80,
                        'posture': 0.88,
                        'elegance': 0.92
                    },
                    'characteristics': ['timeless_style', 'refined_taste', 'sophisticated']
                }
            }
            
            if trend_period not in trending_profiles:
                return {
                    'error': f"Unknown trend period: {trend_period}",
                    'available_periods': list(trending_profiles.keys())
                }
            
            trend_data = trending_profiles[trend_period]
            similarity = self._calculate_score_similarity(current_scores, trend_data['scores'])
            
            return {
                'trend_period': trend_period,
                'trend_profile': trend_data,
                'similarity_to_trend': similarity,
                'score_comparison': self._compare_score_breakdown(current_scores, trend_data['scores']),
                'trend_alignment': self._calculate_trend_alignment(current_data, trend_data),
                'insights': self._generate_trend_insights(similarity, trend_data),
                'recommendations': self._generate_trend_recommendations(current_scores, trend_data),
                'confidence': 0.80
            }
            
        except Exception as e:
            self.logger.error(f"Trend comparison failed: {e}")
            return {'error': str(e)}
    
    def _calculate_similarity(self, input: AgentInput) -> Dict[str, Any]:
        """Calculate similarity between two profiles"""
        try:
            profile_a = input.context.get('profile_a', {})
            profile_b = input.context.get('profile_b', {})
            similarity_method = input.context.get('method', 'cosine')  # 'cosine', 'euclidean', 'weighted'
            
            if not profile_a or not profile_b:
                return {
                    'error': 'Both profile_a and profile_b are required for similarity calculation'
                }
            
            scores_a = self._extract_scores(profile_a)
            scores_b = self._extract_scores(profile_b)
            
            # Calculate similarity using different methods
            similarities = {
                'cosine': self._cosine_similarity(scores_a, scores_b),
                'euclidean': self._euclidean_similarity(scores_a, scores_b),
                'weighted': self._weighted_similarity(scores_a, scores_b),
                'manhattan': self._manhattan_similarity(scores_a, scores_b)
            }
            
            primary_similarity = similarities.get(similarity_method, similarities['cosine'])
            
            return {
                'primary_similarity': primary_similarity,
                'similarity_method': similarity_method,
                'all_similarities': similarities,
                'score_differences': self._calculate_score_differences(scores_a, scores_b),
                'common_strengths': self._find_common_strengths(scores_a, scores_b),
                'common_weaknesses': self._find_common_weaknesses(scores_a, scores_b),
                'insights': self._generate_similarity_insights(similarities, scores_a, scores_b),
                'confidence': 0.92
            }
            
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            return {'error': str(e)}
    
    def _track_improvement(self, input: AgentInput) -> Dict[str, Any]:
        """Track improvement over time"""
        try:
            historical_data = input.context.get('historical_data', [])
            current_data = input.context.get('current_data', {})
            improvement_metrics = input.context.get('metrics', ['all'])
            
            if not historical_data:
                return {
                    'error': 'Historical data is required for improvement tracking'
                }
            
            current_scores = self._extract_scores(current_data)
            
            # Calculate improvement metrics
            improvements = {
                'absolute_change': self._calculate_absolute_improvement(historical_data, current_scores),
                'percentage_change': self._calculate_percentage_improvement(historical_data, current_scores),
                'trend_analysis': self._analyze_improvement_trends(historical_data, current_scores),
                'velocity': self._calculate_improvement_velocity(historical_data, current_scores),
                'consistency': self._calculate_improvement_consistency(historical_data)
            }
            
            # Filter by requested metrics
            if 'all' not in improvement_metrics:
                improvements = {k: v for k, v in improvements.items() if k in improvement_metrics}
            
            return {
                'improvement_metrics': improvements,
                'overall_improvement_score': self._calculate_overall_improvement_score(improvements),
                'best_improving_areas': self._identify_best_improving_areas(improvements),
                'stagnant_areas': self._identify_stagnant_areas(improvements),
                'improvement_predictions': self._predict_future_improvements(historical_data, current_scores),
                'insights': self._generate_improvement_insights(improvements),
                'recommendations': self._generate_improvement_recommendations(improvements),
                'confidence': 0.87
            }
            
        except Exception as e:
            self.logger.error(f"Improvement tracking failed: {e}")
            return {'error': str(e)}
    
    # Helper methods for score extraction and calculation
    
    def _extract_scores(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical scores from analysis data"""
        scores = {}
        
        try:
            # Look for overall scores
            if 'overall_scores' in data:
                overall_scores = data['overall_scores']
                if isinstance(overall_scores, dict):
                    scores.update(overall_scores)
            
            # Look for individual agent scores
            if 'individual_results' in data:
                for agent_name, result in data['individual_results'].items():
                    if isinstance(result, dict) and result.get('available', False):
                        agent_data = result.get('data', {})
                        
                        # Find main score for this agent
                        main_score = self._find_main_score(agent_data)
                        if main_score is not None:
                            scores[agent_name] = main_score
            
            # Look for direct scores in data
            for key, value in data.items():
                if 'score' in key.lower() and isinstance(value, (int, float)):
                    scores[key] = float(value)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Score extraction failed: {e}")
            return {}
    
    def _find_main_score(self, data: Dict[str, Any]) -> Optional[float]:
        """Find the main score in agent data"""
        score_fields = [
            'score', 'overall_score', 'main_score', 'confidence_score',
            'attractiveness_score', 'fashion_score', 'posture_score', 'vibe_score'
        ]
        
        for field in score_fields:
            if field in data and isinstance(data[field], (int, float)):
                return float(data[field])
        
        return None
    
    def _calculate_score_similarity(self, scores_a: Dict[str, float], scores_b: Dict[str, float]) -> float:
        """Calculate similarity between two score dictionaries"""
        try:
            # Find common keys
            common_keys = set(scores_a.keys()) & set(scores_b.keys())
            
            if not common_keys:
                return 0.0
            
            # Calculate cosine similarity
            dot_product = sum(scores_a[key] * scores_b[key] for key in common_keys)
            norm_a = math.sqrt(sum(scores_a[key] ** 2 for key in common_keys))
            norm_b = math.sqrt(sum(scores_b[key] ** 2 for key in common_keys))
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
            
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _cosine_similarity(self, scores_a: Dict[str, float], scores_b: Dict[str, float]) -> float:
        """Calculate cosine similarity"""
        return self._calculate_score_similarity(scores_a, scores_b)
    
    def _euclidean_similarity(self, scores_a: Dict[str, float], scores_b: Dict[str, float]) -> float:
        """Calculate euclidean similarity (converted to 0-1 scale)"""
        try:
            common_keys = set(scores_a.keys()) & set(scores_b.keys())
            
            if not common_keys:
                return 0.0
            
            # Calculate euclidean distance
            distance = math.sqrt(sum((scores_a[key] - scores_b[key]) ** 2 for key in common_keys))
            
            # Convert to similarity (0-1 scale)
            max_distance = math.sqrt(len(common_keys))  # Maximum possible distance
            similarity = 1 - (distance / max_distance)
            
            return max(0.0, similarity)
            
        except Exception as e:
            self.logger.error(f"Euclidean similarity calculation failed: {e}")
            return 0.0
    
    def _weighted_similarity(self, scores_a: Dict[str, float], scores_b: Dict[str, float]) -> float:
        """Calculate weighted similarity (giving more weight to important scores)"""
        try:
            # Define weights for different score types
            weights = {
                'attractiveness': 1.2,
                'fashion': 1.0,
                'posture': 1.1,
                'confidence': 1.3,
                'charisma': 1.2,
                'overall': 1.5
            }
            
            common_keys = set(scores_a.keys()) & set(scores_b.keys())
            
            if not common_keys:
                return 0.0
            
            weighted_dot_product = 0
            weighted_norm_a = 0
            weighted_norm_b = 0
            
            for key in common_keys:
                weight = weights.get(key, 1.0)
                
                weighted_dot_product += weight * scores_a[key] * scores_b[key]
                weighted_norm_a += weight * (scores_a[key] ** 2)
                weighted_norm_b += weight * (scores_b[key] ** 2)
            
            weighted_norm_a = math.sqrt(weighted_norm_a)
            weighted_norm_b = math.sqrt(weighted_norm_b)
            
            if weighted_norm_a == 0 or weighted_norm_b == 0:
                return 0.0
            
            return weighted_dot_product / (weighted_norm_a * weighted_norm_b)
            
        except Exception as e:
            self.logger.error(f"Weighted similarity calculation failed: {e}")
            return 0.0
    
    def _manhattan_similarity(self, scores_a: Dict[str, float], scores_b: Dict[str, float]) -> float:
        """Calculate Manhattan similarity"""
        try:
            common_keys = set(scores_a.keys()) & set(scores_b.keys())
            
            if not common_keys:
                return 0.0
            
            # Calculate Manhattan distance
            distance = sum(abs(scores_a[key] - scores_b[key]) for key in common_keys)
            
            # Convert to similarity (0-1 scale)
            max_distance = len(common_keys)  # Maximum possible distance
            similarity = 1 - (distance / max_distance)
            
            return max(0.0, similarity)
            
        except Exception as e:
            self.logger.error(f"Manhattan similarity calculation failed: {e}")
            return 0.0
    
    def _compare_score_breakdown(self, scores_a: Dict[str, float], scores_b: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Compare scores breakdown"""
        breakdown = {}
        
        common_keys = set(scores_a.keys()) & set(scores_b.keys())
        
        for key in common_keys:
            breakdown[key] = {
                'your_score': scores_a[key],
                'target_score': scores_b[key],
                'difference': scores_a[key] - scores_b[key],
                'percentage_difference': ((scores_a[key] - scores_b[key]) / scores_b[key] * 100) if scores_b[key] != 0 else 0
            }
        
        return breakdown
    
    def _assess_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of input data"""
        try:
            quality_metrics = {
                'completeness': 0.0,
                'consistency': 0.0,
                'reliability': 0.0,
                'freshness': 0.0
            }
            
            # Assess completeness
            expected_fields = ['individual_results', 'overall_scores', 'confidence_metrics']
            present_fields = sum(1 for field in expected_fields if field in data)
            quality_metrics['completeness'] = present_fields / len(expected_fields)
            
            # Assess consistency (check if scores are in valid ranges)
            scores = self._extract_scores(data)
            valid_scores = sum(1 for score in scores.values() if 0 <= score <= 1)
            quality_metrics['consistency'] = valid_scores / len(scores) if scores else 0
            
            # Assess reliability (based on confidence metrics)
            if 'confidence_metrics' in data:
                confidence_data = data['confidence_metrics']
                if isinstance(confidence_data, dict) and 'average' in confidence_data:
                    quality_metrics['reliability'] = confidence_data['average']
            
            # Assess freshness (mock implementation)
            quality_metrics['freshness'] = 1.0  # Assume data is fresh
            
            overall_quality = sum(quality_metrics.values()) / len(quality_metrics)
            
            return {
                'overall_quality': overall_quality,
                'metrics': quality_metrics,
                'quality_level': 'high' if overall_quality > 0.8 else 'medium' if overall_quality > 0.6 else 'low'
            }
            
        except Exception as e:
            self.logger.error(f"Data quality assessment failed: {e}")
            return {'overall_quality': 0.5, 'metrics': {}, 'quality_level': 'unknown'}
    
    # Additional helper methods for specific comparison types
    
    def _generate_celebrity_insights(self, current_data: Dict[str, Any], celebrity_data: Dict[str, Any], similarity: float) -> List[str]:
        """Generate insights for celebrity comparison"""
        insights = []
        
        try:
            if similarity > 0.8:
                insights.append(f"You have a very high similarity to {celebrity_data['name']} ({similarity:.1%})")
                insights.append(f"You share similar {celebrity_data['attributes']['style']} characteristics")
            elif similarity > 0.6:
                insights.append(f"You have moderate similarity to {celebrity_data['name']} ({similarity:.1%})")
                insights.append("There are several areas where you could align more closely")
            else:
                insights.append(f"You have a unique style different from {celebrity_data['name']}")
                insights.append("This comparison can help identify areas for potential improvement")
            
            # Add specific attribute insights
            current_scores = self._extract_scores(current_data)
            celeb_scores = celebrity_data['scores']
            
            for score_type, celeb_score in celeb_scores.items():
                if score_type in current_scores:
                    current_score = current_scores[score_type]
                    if current_score > celeb_score:
                        insights.append(f"Your {score_type} score exceeds {celebrity_data['name']}'s level")
                    elif current_score < celeb_score * 0.8:
                        insights.append(f"Your {score_type} has significant room for improvement")
            
            return insights
            
        except Exception as e:
            return [f"Insight generation failed: {e}"]
    
    def _generate_celebrity_recommendations(self, current_data: Dict[str, Any], celebrity_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations for celebrity comparison"""
        recommendations = []
        
        try:
            current_scores = self._extract_scores(current_data)
            celeb_scores = celebrity_data['scores']
            
            # Find areas with biggest gaps
            gaps = {}
            for score_type, celeb_score in celeb_scores.items():
                if score_type in current_scores:
                    gap = celeb_score - current_scores[score_type]
                    if gap > 0.1:  # Significant gap
                        gaps[score_type] = gap
            
            # Sort by gap size
            sorted_gaps = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
            
            for score_type, gap in sorted_gaps[:3]:  # Top 3 gaps
                if score_type == 'attractiveness':
                    recommendations.append("Focus on skincare routine and grooming habits")
                elif score_type == 'fashion':
                    recommendations.append(f"Study {celebrity_data['name']}'s style choices and color coordination")
                elif score_type == 'posture':
                    recommendations.append("Work on posture exercises and body alignment")
                elif score_type == 'charisma':
                    recommendations.append("Practice confident body language and communication skills")
            
            # Add style-specific recommendations
            style = celebrity_data['attributes'].get('style', '')
            if 'classic' in style:
                recommendations.append("Invest in timeless, well-fitted clothing pieces")
            elif 'casual' in style:
                recommendations.append("Focus on effortless, comfortable yet stylish looks")
            elif 'elegant' in style:
                recommendations.append("Emphasize sophistication and refined choices")
            
            return recommendations[:5]  # Limit to 5 recommendations
            
        except Exception as e:
            return [f"Recommendation generation failed: {e}"]
    
    def _filter_by_time_period(self, past_data: List[Dict[str, Any]], time_period: str) -> List[Dict[str, Any]]:
        """Filter past data by time period"""
        try:
            if time_period == 'all':
                return past_data
            
            now = datetime.now()
            
            if time_period == 'week':
                cutoff = now - timedelta(weeks=1)
            elif time_period == 'month':
                cutoff = now - timedelta(days=30)
            elif time_period == 'year':
                cutoff = now - timedelta(days=365)
            else:
                return past_data
            
            filtered_data = []
            for data_point in past_data:
                timestamp_str = data_point.get('timestamp', '')
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if timestamp >= cutoff:
                            filtered_data.append(data_point)
                    except ValueError:
                        continue
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Time period filtering failed: {e}")
            return past_data
    
    def _calculate_trends(self, past_data: List[Dict[str, Any]], current_scores: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Calculate trends from historical data"""
        trends = {}
        
        try:
            if len(past_data) < 2:
                return trends
            
            # Extract scores from all data points
            all_scores = [self._extract_scores(data) for data in past_data]
            all_scores.append(current_scores)
            
            # Calculate trends for each score type
            score_types = set()
            for scores in all_scores:
                score_types.update(scores.keys())
            
            for score_type in score_types:
                values = []
                for scores in all_scores:
                    if score_type in scores:
                        values.append(scores[score_type])
                
                if len(values) >= 2:
                    # Calculate trend
                    first_value = values[0]
                    last_value = values[-1]
                    
                    trend_direction = 'improving' if last_value > first_value else 'declining' if last_value < first_value else 'stable'
                    trend_strength = abs(last_value - first_value)
                    
                    # Calculate slope (simple linear trend)
                    n = len(values)
                    x_values = list(range(n))
                    slope = sum((x_values[i] - sum(x_values)/n) * (values[i] - sum(values)/n) for i in range(n))
                    slope /= sum((x - sum(x_values)/n) ** 2 for x in x_values)
                    
                    trends[score_type] = {
                        'direction': trend_direction,
                        'strength': trend_strength,
                        'slope': slope,
                        'first_value': first_value,
                        'last_value': last_value,
                        'data_points': len(values)
                    }
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Trend calculation failed: {e}")
            return {}
    
    def _calculate_improvements(self, past_data: List[Dict[str, Any]], current_scores: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Calculate improvements from historical data"""
        improvements = {}
        
        try:
            if not past_data:
                return improvements
            
            # Get the most recent past data point
            recent_past = past_data[-1]
            past_scores = self._extract_scores(recent_past)
            
            # Calculate improvements for each score
            for score_type in current_scores:
                if score_type in past_scores:
                    past_value = past_scores[score_type]
                    current_value = current_scores[score_type]
                    
                    absolute_change = current_value - past_value
                    percentage_change = (absolute_change / past_value * 100) if past_value != 0 else 0
                    
                    improvement_level = 'significant' if abs(percentage_change) > 10 else 'moderate' if abs(percentage_change) > 5 else 'minimal'
                    
                    improvements[score_type] = {
                        'absolute_change': absolute_change,
                        'percentage_change': percentage_change,
                        'improvement_level': improvement_level,
                        'past_value': past_value,
                        'current_value': current_value
                    }
            
            return improvements
            
        except Exception as e:
            self.logger.error(f"Improvement calculation failed: {e}")
            return {}
    
    def _generate_progress_insights(self, trends: Dict[str, Dict[str, Any]], improvements: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate insights about progress"""
        insights = []
        
        try:
            # Analyze trends
            improving_areas = [area for area, trend in trends.items() if trend['direction'] == 'improving']
            declining_areas = [area for area, trend in trends.items() if trend['direction'] == 'declining']
            
            if improving_areas:
                insights.append(f"You're showing improvement in: {', '.join(improving_areas)}")
            
            if declining_areas:
                insights.append(f"Areas needing attention: {', '.join(declining_areas)}")
            
            # Analyze recent improvements
            significant_improvements = [area for area, imp in improvements.items() if imp['improvement_level'] == 'significant' and imp['absolute_change'] > 0]
            
            if significant_improvements:
                insights.append(f"Recent significant improvements in: {', '.join(significant_improvements)}")
            
            # Overall progress assessment
            total_improvements = sum(1 for imp in improvements.values() if imp['absolute_change'] > 0)
            total_declines = sum(1 for imp in improvements.values() if imp['absolute_change'] < 0)
            
            if total_improvements > total_declines:
                insights.append("Overall, you're making positive progress")
            elif total_declines > total_improvements:
                insights.append("Consider focusing more on consistency and improvement strategies")
            else:
                insights.append("Your progress is stable with room for focused improvement")
            
            return insights
            
        except Exception as e:
            return [f"Progress insight generation failed: {e}"]
    
    def _generate_progress_recommendations(self, trends: Dict[str, Dict[str, Any]], improvements: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on progress"""
        recommendations = []
        
        try:
            # Focus on declining areas
            declining_areas = [area for area, trend in trends.items() if trend['direction'] == 'declining']
            
            for area in declining_areas[:3]:  # Top 3 declining areas
                if area == 'attractiveness':
                    recommendations.append("Revisit your skincare and grooming routine")
                elif area == 'fashion':
                    recommendations.append("Experiment with new styles or seek fashion advice")
                elif area == 'posture':
                    recommendations.append("Incorporate posture exercises into your daily routine")
                elif area == 'confidence':
                    recommendations.append("Practice confidence-building activities")
            
            # Leverage improving areas
            improving_areas = [area for area, trend in trends.items() if trend['direction'] == 'improving']
            
            if improving_areas:
                recommendations.append(f"Continue building on your strengths in {improving_areas[0]}")
            
            # General recommendations
            if len(declining_areas) > len(improving_areas):
                recommendations.append("Consider setting specific, measurable goals for improvement")
                recommendations.append("Track your progress more frequently to identify patterns")
            
            return recommendations[:5]  # Limit to 5 recommendations
            
        except Exception as e:
            return [f"Progress recommendation generation failed: {e}"]
