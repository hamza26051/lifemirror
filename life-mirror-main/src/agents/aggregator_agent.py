import statistics
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentInput, AgentOutput

class AggregatorAgent(BaseAgent):
    """Agent for aggregating and combining results from multiple agents"""
    
    def __init__(self):
        super().__init__()
        self.weight_config = {
            'face': 0.25,
            'fashion': 0.20,
            'posture': 0.15,
            'bio': 0.15,
            'vibe': 0.10,
            'social': 0.10,
            'other': 0.05
        }
        
    def run(self, input: AgentInput) -> AgentOutput:
        """Aggregate results from multiple agents"""
        try:
            agent_results = input.context.get('agent_results', {})
            aggregation_type = input.context.get('aggregation_type', 'comprehensive')
            
            if not agent_results:
                return self._create_output(
                    success=False,
                    data={},
                    error="No agent results provided for aggregation",
                    confidence=0.0
                )
            
            # Perform aggregation based on type
            if aggregation_type == 'comprehensive':
                result = self._comprehensive_aggregation(agent_results)
            elif aggregation_type == 'scoring':
                result = self._scoring_aggregation(agent_results)
            elif aggregation_type == 'confidence':
                result = self._confidence_aggregation(agent_results)
            elif aggregation_type == 'summary':
                result = self._summary_aggregation(agent_results)
            else:
                return self._create_output(
                    success=False,
                    data={},
                    error=f"Unknown aggregation type: {aggregation_type}",
                    confidence=0.0
                )
            
            return self._create_output(
                success=True,
                data=result,
                confidence=result.get('overall_confidence', 0.7)
            )
            
        except Exception as e:
            return self._handle_error(e, input)
    
    def _comprehensive_aggregation(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive aggregation of all agent results"""
        try:
            # Extract individual agent data
            face_data = self._extract_agent_data(agent_results, 'face')
            fashion_data = self._extract_agent_data(agent_results, 'fashion')
            posture_data = self._extract_agent_data(agent_results, 'posture')
            bio_data = self._extract_agent_data(agent_results, 'bio')
            vibe_data = self._extract_agent_data(agent_results, 'vibe')
            social_data = self._extract_agent_data(agent_results, 'social')
            
            # Calculate overall scores
            overall_scores = self._calculate_overall_scores({
                'face': face_data,
                'fashion': fashion_data,
                'posture': posture_data,
                'bio': bio_data,
                'vibe': vibe_data,
                'social': social_data
            })
            
            # Generate comprehensive insights
            insights = self._generate_comprehensive_insights({
                'face': face_data,
                'fashion': fashion_data,
                'posture': posture_data,
                'bio': bio_data,
                'vibe': vibe_data,
                'social': social_data
            })
            
            # Calculate confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(agent_results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations({
                'face': face_data,
                'fashion': fashion_data,
                'posture': posture_data,
                'bio': bio_data,
                'vibe': vibe_data,
                'social': social_data
            })
            
            # Create personality profile
            personality_profile = self._create_personality_profile({
                'face': face_data,
                'bio': bio_data,
                'vibe': vibe_data,
                'social': social_data
            })
            
            return {
                'aggregation_type': 'comprehensive',
                'overall_scores': overall_scores,
                'insights': insights,
                'confidence_metrics': confidence_metrics,
                'recommendations': recommendations,
                'personality_profile': personality_profile,
                'individual_results': {
                    'face': face_data,
                    'fashion': fashion_data,
                    'posture': posture_data,
                    'bio': bio_data,
                    'vibe': vibe_data,
                    'social': social_data
                },
                'overall_confidence': confidence_metrics.get('weighted_average', 0.7),
                'analysis_completeness': self._calculate_completeness(agent_results)
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive aggregation failed: {e}")
            return {
                'aggregation_type': 'comprehensive',
                'error': str(e),
                'overall_confidence': 0.0
            }
    
    def _scoring_aggregation(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Focus on scoring and numerical analysis"""
        try:
            scores = {}
            confidences = {}
            
            # Extract scores from each agent
            for agent_name, result in agent_results.items():
                if isinstance(result, dict) and result.get('success', False):
                    data = result.get('data', {})
                    
                    # Extract various score types
                    if 'score' in data:
                        scores[f'{agent_name}_score'] = data['score']
                    if 'confidence_score' in data:
                        scores[f'{agent_name}_confidence'] = data['confidence_score']
                    if 'attractiveness_score' in data:
                        scores[f'{agent_name}_attractiveness'] = data['attractiveness_score']
                    if 'fashion_score' in data:
                        scores[f'{agent_name}_fashion'] = data['fashion_score']
                    if 'posture_score' in data:
                        scores[f'{agent_name}_posture'] = data['posture_score']
                    if 'vibe_score' in data:
                        scores[f'{agent_name}_vibe'] = data['vibe_score']
                    
                    # Extract confidence
                    confidences[agent_name] = result.get('confidence', 0.5)
            
            # Calculate aggregate scores
            aggregate_scores = self._calculate_weighted_scores(scores, confidences)
            
            # Generate score analysis
            score_analysis = self._analyze_scores(scores, aggregate_scores)
            
            return {
                'aggregation_type': 'scoring',
                'individual_scores': scores,
                'aggregate_scores': aggregate_scores,
                'score_analysis': score_analysis,
                'confidence_weights': confidences,
                'overall_confidence': statistics.mean(confidences.values()) if confidences else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Scoring aggregation failed: {e}")
            return {
                'aggregation_type': 'scoring',
                'error': str(e),
                'overall_confidence': 0.0
            }
    
    def _confidence_aggregation(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Focus on confidence analysis and reliability"""
        try:
            confidence_data = {}
            
            for agent_name, result in agent_results.items():
                if isinstance(result, dict):
                    confidence_data[agent_name] = {
                        'success': result.get('success', False),
                        'confidence': result.get('confidence', 0.0),
                        'has_data': bool(result.get('data')),
                        'error': result.get('error'),
                        'data_quality': self._assess_data_quality(result.get('data', {}))
                    }
            
            # Calculate overall confidence metrics
            successful_agents = [name for name, data in confidence_data.items() if data['success']]
            failed_agents = [name for name, data in confidence_data.items() if not data['success']]
            
            confidence_scores = [data['confidence'] for data in confidence_data.values() if data['success']]
            
            overall_metrics = {
                'success_rate': len(successful_agents) / len(confidence_data) if confidence_data else 0,
                'average_confidence': statistics.mean(confidence_scores) if confidence_scores else 0,
                'confidence_std': statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0,
                'min_confidence': min(confidence_scores) if confidence_scores else 0,
                'max_confidence': max(confidence_scores) if confidence_scores else 0,
                'successful_agents': successful_agents,
                'failed_agents': failed_agents
            }
            
            # Generate reliability assessment
            reliability_assessment = self._assess_reliability(confidence_data, overall_metrics)
            
            return {
                'aggregation_type': 'confidence',
                'agent_confidence': confidence_data,
                'overall_metrics': overall_metrics,
                'reliability_assessment': reliability_assessment,
                'overall_confidence': overall_metrics['average_confidence']
            }
            
        except Exception as e:
            self.logger.error(f"Confidence aggregation failed: {e}")
            return {
                'aggregation_type': 'confidence',
                'error': str(e),
                'overall_confidence': 0.0
            }
    
    def _summary_aggregation(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a concise summary of all results"""
        try:
            summary = {
                'total_agents': len(agent_results),
                'successful_agents': 0,
                'failed_agents': 0,
                'key_findings': [],
                'main_scores': {},
                'top_insights': [],
                'critical_issues': []
            }
            
            for agent_name, result in agent_results.items():
                if isinstance(result, dict):
                    if result.get('success', False):
                        summary['successful_agents'] += 1
                        
                        # Extract key findings
                        data = result.get('data', {})
                        
                        # Extract main score
                        main_score = self._extract_main_score(data)
                        if main_score is not None:
                            summary['main_scores'][agent_name] = main_score
                        
                        # Extract insights
                        insights = data.get('insights', [])
                        if insights:
                            summary['key_findings'].extend(insights[:2])  # Top 2 insights per agent
                        
                        # Check for critical issues
                        if main_score is not None and main_score < 0.3:
                            summary['critical_issues'].append(f"Low {agent_name} score: {main_score:.2f}")
                    else:
                        summary['failed_agents'] += 1
                        error = result.get('error', 'Unknown error')
                        summary['critical_issues'].append(f"{agent_name} failed: {error}")
            
            # Generate top insights (remove duplicates and limit)
            summary['top_insights'] = list(set(summary['key_findings']))[:5]
            
            # Calculate overall summary score
            if summary['main_scores']:
                summary['overall_score'] = statistics.mean(summary['main_scores'].values())
            else:
                summary['overall_score'] = 0.0
            
            # Generate summary text
            summary['summary_text'] = self._generate_summary_text(summary)
            
            return {
                'aggregation_type': 'summary',
                'summary': summary,
                'overall_confidence': summary['successful_agents'] / summary['total_agents'] if summary['total_agents'] > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Summary aggregation failed: {e}")
            return {
                'aggregation_type': 'summary',
                'error': str(e),
                'overall_confidence': 0.0
            }
    
    def _extract_agent_data(self, agent_results: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """Extract and normalize data from a specific agent"""
        result = agent_results.get(agent_name, {})
        
        if not isinstance(result, dict) or not result.get('success', False):
            return {
                'available': False,
                'error': result.get('error', 'Agent failed or not available'),
                'confidence': 0.0
            }
        
        data = result.get('data', {})
        confidence = result.get('confidence', 0.5)
        
        return {
            'available': True,
            'data': data,
            'confidence': confidence,
            'raw_result': result
        }
    
    def _calculate_overall_scores(self, agent_data: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate weighted overall scores"""
        scores = {}
        total_weight = 0
        weighted_sum = 0
        
        for agent_name, data in agent_data.items():
            if data.get('available', False):
                agent_weight = self.weight_config.get(agent_name, self.weight_config['other'])
                agent_score = self._extract_main_score(data.get('data', {}))
                
                if agent_score is not None:
                    scores[f'{agent_name}_score'] = agent_score
                    weighted_sum += agent_score * agent_weight
                    total_weight += agent_weight
        
        # Calculate overall weighted score
        if total_weight > 0:
            scores['overall_weighted_score'] = weighted_sum / total_weight
        else:
            scores['overall_weighted_score'] = 0.0
        
        # Calculate simple average
        individual_scores = [score for key, score in scores.items() if key != 'overall_weighted_score']
        if individual_scores:
            scores['overall_average_score'] = statistics.mean(individual_scores)
        else:
            scores['overall_average_score'] = 0.0
        
        return scores
    
    def _extract_main_score(self, data: Dict[str, Any]) -> Optional[float]:
        """Extract the main score from agent data"""
        # Try different score field names
        score_fields = [
            'score', 'overall_score', 'confidence_score', 'attractiveness_score',
            'fashion_score', 'posture_score', 'vibe_score', 'main_score'
        ]
        
        for field in score_fields:
            if field in data and isinstance(data[field], (int, float)):
                return float(data[field])
        
        return None
    
    def _generate_comprehensive_insights(self, agent_data: Dict[str, Dict]) -> List[str]:
        """Generate comprehensive insights from all agents"""
        insights = []
        
        # Collect insights from each agent
        for agent_name, data in agent_data.items():
            if data.get('available', False):
                agent_insights = data.get('data', {}).get('insights', [])
                if isinstance(agent_insights, list):
                    insights.extend(agent_insights)
        
        # Add cross-agent insights
        cross_insights = self._generate_cross_agent_insights(agent_data)
        insights.extend(cross_insights)
        
        # Remove duplicates and limit
        unique_insights = list(set(insights))
        return unique_insights[:10]  # Top 10 insights
    
    def _generate_cross_agent_insights(self, agent_data: Dict[str, Dict]) -> List[str]:
        """Generate insights that combine information from multiple agents"""
        insights = []
        
        # Check for consistency across agents
        face_available = agent_data.get('face', {}).get('available', False)
        fashion_available = agent_data.get('fashion', {}).get('available', False)
        posture_available = agent_data.get('posture', {}).get('available', False)
        bio_available = agent_data.get('bio', {}).get('available', False)
        
        if face_available and fashion_available:
            insights.append("Comprehensive visual analysis completed with both facial and fashion assessment")
        
        if posture_available and face_available:
            insights.append("Full body and facial analysis provides complete physical assessment")
        
        if bio_available and face_available:
            insights.append("Personality analysis combined with facial features for deeper insights")
        
        # Check for score consistency
        scores = []
        for agent_name, data in agent_data.items():
            if data.get('available', False):
                score = self._extract_main_score(data.get('data', {}))
                if score is not None:
                    scores.append(score)
        
        if len(scores) >= 2:
            score_std = statistics.stdev(scores)
            if score_std < 0.1:
                insights.append("Consistent scores across all analysis dimensions")
            elif score_std > 0.3:
                insights.append("Varied performance across different analysis areas")
        
        return insights
    
    def _calculate_confidence_metrics(self, agent_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various confidence metrics"""
        confidences = []
        
        for result in agent_results.values():
            if isinstance(result, dict) and result.get('success', False):
                confidences.append(result.get('confidence', 0.5))
        
        if not confidences:
            return {
                'average': 0.0,
                'weighted_average': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0
            }
        
        return {
            'average': statistics.mean(confidences),
            'weighted_average': statistics.mean(confidences),  # Could implement actual weighting
            'min': min(confidences),
            'max': max(confidences),
            'std': statistics.stdev(confidences) if len(confidences) > 1 else 0.0
        }
    
    def _generate_recommendations(self, agent_data: Dict[str, Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Collect recommendations from each agent
        for agent_name, data in agent_data.items():
            if data.get('available', False):
                agent_recommendations = data.get('data', {}).get('recommendations', [])
                if isinstance(agent_recommendations, list):
                    recommendations.extend(agent_recommendations)
        
        # Add general recommendations based on scores
        scores = {}
        for agent_name, data in agent_data.items():
            if data.get('available', False):
                score = self._extract_main_score(data.get('data', {}))
                if score is not None:
                    scores[agent_name] = score
        
        # Generate recommendations for low scores
        for agent_name, score in scores.items():
            if score < 0.4:
                recommendations.append(f"Focus on improving {agent_name} aspects for better overall results")
        
        # Remove duplicates and limit
        unique_recommendations = list(set(recommendations))
        return unique_recommendations[:8]  # Top 8 recommendations
    
    def _create_personality_profile(self, agent_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Create a comprehensive personality profile"""
        profile = {
            'traits': [],
            'dominant_characteristics': [],
            'communication_style': 'unknown',
            'social_tendencies': 'unknown',
            'confidence_level': 'medium'
        }
        
        # Extract from bio agent
        bio_data = agent_data.get('bio', {})
        if bio_data.get('available', False):
            bio_info = bio_data.get('data', {})
            
            personality_traits = bio_info.get('personality_traits', {})
            if isinstance(personality_traits, dict):
                dominant_traits = personality_traits.get('dominant_traits', [])
                profile['traits'].extend(dominant_traits)
            
            communication_style = bio_info.get('communication_style', {})
            if isinstance(communication_style, dict):
                profile['communication_style'] = communication_style.get('style', 'unknown')
        
        # Extract from face agent
        face_data = agent_data.get('face', {})
        if face_data.get('available', False):
            face_info = face_data.get('data', {})
            
            # Add facial expression insights
            emotion = face_info.get('emotion', {})
            if isinstance(emotion, dict):
                dominant_emotion = emotion.get('dominant_emotion')
                if dominant_emotion:
                    profile['dominant_characteristics'].append(f"Often appears {dominant_emotion}")
        
        # Extract from vibe agent
        vibe_data = agent_data.get('vibe', {})
        if vibe_data.get('available', False):
            vibe_info = vibe_data.get('data', {})
            vibe_score = vibe_info.get('vibe_score', 0.5)
            
            if vibe_score > 0.7:
                profile['confidence_level'] = 'high'
            elif vibe_score < 0.3:
                profile['confidence_level'] = 'low'
        
        return profile
    
    def _calculate_completeness(self, agent_results: Dict[str, Any]) -> float:
        """Calculate how complete the analysis is"""
        total_agents = len(agent_results)
        successful_agents = sum(1 for result in agent_results.values() 
                              if isinstance(result, dict) and result.get('success', False))
        
        return successful_agents / total_agents if total_agents > 0 else 0.0
    
    def _calculate_weighted_scores(self, scores: Dict[str, float], confidences: Dict[str, float]) -> Dict[str, float]:
        """Calculate weighted aggregate scores"""
        if not scores:
            return {}
        
        # Simple average
        aggregate = {
            'simple_average': statistics.mean(scores.values())
        }
        
        # Confidence-weighted average
        if confidences:
            weighted_sum = 0
            total_weight = 0
            
            for score_name, score_value in scores.items():
                # Try to match score name to agent name
                agent_name = score_name.split('_')[0]
                weight = confidences.get(agent_name, 0.5)
                
                weighted_sum += score_value * weight
                total_weight += weight
            
            if total_weight > 0:
                aggregate['confidence_weighted'] = weighted_sum / total_weight
            else:
                aggregate['confidence_weighted'] = aggregate['simple_average']
        
        return aggregate
    
    def _analyze_scores(self, individual_scores: Dict[str, float], aggregate_scores: Dict[str, float]) -> Dict[str, Any]:
        """Analyze score patterns and distributions"""
        if not individual_scores:
            return {}
        
        scores_list = list(individual_scores.values())
        
        analysis = {
            'distribution': {
                'min': min(scores_list),
                'max': max(scores_list),
                'mean': statistics.mean(scores_list),
                'std': statistics.stdev(scores_list) if len(scores_list) > 1 else 0.0
            },
            'consistency': 'high' if statistics.stdev(scores_list) < 0.1 else 'medium' if statistics.stdev(scores_list) < 0.3 else 'low',
            'overall_performance': 'excellent' if statistics.mean(scores_list) > 0.8 else 'good' if statistics.mean(scores_list) > 0.6 else 'average' if statistics.mean(scores_list) > 0.4 else 'needs_improvement'
        }
        
        return analysis
    
    def _assess_data_quality(self, data: Dict[str, Any]) -> str:
        """Assess the quality of agent data"""
        if not data:
            return 'no_data'
        
        # Count non-empty fields
        non_empty_fields = sum(1 for value in data.values() if value not in [None, '', [], {}])
        total_fields = len(data)
        
        if total_fields == 0:
            return 'no_data'
        
        completeness = non_empty_fields / total_fields
        
        if completeness > 0.8:
            return 'high'
        elif completeness > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _assess_reliability(self, confidence_data: Dict[str, Dict], overall_metrics: Dict[str, Any]) -> str:
        """Assess overall reliability of the analysis"""
        success_rate = overall_metrics.get('success_rate', 0)
        avg_confidence = overall_metrics.get('average_confidence', 0)
        
        if success_rate > 0.8 and avg_confidence > 0.7:
            return 'high'
        elif success_rate > 0.6 and avg_confidence > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _generate_summary_text(self, summary: Dict[str, Any]) -> str:
        """Generate a human-readable summary text"""
        total = summary['total_agents']
        successful = summary['successful_agents']
        overall_score = summary.get('overall_score', 0)
        
        text = f"Analysis completed with {successful}/{total} agents successful. "
        
        if overall_score > 0.7:
            text += "Overall performance is excellent. "
        elif overall_score > 0.5:
            text += "Overall performance is good. "
        else:
            text += "Overall performance needs improvement. "
        
        if summary['critical_issues']:
            text += f"Found {len(summary['critical_issues'])} critical issues to address."
        else:
            text += "No critical issues detected."
        
        return text
