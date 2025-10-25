import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from .base_agent import BaseAgent, AgentInput, AgentOutput

class FormatterAgent(BaseAgent):
    """Agent for formatting and presenting analysis results"""
    
    def __init__(self):
        super().__init__()
        self.format_templates = {
            'json': self._format_json,
            'summary': self._format_summary,
            'detailed': self._format_detailed,
            'report': self._format_report,
            'api': self._format_api_response,
            'user_friendly': self._format_user_friendly
        }
        
    def run(self, input: AgentInput) -> AgentOutput:
        """Format analysis results according to specified format"""
        try:
            data_to_format = input.context.get('data', {})
            format_type = input.context.get('format_type', 'summary')
            include_metadata = input.context.get('include_metadata', True)
            
            if not data_to_format:
                return self._create_output(
                    success=False,
                    data={},
                    error="No data provided for formatting",
                    confidence=0.0
                )
            
            # Get formatter function
            formatter = self.format_templates.get(format_type)
            if not formatter:
                return self._create_output(
                    success=False,
                    data={},
                    error=f"Unknown format type: {format_type}",
                    confidence=0.0
                )
            
            # Format the data
            formatted_result = formatter(data_to_format, include_metadata)
            
            return self._create_output(
                success=True,
                data={
                    'formatted_output': formatted_result,
                    'format_type': format_type,
                    'timestamp': datetime.now().isoformat(),
                    'metadata': {
                        'original_data_size': len(str(data_to_format)),
                        'formatted_size': len(str(formatted_result)),
                        'include_metadata': include_metadata
                    }
                },
                confidence=0.9
            )
            
        except Exception as e:
            return self._handle_error(e, input)
    
    def _format_json(self, data: Dict[str, Any], include_metadata: bool = True) -> Dict[str, Any]:
        """Format as clean JSON structure"""
        try:
            formatted = {
                'analysis_results': data,
                'format': 'json'
            }
            
            if include_metadata:
                formatted['metadata'] = {
                    'timestamp': datetime.now().isoformat(),
                    'format_version': '1.0',
                    'data_structure': self._analyze_data_structure(data)
                }
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"JSON formatting failed: {e}")
            return {'error': str(e), 'format': 'json'}
    
    def _format_summary(self, data: Dict[str, Any], include_metadata: bool = True) -> Dict[str, Any]:
        """Format as concise summary"""
        try:
            summary = {
                'format': 'summary',
                'overview': self._extract_overview(data),
                'key_scores': self._extract_key_scores(data),
                'main_insights': self._extract_main_insights(data),
                'recommendations': self._extract_recommendations(data)
            }
            
            if include_metadata:
                summary['metadata'] = {
                    'summary_generated': datetime.now().isoformat(),
                    'data_sources': self._identify_data_sources(data),
                    'confidence_level': self._calculate_overall_confidence(data)
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Summary formatting failed: {e}")
            return {'error': str(e), 'format': 'summary'}
    
    def _format_detailed(self, data: Dict[str, Any], include_metadata: bool = True) -> Dict[str, Any]:
        """Format as detailed analysis report"""
        try:
            detailed = {
                'format': 'detailed',
                'executive_summary': self._create_executive_summary(data),
                'detailed_analysis': {
                    'facial_analysis': self._format_facial_analysis(data),
                    'fashion_analysis': self._format_fashion_analysis(data),
                    'posture_analysis': self._format_posture_analysis(data),
                    'personality_analysis': self._format_personality_analysis(data),
                    'overall_assessment': self._format_overall_assessment(data)
                },
                'scoring_breakdown': self._create_scoring_breakdown(data),
                'improvement_areas': self._identify_improvement_areas(data),
                'strengths': self._identify_strengths(data)
            }
            
            if include_metadata:
                detailed['metadata'] = {
                    'report_generated': datetime.now().isoformat(),
                    'analysis_depth': 'detailed',
                    'sections_included': list(detailed['detailed_analysis'].keys()),
                    'data_completeness': self._assess_data_completeness(data)
                }
            
            return detailed
            
        except Exception as e:
            self.logger.error(f"Detailed formatting failed: {e}")
            return {'error': str(e), 'format': 'detailed'}
    
    def _format_report(self, data: Dict[str, Any], include_metadata: bool = True) -> Dict[str, Any]:
        """Format as professional report"""
        try:
            report = {
                'format': 'report',
                'title': 'Life Mirror Analysis Report',
                'report_sections': {
                    'introduction': self._create_introduction(data),
                    'methodology': self._create_methodology_section(data),
                    'findings': self._create_findings_section(data),
                    'analysis': self._create_analysis_section(data),
                    'conclusions': self._create_conclusions_section(data),
                    'recommendations': self._create_recommendations_section(data)
                },
                'appendices': {
                    'raw_scores': self._extract_all_scores(data),
                    'confidence_metrics': self._extract_confidence_metrics(data),
                    'technical_details': self._extract_technical_details(data)
                }
            }
            
            if include_metadata:
                report['metadata'] = {
                    'report_date': datetime.now().isoformat(),
                    'report_type': 'comprehensive_analysis',
                    'version': '1.0',
                    'page_count': len(report['report_sections']),
                    'data_sources': self._identify_data_sources(data)
                }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Report formatting failed: {e}")
            return {'error': str(e), 'format': 'report'}
    
    def _format_api_response(self, data: Dict[str, Any], include_metadata: bool = True) -> Dict[str, Any]:
        """Format for API consumption"""
        try:
            api_response = {
                'status': 'success',
                'data': {
                    'scores': self._extract_api_scores(data),
                    'insights': self._extract_api_insights(data),
                    'recommendations': self._extract_api_recommendations(data),
                    'confidence': self._calculate_overall_confidence(data)
                },
                'format': 'api'
            }
            
            if include_metadata:
                api_response['metadata'] = {
                    'timestamp': datetime.now().isoformat(),
                    'api_version': '1.0',
                    'response_time_ms': 0,  # Would be calculated by API layer
                    'data_freshness': 'real_time'
                }
            
            return api_response
            
        except Exception as e:
            self.logger.error(f"API formatting failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'format': 'api'
            }
    
    def _format_user_friendly(self, data: Dict[str, Any], include_metadata: bool = True) -> Dict[str, Any]:
        """Format for end-user consumption"""
        try:
            user_friendly = {
                'format': 'user_friendly',
                'your_results': {
                    'overall_score': self._get_simple_overall_score(data),
                    'key_strengths': self._get_simple_strengths(data),
                    'areas_to_improve': self._get_simple_improvements(data),
                    'personality_highlights': self._get_personality_highlights(data)
                },
                'detailed_breakdown': {
                    'appearance': self._get_appearance_summary(data),
                    'style': self._get_style_summary(data),
                    'posture': self._get_posture_summary(data),
                    'personality': self._get_personality_summary(data)
                },
                'next_steps': self._get_actionable_next_steps(data),
                'encouragement': self._generate_encouragement(data)
            }
            
            if include_metadata:
                user_friendly['about_this_analysis'] = {
                    'analysis_date': datetime.now().strftime('%B %d, %Y'),
                    'confidence_level': self._get_confidence_description(data),
                    'analysis_completeness': self._get_completeness_description(data)
                }
            
            return user_friendly
            
        except Exception as e:
            self.logger.error(f"User-friendly formatting failed: {e}")
            return {'error': str(e), 'format': 'user_friendly'}
    
    # Helper methods for data extraction and analysis
    
    def _extract_overview(self, data: Dict[str, Any]) -> str:
        """Extract high-level overview"""
        try:
            # Look for aggregated results
            if 'aggregation_type' in data and data.get('aggregation_type') == 'summary':
                summary_data = data.get('summary', {})
                return summary_data.get('summary_text', 'Analysis completed successfully')
            
            # Fallback to general overview
            successful_analyses = []
            if 'individual_results' in data:
                for agent_name, result in data['individual_results'].items():
                    if isinstance(result, dict) and result.get('available', False):
                        successful_analyses.append(agent_name)
            
            if successful_analyses:
                return f"Comprehensive analysis completed covering {', '.join(successful_analyses)} aspects."
            else:
                return "Analysis completed with available data."
                
        except Exception as e:
            return f"Overview generation failed: {e}"
    
    def _extract_key_scores(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key numerical scores"""
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
                        main_score = self._find_main_score(agent_data)
                        if main_score is not None:
                            scores[f'{agent_name}_score'] = main_score
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Score extraction failed: {e}")
            return {}
    
    def _extract_main_insights(self, data: Dict[str, Any]) -> List[str]:
        """Extract main insights"""
        insights = []
        
        try:
            # Look for aggregated insights
            if 'insights' in data and isinstance(data['insights'], list):
                insights.extend(data['insights'][:5])  # Top 5 insights
            
            # Look for individual agent insights
            if 'individual_results' in data:
                for agent_name, result in data['individual_results'].items():
                    if isinstance(result, dict) and result.get('available', False):
                        agent_data = result.get('data', {})
                        agent_insights = agent_data.get('insights', [])
                        if isinstance(agent_insights, list):
                            insights.extend(agent_insights[:2])  # Top 2 per agent
            
            # Remove duplicates and limit
            unique_insights = list(set(insights))
            return unique_insights[:8]
            
        except Exception as e:
            self.logger.error(f"Insights extraction failed: {e}")
            return []
    
    def _extract_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Extract recommendations"""
        recommendations = []
        
        try:
            # Look for aggregated recommendations
            if 'recommendations' in data and isinstance(data['recommendations'], list):
                recommendations.extend(data['recommendations'])
            
            # Look for individual agent recommendations
            if 'individual_results' in data:
                for agent_name, result in data['individual_results'].items():
                    if isinstance(result, dict) and result.get('available', False):
                        agent_data = result.get('data', {})
                        agent_recs = agent_data.get('recommendations', [])
                        if isinstance(agent_recs, list):
                            recommendations.extend(agent_recs)
            
            # Remove duplicates and limit
            unique_recommendations = list(set(recommendations))
            return unique_recommendations[:6]
            
        except Exception as e:
            self.logger.error(f"Recommendations extraction failed: {e}")
            return []
    
    def _find_main_score(self, data: Dict[str, Any]) -> Optional[float]:
        """Find the main score in agent data"""
        score_fields = [
            'score', 'overall_score', 'confidence_score', 'attractiveness_score',
            'fashion_score', 'posture_score', 'vibe_score', 'main_score'
        ]
        
        for field in score_fields:
            if field in data and isinstance(data[field], (int, float)):
                return float(data[field])
        
        return None
    
    def _calculate_overall_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        try:
            # Look for confidence metrics
            if 'confidence_metrics' in data:
                metrics = data['confidence_metrics']
                if isinstance(metrics, dict):
                    return metrics.get('average', 0.5)
            
            # Look for overall confidence
            if 'overall_confidence' in data:
                return data['overall_confidence']
            
            # Calculate from individual results
            confidences = []
            if 'individual_results' in data:
                for result in data['individual_results'].values():
                    if isinstance(result, dict) and result.get('available', False):
                        confidences.append(result.get('confidence', 0.5))
            
            if confidences:
                return sum(confidences) / len(confidences)
            
            return 0.5  # Default
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _analyze_data_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the structure of the data"""
        try:
            structure = {
                'total_keys': len(data),
                'nested_objects': 0,
                'arrays': 0,
                'primitive_values': 0,
                'data_types': {}
            }
            
            for key, value in data.items():
                value_type = type(value).__name__
                structure['data_types'][key] = value_type
                
                if isinstance(value, dict):
                    structure['nested_objects'] += 1
                elif isinstance(value, list):
                    structure['arrays'] += 1
                else:
                    structure['primitive_values'] += 1
            
            return structure
            
        except Exception as e:
            return {'error': str(e)}
    
    def _identify_data_sources(self, data: Dict[str, Any]) -> List[str]:
        """Identify data sources from the analysis"""
        sources = []
        
        try:
            if 'individual_results' in data:
                for agent_name, result in data['individual_results'].items():
                    if isinstance(result, dict) and result.get('available', False):
                        sources.append(agent_name)
            
            return sources
            
        except Exception as e:
            return []
    
    def _create_executive_summary(self, data: Dict[str, Any]) -> str:
        """Create executive summary"""
        try:
            overview = self._extract_overview(data)
            key_scores = self._extract_key_scores(data)
            
            if key_scores:
                avg_score = sum(key_scores.values()) / len(key_scores)
                performance = "excellent" if avg_score > 0.8 else "good" if avg_score > 0.6 else "average"
                return f"{overview} Overall performance is {performance} with an average score of {avg_score:.2f}."
            else:
                return overview
                
        except Exception as e:
            return f"Executive summary generation failed: {e}"
    
    def _format_facial_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format facial analysis section"""
        try:
            face_data = data.get('individual_results', {}).get('face', {})
            
            if not face_data.get('available', False):
                return {'status': 'not_available', 'reason': 'Facial analysis not performed'}
            
            face_info = face_data.get('data', {})
            
            return {
                'status': 'available',
                'attractiveness_score': face_info.get('attractiveness_score', 0),
                'confidence_score': face_info.get('confidence_score', 0),
                'detected_features': face_info.get('face_attributes', {}),
                'emotion_analysis': face_info.get('emotion', {}),
                'age_estimation': face_info.get('age', {}),
                'insights': face_info.get('insights', [])
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _format_fashion_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format fashion analysis section"""
        try:
            fashion_data = data.get('individual_results', {}).get('fashion', {})
            
            if not fashion_data.get('available', False):
                return {'status': 'not_available', 'reason': 'Fashion analysis not performed'}
            
            fashion_info = fashion_data.get('data', {})
            
            return {
                'status': 'available',
                'fashion_score': fashion_info.get('fashion_score', 0),
                'style_assessment': fashion_info.get('style_assessment', {}),
                'color_analysis': fashion_info.get('color_analysis', {}),
                'clothing_items': fashion_info.get('clothing_items', []),
                'recommendations': fashion_info.get('recommendations', [])
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _format_posture_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format posture analysis section"""
        try:
            posture_data = data.get('individual_results', {}).get('posture', {})
            
            if not posture_data.get('available', False):
                return {'status': 'not_available', 'reason': 'Posture analysis not performed'}
            
            posture_info = posture_data.get('data', {})
            
            return {
                'status': 'available',
                'posture_score': posture_info.get('posture_score', 0),
                'alignment_analysis': posture_info.get('alignment_analysis', {}),
                'detected_issues': posture_info.get('detected_issues', []),
                'recommendations': posture_info.get('recommendations', [])
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _format_personality_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format personality analysis section"""
        try:
            bio_data = data.get('individual_results', {}).get('bio', {})
            
            if not bio_data.get('available', False):
                return {'status': 'not_available', 'reason': 'Personality analysis not performed'}
            
            bio_info = bio_data.get('data', {})
            
            return {
                'status': 'available',
                'personality_traits': bio_info.get('personality_traits', {}),
                'communication_style': bio_info.get('communication_style', {}),
                'interests': bio_info.get('interests', []),
                'vibe_score': bio_info.get('vibe_score', 0),
                'insights': bio_info.get('insights', [])
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _format_overall_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format overall assessment section"""
        try:
            overall_scores = data.get('overall_scores', {})
            confidence_metrics = data.get('confidence_metrics', {})
            
            return {
                'overall_score': overall_scores.get('overall_weighted_score', 0),
                'confidence_level': confidence_metrics.get('average', 0),
                'analysis_completeness': data.get('analysis_completeness', 0),
                'key_strengths': self._identify_strengths(data),
                'improvement_areas': self._identify_improvement_areas(data)
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _identify_strengths(self, data: Dict[str, Any]) -> List[str]:
        """Identify key strengths"""
        strengths = []
        
        try:
            # Look at individual scores
            if 'individual_results' in data:
                for agent_name, result in data['individual_results'].items():
                    if isinstance(result, dict) and result.get('available', False):
                        agent_data = result.get('data', {})
                        score = self._find_main_score(agent_data)
                        
                        if score is not None and score > 0.7:
                            strengths.append(f"Strong {agent_name} performance")
            
            # Add general strengths
            overall_confidence = self._calculate_overall_confidence(data)
            if overall_confidence > 0.8:
                strengths.append("High analysis confidence")
            
            return strengths[:5]  # Top 5 strengths
            
        except Exception as e:
            return []
    
    def _identify_improvement_areas(self, data: Dict[str, Any]) -> List[str]:
        """Identify areas for improvement"""
        improvements = []
        
        try:
            # Look at individual scores
            if 'individual_results' in data:
                for agent_name, result in data['individual_results'].items():
                    if isinstance(result, dict) and result.get('available', False):
                        agent_data = result.get('data', {})
                        score = self._find_main_score(agent_data)
                        
                        if score is not None and score < 0.5:
                            improvements.append(f"Enhance {agent_name} aspects")
            
            return improvements[:5]  # Top 5 improvement areas
            
        except Exception as e:
            return []
    
    def _get_simple_overall_score(self, data: Dict[str, Any]) -> str:
        """Get simple overall score description"""
        try:
            overall_scores = data.get('overall_scores', {})
            score = overall_scores.get('overall_weighted_score', 0)
            
            if score > 0.8:
                return f"Excellent ({score:.1f}/1.0)"
            elif score > 0.6:
                return f"Good ({score:.1f}/1.0)"
            elif score > 0.4:
                return f"Average ({score:.1f}/1.0)"
            else:
                return f"Needs Improvement ({score:.1f}/1.0)"
                
        except Exception as e:
            return "Score unavailable"
    
    def _get_confidence_description(self, data: Dict[str, Any]) -> str:
        """Get confidence level description"""
        try:
            confidence = self._calculate_overall_confidence(data)
            
            if confidence > 0.8:
                return "Very High"
            elif confidence > 0.6:
                return "High"
            elif confidence > 0.4:
                return "Medium"
            else:
                return "Low"
                
        except Exception as e:
            return "Unknown"
    
    def _generate_encouragement(self, data: Dict[str, Any]) -> str:
        """Generate encouraging message"""
        try:
            overall_scores = data.get('overall_scores', {})
            score = overall_scores.get('overall_weighted_score', 0.5)
            
            if score > 0.7:
                return "You're doing great! Keep up the excellent work and continue building on your strengths."
            elif score > 0.5:
                return "You have a solid foundation with room for growth. Focus on the improvement areas to reach your full potential."
            else:
                return "Everyone starts somewhere! Use these insights as stepping stones toward positive change and growth."
                
        except Exception as e:
            return "Remember, every analysis is an opportunity for growth and self-improvement!"
    
    # Additional helper methods for comprehensive formatting
    
    def _create_scoring_breakdown(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed scoring breakdown"""
        try:
            breakdown = {
                'individual_scores': {},
                'weighted_scores': {},
                'confidence_scores': {},
                'score_distribution': {}
            }
            
            if 'individual_results' in data:
                for agent_name, result in data['individual_results'].items():
                    if isinstance(result, dict) and result.get('available', False):
                        agent_data = result.get('data', {})
                        score = self._find_main_score(agent_data)
                        confidence = result.get('confidence', 0.5)
                        
                        if score is not None:
                            breakdown['individual_scores'][agent_name] = score
                            breakdown['confidence_scores'][agent_name] = confidence
            
            # Add overall scores
            if 'overall_scores' in data:
                breakdown['weighted_scores'] = data['overall_scores']
            
            return breakdown
            
        except Exception as e:
            return {'error': str(e)}
    
    def _assess_data_completeness(self, data: Dict[str, Any]) -> float:
        """Assess how complete the data is"""
        try:
            if 'analysis_completeness' in data:
                return data['analysis_completeness']
            
            # Calculate based on available agents
            total_expected = 6  # Expected number of core agents
            available_count = 0
            
            if 'individual_results' in data:
                for result in data['individual_results'].values():
                    if isinstance(result, dict) and result.get('available', False):
                        available_count += 1
            
            return available_count / total_expected
            
        except Exception as e:
            return 0.0
    
    def _extract_all_scores(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all available scores"""
        all_scores = {}
        
        try:
            # Individual agent scores
            if 'individual_results' in data:
                for agent_name, result in data['individual_results'].items():
                    if isinstance(result, dict) and result.get('available', False):
                        agent_data = result.get('data', {})
                        
                        # Extract all score-like fields
                        for key, value in agent_data.items():
                            if 'score' in key.lower() and isinstance(value, (int, float)):
                                all_scores[f'{agent_name}_{key}'] = value
            
            # Overall scores
            if 'overall_scores' in data:
                all_scores.update(data['overall_scores'])
            
            return all_scores
            
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_confidence_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract confidence metrics"""
        try:
            if 'confidence_metrics' in data:
                return data['confidence_metrics']
            
            # Calculate basic confidence metrics
            confidences = []
            if 'individual_results' in data:
                for result in data['individual_results'].values():
                    if isinstance(result, dict) and result.get('available', False):
                        confidences.append(result.get('confidence', 0.5))
            
            if confidences:
                return {
                    'average': sum(confidences) / len(confidences),
                    'min': min(confidences),
                    'max': max(confidences),
                    'count': len(confidences)
                }
            
            return {}
            
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_technical_details(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical details"""
        technical = {
            'analysis_methods': [],
            'data_sources': self._identify_data_sources(data),
            'processing_info': {}
        }
        
        try:
            if 'individual_results' in data:
                for agent_name, result in data['individual_results'].items():
                    if isinstance(result, dict) and result.get('available', False):
                        agent_data = result.get('data', {})
                        
                        # Extract method information
                        if 'analysis_method' in agent_data:
                            technical['analysis_methods'].append({
                                'agent': agent_name,
                                'method': agent_data['analysis_method']
                            })
            
            return technical
            
        except Exception as e:
            return {'error': str(e)}
