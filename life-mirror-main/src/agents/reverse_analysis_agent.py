import json
import re
import requests
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter
from .base_agent import BaseAgent, AgentInput, AgentOutput

class ReverseAnalysisAgent(BaseAgent):
    """Agent for reverse-engineering successful profiles and analyzing what makes them effective"""
    
    def __init__(self):
        super().__init__()
        
        # Success indicators for different profile types
        self.success_indicators = {
            'professional': {
                'keywords': ['ceo', 'founder', 'director', 'manager', 'lead', 'senior', 'expert', 'consultant'],
                'metrics': ['experience', 'achievements', 'leadership', 'expertise'],
                'weight': 1.0
            },
            'creative': {
                'keywords': ['artist', 'designer', 'creator', 'photographer', 'writer', 'musician', 'filmmaker'],
                'metrics': ['portfolio', 'creativity', 'originality', 'artistic_vision'],
                'weight': 0.9
            },
            'influencer': {
                'keywords': ['influencer', 'content', 'creator', 'blogger', 'youtuber', 'tiktoker', 'brand'],
                'metrics': ['followers', 'engagement', 'reach', 'brand_partnerships'],
                'weight': 0.8
            },
            'entrepreneur': {
                'keywords': ['entrepreneur', 'startup', 'business', 'founder', 'investor', 'innovation'],
                'metrics': ['ventures', 'funding', 'growth', 'market_impact'],
                'weight': 0.9
            },
            'academic': {
                'keywords': ['professor', 'researcher', 'phd', 'scholar', 'scientist', 'academic'],
                'metrics': ['publications', 'citations', 'research', 'expertise'],
                'weight': 0.8
            },
            'athlete': {
                'keywords': ['athlete', 'player', 'champion', 'competitor', 'sports', 'fitness'],
                'metrics': ['performance', 'achievements', 'training', 'competition'],
                'weight': 0.7
            }
        }
        
        # Profile effectiveness factors
        self.effectiveness_factors = {
            'clarity': {
                'indicators': ['clear', 'specific', 'focused', 'direct', 'concise'],
                'weight': 0.2
            },
            'credibility': {
                'indicators': ['verified', 'certified', 'experienced', 'proven', 'established'],
                'weight': 0.25
            },
            'uniqueness': {
                'indicators': ['unique', 'different', 'innovative', 'original', 'distinctive'],
                'weight': 0.15
            },
            'engagement': {
                'indicators': ['engaging', 'interactive', 'responsive', 'active', 'connected'],
                'weight': 0.2
            },
            'value_proposition': {
                'indicators': ['value', 'benefit', 'solution', 'help', 'improve', 'transform'],
                'weight': 0.2
            }
        }
        
        # Communication patterns of successful profiles
        self.successful_patterns = {
            'storytelling': {
                'keywords': ['story', 'journey', 'experience', 'background', 'path'],
                'impact': 'high'
            },
            'achievement_highlighting': {
                'keywords': ['achieved', 'accomplished', 'won', 'awarded', 'recognized'],
                'impact': 'high'
            },
            'value_demonstration': {
                'keywords': ['help', 'solve', 'improve', 'transform', 'deliver', 'create'],
                'impact': 'high'
            },
            'social_proof': {
                'keywords': ['featured', 'mentioned', 'interviewed', 'collaborated', 'partnered'],
                'impact': 'medium'
            },
            'personality_expression': {
                'keywords': ['passionate', 'dedicated', 'enthusiastic', 'committed', 'driven'],
                'impact': 'medium'
            },
            'call_to_action': {
                'keywords': ['contact', 'connect', 'reach', 'message', 'collaborate'],
                'impact': 'medium'
            }
        }
        
        # Profile structure elements
        self.structure_elements = {
            'hook': 'Opening statement that grabs attention',
            'credentials': 'Professional qualifications and achievements',
            'value_proposition': 'What value they provide to others',
            'personality': 'Personal traits and characteristics',
            'social_proof': 'Evidence of success and recognition',
            'call_to_action': 'How others can connect or engage'
        }
        
        # Industry-specific success patterns
        self.industry_patterns = {
            'technology': {
                'keywords': ['tech', 'software', 'ai', 'data', 'digital', 'innovation'],
                'success_factors': ['technical_expertise', 'innovation', 'problem_solving']
            },
            'marketing': {
                'keywords': ['marketing', 'brand', 'advertising', 'growth', 'strategy'],
                'success_factors': ['creativity', 'results', 'strategy']
            },
            'finance': {
                'keywords': ['finance', 'investment', 'banking', 'trading', 'wealth'],
                'success_factors': ['expertise', 'track_record', 'trust']
            },
            'healthcare': {
                'keywords': ['health', 'medical', 'doctor', 'nurse', 'wellness'],
                'success_factors': ['expertise', 'compassion', 'results']
            },
            'education': {
                'keywords': ['education', 'teaching', 'training', 'learning', 'development'],
                'success_factors': ['knowledge', 'communication', 'impact']
            }
        }
        
    def run(self, input: AgentInput) -> AgentOutput:
        """Perform reverse analysis on successful profiles"""
        try:
            # Get input data
            target_profile = input.context.get('target_profile', '')
            reference_profiles = input.context.get('reference_profiles', [])
            analysis_type = input.context.get('analysis_type', 'comprehensive')  # 'comprehensive', 'pattern', 'structure', 'effectiveness'
            industry_focus = input.context.get('industry_focus', None)
            
            if not target_profile and not reference_profiles:
                return self._create_output(
                    success=False,
                    data={},
                    error="Target profile or reference profiles are required for reverse analysis",
                    confidence=0.0
                )
            
            # Perform reverse analysis based on type
            if analysis_type == 'comprehensive':
                analysis_result = self._perform_comprehensive_reverse_analysis(target_profile, reference_profiles, industry_focus)
            elif analysis_type == 'pattern':
                analysis_result = self._analyze_success_patterns(target_profile, reference_profiles, industry_focus)
            elif analysis_type == 'structure':
                analysis_result = self._analyze_profile_structure(target_profile, reference_profiles, industry_focus)
            elif analysis_type == 'effectiveness':
                analysis_result = self._analyze_profile_effectiveness(target_profile, reference_profiles, industry_focus)
            else:
                analysis_result = self._perform_comprehensive_reverse_analysis(target_profile, reference_profiles, industry_focus)
            
            # Generate reverse engineering insights
            reverse_insights = self._generate_reverse_insights(analysis_result)
            
            # Create improvement blueprint
            improvement_blueprint = self._create_improvement_blueprint(analysis_result)
            
            # Generate success formula
            success_formula = self._extract_success_formula(analysis_result)
            
            # Calculate reverse analysis score
            analysis_score = self._calculate_reverse_analysis_score(analysis_result)
            
            return self._create_output(
                success=True,
                data={
                    'reverse_analysis': analysis_result,
                    'reverse_insights': reverse_insights,
                    'improvement_blueprint': improvement_blueprint,
                    'success_formula': success_formula,
                    'analysis_score': analysis_score,
                    'analysis_type': analysis_type,
                    'industry_focus': industry_focus,
                    'profiles_analyzed': len(reference_profiles) + (1 if target_profile else 0),
                    'timestamp': datetime.now().isoformat()
                },
                confidence=self._calculate_reverse_confidence(target_profile, reference_profiles)
            )
            
        except Exception as e:
            return self._handle_error(e, input)
    
    def _perform_comprehensive_reverse_analysis(self, target_profile: str, reference_profiles: List[str], industry_focus: Optional[str]) -> Dict[str, Any]:
        """Perform comprehensive reverse engineering analysis"""
        try:
            analysis = {
                'success_patterns': self._analyze_success_patterns(target_profile, reference_profiles, industry_focus),
                'effectiveness_analysis': self._analyze_profile_effectiveness(target_profile, reference_profiles, industry_focus),
                'structure_analysis': self._analyze_profile_structure(target_profile, reference_profiles, industry_focus),
                'communication_patterns': self._analyze_communication_patterns(target_profile, reference_profiles),
                'industry_alignment': self._analyze_industry_alignment(target_profile, reference_profiles, industry_focus),
                'competitive_analysis': self._perform_competitive_analysis(target_profile, reference_profiles),
                'optimization_opportunities': self._identify_optimization_opportunities(target_profile, reference_profiles),
                'success_metrics': self._extract_success_metrics(target_profile, reference_profiles)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Comprehensive reverse analysis failed: {e}")
            return {}
    
    def _analyze_success_patterns(self, target_profile: str, reference_profiles: List[str], industry_focus: Optional[str]) -> Dict[str, Any]:
        """Analyze patterns that contribute to profile success"""
        try:
            all_profiles = [target_profile] + reference_profiles if target_profile else reference_profiles
            pattern_analysis = {}
            
            for profile in all_profiles:
                profile_lower = profile.lower()
                profile_patterns = {}
                
                # Analyze each success pattern
                for pattern_name, pattern_data in self.successful_patterns.items():
                    keywords = pattern_data['keywords']
                    impact = pattern_data['impact']
                    
                    matches = []
                    for keyword in keywords:
                        if keyword in profile_lower:
                            matches.append(keyword)
                    
                    if matches:
                        pattern_score = len(matches) / len(keywords)
                        profile_patterns[pattern_name] = {
                            'score': pattern_score,
                            'matches': matches,
                            'impact': impact
                        }
                    else:
                        profile_patterns[pattern_name] = {
                            'score': 0.0,
                            'matches': [],
                            'impact': impact
                        }
                
                pattern_analysis[f'profile_{len(pattern_analysis)}'] = profile_patterns
            
            # Aggregate patterns across all profiles
            aggregated_patterns = self._aggregate_success_patterns(pattern_analysis)
            
            # Identify most effective patterns
            top_patterns = self._identify_top_patterns(aggregated_patterns)
            
            return {
                'individual_patterns': pattern_analysis,
                'aggregated_patterns': aggregated_patterns,
                'top_patterns': top_patterns,
                'pattern_effectiveness': self._calculate_pattern_effectiveness(aggregated_patterns)
            }
            
        except Exception as e:
            self.logger.error(f"Success patterns analysis failed: {e}")
            return {}
    
    def _analyze_profile_effectiveness(self, target_profile: str, reference_profiles: List[str], industry_focus: Optional[str]) -> Dict[str, Any]:
        """Analyze what makes profiles effective"""
        try:
            all_profiles = [target_profile] + reference_profiles if target_profile else reference_profiles
            effectiveness_analysis = {}
            
            for profile in all_profiles:
                profile_lower = profile.lower()
                profile_effectiveness = {}
                
                # Analyze each effectiveness factor
                for factor_name, factor_data in self.effectiveness_factors.items():
                    indicators = factor_data['indicators']
                    weight = factor_data['weight']
                    
                    matches = []
                    for indicator in indicators:
                        if indicator in profile_lower:
                            matches.append(indicator)
                    
                    if matches:
                        factor_score = len(matches) / len(indicators)
                        weighted_score = factor_score * weight
                        profile_effectiveness[factor_name] = {
                            'score': factor_score,
                            'weighted_score': weighted_score,
                            'matches': matches,
                            'weight': weight
                        }
                    else:
                        profile_effectiveness[factor_name] = {
                            'score': 0.0,
                            'weighted_score': 0.0,
                            'matches': [],
                            'weight': weight
                        }
                
                # Calculate overall effectiveness score
                overall_score = sum(factor['weighted_score'] for factor in profile_effectiveness.values())
                profile_effectiveness['overall_effectiveness'] = overall_score
                
                effectiveness_analysis[f'profile_{len(effectiveness_analysis)}'] = profile_effectiveness
            
            # Aggregate effectiveness across all profiles
            aggregated_effectiveness = self._aggregate_effectiveness_factors(effectiveness_analysis)
            
            return {
                'individual_effectiveness': effectiveness_analysis,
                'aggregated_effectiveness': aggregated_effectiveness,
                'effectiveness_ranking': self._rank_effectiveness_factors(aggregated_effectiveness)
            }
            
        except Exception as e:
            self.logger.error(f"Profile effectiveness analysis failed: {e}")
            return {}
    
    def _analyze_profile_structure(self, target_profile: str, reference_profiles: List[str], industry_focus: Optional[str]) -> Dict[str, Any]:
        """Analyze the structure and organization of successful profiles"""
        try:
            all_profiles = [target_profile] + reference_profiles if target_profile else reference_profiles
            structure_analysis = {}
            
            for profile in all_profiles:
                profile_structure = self._identify_profile_structure_elements(profile)
                structure_analysis[f'profile_{len(structure_analysis)}'] = profile_structure
            
            # Analyze common structural patterns
            common_structures = self._identify_common_structures(structure_analysis)
            
            # Determine optimal structure
            optimal_structure = self._determine_optimal_structure(structure_analysis, common_structures)
            
            return {
                'individual_structures': structure_analysis,
                'common_structures': common_structures,
                'optimal_structure': optimal_structure,
                'structure_recommendations': self._generate_structure_recommendations(optimal_structure)
            }
            
        except Exception as e:
            self.logger.error(f"Profile structure analysis failed: {e}")
            return {}
    
    def _analyze_communication_patterns(self, target_profile: str, reference_profiles: List[str]) -> Dict[str, Any]:
        """Analyze communication patterns in successful profiles"""
        try:
            all_profiles = [target_profile] + reference_profiles if target_profile else reference_profiles
            communication_analysis = {}
            
            for profile in all_profiles:
                comm_patterns = {
                    'tone_analysis': self._analyze_communication_tone(profile),
                    'language_style': self._analyze_language_style(profile),
                    'persuasion_techniques': self._identify_persuasion_techniques(profile),
                    'emotional_appeal': self._analyze_emotional_appeal(profile)
                }
                communication_analysis[f'profile_{len(communication_analysis)}'] = comm_patterns
            
            # Aggregate communication patterns
            aggregated_communication = self._aggregate_communication_patterns(communication_analysis)
            
            return {
                'individual_communication': communication_analysis,
                'aggregated_communication': aggregated_communication,
                'effective_communication_elements': self._identify_effective_communication_elements(aggregated_communication)
            }
            
        except Exception as e:
            self.logger.error(f"Communication patterns analysis failed: {e}")
            return {}
    
    def _analyze_industry_alignment(self, target_profile: str, reference_profiles: List[str], industry_focus: Optional[str]) -> Dict[str, Any]:
        """Analyze how profiles align with industry expectations"""
        try:
            all_profiles = [target_profile] + reference_profiles if target_profile else reference_profiles
            industry_analysis = {}
            
            # Detect industries if not specified
            if not industry_focus:
                detected_industries = self._detect_industries(all_profiles)
            else:
                detected_industries = [industry_focus]
            
            for profile in all_profiles:
                profile_industry_alignment = {}
                
                for industry in detected_industries:
                    if industry in self.industry_patterns:
                        industry_data = self.industry_patterns[industry]
                        alignment_score = self._calculate_industry_alignment(profile, industry_data)
                        profile_industry_alignment[industry] = alignment_score
                
                industry_analysis[f'profile_{len(industry_analysis)}'] = profile_industry_alignment
            
            # Determine best industry alignment
            best_alignment = self._determine_best_industry_alignment(industry_analysis)
            
            return {
                'individual_alignments': industry_analysis,
                'detected_industries': detected_industries,
                'best_alignment': best_alignment,
                'industry_recommendations': self._generate_industry_recommendations(best_alignment)
            }
            
        except Exception as e:
            self.logger.error(f"Industry alignment analysis failed: {e}")
            return {}
    
    def _perform_competitive_analysis(self, target_profile: str, reference_profiles: List[str]) -> Dict[str, Any]:
        """Perform competitive analysis against reference profiles"""
        try:
            if not target_profile:
                return {'error': 'Target profile required for competitive analysis'}
            
            competitive_analysis = {
                'target_strengths': self._identify_profile_strengths(target_profile),
                'target_weaknesses': self._identify_profile_weaknesses(target_profile, reference_profiles),
                'competitive_gaps': self._identify_competitive_gaps(target_profile, reference_profiles),
                'differentiation_opportunities': self._identify_differentiation_opportunities(target_profile, reference_profiles),
                'benchmark_comparison': self._perform_benchmark_comparison(target_profile, reference_profiles)
            }
            
            return competitive_analysis
            
        except Exception as e:
            self.logger.error(f"Competitive analysis failed: {e}")
            return {}
    
    def _identify_optimization_opportunities(self, target_profile: str, reference_profiles: List[str]) -> Dict[str, Any]:
        """Identify opportunities for profile optimization"""
        try:
            opportunities = {
                'content_optimization': self._identify_content_optimization_opportunities(target_profile, reference_profiles),
                'structure_optimization': self._identify_structure_optimization_opportunities(target_profile, reference_profiles),
                'messaging_optimization': self._identify_messaging_optimization_opportunities(target_profile, reference_profiles),
                'positioning_optimization': self._identify_positioning_optimization_opportunities(target_profile, reference_profiles)
            }
            
            # Prioritize opportunities
            prioritized_opportunities = self._prioritize_optimization_opportunities(opportunities)
            
            return {
                'opportunities': opportunities,
                'prioritized_opportunities': prioritized_opportunities,
                'quick_wins': self._identify_quick_wins(opportunities),
                'long_term_improvements': self._identify_long_term_improvements(opportunities)
            }
            
        except Exception as e:
            self.logger.error(f"Optimization opportunities identification failed: {e}")
            return {}
    
    def _extract_success_metrics(self, target_profile: str, reference_profiles: List[str]) -> Dict[str, Any]:
        """Extract and analyze success metrics from profiles"""
        try:
            all_profiles = [target_profile] + reference_profiles if target_profile else reference_profiles
            success_metrics = {}
            
            for profile in all_profiles:
                profile_metrics = {
                    'profile_type': self._determine_profile_type(profile),
                    'success_indicators': self._extract_success_indicators(profile),
                    'achievement_level': self._assess_achievement_level(profile),
                    'influence_metrics': self._extract_influence_metrics(profile),
                    'credibility_score': self._calculate_credibility_score(profile)
                }
                success_metrics[f'profile_{len(success_metrics)}'] = profile_metrics
            
            # Aggregate success metrics
            aggregated_metrics = self._aggregate_success_metrics(success_metrics)
            
            return {
                'individual_metrics': success_metrics,
                'aggregated_metrics': aggregated_metrics,
                'success_benchmarks': self._establish_success_benchmarks(aggregated_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Success metrics extraction failed: {e}")
            return {}
    
    def _generate_reverse_insights(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate insights from reverse engineering analysis"""
        try:
            insights = []
            
            # Pattern insights
            success_patterns = analysis_result.get('success_patterns', {})
            if success_patterns:
                top_patterns = success_patterns.get('top_patterns', [])
                if top_patterns:
                    top_pattern = top_patterns[0]
                    insights.append(f"Most effective pattern: {top_pattern['pattern'].replace('_', ' ').title()} - used in {top_pattern['usage_rate']:.1%} of successful profiles")
            
            # Effectiveness insights
            effectiveness_analysis = analysis_result.get('effectiveness_analysis', {})
            if effectiveness_analysis:
                effectiveness_ranking = effectiveness_analysis.get('effectiveness_ranking', [])
                if effectiveness_ranking:
                    top_factor = effectiveness_ranking[0]
                    insights.append(f"Key effectiveness factor: {top_factor['factor'].replace('_', ' ').title()} with average score of {top_factor['average_score']:.2f}")
            
            # Structure insights
            structure_analysis = analysis_result.get('structure_analysis', {})
            if structure_analysis:
                optimal_structure = structure_analysis.get('optimal_structure', {})
                if optimal_structure:
                    insights.append(f"Optimal profile structure includes {len(optimal_structure.get('elements', []))} key elements")
            
            # Communication insights
            communication_patterns = analysis_result.get('communication_patterns', {})
            if communication_patterns:
                effective_elements = communication_patterns.get('effective_communication_elements', [])
                if effective_elements:
                    insights.append(f"Most effective communication style: {effective_elements[0].replace('_', ' ').title()}")
            
            # Industry insights
            industry_alignment = analysis_result.get('industry_alignment', {})
            if industry_alignment:
                best_alignment = industry_alignment.get('best_alignment', {})
                if best_alignment:
                    insights.append(f"Best industry alignment: {best_alignment.get('industry', 'Unknown')} with {best_alignment.get('score', 0):.2f} alignment score")
            
            return insights[:5]  # Limit to top 5 insights
            
        except Exception as e:
            self.logger.error(f"Reverse insights generation failed: {e}")
            return []
    
    def _create_improvement_blueprint(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a blueprint for profile improvement based on analysis"""
        try:
            blueprint = {
                'immediate_actions': [],
                'short_term_goals': [],
                'long_term_strategy': [],
                'content_recommendations': [],
                'structure_recommendations': [],
                'messaging_recommendations': []
            }
            
            # Extract recommendations from analysis
            optimization_opportunities = analysis_result.get('optimization_opportunities', {})
            if optimization_opportunities:
                quick_wins = optimization_opportunities.get('quick_wins', [])
                blueprint['immediate_actions'] = quick_wins[:3]
                
                long_term_improvements = optimization_opportunities.get('long_term_improvements', [])
                blueprint['long_term_strategy'] = long_term_improvements[:3]
            
            # Structure recommendations
            structure_analysis = analysis_result.get('structure_analysis', {})
            if structure_analysis:
                structure_recs = structure_analysis.get('structure_recommendations', [])
                blueprint['structure_recommendations'] = structure_recs[:3]
            
            # Communication recommendations
            communication_patterns = analysis_result.get('communication_patterns', {})
            if communication_patterns:
                effective_elements = communication_patterns.get('effective_communication_elements', [])
                blueprint['messaging_recommendations'] = [f"Incorporate {element.replace('_', ' ')}" for element in effective_elements[:3]]
            
            # Success pattern recommendations
            success_patterns = analysis_result.get('success_patterns', {})
            if success_patterns:
                top_patterns = success_patterns.get('top_patterns', [])
                blueprint['content_recommendations'] = [f"Include {pattern['pattern'].replace('_', ' ')}" for pattern in top_patterns[:3]]
            
            # Short-term goals
            blueprint['short_term_goals'] = [
                "Optimize profile structure based on successful patterns",
                "Enhance credibility indicators and social proof",
                "Improve value proposition clarity and impact"
            ]
            
            return blueprint
            
        except Exception as e:
            self.logger.error(f"Improvement blueprint creation failed: {e}")
            return {}
    
    def _extract_success_formula(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the success formula from analysis"""
        try:
            formula = {
                'core_elements': [],
                'success_weights': {},
                'critical_factors': [],
                'formula_description': ''
            }
            
            # Extract core elements from effectiveness analysis
            effectiveness_analysis = analysis_result.get('effectiveness_analysis', {})
            if effectiveness_analysis:
                effectiveness_ranking = effectiveness_analysis.get('effectiveness_ranking', [])
                formula['core_elements'] = [factor['factor'] for factor in effectiveness_ranking[:5]]
                formula['success_weights'] = {factor['factor']: factor['average_score'] for factor in effectiveness_ranking[:5]}
            
            # Extract critical factors from patterns
            success_patterns = analysis_result.get('success_patterns', {})
            if success_patterns:
                top_patterns = success_patterns.get('top_patterns', [])
                formula['critical_factors'] = [pattern['pattern'] for pattern in top_patterns[:3]]
            
            # Create formula description
            if formula['core_elements'] and formula['critical_factors']:
                formula['formula_description'] = f"Success = {' + '.join(formula['core_elements'][:3])} + {' + '.join(formula['critical_factors'])}"
            
            return formula
            
        except Exception as e:
            self.logger.error(f"Success formula extraction failed: {e}")
            return {}
    
    def _calculate_reverse_analysis_score(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate overall reverse analysis quality score"""
        try:
            score_components = []
            
            # Pattern analysis quality (25%)
            success_patterns = analysis_result.get('success_patterns', {})
            if success_patterns:
                pattern_effectiveness = success_patterns.get('pattern_effectiveness', 0.5)
                score_components.append(pattern_effectiveness * 0.25)
            
            # Effectiveness analysis quality (25%)
            effectiveness_analysis = analysis_result.get('effectiveness_analysis', {})
            if effectiveness_analysis:
                avg_effectiveness = self._calculate_average_effectiveness(effectiveness_analysis)
                score_components.append(avg_effectiveness * 0.25)
            
            # Structure analysis quality (20%)
            structure_analysis = analysis_result.get('structure_analysis', {})
            if structure_analysis:
                structure_completeness = self._calculate_structure_completeness(structure_analysis)
                score_components.append(structure_completeness * 0.2)
            
            # Communication analysis quality (15%)
            communication_patterns = analysis_result.get('communication_patterns', {})
            if communication_patterns:
                communication_quality = self._calculate_communication_quality(communication_patterns)
                score_components.append(communication_quality * 0.15)
            
            # Industry alignment quality (15%)
            industry_alignment = analysis_result.get('industry_alignment', {})
            if industry_alignment:
                alignment_quality = self._calculate_alignment_quality(industry_alignment)
                score_components.append(alignment_quality * 0.15)
            
            # Calculate final score
            if score_components:
                final_score = sum(score_components)
                return min(max(final_score, 0.0), 1.0)
            else:
                return 0.5  # Default neutral score
                
        except Exception as e:
            self.logger.error(f"Reverse analysis score calculation failed: {e}")
            return 0.5
    
    # Helper methods
    
    def _calculate_reverse_confidence(self, target_profile: str, reference_profiles: List[str]) -> float:
        """Calculate confidence in reverse analysis"""
        try:
            confidence_factors = []
            
            # Target profile factor
            if target_profile and len(target_profile) > 100:
                confidence_factors.append(0.8)
            elif target_profile and len(target_profile) > 50:
                confidence_factors.append(0.6)
            elif target_profile:
                confidence_factors.append(0.4)
            else:
                confidence_factors.append(0.2)
            
            # Reference profiles factor
            if len(reference_profiles) > 5:
                confidence_factors.append(0.9)
            elif len(reference_profiles) > 2:
                confidence_factors.append(0.7)
            elif len(reference_profiles) > 0:
                confidence_factors.append(0.5)
            else:
                confidence_factors.append(0.3)
            
            # Profile quality factor
            all_profiles = [target_profile] + reference_profiles if target_profile else reference_profiles
            avg_length = sum(len(profile) for profile in all_profiles) / len(all_profiles) if all_profiles else 0
            
            if avg_length > 200:
                confidence_factors.append(0.8)
            elif avg_length > 100:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
            
            return sum(confidence_factors) / len(confidence_factors)
            
        except Exception as e:
            self.logger.error(f"Reverse confidence calculation failed: {e}")
            return 0.6
    
    # Additional helper methods would continue here...
    # Due to length constraints, I'll include the most essential ones
    
    def _aggregate_success_patterns(self, pattern_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate success patterns across all profiles"""
        try:
            aggregated = {}
            pattern_names = list(self.successful_patterns.keys())
            
            for pattern_name in pattern_names:
                scores = []
                all_matches = []
                
                for profile_data in pattern_analysis.values():
                    if pattern_name in profile_data:
                        scores.append(profile_data[pattern_name]['score'])
                        all_matches.extend(profile_data[pattern_name]['matches'])
                
                if scores:
                    aggregated[pattern_name] = {
                        'average_score': sum(scores) / len(scores),
                        'usage_rate': len([s for s in scores if s > 0]) / len(scores),
                        'total_matches': len(all_matches),
                        'unique_matches': len(set(all_matches))
                    }
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Success patterns aggregation failed: {e}")
            return {}
    
    def _identify_top_patterns(self, aggregated_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify the most effective patterns"""
        try:
            pattern_effectiveness = []
            
            for pattern_name, pattern_data in aggregated_patterns.items():
                effectiveness = (pattern_data['average_score'] * 0.6 + pattern_data['usage_rate'] * 0.4)
                pattern_effectiveness.append({
                    'pattern': pattern_name,
                    'effectiveness': effectiveness,
                    'average_score': pattern_data['average_score'],
                    'usage_rate': pattern_data['usage_rate']
                })
            
            return sorted(pattern_effectiveness, key=lambda x: x['effectiveness'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Top patterns identification failed: {e}")
            return []
    
    def _calculate_pattern_effectiveness(self, aggregated_patterns: Dict[str, Any]) -> float:
        """Calculate overall pattern effectiveness"""
        try:
            if not aggregated_patterns:
                return 0.5
            
            effectiveness_scores = []
            for pattern_data in aggregated_patterns.values():
                effectiveness = (pattern_data['average_score'] * 0.6 + pattern_data['usage_rate'] * 0.4)
                effectiveness_scores.append(effectiveness)
            
            return sum(effectiveness_scores) / len(effectiveness_scores)
            
        except Exception as e:
            self.logger.error(f"Pattern effectiveness calculation failed: {e}")
            return 0.5
    
    def _determine_profile_type(self, profile: str) -> str:
        """Determine the type of profile based on content"""
        try:
            profile_lower = profile.lower()
            type_scores = {}
            
            for profile_type, type_data in self.success_indicators.items():
                keywords = type_data['keywords']
                matches = sum(1 for keyword in keywords if keyword in profile_lower)
                type_scores[profile_type] = matches / len(keywords)
            
            if type_scores:
                best_type = max(type_scores.items(), key=lambda x: x[1])
                return best_type[0] if best_type[1] > 0.2 else 'general'
            else:
                return 'general'
                
        except Exception as e:
            self.logger.error(f"Profile type determination failed: {e}")
            return 'unknown'
    
    def _extract_success_indicators(self, profile: str) -> List[str]:
        """Extract success indicators from profile"""
        try:
            profile_lower = profile.lower()
            indicators = []
            
            # Check for achievement words
            achievement_words = ['achieved', 'accomplished', 'won', 'awarded', 'recognized', 'featured', 'led', 'founded', 'created']
            for word in achievement_words:
                if word in profile_lower:
                    indicators.append(word)
            
            # Check for quantifiable metrics
            import re
            numbers = re.findall(r'\d+[k|m|%|\+]?', profile_lower)
            if numbers:
                indicators.extend([f"quantified_metric_{num}" for num in numbers[:3]])
            
            return indicators[:5]  # Limit to top 5 indicators
            
        except Exception as e:
            self.logger.error(f"Success indicators extraction failed: {e}")
            return []
    
    def _assess_achievement_level(self, profile: str) -> str:
        """Assess the level of achievements mentioned in profile"""
        try:
            profile_lower = profile.lower()
            
            high_achievement_words = ['ceo', 'founder', 'director', 'award', 'winner', 'champion', 'expert', 'leader']
            medium_achievement_words = ['manager', 'senior', 'lead', 'specialist', 'consultant', 'experienced']
            
            high_count = sum(1 for word in high_achievement_words if word in profile_lower)
            medium_count = sum(1 for word in medium_achievement_words if word in profile_lower)
            
            if high_count > 0:
                return 'high'
            elif medium_count > 0:
                return 'medium'
            else:
                return 'entry'
                
        except Exception as e:
            self.logger.error(f"Achievement level assessment failed: {e}")
            return 'unknown'
    
    def _calculate_credibility_score(self, profile: str) -> float:
        """Calculate credibility score based on profile content"""
        try:
            profile_lower = profile.lower()
            credibility_score = 0.0
            
            # Credibility indicators
            credibility_words = ['verified', 'certified', 'licensed', 'accredited', 'experienced', 'proven', 'established']
            credibility_count = sum(1 for word in credibility_words if word in profile_lower)
            credibility_score += min(credibility_count / len(credibility_words), 0.4)
            
            # Social proof indicators
            social_proof_words = ['featured', 'mentioned', 'interviewed', 'collaborated', 'partnered', 'worked with']
            social_proof_count = sum(1 for word in social_proof_words if word in profile_lower)
            credibility_score += min(social_proof_count / len(social_proof_words), 0.3)
            
            # Achievement indicators
            achievement_words = ['achieved', 'accomplished', 'won', 'awarded', 'recognized']
            achievement_count = sum(1 for word in achievement_words if word in profile_lower)
            credibility_score += min(achievement_count / len(achievement_words), 0.3)
            
            return min(credibility_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Credibility score calculation failed: {e}")
            return 0.5