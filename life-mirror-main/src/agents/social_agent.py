import json
import re
import requests
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter
from .base_agent import BaseAgent, AgentInput, AgentOutput

class SocialAgent(BaseAgent):
    """Agent for analyzing social media presence and social characteristics"""
    
    def __init__(self):
        super().__init__()
        
        # Social platform indicators
        self.platform_indicators = {
            'instagram': ['instagram', 'insta', 'ig', '@', '#', 'story', 'reel', 'igtv'],
            'twitter': ['twitter', 'tweet', 'retweet', 'hashtag', 'trending', 'thread'],
            'facebook': ['facebook', 'fb', 'post', 'share', 'like', 'comment'],
            'linkedin': ['linkedin', 'professional', 'network', 'career', 'business'],
            'tiktok': ['tiktok', 'viral', 'dance', 'trend', 'fyp', 'duet'],
            'youtube': ['youtube', 'video', 'subscribe', 'channel', 'vlog'],
            'snapchat': ['snapchat', 'snap', 'streak', 'filter'],
            'discord': ['discord', 'server', 'gaming', 'chat', 'voice'],
            'reddit': ['reddit', 'subreddit', 'upvote', 'karma', 'thread']
        }
        
        # Social behavior indicators
        self.social_behaviors = {
            'content_creator': {
                'keywords': ['create', 'content', 'post', 'share', 'video', 'photo', 'blog', 'vlog'],
                'weight': 1.0
            },
            'social_influencer': {
                'keywords': ['influence', 'followers', 'audience', 'brand', 'sponsor', 'collab'],
                'weight': 0.9
            },
            'community_builder': {
                'keywords': ['community', 'group', 'organize', 'event', 'meetup', 'network'],
                'weight': 0.8
            },
            'social_consumer': {
                'keywords': ['follow', 'watch', 'read', 'browse', 'scroll', 'consume'],
                'weight': 0.6
            },
            'social_connector': {
                'keywords': ['connect', 'friend', 'message', 'chat', 'call', 'meet'],
                'weight': 0.8
            },
            'trend_follower': {
                'keywords': ['trend', 'viral', 'popular', 'latest', 'new', 'hot'],
                'weight': 0.7
            },
            'opinion_leader': {
                'keywords': ['opinion', 'review', 'recommend', 'advice', 'guide', 'expert'],
                'weight': 0.8
            },
            'social_activist': {
                'keywords': ['cause', 'activism', 'awareness', 'change', 'support', 'advocate'],
                'weight': 0.7
            }
        }
        
        # Engagement patterns
        self.engagement_patterns = {
            'high_engagement': ['active', 'daily', 'frequent', 'regular', 'constant', 'always'],
            'moderate_engagement': ['sometimes', 'occasional', 'weekly', 'often', 'usually'],
            'low_engagement': ['rarely', 'seldom', 'minimal', 'quiet', 'lurker', 'observer']
        }
        
        # Social interests categories
        self.social_interests = {
            'lifestyle': ['lifestyle', 'fashion', 'beauty', 'food', 'travel', 'home', 'wellness'],
            'entertainment': ['movies', 'music', 'games', 'tv', 'celebrity', 'comedy', 'memes'],
            'technology': ['tech', 'gadgets', 'ai', 'coding', 'startup', 'innovation', 'digital'],
            'sports': ['sports', 'fitness', 'workout', 'team', 'athlete', 'competition', 'game'],
            'education': ['learning', 'education', 'knowledge', 'skill', 'course', 'tutorial'],
            'business': ['business', 'entrepreneur', 'marketing', 'finance', 'career', 'professional'],
            'arts': ['art', 'design', 'creative', 'photography', 'music', 'writing', 'culture'],
            'news': ['news', 'politics', 'current', 'events', 'world', 'society', 'issues'],
            'personal': ['personal', 'life', 'family', 'relationships', 'thoughts', 'feelings']
        }
        
        # Communication styles on social media
        self.social_communication_styles = {
            'casual': ['lol', 'omg', 'btw', 'tbh', 'imo', 'casual', 'chill'],
            'professional': ['professional', 'business', 'formal', 'corporate', 'industry'],
            'humorous': ['funny', 'joke', 'meme', 'lol', 'haha', 'humor', 'comedy'],
            'inspirational': ['inspire', 'motivate', 'positive', 'quote', 'wisdom', 'growth'],
            'educational': ['learn', 'teach', 'explain', 'tutorial', 'guide', 'tip'],
            'personal': ['personal', 'share', 'story', 'experience', 'life', 'journey'],
            'promotional': ['promote', 'brand', 'product', 'service', 'buy', 'sale']
        }
        
        # Social influence indicators
        self.influence_indicators = {
            'micro_influencer': ['followers', 'engagement', 'niche', 'authentic', 'community'],
            'thought_leader': ['expert', 'authority', 'knowledge', 'insight', 'opinion'],
            'brand_ambassador': ['brand', 'partner', 'sponsor', 'collaborate', 'represent'],
            'community_leader': ['leader', 'organize', 'group', 'community', 'moderate'],
            'content_creator': ['create', 'original', 'content', 'produce', 'publish']
        }
        
    def run(self, input: AgentInput) -> AgentOutput:
        """Analyze social media presence and characteristics"""
        try:
            # Get input data
            bio_text = input.context.get('bio_text', '')
            social_links = input.context.get('social_links', [])
            additional_context = input.context.get('additional_context', {})
            analysis_type = input.context.get('analysis_type', 'comprehensive')  # 'comprehensive', 'platforms', 'behavior', 'influence'
            
            if not bio_text and not social_links and not additional_context:
                return self._create_output(
                    success=False,
                    data={},
                    error="Bio text, social links, or additional context is required for social analysis",
                    confidence=0.0
                )
            
            # Perform social analysis based on type
            if analysis_type == 'comprehensive':
                analysis_result = self._perform_comprehensive_social_analysis(bio_text, social_links, additional_context)
            elif analysis_type == 'platforms':
                analysis_result = self._analyze_platform_presence(bio_text, social_links, additional_context)
            elif analysis_type == 'behavior':
                analysis_result = self._analyze_social_behavior(bio_text, social_links, additional_context)
            elif analysis_type == 'influence':
                analysis_result = self._analyze_social_influence(bio_text, social_links, additional_context)
            else:
                analysis_result = self._perform_comprehensive_social_analysis(bio_text, social_links, additional_context)
            
            # Calculate social score
            social_score = self._calculate_social_score(analysis_result)
            
            # Generate social profile
            social_profile = self._create_social_profile(analysis_result)
            
            # Generate insights and recommendations
            insights = self._generate_social_insights(analysis_result)
            recommendations = self._generate_social_recommendations(analysis_result)
            
            return self._create_output(
                success=True,
                data={
                    'social_analysis': analysis_result,
                    'social_score': social_score,
                    'social_profile': social_profile,
                    'insights': insights,
                    'recommendations': recommendations,
                    'analysis_type': analysis_type,
                    'platforms_detected': len(analysis_result.get('platform_presence', {}).get('active_platforms', [])),
                    'social_engagement_level': analysis_result.get('engagement_analysis', {}).get('engagement_level', 'unknown'),
                    'timestamp': datetime.now().isoformat()
                },
                confidence=self._calculate_social_confidence(bio_text, social_links, additional_context)
            )
            
        except Exception as e:
            return self._handle_error(e, input)
    
    def _perform_comprehensive_social_analysis(self, bio_text: str, social_links: List[str], additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive social media analysis"""
        try:
            analysis = {
                'platform_presence': self._analyze_platform_presence(bio_text, social_links, additional_context),
                'social_behavior': self._analyze_social_behavior(bio_text, social_links, additional_context),
                'engagement_analysis': self._analyze_engagement_patterns(bio_text, social_links, additional_context),
                'content_analysis': self._analyze_content_preferences(bio_text, social_links, additional_context),
                'communication_style': self._analyze_social_communication_style(bio_text, social_links, additional_context),
                'influence_potential': self._analyze_social_influence(bio_text, social_links, additional_context),
                'network_characteristics': self._analyze_network_characteristics(bio_text, social_links, additional_context),
                'digital_footprint': self._analyze_digital_footprint(bio_text, social_links, additional_context)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Comprehensive social analysis failed: {e}")
            return {}
    
    def _analyze_platform_presence(self, bio_text: str, social_links: List[str], additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze presence across different social platforms"""
        try:
            text_lower = bio_text.lower()
            platform_scores = {}
            platform_evidence = {}
            detected_platforms = []
            
            # Analyze text for platform indicators
            for platform, keywords in self.platform_indicators.items():
                matches = []
                for keyword in keywords:
                    if keyword in text_lower:
                        matches.append(keyword)
                
                if matches:
                    platform_scores[platform] = len(matches) / len(keywords)
                    platform_evidence[platform] = matches
                    detected_platforms.append(platform)
                else:
                    platform_scores[platform] = 0.0
                    platform_evidence[platform] = []
            
            # Analyze social links
            link_platforms = []
            for link in social_links:
                link_lower = link.lower()
                for platform in self.platform_indicators.keys():
                    if platform in link_lower:
                        link_platforms.append(platform)
                        if platform not in detected_platforms:
                            detected_platforms.append(platform)
                        platform_scores[platform] = max(platform_scores.get(platform, 0), 0.8)
            
            # Determine primary platforms
            primary_platforms = sorted(platform_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            primary_platforms = [{'platform': platform, 'score': score} for platform, score in primary_platforms if score > 0.2]
            
            # Calculate platform diversity
            active_platforms = [platform for platform, score in platform_scores.items() if score > 0.1]
            platform_diversity = len(active_platforms) / len(self.platform_indicators)
            
            return {
                'platform_scores': platform_scores,
                'platform_evidence': platform_evidence,
                'detected_platforms': detected_platforms,
                'link_platforms': link_platforms,
                'primary_platforms': primary_platforms,
                'active_platforms': active_platforms,
                'platform_diversity': platform_diversity,
                'platform_focus': self._determine_platform_focus(platform_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Platform presence analysis failed: {e}")
            return {}
    
    def _analyze_social_behavior(self, bio_text: str, social_links: List[str], additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze social media behavior patterns"""
        try:
            text_lower = bio_text.lower()
            behavior_scores = {}
            behavior_evidence = {}
            
            for behavior, behavior_data in self.social_behaviors.items():
                keywords = behavior_data['keywords']
                weight = behavior_data['weight']
                
                matches = []
                for keyword in keywords:
                    if keyword in text_lower:
                        matches.append(keyword)
                
                if matches:
                    base_score = len(matches) / len(keywords)
                    behavior_scores[behavior] = min(base_score * weight, 1.0)
                    behavior_evidence[behavior] = matches
                else:
                    behavior_scores[behavior] = 0.0
                    behavior_evidence[behavior] = []
            
            # Identify dominant behaviors
            dominant_behaviors = sorted(behavior_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            dominant_behaviors = [{'behavior': behavior, 'score': score} for behavior, score in dominant_behaviors if score > 0.2]
            
            # Determine social role
            social_role = self._determine_social_role(behavior_scores)
            
            return {
                'behavior_scores': behavior_scores,
                'behavior_evidence': behavior_evidence,
                'dominant_behaviors': dominant_behaviors,
                'social_role': social_role,
                'behavior_profile': self._generate_behavior_profile(behavior_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Social behavior analysis failed: {e}")
            return {}
    
    def _analyze_engagement_patterns(self, bio_text: str, social_links: List[str], additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze social media engagement patterns"""
        try:
            text_lower = bio_text.lower()
            engagement_scores = {}
            engagement_evidence = {}
            
            for level, keywords in self.engagement_patterns.items():
                matches = []
                for keyword in keywords:
                    if keyword in text_lower:
                        matches.append(keyword)
                
                if matches:
                    engagement_scores[level] = len(matches) / len(keywords)
                    engagement_evidence[level] = matches
                else:
                    engagement_scores[level] = 0.0
                    engagement_evidence[level] = []
            
            # Determine engagement level
            if engagement_scores['high_engagement'] > max(engagement_scores['moderate_engagement'], engagement_scores['low_engagement']):
                engagement_level = 'high'
                engagement_score = engagement_scores['high_engagement']
            elif engagement_scores['moderate_engagement'] > engagement_scores['low_engagement']:
                engagement_level = 'moderate'
                engagement_score = engagement_scores['moderate_engagement']
            else:
                engagement_level = 'low'
                engagement_score = engagement_scores['low_engagement']
            
            # Analyze engagement quality indicators
            quality_indicators = self._analyze_engagement_quality(text_lower)
            
            return {
                'engagement_scores': engagement_scores,
                'engagement_evidence': engagement_evidence,
                'engagement_level': engagement_level,
                'engagement_score': engagement_score,
                'quality_indicators': quality_indicators,
                'engagement_style': self._determine_engagement_style(engagement_scores, quality_indicators)
            }
            
        except Exception as e:
            self.logger.error(f"Engagement patterns analysis failed: {e}")
            return {}
    
    def _analyze_content_preferences(self, bio_text: str, social_links: List[str], additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content preferences and interests"""
        try:
            text_lower = bio_text.lower()
            interest_scores = {}
            interest_evidence = {}
            
            for category, keywords in self.social_interests.items():
                matches = []
                for keyword in keywords:
                    if keyword in text_lower:
                        matches.append(keyword)
                
                if matches:
                    interest_scores[category] = len(matches) / len(keywords)
                    interest_evidence[category] = matches
                else:
                    interest_scores[category] = 0.0
                    interest_evidence[category] = []
            
            # Identify primary content interests
            primary_interests = sorted(interest_scores.items(), key=lambda x: x[1], reverse=True)[:4]
            primary_interests = [{'category': cat, 'score': score} for cat, score in primary_interests if score > 0.1]
            
            # Calculate content diversity
            active_interests = len([score for score in interest_scores.values() if score > 0.1])
            content_diversity = active_interests / len(self.social_interests)
            
            # Determine content strategy
            content_strategy = self._determine_content_strategy(interest_scores)
            
            return {
                'interest_scores': interest_scores,
                'interest_evidence': interest_evidence,
                'primary_interests': primary_interests,
                'content_diversity': content_diversity,
                'content_strategy': content_strategy,
                'content_niche': self._identify_content_niche(interest_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Content preferences analysis failed: {e}")
            return {}
    
    def _analyze_social_communication_style(self, bio_text: str, social_links: List[str], additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze social media communication style"""
        try:
            text_lower = bio_text.lower()
            style_scores = {}
            style_evidence = {}
            
            for style, keywords in self.social_communication_styles.items():
                matches = []
                for keyword in keywords:
                    if keyword in text_lower:
                        matches.append(keyword)
                
                if matches:
                    style_scores[style] = len(matches) / len(keywords)
                    style_evidence[style] = matches
                else:
                    style_scores[style] = 0.0
                    style_evidence[style] = []
            
            # Determine primary communication style
            primary_style = max(style_scores.items(), key=lambda x: x[1])
            
            # Analyze tone and personality
            tone_analysis = self._analyze_communication_tone(bio_text)
            
            return {
                'style_scores': style_scores,
                'style_evidence': style_evidence,
                'primary_style': {'style': primary_style[0], 'score': primary_style[1]} if primary_style[1] > 0.1 else None,
                'tone_analysis': tone_analysis,
                'communication_effectiveness': self._calculate_social_communication_effectiveness(style_scores, tone_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Social communication style analysis failed: {e}")
            return {}
    
    def _analyze_social_influence(self, bio_text: str, social_links: List[str], additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze social influence potential and characteristics"""
        try:
            text_lower = bio_text.lower()
            influence_scores = {}
            influence_evidence = {}
            
            for influence_type, keywords in self.influence_indicators.items():
                matches = []
                for keyword in keywords:
                    if keyword in text_lower:
                        matches.append(keyword)
                
                if matches:
                    influence_scores[influence_type] = len(matches) / len(keywords)
                    influence_evidence[influence_type] = matches
                else:
                    influence_scores[influence_type] = 0.0
                    influence_evidence[influence_type] = []
            
            # Determine influence potential
            influence_potential = max(influence_scores.values()) if influence_scores else 0.0
            primary_influence_type = max(influence_scores.items(), key=lambda x: x[1])[0] if influence_potential > 0.1 else None
            
            # Analyze influence factors
            influence_factors = self._analyze_influence_factors(bio_text, social_links)
            
            return {
                'influence_scores': influence_scores,
                'influence_evidence': influence_evidence,
                'influence_potential': influence_potential,
                'primary_influence_type': primary_influence_type,
                'influence_factors': influence_factors,
                'influence_category': self._categorize_influence_level(influence_potential, influence_factors)
            }
            
        except Exception as e:
            self.logger.error(f"Social influence analysis failed: {e}")
            return {}
    
    def _analyze_network_characteristics(self, bio_text: str, social_links: List[str], additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze social network characteristics"""
        try:
            text_lower = bio_text.lower()
            
            # Network size indicators
            size_indicators = {
                'large_network': ['thousands', 'many', 'lots', 'numerous', 'extensive', 'wide'],
                'medium_network': ['some', 'several', 'moderate', 'decent', 'good'],
                'small_network': ['few', 'small', 'close', 'intimate', 'select']
            }
            
            network_size_scores = {}
            for size, keywords in size_indicators.items():
                count = sum(1 for keyword in keywords if keyword in text_lower)
                network_size_scores[size] = count / len(keywords)
            
            # Network quality indicators
            quality_keywords = ['meaningful', 'deep', 'authentic', 'genuine', 'quality', 'close', 'trusted']
            quality_score = sum(1 for keyword in quality_keywords if keyword in text_lower) / len(quality_keywords)
            
            # Network diversity indicators
            diversity_keywords = ['diverse', 'different', 'various', 'international', 'multicultural', 'varied']
            diversity_score = sum(1 for keyword in diversity_keywords if keyword in text_lower) / len(diversity_keywords)
            
            # Determine network characteristics
            network_size = max(network_size_scores.items(), key=lambda x: x[1])[0] if max(network_size_scores.values()) > 0.1 else 'unknown'
            
            return {
                'network_size_scores': network_size_scores,
                'network_size': network_size,
                'network_quality': quality_score,
                'network_diversity': diversity_score,
                'network_profile': self._generate_network_profile(network_size, quality_score, diversity_score)
            }
            
        except Exception as e:
            self.logger.error(f"Network characteristics analysis failed: {e}")
            return {}
    
    def _analyze_digital_footprint(self, bio_text: str, social_links: List[str], additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall digital footprint"""
        try:
            # Calculate footprint size based on available data
            footprint_indicators = {
                'bio_length': len(bio_text),
                'social_links_count': len(social_links),
                'context_richness': len(additional_context)
            }
            
            # Determine footprint size
            total_indicators = sum(footprint_indicators.values())
            if total_indicators > 500:
                footprint_size = 'large'
            elif total_indicators > 200:
                footprint_size = 'medium'
            else:
                footprint_size = 'small'
            
            # Analyze digital presence quality
            presence_quality = self._analyze_digital_presence_quality(bio_text, social_links)
            
            # Calculate digital maturity
            digital_maturity = self._calculate_digital_maturity(bio_text, social_links, additional_context)
            
            return {
                'footprint_indicators': footprint_indicators,
                'footprint_size': footprint_size,
                'presence_quality': presence_quality,
                'digital_maturity': digital_maturity,
                'footprint_score': self._calculate_footprint_score(footprint_indicators, presence_quality, digital_maturity)
            }
            
        except Exception as e:
            self.logger.error(f"Digital footprint analysis failed: {e}")
            return {}
    
    def _calculate_social_score(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate overall social media score"""
        try:
            score_components = []
            
            # Platform presence contribution (20%)
            platform_data = analysis_result.get('platform_presence', {})
            if platform_data:
                diversity = platform_data.get('platform_diversity', 0)
                score_components.append(diversity * 0.2)
            
            # Social behavior contribution (25%)
            behavior_data = analysis_result.get('social_behavior', {})
            if behavior_data:
                behavior_scores = behavior_data.get('behavior_scores', {})
                avg_behavior = sum(behavior_scores.values()) / len(behavior_scores) if behavior_scores else 0
                score_components.append(avg_behavior * 0.25)
            
            # Engagement contribution (20%)
            engagement_data = analysis_result.get('engagement_analysis', {})
            if engagement_data:
                engagement_score = engagement_data.get('engagement_score', 0.5)
                score_components.append(engagement_score * 0.2)
            
            # Content quality contribution (15%)
            content_data = analysis_result.get('content_analysis', {})
            if content_data:
                content_diversity = content_data.get('content_diversity', 0.5)
                score_components.append(content_diversity * 0.15)
            
            # Influence potential contribution (10%)
            influence_data = analysis_result.get('influence_potential', {})
            if influence_data:
                influence_potential = influence_data.get('influence_potential', 0.3)
                score_components.append(influence_potential * 0.1)
            
            # Digital footprint contribution (10%)
            footprint_data = analysis_result.get('digital_footprint', {})
            if footprint_data:
                footprint_score = footprint_data.get('footprint_score', 0.5)
                score_components.append(footprint_score * 0.1)
            
            # Calculate final score
            if score_components:
                final_score = sum(score_components)
                return min(max(final_score, 0.0), 1.0)
            else:
                return 0.5  # Default neutral score
                
        except Exception as e:
            self.logger.error(f"Social score calculation failed: {e}")
            return 0.5
    
    def _create_social_profile(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive social media profile"""
        try:
            profile = {
                'social_persona': self._determine_social_persona(analysis_result),
                'platform_strategy': self._analyze_platform_strategy(analysis_result),
                'content_themes': self._identify_content_themes(analysis_result),
                'engagement_style': self._determine_overall_engagement_style(analysis_result),
                'influence_level': self._categorize_overall_influence(analysis_result),
                'social_strengths': self._identify_social_strengths(analysis_result),
                'growth_opportunities': self._identify_social_growth_opportunities(analysis_result)
            }
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Social profile creation failed: {e}")
            return {}
    
    def _generate_social_insights(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate insights about social media presence"""
        try:
            insights = []
            
            # Platform insights
            platform_data = analysis_result.get('platform_presence', {})
            if platform_data:
                active_platforms = platform_data.get('active_platforms', [])
                if len(active_platforms) > 3:
                    insights.append(f"Multi-platform presence across {len(active_platforms)} platforms shows strong digital engagement")
                elif len(active_platforms) == 1:
                    insights.append(f"Focused presence on {active_platforms[0]} suggests specialized audience targeting")
            
            # Behavior insights
            behavior_data = analysis_result.get('social_behavior', {})
            if behavior_data:
                dominant_behaviors = behavior_data.get('dominant_behaviors', [])
                if dominant_behaviors:
                    top_behavior = dominant_behaviors[0]['behavior']
                    insights.append(f"Strong {top_behavior.replace('_', ' ')} behavior indicates {self._get_behavior_insight(top_behavior)}")
            
            # Engagement insights
            engagement_data = analysis_result.get('engagement_analysis', {})
            if engagement_data:
                engagement_level = engagement_data.get('engagement_level', 'unknown')
                if engagement_level == 'low':
                    opportunities.append("Increase audience engagement")
                elif engagement_level == 'moderate':
                    opportunities.append("Enhance engagement quality")
            
            # Content opportunities
            content_data = analysis_result.get('content_analysis', {})
            if content_data:
                content_diversity = content_data.get('content_diversity', 0)
                if content_diversity < 0.4:
                    opportunities.append("Diversify content themes")
                elif content_diversity > 0.7:
                    opportunities.append("Develop content specialization")
            
            # Influence opportunities
            influence_data = analysis_result.get('influence_potential', {})
            if influence_data:
                influence_potential = influence_data.get('influence_potential', 0)
                if influence_potential < 0.5:
                    opportunities.append("Build thought leadership")
                elif influence_potential > 0.6:
                    opportunities.append("Leverage influence for partnerships")
            
            # Communication opportunities
            communication_data = analysis_result.get('communication_style', {})
            if communication_data:
                style_scores = communication_data.get('style_scores', {})
                if max(style_scores.values()) < 0.4 if style_scores else True:
                    opportunities.append("Develop distinctive communication style")
            
            return opportunities[:4]  # Limit to top 4 opportunities
             
        except Exception as e:
            self.logger.error(f"Social growth opportunities identification failed: {e}")
            return []
    
    def _generate_social_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving social media presence"""
        try:
            recommendations = []
            
            # Platform recommendations
            platform_data = analysis_result.get('platform_presence', {})
            if platform_data:
                platform_diversity = platform_data.get('platform_diversity', 0)
                if platform_diversity < 0.3:
                    recommendations.append("Consider expanding to additional platforms to reach broader audiences")
                elif platform_diversity > 0.7:
                    recommendations.append("Focus on optimizing performance on your top 2-3 platforms for better results")
            
            # Engagement recommendations
            engagement_data = analysis_result.get('engagement_analysis', {})
            if engagement_data:
                engagement_level = engagement_data.get('engagement_level', 'unknown')
                if engagement_level == 'low':
                    recommendations.append("Increase engagement by responding to comments and participating in conversations")
                elif engagement_level == 'high':
                    recommendations.append("Leverage your high engagement to build stronger community connections")
            
            # Content recommendations
            content_data = analysis_result.get('content_analysis', {})
            if content_data:
                content_diversity = content_data.get('content_diversity', 0)
                if content_diversity < 0.4:
                    recommendations.append("Diversify content themes to appeal to broader audience interests")
            
            # Influence recommendations
            influence_data = analysis_result.get('influence_potential', {})
            if influence_data:
                influence_potential = influence_data.get('influence_potential', 0)
                if influence_potential > 0.5:
                    recommendations.append("Develop thought leadership content to maximize your influence potential")
                else:
                    recommendations.append("Build authority by sharing expertise and valuable insights consistently")
            
            # General recommendations
            recommendations.extend([
                "Maintain consistent posting schedule to build audience expectations",
                "Engage authentically with your community to build genuine connections",
                "Use analytics to understand what content resonates with your audience",
                "Collaborate with others in your niche to expand reach and credibility"
            ])
            
            return recommendations[:6]  # Limit to top 6 recommendations
            
        except Exception as e:
            self.logger.error(f"Social recommendations generation failed: {e}")
            return []
    
    # Helper methods
    
    def _calculate_social_confidence(self, bio_text: str, social_links: List[str], additional_context: Dict[str, Any]) -> float:
        """Calculate confidence in social analysis"""
        try:
            confidence_factors = []
            
            # Bio text factor
            if len(bio_text) > 100:
                confidence_factors.append(0.8)
            elif len(bio_text) > 50:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
            
            # Social links factor
            if len(social_links) > 2:
                confidence_factors.append(0.9)
            elif len(social_links) > 0:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.3)
            
            # Additional context factor
            if additional_context:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
            
            return sum(confidence_factors) / len(confidence_factors)
            
        except Exception as e:
            self.logger.error(f"Social confidence calculation failed: {e}")
            return 0.6
    
    def _determine_platform_focus(self, platform_scores: Dict[str, float]) -> str:
        """Determine platform focus strategy"""
        try:
            active_platforms = [platform for platform, score in platform_scores.items() if score > 0.2]
            
            if len(active_platforms) > 4:
                return 'multi-platform'
            elif len(active_platforms) > 2:
                return 'diversified'
            elif len(active_platforms) == 1:
                return 'specialized'
            else:
                return 'minimal'
                
        except Exception as e:
            self.logger.error(f"Platform focus determination failed: {e}")
            return 'unknown'
    
    def _determine_social_role(self, behavior_scores: Dict[str, float]) -> str:
        """Determine primary social media role"""
        try:
            top_behavior = max(behavior_scores.items(), key=lambda x: x[1])
            
            role_mapping = {
                'content_creator': 'Creator',
                'social_influencer': 'Influencer',
                'community_builder': 'Community Leader',
                'social_consumer': 'Consumer',
                'social_connector': 'Connector',
                'trend_follower': 'Trend Follower',
                'opinion_leader': 'Opinion Leader',
                'social_activist': 'Activist'
            }
            
            return role_mapping.get(top_behavior[0], 'Balanced User') if top_behavior[1] > 0.3 else 'Balanced User'
            
        except Exception as e:
            self.logger.error(f"Social role determination failed: {e}")
            return 'Unknown'
    
    def _generate_behavior_profile(self, behavior_scores: Dict[str, float]) -> str:
        """Generate behavior profile summary"""
        try:
            top_behaviors = sorted(behavior_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            top_behaviors = [behavior for behavior, score in top_behaviors if score > 0.2]
            
            if len(top_behaviors) >= 2:
                return f"Combines {top_behaviors[0].replace('_', ' ')} with {top_behaviors[1].replace('_', ' ')} behaviors"
            elif len(top_behaviors) == 1:
                return f"Primarily exhibits {top_behaviors[0].replace('_', ' ')} behavior"
            else:
                return "Balanced social media behavior across multiple areas"
                
        except Exception as e:
            self.logger.error(f"Behavior profile generation failed: {e}")
            return "Complex behavior pattern"
    
    def _analyze_engagement_quality(self, text_lower: str) -> Dict[str, float]:
        """Analyze quality indicators of engagement"""
        try:
            quality_indicators = {
                'authentic_engagement': 0.0,
                'meaningful_interactions': 0.0,
                'community_building': 0.0,
                'value_creation': 0.0
            }
            
            # Authentic engagement indicators
            authentic_keywords = ['genuine', 'authentic', 'real', 'honest', 'sincere']
            quality_indicators['authentic_engagement'] = sum(1 for keyword in authentic_keywords if keyword in text_lower) / len(authentic_keywords)
            
            # Meaningful interactions indicators
            meaningful_keywords = ['meaningful', 'deep', 'thoughtful', 'insightful', 'valuable']
            quality_indicators['meaningful_interactions'] = sum(1 for keyword in meaningful_keywords if keyword in text_lower) / len(meaningful_keywords)
            
            # Community building indicators
            community_keywords = ['community', 'together', 'support', 'help', 'connect']
            quality_indicators['community_building'] = sum(1 for keyword in community_keywords if keyword in text_lower) / len(community_keywords)
            
            # Value creation indicators
            value_keywords = ['value', 'useful', 'helpful', 'educational', 'informative']
            quality_indicators['value_creation'] = sum(1 for keyword in value_keywords if keyword in text_lower) / len(value_keywords)
            
            return quality_indicators
            
        except Exception as e:
            self.logger.error(f"Engagement quality analysis failed: {e}")
            return {}
    
    def _determine_engagement_style(self, engagement_scores: Dict[str, float], quality_indicators: Dict[str, float]) -> str:
        """Determine overall engagement style"""
        try:
            # Get primary engagement level
            primary_level = max(engagement_scores.items(), key=lambda x: x[1])[0]
            
            # Check quality factors
            avg_quality = sum(quality_indicators.values()) / len(quality_indicators) if quality_indicators else 0
            
            if primary_level == 'high_engagement' and avg_quality > 0.5:
                return 'High-Quality Engager'
            elif primary_level == 'high_engagement':
                return 'Active Engager'
            elif primary_level == 'moderate_engagement' and avg_quality > 0.4:
                return 'Thoughtful Participant'
            elif primary_level == 'moderate_engagement':
                return 'Casual Participant'
            else:
                return 'Observer'
                
        except Exception as e:
            self.logger.error(f"Engagement style determination failed: {e}")
            return 'Unknown'
    
    def _determine_content_strategy(self, interest_scores: Dict[str, float]) -> str:
        """Determine content strategy based on interests"""
        try:
            active_interests = [interest for interest, score in interest_scores.items() if score > 0.2]
            
            if len(active_interests) > 5:
                return 'Broad Appeal Strategy'
            elif len(active_interests) > 2:
                return 'Multi-Niche Strategy'
            elif len(active_interests) == 1:
                return 'Niche Specialist Strategy'
            else:
                return 'Undefined Strategy'
                
        except Exception as e:
            self.logger.error(f"Content strategy determination failed: {e}")
            return 'Unknown Strategy'
    
    def _identify_content_niche(self, interest_scores: Dict[str, float]) -> str:
        """Identify primary content niche"""
        try:
            top_interest = max(interest_scores.items(), key=lambda x: x[1])
            
            if top_interest[1] > 0.4:
                return top_interest[0].replace('_', ' ').title()
            else:
                return 'Multi-Interest'
                
        except Exception as e:
            self.logger.error(f"Content niche identification failed: {e}")
            return 'General'
    
    def _analyze_communication_tone(self, bio_text: str) -> Dict[str, Any]:
        """Analyze communication tone"""
        try:
            text_lower = bio_text.lower()
            
            tone_indicators = {
                'positive': ['positive', 'happy', 'excited', 'love', 'amazing', 'great', 'awesome'],
                'professional': ['professional', 'business', 'career', 'industry', 'expertise'],
                'casual': ['casual', 'chill', 'relaxed', 'fun', 'easy', 'simple'],
                'passionate': ['passionate', 'love', 'obsessed', 'dedicated', 'committed'],
                'humorous': ['funny', 'humor', 'joke', 'laugh', 'witty', 'sarcastic']
            }
            
            tone_scores = {}
            for tone, keywords in tone_indicators.items():
                score = sum(1 for keyword in keywords if keyword in text_lower) / len(keywords)
                tone_scores[tone] = score
            
            primary_tone = max(tone_scores.items(), key=lambda x: x[1])
            
            return {
                'tone_scores': tone_scores,
                'primary_tone': primary_tone[0] if primary_tone[1] > 0.1 else 'neutral',
                'tone_strength': primary_tone[1]
            }
            
        except Exception as e:
            self.logger.error(f"Communication tone analysis failed: {e}")
            return {}
    
    def _calculate_social_communication_effectiveness(self, style_scores: Dict[str, float], tone_analysis: Dict[str, Any]) -> float:
        """Calculate social communication effectiveness"""
        try:
            effectiveness = 0.5  # Base score
            
            # Style diversity bonus
            active_styles = len([score for score in style_scores.values() if score > 0.2])
            if active_styles > 2:
                effectiveness += 0.2
            
            # Tone consistency bonus
            if tone_analysis:
                tone_strength = tone_analysis.get('tone_strength', 0)
                if tone_strength > 0.3:
                    effectiveness += 0.2
            
            # Professional + personal balance bonus
            if style_scores.get('professional', 0) > 0.2 and style_scores.get('personal', 0) > 0.2:
                effectiveness += 0.1
            
            return min(effectiveness, 1.0)
            
        except Exception as e:
            self.logger.error(f"Social communication effectiveness calculation failed: {e}")
            return 0.5
    
    def _analyze_influence_factors(self, bio_text: str, social_links: List[str]) -> Dict[str, float]:
        """Analyze factors that contribute to social influence"""
        try:
            text_lower = bio_text.lower()
            
            factors = {
                'expertise': 0.0,
                'authenticity': 0.0,
                'engagement': 0.0,
                'reach': 0.0,
                'consistency': 0.0
            }
            
            # Expertise indicators
            expertise_keywords = ['expert', 'professional', 'specialist', 'authority', 'experienced']
            factors['expertise'] = sum(1 for keyword in expertise_keywords if keyword in text_lower) / len(expertise_keywords)
            
            # Authenticity indicators
            authenticity_keywords = ['authentic', 'genuine', 'real', 'honest', 'transparent']
            factors['authenticity'] = sum(1 for keyword in authenticity_keywords if keyword in text_lower) / len(authenticity_keywords)
            
            # Engagement indicators
            engagement_keywords = ['engage', 'interact', 'connect', 'community', 'respond']
            factors['engagement'] = sum(1 for keyword in engagement_keywords if keyword in text_lower) / len(engagement_keywords)
            
            # Reach indicators (based on platform presence)
            factors['reach'] = min(len(social_links) / 5.0, 1.0)  # Normalize to 0-1
            
            # Consistency indicators
            consistency_keywords = ['consistent', 'regular', 'daily', 'weekly', 'schedule']
            factors['consistency'] = sum(1 for keyword in consistency_keywords if keyword in text_lower) / len(consistency_keywords)
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Influence factors analysis failed: {e}")
            return {}
    
    def _categorize_influence_level(self, influence_potential: float, influence_factors: Dict[str, float]) -> str:
        """Categorize overall influence level"""
        try:
            avg_factors = sum(influence_factors.values()) / len(influence_factors) if influence_factors else 0
            combined_score = (influence_potential + avg_factors) / 2
            
            if combined_score > 0.7:
                return 'High Influence'
            elif combined_score > 0.4:
                return 'Moderate Influence'
            elif combined_score > 0.2:
                return 'Emerging Influence'
            else:
                return 'Limited Influence'
                
        except Exception as e:
            self.logger.error(f"Influence level categorization failed: {e}")
            return 'Unknown'
    
    def _generate_network_profile(self, network_size: str, quality_score: float, diversity_score: float) -> str:
        """Generate network profile description"""
        try:
            profile_parts = []
            
            # Size component
            profile_parts.append(f"{network_size} network")
            
            # Quality component
            if quality_score > 0.6:
                profile_parts.append("high-quality connections")
            elif quality_score > 0.3:
                profile_parts.append("meaningful relationships")
            
            # Diversity component
            if diversity_score > 0.5:
                profile_parts.append("diverse community")
            
            return " with ".join(profile_parts) if len(profile_parts) > 1 else profile_parts[0]
            
        except Exception as e:
            self.logger.error(f"Network profile generation failed: {e}")
            return "Balanced network"
    
    def _analyze_digital_presence_quality(self, bio_text: str, social_links: List[str]) -> float:
        """Analyze quality of digital presence"""
        try:
            quality_score = 0.0
            
            # Bio quality (30%)
            if len(bio_text) > 100:
                quality_score += 0.3
            elif len(bio_text) > 50:
                quality_score += 0.2
            
            # Link diversity (40%)
            if len(social_links) > 3:
                quality_score += 0.4
            elif len(social_links) > 1:
                quality_score += 0.3
            elif len(social_links) > 0:
                quality_score += 0.2
            
            # Content indicators (30%)
            quality_keywords = ['professional', 'creative', 'authentic', 'engaging', 'valuable']
            text_lower = bio_text.lower()
            quality_matches = sum(1 for keyword in quality_keywords if keyword in text_lower)
            quality_score += (quality_matches / len(quality_keywords)) * 0.3
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Digital presence quality analysis failed: {e}")
            return 0.5
    
    def _calculate_digital_maturity(self, bio_text: str, social_links: List[str], additional_context: Dict[str, Any]) -> float:
        """Calculate digital maturity level"""
        try:
            maturity_score = 0.0
            
            # Platform sophistication
            professional_platforms = ['linkedin', 'github', 'behance', 'dribbble']
            professional_count = sum(1 for link in social_links if any(platform in link.lower() for platform in professional_platforms))
            maturity_score += min(professional_count / 2.0, 0.4)
            
            # Content sophistication
            sophisticated_keywords = ['strategy', 'brand', 'professional', 'portfolio', 'expertise']
            text_lower = bio_text.lower()
            sophisticated_matches = sum(1 for keyword in sophisticated_keywords if keyword in text_lower)
            maturity_score += (sophisticated_matches / len(sophisticated_keywords)) * 0.3
            
            # Context richness
            if additional_context:
                maturity_score += min(len(additional_context) / 10.0, 0.3)
            
            return min(maturity_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Digital maturity calculation failed: {e}")
            return 0.5
    
    def _calculate_footprint_score(self, footprint_indicators: Dict[str, int], presence_quality: float, digital_maturity: float) -> float:
        """Calculate overall digital footprint score"""
        try:
            # Normalize footprint size (40%)
            total_indicators = sum(footprint_indicators.values())
            size_score = min(total_indicators / 1000.0, 1.0) * 0.4
            
            # Quality contribution (35%)
            quality_contribution = presence_quality * 0.35
            
            # Maturity contribution (25%)
            maturity_contribution = digital_maturity * 0.25
            
            return size_score + quality_contribution + maturity_contribution
            
        except Exception as e:
            self.logger.error(f"Footprint score calculation failed: {e}")
            return 0.5
    
    def _get_behavior_insight(self, behavior: str) -> str:
        """Get insight for a specific behavior"""
        insights = {
            'content_creator': 'strong creative abilities and audience engagement',
            'social_influencer': 'leadership potential and community impact',
            'community_builder': 'excellent networking and relationship-building skills',
            'social_consumer': 'good curation skills and trend awareness',
            'social_connector': 'strong interpersonal skills and networking ability',
            'trend_follower': 'awareness of current trends and cultural relevance',
            'opinion_leader': 'thought leadership and expertise in specific areas',
            'social_activist': 'passion for causes and ability to drive change'
        }
        return insights.get(behavior, 'unique social media approach')
    
    # Additional helper methods for profile creation
    
    def _determine_social_persona(self, analysis_result: Dict[str, Any]) -> str:
        """Determine overall social media persona"""
        try:
            behavior_data = analysis_result.get('social_behavior', {})
            influence_data = analysis_result.get('influence_potential', {})
            engagement_data = analysis_result.get('engagement_analysis', {})
            
            social_role = behavior_data.get('social_role', 'Balanced User')
            influence_level = influence_data.get('influence_potential', 0)
            engagement_level = engagement_data.get('engagement_level', 'moderate')
            
            if influence_level > 0.6 and engagement_level == 'high':
                return f"Influential {social_role}"
            elif engagement_level == 'high':
                return f"Active {social_role}"
            elif influence_level > 0.4:
                return f"Emerging {social_role}"
            else:
                return social_role
                
        except Exception as e:
            self.logger.error(f"Social persona determination failed: {e}")
            return "Social Media User"
    
    def _analyze_platform_strategy(self, analysis_result: Dict[str, Any]) -> str:
        """Analyze platform strategy"""
        try:
            platform_data = analysis_result.get('platform_presence', {})
            if not platform_data:
                return "Undefined Strategy"
            
            platform_focus = platform_data.get('platform_focus', 'unknown')
            primary_platforms = platform_data.get('primary_platforms', [])
            
            if platform_focus == 'specialized' and primary_platforms:
                return f"Specialized {primary_platforms[0]['platform'].title()} Strategy"
            elif platform_focus == 'diversified':
                return "Multi-Platform Diversification Strategy"
            elif platform_focus == 'multi-platform':
                return "Broad Multi-Platform Strategy"
            else:
                return "Emerging Platform Strategy"
                
        except Exception as e:
            self.logger.error(f"Platform strategy analysis failed: {e}")
            return "Unknown Strategy"
    
    def _identify_content_themes(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Identify primary content themes"""
        try:
            content_data = analysis_result.get('content_analysis', {})
            if not content_data:
                return []
            
            primary_interests = content_data.get('primary_interests', [])
            return [interest['category'].replace('_', ' ').title() for interest in primary_interests[:3]]
            
        except Exception as e:
            self.logger.error(f"Content themes identification failed: {e}")
            return []
    
    def _determine_overall_engagement_style(self, analysis_result: Dict[str, Any]) -> str:
        """Determine overall engagement style"""
        try:
            engagement_data = analysis_result.get('engagement_analysis', {})
            communication_data = analysis_result.get('communication_style', {})
            
            engagement_style = engagement_data.get('engagement_style', 'Unknown')
            primary_style = communication_data.get('primary_style', {})
            
            if primary_style:
                comm_style = primary_style['style']
                return f"{engagement_style} with {comm_style.title()} Communication"
            else:
                return engagement_style
                
        except Exception as e:
            self.logger.error(f"Overall engagement style determination failed: {e}")
            return "Balanced Engagement"
    
    def _categorize_overall_influence(self, analysis_result: Dict[str, Any]) -> str:
        """Categorize overall influence level"""
        try:
            influence_data = analysis_result.get('influence_potential', {})
            return influence_data.get('influence_category', 'Unknown Influence')
            
        except Exception as e:
            self.logger.error(f"Overall influence categorization failed: {e}")
            return "Unknown Influence"
    
    def _identify_social_strengths(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Identify social media strengths"""
        try:
            strengths = []
            
            # Platform strengths
            platform_data = analysis_result.get('platform_presence', {})
            if platform_data:
                platform_diversity = platform_data.get('platform_diversity', 0)
                if platform_diversity > 0.6:
                    strengths.append("Multi-platform presence")
                elif platform_diversity < 0.3:
                    strengths.append("Platform specialization")
            
            # Behavior strengths
            behavior_data = analysis_result.get('social_behavior', {})
            if behavior_data:
                dominant_behaviors = behavior_data.get('dominant_behaviors', [])
                for behavior_info in dominant_behaviors[:2]:
                    behavior = behavior_info['behavior']
                    if behavior_info['score'] > 0.6:
                        strengths.append(f"Strong {behavior.replace('_', ' ')} skills")
            
            # Engagement strengths
            engagement_data = analysis_result.get('engagement_analysis', {})
            if engagement_data:
                engagement_level = engagement_data.get('engagement_level', 'unknown')
                if engagement_level == 'high':
                    strengths.append("High audience engagement")
            
            # Influence strengths
            influence_data = analysis_result.get('influence_potential', {})
            if influence_data:
                influence_potential = influence_data.get('influence_potential', 0)
                if influence_potential > 0.6:
                    strengths.append("Strong influence potential")
            
            return strengths[:4]  # Limit to top 4 strengths
            
        except Exception as e:
            self.logger.error(f"Social strengths identification failed: {e}")
            return []
    
    def _identify_social_growth_opportunities(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Identify social media growth opportunities"""
        try:
            opportunities = []
            
            # Platform opportunities
            platform_data = analysis_result.get('platform_presence', {})
            if platform_data:
                platform_diversity = platform_data.get('platform_diversity', 0)
                if platform_diversity < 0.4:
                    opportunities.append("Expand to additional platforms")
                elif platform_diversity > 0.7:
                    opportunities.append("Focus and optimize top-performing platforms")
            
            # Engagement opportunities
            engagement_data = analysis_result.get('engagement_analysis', {})
            if engagement_data:
                engagement_level = engagement_data.get('engagement_level', 'unknown')
                if engagement_level == 'low':
                    recommendations.append("Increase posting frequency and interaction")
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            self.logger.error(f"Social recommendations generation failed: {e}")
            return []