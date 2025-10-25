import json
import re
import random
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import Counter
from .base_agent import BaseAgent, AgentInput, AgentOutput

class VibeAnalysisAgent(BaseAgent):
    """Agent for analyzing personality, emotional characteristics, and overall vibe"""
    
    def __init__(self):
        super().__init__()
        
        # Personality trait keywords
        self.personality_traits = {
            'extroversion': {
                'keywords': ['outgoing', 'social', 'party', 'people', 'friends', 'talkative', 'energetic', 'confident'],
                'weight': 1.0
            },
            'introversion': {
                'keywords': ['quiet', 'reserved', 'thoughtful', 'introspective', 'reading', 'alone', 'peaceful', 'calm'],
                'weight': 1.0
            },
            'openness': {
                'keywords': ['creative', 'artistic', 'curious', 'adventure', 'travel', 'new', 'explore', 'innovative'],
                'weight': 0.9
            },
            'conscientiousness': {
                'keywords': ['organized', 'responsible', 'disciplined', 'goal', 'plan', 'achievement', 'work', 'dedicated'],
                'weight': 0.9
            },
            'agreeableness': {
                'keywords': ['kind', 'helpful', 'caring', 'empathetic', 'supportive', 'friendly', 'compassionate', 'understanding'],
                'weight': 0.8
            },
            'neuroticism': {
                'keywords': ['anxious', 'stressed', 'worried', 'emotional', 'sensitive', 'moody', 'nervous', 'tense'],
                'weight': 0.7
            },
            'optimism': {
                'keywords': ['positive', 'happy', 'hopeful', 'bright', 'cheerful', 'upbeat', 'enthusiastic', 'motivated'],
                'weight': 0.9
            },
            'intelligence': {
                'keywords': ['smart', 'intelligent', 'clever', 'analytical', 'logical', 'strategic', 'wise', 'knowledgeable'],
                'weight': 0.8
            },
            'humor': {
                'keywords': ['funny', 'witty', 'humorous', 'joke', 'laugh', 'sarcastic', 'playful', 'entertaining'],
                'weight': 0.8
            },
            'ambition': {
                'keywords': ['ambitious', 'driven', 'success', 'career', 'goals', 'achievement', 'leadership', 'competitive'],
                'weight': 0.9
            }
        }
        
        # Emotional state indicators
        self.emotional_indicators = {
            'happiness': ['happy', 'joy', 'excited', 'thrilled', 'delighted', 'pleased', 'content', 'cheerful'],
            'sadness': ['sad', 'depressed', 'down', 'blue', 'melancholy', 'disappointed', 'heartbroken', 'grief'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated', 'rage', 'upset'],
            'fear': ['scared', 'afraid', 'anxious', 'worried', 'nervous', 'terrified', 'panic', 'concerned'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'bewildered', 'confused'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sick', 'nauseated', 'appalled'],
            'love': ['love', 'adore', 'cherish', 'affection', 'romance', 'passion', 'devoted', 'infatuated'],
            'excitement': ['excited', 'thrilled', 'pumped', 'energized', 'enthusiastic', 'eager', 'anticipation']
        }
        
        # Interest categories
        self.interest_categories = {
            'sports': ['football', 'basketball', 'soccer', 'tennis', 'golf', 'swimming', 'running', 'gym', 'fitness', 'workout'],
            'arts': ['painting', 'drawing', 'music', 'singing', 'dancing', 'theater', 'photography', 'writing', 'poetry'],
            'technology': ['coding', 'programming', 'computer', 'tech', 'gaming', 'AI', 'software', 'hardware', 'internet'],
            'travel': ['travel', 'vacation', 'adventure', 'explore', 'countries', 'culture', 'backpacking', 'tourism'],
            'food': ['cooking', 'baking', 'food', 'restaurant', 'cuisine', 'recipe', 'chef', 'dining', 'wine'],
            'nature': ['hiking', 'camping', 'outdoors', 'nature', 'animals', 'environment', 'gardening', 'wildlife'],
            'learning': ['reading', 'books', 'education', 'learning', 'studying', 'knowledge', 'research', 'academic'],
            'social': ['friends', 'family', 'relationships', 'community', 'networking', 'socializing', 'parties'],
            'business': ['business', 'entrepreneur', 'startup', 'investing', 'finance', 'marketing', 'sales', 'leadership'],
            'wellness': ['meditation', 'yoga', 'mindfulness', 'health', 'wellness', 'spirituality', 'self-care']
        }
        
        # Communication style indicators
        self.communication_styles = {
            'formal': ['professional', 'formal', 'respectful', 'polite', 'courteous', 'proper'],
            'casual': ['casual', 'relaxed', 'informal', 'laid-back', 'chill', 'easy-going'],
            'humorous': ['funny', 'witty', 'sarcastic', 'joke', 'humor', 'playful'],
            'intellectual': ['analytical', 'thoughtful', 'deep', 'philosophical', 'complex', 'sophisticated'],
            'emotional': ['passionate', 'emotional', 'expressive', 'heartfelt', 'sincere', 'genuine'],
            'direct': ['direct', 'straightforward', 'honest', 'blunt', 'clear', 'concise'],
            'diplomatic': ['diplomatic', 'tactful', 'considerate', 'careful', 'balanced', 'measured']
        }
        
        # Vibe categories
        self.vibe_categories = {
            'energetic': {
                'indicators': ['energy', 'active', 'dynamic', 'vibrant', 'lively', 'enthusiastic'],
                'score_weight': 0.9
            },
            'calm': {
                'indicators': ['calm', 'peaceful', 'serene', 'tranquil', 'relaxed', 'zen'],
                'score_weight': 0.8
            },
            'creative': {
                'indicators': ['creative', 'artistic', 'innovative', 'imaginative', 'original', 'unique'],
                'score_weight': 0.9
            },
            'intellectual': {
                'indicators': ['smart', 'intelligent', 'analytical', 'thoughtful', 'wise', 'knowledgeable'],
                'score_weight': 0.8
            },
            'social': {
                'indicators': ['social', 'friendly', 'outgoing', 'charismatic', 'popular', 'connected'],
                'score_weight': 0.9
            },
            'adventurous': {
                'indicators': ['adventure', 'bold', 'daring', 'explorer', 'risk-taker', 'spontaneous'],
                'score_weight': 0.8
            },
            'nurturing': {
                'indicators': ['caring', 'supportive', 'helpful', 'kind', 'compassionate', 'empathetic'],
                'score_weight': 0.8
            },
            'ambitious': {
                'indicators': ['ambitious', 'driven', 'goal-oriented', 'successful', 'determined', 'focused'],
                'score_weight': 0.9
            }
        }
        
    def run(self, input: AgentInput) -> AgentOutput:
        """Analyze personality and vibe characteristics"""
        try:
            # Get input data
            bio_text = input.context.get('bio_text', '')
            image_url = input.context.get('image_url', '')
            additional_context = input.context.get('additional_context', {})
            analysis_type = input.context.get('analysis_type', 'comprehensive')  # 'comprehensive', 'personality', 'emotional', 'interests'
            
            if not bio_text and not additional_context:
                return self._create_output(
                    success=False,
                    data={},
                    error="Bio text or additional context is required for vibe analysis",
                    confidence=0.0
                )
            
            # Perform comprehensive vibe analysis
            if analysis_type == 'comprehensive':
                analysis_result = self._perform_comprehensive_analysis(bio_text, additional_context)
            elif analysis_type == 'personality':
                analysis_result = self._analyze_personality_traits(bio_text, additional_context)
            elif analysis_type == 'emotional':
                analysis_result = self._analyze_emotional_characteristics(bio_text, additional_context)
            elif analysis_type == 'interests':
                analysis_result = self._analyze_interests_and_hobbies(bio_text, additional_context)
            else:
                analysis_result = self._perform_comprehensive_analysis(bio_text, additional_context)
            
            # Calculate overall vibe score
            vibe_score = self._calculate_overall_vibe_score(analysis_result)
            
            # Generate vibe summary
            vibe_summary = self._generate_vibe_summary(analysis_result)
            
            # Create personality profile
            personality_profile = self._create_personality_profile(analysis_result)
            
            # Generate insights and recommendations
            insights = self._generate_vibe_insights(analysis_result)
            recommendations = self._generate_vibe_recommendations(analysis_result)
            
            return self._create_output(
                success=True,
                data={
                    'vibe_analysis': analysis_result,
                    'vibe_score': vibe_score,
                    'vibe_summary': vibe_summary,
                    'personality_profile': personality_profile,
                    'insights': insights,
                    'recommendations': recommendations,
                    'analysis_type': analysis_type,
                    'text_length': len(bio_text),
                    'analysis_depth': self._calculate_analysis_depth(bio_text, additional_context),
                    'timestamp': datetime.now().isoformat()
                },
                confidence=self._calculate_analysis_confidence(bio_text, additional_context)
            )
            
        except Exception as e:
            return self._handle_error(e, input)
    
    def _perform_comprehensive_analysis(self, bio_text: str, additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive vibe analysis"""
        try:
            analysis = {
                'personality_traits': self._analyze_personality_traits(bio_text, additional_context),
                'emotional_characteristics': self._analyze_emotional_characteristics(bio_text, additional_context),
                'interests_and_hobbies': self._analyze_interests_and_hobbies(bio_text, additional_context),
                'communication_style': self._analyze_communication_style(bio_text, additional_context),
                'vibe_categories': self._analyze_vibe_categories(bio_text, additional_context),
                'energy_level': self._analyze_energy_level(bio_text, additional_context),
                'social_orientation': self._analyze_social_orientation(bio_text, additional_context),
                'life_approach': self._analyze_life_approach(bio_text, additional_context)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
            return {}
    
    def _analyze_personality_traits(self, bio_text: str, additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze personality traits from text"""
        try:
            text_lower = bio_text.lower()
            trait_scores = {}
            trait_evidence = {}
            
            for trait, trait_data in self.personality_traits.items():
                keywords = trait_data['keywords']
                weight = trait_data['weight']
                
                # Count keyword matches
                matches = []
                for keyword in keywords:
                    if keyword in text_lower:
                        matches.append(keyword)
                
                # Calculate trait score
                if matches:
                    base_score = len(matches) / len(keywords)
                    trait_scores[trait] = min(base_score * weight, 1.0)
                    trait_evidence[trait] = matches
                else:
                    trait_scores[trait] = 0.0
                    trait_evidence[trait] = []
            
            # Identify dominant traits
            dominant_traits = sorted(trait_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            return {
                'trait_scores': trait_scores,
                'trait_evidence': trait_evidence,
                'dominant_traits': [{'trait': trait, 'score': score} for trait, score in dominant_traits if score > 0.1],
                'personality_summary': self._generate_personality_summary(trait_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Personality trait analysis failed: {e}")
            return {}
    
    def _analyze_emotional_characteristics(self, bio_text: str, additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze emotional characteristics and states"""
        try:
            text_lower = bio_text.lower()
            emotional_scores = {}
            emotional_evidence = {}
            
            for emotion, keywords in self.emotional_indicators.items():
                matches = []
                for keyword in keywords:
                    if keyword in text_lower:
                        matches.append(keyword)
                
                if matches:
                    emotional_scores[emotion] = len(matches) / len(keywords)
                    emotional_evidence[emotion] = matches
                else:
                    emotional_scores[emotion] = 0.0
                    emotional_evidence[emotion] = []
            
            # Determine emotional tone
            positive_emotions = ['happiness', 'love', 'excitement']
            negative_emotions = ['sadness', 'anger', 'fear', 'disgust']
            
            positive_score = sum(emotional_scores[emotion] for emotion in positive_emotions) / len(positive_emotions)
            negative_score = sum(emotional_scores[emotion] for emotion in negative_emotions) / len(negative_emotions)
            
            emotional_tone = 'positive' if positive_score > negative_score else 'negative' if negative_score > positive_score else 'neutral'
            emotional_intensity = max(max(emotional_scores.values()), 0.1)
            
            return {
                'emotional_scores': emotional_scores,
                'emotional_evidence': emotional_evidence,
                'emotional_tone': emotional_tone,
                'emotional_intensity': emotional_intensity,
                'positive_score': positive_score,
                'negative_score': negative_score,
                'dominant_emotions': sorted(emotional_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            }
            
        except Exception as e:
            self.logger.error(f"Emotional characteristics analysis failed: {e}")
            return {}
    
    def _analyze_interests_and_hobbies(self, bio_text: str, additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze interests and hobbies"""
        try:
            text_lower = bio_text.lower()
            interest_scores = {}
            interest_evidence = {}
            
            for category, keywords in self.interest_categories.items():
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
            
            # Identify primary interests
            primary_interests = sorted(interest_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            primary_interests = [{'category': cat, 'score': score} for cat, score in primary_interests if score > 0.1]
            
            # Calculate interest diversity
            active_interests = len([score for score in interest_scores.values() if score > 0.1])
            interest_diversity = active_interests / len(self.interest_categories)
            
            return {
                'interest_scores': interest_scores,
                'interest_evidence': interest_evidence,
                'primary_interests': primary_interests,
                'interest_diversity': interest_diversity,
                'interest_profile': self._generate_interest_profile(interest_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Interests and hobbies analysis failed: {e}")
            return {}
    
    def _analyze_communication_style(self, bio_text: str, additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze communication style"""
        try:
            text_lower = bio_text.lower()
            style_scores = {}
            style_evidence = {}
            
            for style, keywords in self.communication_styles.items():
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
            
            # Analyze text characteristics
            text_characteristics = self._analyze_text_characteristics(bio_text)
            
            # Determine primary communication style
            primary_style = max(style_scores.items(), key=lambda x: x[1])
            
            return {
                'style_scores': style_scores,
                'style_evidence': style_evidence,
                'primary_style': {'style': primary_style[0], 'score': primary_style[1]} if primary_style[1] > 0.1 else None,
                'text_characteristics': text_characteristics,
                'communication_effectiveness': self._calculate_communication_effectiveness(style_scores, text_characteristics)
            }
            
        except Exception as e:
            self.logger.error(f"Communication style analysis failed: {e}")
            return {}
    
    def _analyze_vibe_categories(self, bio_text: str, additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall vibe categories"""
        try:
            text_lower = bio_text.lower()
            vibe_scores = {}
            vibe_evidence = {}
            
            for vibe, vibe_data in self.vibe_categories.items():
                indicators = vibe_data['indicators']
                weight = vibe_data['score_weight']
                
                matches = []
                for indicator in indicators:
                    if indicator in text_lower:
                        matches.append(indicator)
                
                if matches:
                    base_score = len(matches) / len(indicators)
                    vibe_scores[vibe] = min(base_score * weight, 1.0)
                    vibe_evidence[vibe] = matches
                else:
                    vibe_scores[vibe] = 0.0
                    vibe_evidence[vibe] = []
            
            # Identify dominant vibes
            dominant_vibes = sorted(vibe_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            dominant_vibes = [{'vibe': vibe, 'score': score} for vibe, score in dominant_vibes if score > 0.1]
            
            return {
                'vibe_scores': vibe_scores,
                'vibe_evidence': vibe_evidence,
                'dominant_vibes': dominant_vibes,
                'vibe_blend': self._generate_vibe_blend(vibe_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Vibe categories analysis failed: {e}")
            return {}
    
    def _analyze_energy_level(self, bio_text: str, additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze energy level indicators"""
        try:
            text_lower = bio_text.lower()
            
            high_energy_keywords = ['energetic', 'active', 'dynamic', 'vibrant', 'enthusiastic', 'passionate', 'excited']
            low_energy_keywords = ['calm', 'peaceful', 'relaxed', 'quiet', 'gentle', 'serene', 'tranquil']
            
            high_energy_count = sum(1 for keyword in high_energy_keywords if keyword in text_lower)
            low_energy_count = sum(1 for keyword in low_energy_keywords if keyword in text_lower)
            
            if high_energy_count > low_energy_count:
                energy_level = 'high'
                energy_score = min(high_energy_count / len(high_energy_keywords), 1.0)
            elif low_energy_count > high_energy_count:
                energy_level = 'low'
                energy_score = min(low_energy_count / len(low_energy_keywords), 1.0)
            else:
                energy_level = 'moderate'
                energy_score = 0.5
            
            return {
                'energy_level': energy_level,
                'energy_score': energy_score,
                'high_energy_indicators': high_energy_count,
                'low_energy_indicators': low_energy_count
            }
            
        except Exception as e:
            self.logger.error(f"Energy level analysis failed: {e}")
            return {}
    
    def _analyze_social_orientation(self, bio_text: str, additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze social orientation (extroversion vs introversion)"""
        try:
            text_lower = bio_text.lower()
            
            extrovert_keywords = ['social', 'outgoing', 'people', 'friends', 'party', 'networking', 'team', 'group']
            introvert_keywords = ['quiet', 'alone', 'reading', 'thinking', 'introspective', 'private', 'solitude']
            
            extrovert_count = sum(1 for keyword in extrovert_keywords if keyword in text_lower)
            introvert_count = sum(1 for keyword in introvert_keywords if keyword in text_lower)
            
            if extrovert_count > introvert_count:
                orientation = 'extroverted'
                orientation_score = min(extrovert_count / len(extrovert_keywords), 1.0)
            elif introvert_count > extrovert_count:
                orientation = 'introverted'
                orientation_score = min(introvert_count / len(introvert_keywords), 1.0)
            else:
                orientation = 'ambivert'
                orientation_score = 0.5
            
            return {
                'social_orientation': orientation,
                'orientation_score': orientation_score,
                'extrovert_indicators': extrovert_count,
                'introvert_indicators': introvert_count
            }
            
        except Exception as e:
            self.logger.error(f"Social orientation analysis failed: {e}")
            return {}
    
    def _analyze_life_approach(self, bio_text: str, additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze approach to life and goals"""
        try:
            text_lower = bio_text.lower()
            
            approach_indicators = {
                'goal_oriented': ['goal', 'achievement', 'success', 'ambition', 'driven', 'focused'],
                'experience_oriented': ['experience', 'adventure', 'explore', 'travel', 'discover', 'journey'],
                'relationship_oriented': ['family', 'friends', 'love', 'relationships', 'connection', 'community'],
                'growth_oriented': ['learning', 'growth', 'development', 'improvement', 'challenge', 'evolve'],
                'stability_oriented': ['stable', 'secure', 'consistent', 'reliable', 'steady', 'balanced']
            }
            
            approach_scores = {}
            for approach, keywords in approach_indicators.items():
                count = sum(1 for keyword in keywords if keyword in text_lower)
                approach_scores[approach] = count / len(keywords)
            
            primary_approach = max(approach_scores.items(), key=lambda x: x[1])
            
            return {
                'approach_scores': approach_scores,
                'primary_approach': {'approach': primary_approach[0], 'score': primary_approach[1]} if primary_approach[1] > 0.1 else None,
                'life_philosophy': self._infer_life_philosophy(approach_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Life approach analysis failed: {e}")
            return {}
    
    def _analyze_text_characteristics(self, bio_text: str) -> Dict[str, Any]:
        """Analyze characteristics of the text itself"""
        try:
            if not bio_text:
                return {}
            
            # Basic text statistics
            word_count = len(bio_text.split())
            sentence_count = len(re.split(r'[.!?]+', bio_text))
            avg_word_length = sum(len(word) for word in bio_text.split()) / word_count if word_count > 0 else 0
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Punctuation analysis
            exclamation_count = bio_text.count('!')
            question_count = bio_text.count('?')
            emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]')
            emoji_count = len(emoji_pattern.findall(bio_text))
            
            # Determine writing style
            if exclamation_count > 2 or emoji_count > 3:
                writing_style = 'enthusiastic'
            elif question_count > 1:
                writing_style = 'inquisitive'
            elif avg_word_length > 5:
                writing_style = 'sophisticated'
            elif avg_sentence_length > 15:
                writing_style = 'detailed'
            else:
                writing_style = 'casual'
            
            return {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_word_length': avg_word_length,
                'avg_sentence_length': avg_sentence_length,
                'exclamation_count': exclamation_count,
                'question_count': question_count,
                'emoji_count': emoji_count,
                'writing_style': writing_style,
                'text_complexity': self._calculate_text_complexity(avg_word_length, avg_sentence_length)
            }
            
        except Exception as e:
            self.logger.error(f"Text characteristics analysis failed: {e}")
            return {}
    
    def _calculate_overall_vibe_score(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate overall vibe score"""
        try:
            score_components = []
            
            # Personality traits contribution
            personality_data = analysis_result.get('personality_traits', {})
            if personality_data:
                trait_scores = personality_data.get('trait_scores', {})
                positive_traits = ['extroversion', 'openness', 'conscientiousness', 'agreeableness', 'optimism']
                positive_score = sum(trait_scores.get(trait, 0) for trait in positive_traits) / len(positive_traits)
                score_components.append(positive_score * 0.3)
            
            # Emotional characteristics contribution
            emotional_data = analysis_result.get('emotional_characteristics', {})
            if emotional_data:
                positive_score = emotional_data.get('positive_score', 0.5)
                score_components.append(positive_score * 0.25)
            
            # Interests diversity contribution
            interests_data = analysis_result.get('interests_and_hobbies', {})
            if interests_data:
                diversity_score = interests_data.get('interest_diversity', 0.5)
                score_components.append(diversity_score * 0.2)
            
            # Communication effectiveness contribution
            communication_data = analysis_result.get('communication_style', {})
            if communication_data:
                effectiveness = communication_data.get('communication_effectiveness', 0.5)
                score_components.append(effectiveness * 0.15)
            
            # Energy level contribution
            energy_data = analysis_result.get('energy_level', {})
            if energy_data:
                energy_score = energy_data.get('energy_score', 0.5)
                score_components.append(energy_score * 0.1)
            
            # Calculate final score
            if score_components:
                final_score = sum(score_components)
                return min(max(final_score, 0.0), 1.0)
            else:
                return 0.5  # Default neutral score
                
        except Exception as e:
            self.logger.error(f"Vibe score calculation failed: {e}")
            return 0.5
    
    def _generate_vibe_summary(self, analysis_result: Dict[str, Any]) -> str:
        """Generate a summary of the overall vibe"""
        try:
            summary_parts = []
            
            # Personality summary
            personality_data = analysis_result.get('personality_traits', {})
            if personality_data:
                dominant_traits = personality_data.get('dominant_traits', [])
                if dominant_traits:
                    trait_names = [trait['trait'] for trait in dominant_traits[:2]]
                    summary_parts.append(f"Shows strong {' and '.join(trait_names)} characteristics")
            
            # Emotional tone
            emotional_data = analysis_result.get('emotional_characteristics', {})
            if emotional_data:
                tone = emotional_data.get('emotional_tone', 'neutral')
                summary_parts.append(f"Has a {tone} emotional tone")
            
            # Energy level
            energy_data = analysis_result.get('energy_level', {})
            if energy_data:
                energy_level = energy_data.get('energy_level', 'moderate')
                summary_parts.append(f"Displays {energy_level} energy levels")
            
            # Social orientation
            social_data = analysis_result.get('social_orientation', {})
            if social_data:
                orientation = social_data.get('social_orientation', 'balanced')
                summary_parts.append(f"Tends to be {orientation}")
            
            # Primary interests
            interests_data = analysis_result.get('interests_and_hobbies', {})
            if interests_data:
                primary_interests = interests_data.get('primary_interests', [])
                if primary_interests:
                    interest_names = [interest['category'] for interest in primary_interests[:2]]
                    summary_parts.append(f"Interested in {' and '.join(interest_names)}")
            
            if summary_parts:
                return '. '.join(summary_parts) + '.'
            else:
                return "Balanced personality with diverse interests and moderate energy levels."
                
        except Exception as e:
            self.logger.error(f"Vibe summary generation failed: {e}")
            return "Unable to generate vibe summary."
    
    def _create_personality_profile(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive personality profile"""
        try:
            profile = {
                'personality_type': self._determine_personality_type(analysis_result),
                'core_traits': self._extract_core_traits(analysis_result),
                'strengths': self._identify_strengths(analysis_result),
                'growth_areas': self._identify_growth_areas(analysis_result),
                'ideal_environments': self._suggest_ideal_environments(analysis_result),
                'compatibility_factors': self._analyze_compatibility_factors(analysis_result)
            }
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Personality profile creation failed: {e}")
            return {}
    
    def _generate_vibe_insights(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate insights about the person's vibe"""
        try:
            insights = []
            
            # Personality insights
            personality_data = analysis_result.get('personality_traits', {})
            if personality_data:
                dominant_traits = personality_data.get('dominant_traits', [])
                for trait_info in dominant_traits[:2]:
                    trait = trait_info['trait']
                    score = trait_info['score']
                    if score > 0.7:
                        insights.append(f"Strong {trait} suggests excellent {self._get_trait_benefit(trait)}")
            
            # Emotional insights
            emotional_data = analysis_result.get('emotional_characteristics', {})
            if emotional_data:
                tone = emotional_data.get('emotional_tone', 'neutral')
                if tone == 'positive':
                    insights.append("Positive emotional tone indicates resilience and optimism")
                elif tone == 'negative':
                    insights.append("Emotional awareness suggests depth and authenticity")
            
            # Interest insights
            interests_data = analysis_result.get('interests_and_hobbies', {})
            if interests_data:
                diversity = interests_data.get('interest_diversity', 0)
                if diversity > 0.6:
                    insights.append("Diverse interests indicate curiosity and adaptability")
                elif diversity < 0.3:
                    insights.append("Focused interests suggest deep expertise and passion")
            
            # Communication insights
            communication_data = analysis_result.get('communication_style', {})
            if communication_data:
                primary_style = communication_data.get('primary_style', {})
                if primary_style:
                    style = primary_style['style']
                    insights.append(f"{style.capitalize()} communication style enhances {self._get_style_benefit(style)}")
            
            return insights[:5]  # Limit to top 5 insights
            
        except Exception as e:
            self.logger.error(f"Vibe insights generation failed: {e}")
            return []
    
    def _generate_vibe_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for enhancing vibe"""
        try:
            recommendations = []
            
            # Based on personality traits
            personality_data = analysis_result.get('personality_traits', {})
            if personality_data:
                trait_scores = personality_data.get('trait_scores', {})
                
                # Low extroversion recommendations
                if trait_scores.get('extroversion', 0) < 0.3:
                    recommendations.append("Consider joining social groups or activities to expand your network")
                
                # Low openness recommendations
                if trait_scores.get('openness', 0) < 0.3:
                    recommendations.append("Try new experiences or creative activities to broaden your perspective")
                
                # Low optimism recommendations
                if trait_scores.get('optimism', 0) < 0.3:
                    recommendations.append("Practice gratitude and positive thinking exercises")
            
            # Based on emotional characteristics
            emotional_data = analysis_result.get('emotional_characteristics', {})
            if emotional_data:
                negative_score = emotional_data.get('negative_score', 0)
                if negative_score > 0.5:
                    recommendations.append("Consider mindfulness or stress management techniques")
            
            # Based on interests
            interests_data = analysis_result.get('interests_and_hobbies', {})
            if interests_data:
                diversity = interests_data.get('interest_diversity', 0)
                if diversity < 0.3:
                    recommendations.append("Explore new hobbies or interests to add variety to your life")
            
            # Based on energy level
            energy_data = analysis_result.get('energy_level', {})
            if energy_data:
                energy_level = energy_data.get('energy_level', 'moderate')
                if energy_level == 'low':
                    recommendations.append("Consider activities that boost energy like exercise or outdoor time")
            
            # General recommendations
            recommendations.extend([
                "Showcase your authentic personality in social interactions",
                "Develop your unique strengths and interests",
                "Practice active listening to enhance communication",
                "Maintain a balance between different aspects of your personality"
            ])
            
            return recommendations[:6]  # Limit to top 6 recommendations
            
        except Exception as e:
            self.logger.error(f"Vibe recommendations generation failed: {e}")
            return []
    
    # Helper methods
    
    def _calculate_analysis_confidence(self, bio_text: str, additional_context: Dict[str, Any]) -> float:
        """Calculate confidence in the analysis"""
        try:
            confidence_factors = []
            
            # Text length factor
            text_length = len(bio_text)
            if text_length > 200:
                confidence_factors.append(0.9)
            elif text_length > 100:
                confidence_factors.append(0.7)
            elif text_length > 50:
                confidence_factors.append(0.5)
            else:
                confidence_factors.append(0.3)
            
            # Additional context factor
            if additional_context:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.6)
            
            # Text quality factor (based on word variety)
            words = bio_text.lower().split()
            unique_words = len(set(words))
            total_words = len(words)
            word_variety = unique_words / total_words if total_words > 0 else 0
            
            if word_variety > 0.8:
                confidence_factors.append(0.9)
            elif word_variety > 0.6:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
            
            return sum(confidence_factors) / len(confidence_factors)
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.6
    
    def _calculate_analysis_depth(self, bio_text: str, additional_context: Dict[str, Any]) -> str:
        """Calculate the depth of analysis possible"""
        try:
            text_length = len(bio_text)
            context_richness = len(additional_context)
            
            if text_length > 300 and context_richness > 3:
                return 'comprehensive'
            elif text_length > 150 and context_richness > 1:
                return 'detailed'
            elif text_length > 50:
                return 'moderate'
            else:
                return 'basic'
                
        except Exception as e:
            self.logger.error(f"Analysis depth calculation failed: {e}")
            return 'moderate'
    
    def _generate_personality_summary(self, trait_scores: Dict[str, float]) -> str:
        """Generate a personality summary"""
        try:
            top_traits = sorted(trait_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            top_traits = [trait for trait, score in top_traits if score > 0.2]
            
            if len(top_traits) >= 2:
                return f"Primarily {top_traits[0]} with {top_traits[1]} tendencies"
            elif len(top_traits) == 1:
                return f"Shows strong {top_traits[0]} characteristics"
            else:
                return "Balanced personality across multiple traits"
                
        except Exception as e:
            self.logger.error(f"Personality summary generation failed: {e}")
            return "Complex personality profile"
    
    def _generate_interest_profile(self, interest_scores: Dict[str, float]) -> str:
        """Generate an interest profile summary"""
        try:
            top_interests = sorted(interest_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            top_interests = [interest for interest, score in top_interests if score > 0.2]
            
            if len(top_interests) >= 2:
                return f"Passionate about {' and '.join(top_interests[:2])}"
            elif len(top_interests) == 1:
                return f"Strong interest in {top_interests[0]}"
            else:
                return "Diverse range of interests"
                
        except Exception as e:
            self.logger.error(f"Interest profile generation failed: {e}")
            return "Varied interests"
    
    def _calculate_communication_effectiveness(self, style_scores: Dict[str, float], text_characteristics: Dict[str, Any]) -> float:
        """Calculate communication effectiveness"""
        try:
            effectiveness_score = 0.5  # Base score
            
            # Style diversity bonus
            active_styles = len([score for score in style_scores.values() if score > 0.2])
            if active_styles > 2:
                effectiveness_score += 0.2
            
            # Text quality factors
            if text_characteristics:
                word_count = text_characteristics.get('word_count', 0)
                if 50 <= word_count <= 200:  # Optimal length
                    effectiveness_score += 0.1
                
                writing_style = text_characteristics.get('writing_style', '')
                if writing_style in ['sophisticated', 'detailed']:
                    effectiveness_score += 0.1
                elif writing_style == 'enthusiastic':
                    effectiveness_score += 0.05
            
            return min(effectiveness_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Communication effectiveness calculation failed: {e}")
            return 0.5
    
    def _calculate_text_complexity(self, avg_word_length: float, avg_sentence_length: float) -> str:
        """Calculate text complexity level"""
        try:
            if avg_word_length > 6 and avg_sentence_length > 20:
                return 'high'
            elif avg_word_length > 4 and avg_sentence_length > 12:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            self.logger.error(f"Text complexity calculation failed: {e}")
            return 'medium'
    
    def _generate_vibe_blend(self, vibe_scores: Dict[str, float]) -> str:
        """Generate a description of the vibe blend"""
        try:
            top_vibes = sorted(vibe_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            top_vibes = [vibe for vibe, score in top_vibes if score > 0.3]
            
            if len(top_vibes) >= 2:
                return f"{top_vibes[0]}-{top_vibes[1]} blend"
            elif len(top_vibes) == 1:
                return f"Primarily {top_vibes[0]}"
            else:
                return "Balanced vibe profile"
                
        except Exception as e:
            self.logger.error(f"Vibe blend generation failed: {e}")
            return "Complex vibe"
    
    def _infer_life_philosophy(self, approach_scores: Dict[str, float]) -> str:
        """Infer life philosophy from approach scores"""
        try:
            top_approach = max(approach_scores.items(), key=lambda x: x[1])
            
            philosophy_map = {
                'goal_oriented': 'Achievement-focused with clear objectives',
                'experience_oriented': 'Adventure-seeking and experience-rich',
                'relationship_oriented': 'Connection-focused and community-minded',
                'growth_oriented': 'Learning-focused and self-improving',
                'stability_oriented': 'Security-focused and consistency-valued'
            }
            
            return philosophy_map.get(top_approach[0], 'Balanced life approach')
            
        except Exception as e:
            self.logger.error(f"Life philosophy inference failed: {e}")
            return "Thoughtful life approach"
    
    def _determine_personality_type(self, analysis_result: Dict[str, Any]) -> str:
        """Determine overall personality type"""
        try:
            # Simplified personality typing based on dominant characteristics
            personality_data = analysis_result.get('personality_traits', {})
            social_data = analysis_result.get('social_orientation', {})
            energy_data = analysis_result.get('energy_level', {})
            
            if not personality_data:
                return 'Balanced'
            
            trait_scores = personality_data.get('trait_scores', {})
            orientation = social_data.get('social_orientation', 'balanced')
            energy_level = energy_data.get('energy_level', 'moderate')
            
            # Determine type based on key characteristics
            if trait_scores.get('extroversion', 0) > 0.6 and energy_level == 'high':
                return 'Energetic Extrovert'
            elif trait_scores.get('introversion', 0) > 0.6 and trait_scores.get('openness', 0) > 0.6:
                return 'Creative Introvert'
            elif trait_scores.get('conscientiousness', 0) > 0.6 and trait_scores.get('ambition', 0) > 0.6:
                return 'Driven Achiever'
            elif trait_scores.get('agreeableness', 0) > 0.6 and orientation == 'extroverted':
                return 'Social Connector'
            elif trait_scores.get('openness', 0) > 0.6 and trait_scores.get('intelligence', 0) > 0.6:
                return 'Intellectual Explorer'
            else:
                return 'Balanced Individual'
                
        except Exception as e:
            self.logger.error(f"Personality type determination failed: {e}")
            return 'Complex Personality'
    
    def _extract_core_traits(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Extract core personality traits"""
        try:
            personality_data = analysis_result.get('personality_traits', {})
            if not personality_data:
                return []
            
            dominant_traits = personality_data.get('dominant_traits', [])
            return [trait['trait'] for trait in dominant_traits[:3]]
            
        except Exception as e:
            self.logger.error(f"Core traits extraction failed: {e}")
            return []
    
    def _identify_strengths(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Identify personality strengths"""
        try:
            strengths = []
            
            # From personality traits
            personality_data = analysis_result.get('personality_traits', {})
            if personality_data:
                trait_scores = personality_data.get('trait_scores', {})
                
                if trait_scores.get('extroversion', 0) > 0.6:
                    strengths.append('Strong social skills')
                if trait_scores.get('openness', 0) > 0.6:
                    strengths.append('Creative and innovative thinking')
                if trait_scores.get('conscientiousness', 0) > 0.6:
                    strengths.append('Reliable and organized')
                if trait_scores.get('agreeableness', 0) > 0.6:
                    strengths.append('Empathetic and collaborative')
                if trait_scores.get('optimism', 0) > 0.6:
                    strengths.append('Positive outlook and resilience')
            
            # From interests
            interests_data = analysis_result.get('interests_and_hobbies', {})
            if interests_data:
                diversity = interests_data.get('interest_diversity', 0)
                if diversity > 0.6:
                    strengths.append('Diverse interests and adaptability')
            
            return strengths[:4]  # Limit to top 4 strengths
            
        except Exception as e:
            self.logger.error(f"Strengths identification failed: {e}")
            return []
    
    def _identify_growth_areas(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Identify areas for growth"""
        try:
            growth_areas = []
            
            personality_data = analysis_result.get('personality_traits', {})
            if personality_data:
                trait_scores = personality_data.get('trait_scores', {})
                
                if trait_scores.get('extroversion', 0) < 0.3:
                    growth_areas.append('Developing social confidence')
                if trait_scores.get('openness', 0) < 0.3:
                    growth_areas.append('Embracing new experiences')
                if trait_scores.get('conscientiousness', 0) < 0.3:
                    growth_areas.append('Improving organization and planning')
                if trait_scores.get('optimism', 0) < 0.3:
                    growth_areas.append('Building positive mindset')
            
            return growth_areas[:3]  # Limit to top 3 growth areas
            
        except Exception as e:
            self.logger.error(f"Growth areas identification failed: {e}")
            return []
    
    def _suggest_ideal_environments(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Suggest ideal environments for the person"""
        try:
            environments = []
            
            social_data = analysis_result.get('social_orientation', {})
            energy_data = analysis_result.get('energy_level', {})
            interests_data = analysis_result.get('interests_and_hobbies', {})
            
            orientation = social_data.get('social_orientation', 'balanced')
            energy_level = energy_data.get('energy_level', 'moderate')
            
            if orientation == 'extroverted':
                environments.append('Collaborative team environments')
                environments.append('Social and networking events')
            elif orientation == 'introverted':
                environments.append('Quiet, focused work spaces')
                environments.append('Small group or one-on-one settings')
            
            if energy_level == 'high':
                environments.append('Dynamic, fast-paced environments')
            elif energy_level == 'low':
                environments.append('Calm, structured environments')
            
            # Based on interests
            if interests_data:
                primary_interests = interests_data.get('primary_interests', [])
                for interest in primary_interests[:2]:
                    category = interest['category']
                    if category == 'technology':
                        environments.append('Innovation-focused workplaces')
                    elif category == 'arts':
                        environments.append('Creative and artistic spaces')
                    elif category == 'nature':
                        environments.append('Outdoor or nature-connected settings')
            
            return environments[:4]  # Limit to top 4 environments
            
        except Exception as e:
            self.logger.error(f"Ideal environments suggestion failed: {e}")
            return []
    
    def _analyze_compatibility_factors(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze compatibility factors"""
        try:
            compatibility = {
                'communication_style': [],
                'energy_matching': [],
                'interest_alignment': [],
                'personality_complement': []
            }
            
            # Communication compatibility
            communication_data = analysis_result.get('communication_style', {})
            if communication_data:
                primary_style = communication_data.get('primary_style', {})
                if primary_style:
                    style = primary_style['style']
                    compatibility['communication_style'].append(f"Works well with {style} communicators")
            
            # Energy compatibility
            energy_data = analysis_result.get('energy_level', {})
            if energy_data:
                energy_level = energy_data.get('energy_level', 'moderate')
                compatibility['energy_matching'].append(f"Compatible with {energy_level} energy individuals")
            
            # Interest compatibility
            interests_data = analysis_result.get('interests_and_hobbies', {})
            if interests_data:
                primary_interests = interests_data.get('primary_interests', [])
                for interest in primary_interests[:2]:
                    category = interest['category']
                    compatibility['interest_alignment'].append(f"Connects with {category} enthusiasts")
            
            return compatibility
            
        except Exception as e:
            self.logger.error(f"Compatibility factors analysis failed: {e}")
            return {}
    
    def _get_trait_benefit(self, trait: str) -> str:
        """Get the benefit of a personality trait"""
        benefits = {
            'extroversion': 'social networking and team collaboration',
            'introversion': 'deep thinking and focused work',
            'openness': 'creativity and adaptability',
            'conscientiousness': 'reliability and goal achievement',
            'agreeableness': 'relationship building and teamwork',
            'optimism': 'resilience and positive influence',
            'intelligence': 'problem-solving and strategic thinking',
            'humor': 'social connection and stress relief',
            'ambition': 'goal achievement and leadership'
        }
        return benefits.get(trait, 'personal development')
    
    def _get_style_benefit(self, style: str) -> str:
        """Get the benefit of a communication style"""
        benefits = {
            'formal': 'professional relationships and credibility',
            'casual': 'approachability and comfort',
            'humorous': 'social connection and engagement',
            'intellectual': 'depth and thoughtful discourse',
            'emotional': 'authenticity and connection',
            'direct': 'clarity and efficiency',
            'diplomatic': 'conflict resolution and harmony'
        }
        return benefits.get(style, 'effective communication')
