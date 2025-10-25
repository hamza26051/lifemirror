import re
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentInput, AgentOutput

class BioAgent(BaseAgent):
    """Bio and text analysis agent for profile descriptions"""
    
    def __init__(self):
        super().__init__()
        self.sentiment_keywords = {
            'positive': ['happy', 'love', 'amazing', 'great', 'awesome', 'fantastic', 'wonderful', 
                        'excellent', 'perfect', 'beautiful', 'incredible', 'outstanding', 'brilliant'],
            'negative': ['sad', 'hate', 'terrible', 'awful', 'horrible', 'disgusting', 'worst', 
                        'pathetic', 'useless', 'annoying', 'frustrating', 'disappointing'],
            'confident': ['confident', 'strong', 'powerful', 'leader', 'successful', 'ambitious', 
                         'determined', 'fearless', 'bold', 'assertive', 'independent'],
            'creative': ['creative', 'artistic', 'innovative', 'imaginative', 'original', 
                        'unique', 'expressive', 'inspired', 'visionary', 'inventive'],
            'social': ['friendly', 'outgoing', 'social', 'extroverted', 'talkative', 'charismatic', 
                      'popular', 'networking', 'community', 'team', 'collaborative'],
            'intellectual': ['smart', 'intelligent', 'genius', 'brilliant', 'analytical', 
                           'logical', 'strategic', 'thoughtful', 'wise', 'knowledgeable']
        }
        
    def run(self, input: AgentInput) -> AgentOutput:
        """Analyze bio text for personality traits and sentiment"""
        try:
            # Get bio text from context
            bio_text = input.context.get('bio_text', '')
            
            if not bio_text or not isinstance(bio_text, str):
                return self._create_output(
                    success=False,
                    data={},
                    error="No bio text provided",
                    confidence=0.0
                )
            
            # Perform comprehensive bio analysis
            analysis_result = self._analyze_bio_text(bio_text)
            
            return self._create_output(
                success=True,
                data=analysis_result,
                confidence=analysis_result.get('confidence', 0.5)
            )
            
        except Exception as e:
            return self._handle_error(e, input)
    
    def _analyze_bio_text(self, bio_text: str) -> Dict[str, Any]:
        """Comprehensive bio text analysis"""
        try:
            # Clean and normalize text
            cleaned_text = self._clean_text(bio_text)
            
            # Basic text statistics
            text_stats = self._calculate_text_stats(cleaned_text)
            
            # Sentiment analysis
            sentiment_analysis = self._analyze_sentiment(cleaned_text)
            
            # Personality trait detection
            personality_traits = self._detect_personality_traits(cleaned_text)
            
            # Interest and hobby detection
            interests = self._detect_interests(cleaned_text)
            
            # Communication style analysis
            communication_style = self._analyze_communication_style(cleaned_text)
            
            # Calculate overall vibe score
            vibe_score = self._calculate_vibe_score(sentiment_analysis, personality_traits)
            
            # Generate insights
            insights = self._generate_insights(sentiment_analysis, personality_traits, interests)
            
            # Calculate confidence
            confidence = self._calculate_analysis_confidence(text_stats, sentiment_analysis)
            
            return {
                'text_stats': text_stats,
                'sentiment_analysis': sentiment_analysis,
                'personality_traits': personality_traits,
                'interests': interests,
                'communication_style': communication_style,
                'vibe_score': vibe_score,
                'insights': insights,
                'confidence': confidence,
                'analysis_method': 'keyword_based'
            }
            
        except Exception as e:
            self.logger.error(f"Bio analysis failed: {e}")
            return {
                'text_stats': {},
                'sentiment_analysis': {},
                'personality_traits': {},
                'interests': [],
                'communication_style': {},
                'vibe_score': 0.5,
                'insights': [],
                'confidence': 0.0,
                'analysis_method': 'failed',
                'error': str(e)
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
        
        return text.strip()
    
    def _calculate_text_stats(self, text: str) -> Dict[str, Any]:
        """Calculate basic text statistics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using keyword matching"""
        words = text.split()
        
        sentiment_scores = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        
        # Count sentiment keywords
        for word in words:
            if word in self.sentiment_keywords['positive']:
                sentiment_scores['positive'] += 1
            elif word in self.sentiment_keywords['negative']:
                sentiment_scores['negative'] += 1
            else:
                sentiment_scores['neutral'] += 1
        
        # Calculate percentages
        total_words = len(words)
        if total_words > 0:
            sentiment_percentages = {
                key: (count / total_words) * 100 
                for key, count in sentiment_scores.items()
            }
        else:
            sentiment_percentages = {'positive': 33.3, 'negative': 33.3, 'neutral': 33.3}
        
        # Determine dominant sentiment
        dominant_sentiment = max(sentiment_percentages, key=sentiment_percentages.get)
        
        # Calculate overall sentiment score (-1 to 1)
        sentiment_score = (
            sentiment_percentages['positive'] - sentiment_percentages['negative']
        ) / 100
        
        return {
            'scores': sentiment_scores,
            'percentages': sentiment_percentages,
            'dominant_sentiment': dominant_sentiment,
            'sentiment_score': sentiment_score,
            'confidence': min(abs(sentiment_score) + 0.3, 1.0)
        }
    
    def _detect_personality_traits(self, text: str) -> Dict[str, Any]:
        """Detect personality traits from text"""
        words = text.split()
        
        trait_scores = {}
        
        for trait, keywords in self.sentiment_keywords.items():
            if trait in ['positive', 'negative']:  # Skip sentiment categories
                continue
                
            score = 0
            matched_keywords = []
            
            for word in words:
                if word in keywords:
                    score += 1
                    matched_keywords.append(word)
            
            # Normalize score
            normalized_score = min(score / max(len(words) * 0.1, 1), 1.0)
            
            trait_scores[trait] = {
                'score': normalized_score,
                'raw_count': score,
                'matched_keywords': matched_keywords,
                'confidence': min(normalized_score + 0.2, 1.0) if score > 0 else 0.1
            }
        
        # Find dominant traits
        dominant_traits = sorted(
            trait_scores.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )[:3]
        
        return {
            'trait_scores': trait_scores,
            'dominant_traits': [trait for trait, _ in dominant_traits if _['score'] > 0.1],
            'trait_summary': self._generate_trait_summary(trait_scores)
        }
    
    def _detect_interests(self, text: str) -> List[Dict[str, Any]]:
        """Detect interests and hobbies from text"""
        interest_keywords = {
            'sports': ['sports', 'football', 'basketball', 'soccer', 'tennis', 'gym', 'fitness', 
                      'running', 'swimming', 'cycling', 'yoga', 'workout'],
            'music': ['music', 'singing', 'guitar', 'piano', 'drums', 'concert', 'band', 
                     'song', 'album', 'artist', 'musician'],
            'travel': ['travel', 'traveling', 'vacation', 'trip', 'adventure', 'explore', 
                      'journey', 'wanderlust', 'backpacking', 'tourism'],
            'technology': ['tech', 'technology', 'coding', 'programming', 'computer', 'software', 
                          'app', 'digital', 'innovation', 'startup'],
            'arts': ['art', 'painting', 'drawing', 'photography', 'design', 'creative', 
                    'artistic', 'gallery', 'exhibition', 'craft'],
            'food': ['food', 'cooking', 'chef', 'restaurant', 'cuisine', 'recipe', 
                    'foodie', 'culinary', 'baking', 'dining'],
            'reading': ['reading', 'books', 'literature', 'novel', 'author', 'library', 
                       'bookworm', 'story', 'poetry', 'writing'],
            'nature': ['nature', 'hiking', 'camping', 'outdoors', 'mountains', 'beach', 
                      'forest', 'wildlife', 'environment', 'conservation']
        }
        
        detected_interests = []
        words = text.split()
        
        for interest_category, keywords in interest_keywords.items():
            matches = [word for word in words if word in keywords]
            
            if matches:
                confidence = min(len(matches) / len(keywords), 1.0)
                detected_interests.append({
                    'category': interest_category,
                    'matched_keywords': matches,
                    'confidence': confidence,
                    'relevance_score': len(matches) / len(words) if words else 0
                })
        
        # Sort by relevance
        detected_interests.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return detected_interests
    
    def _analyze_communication_style(self, text: str) -> Dict[str, Any]:
        """Analyze communication style"""
        # Count different types of punctuation and patterns
        exclamation_count = text.count('!')
        question_count = text.count('?')
        emoji_pattern = re.compile(r'[😀-🙏]|[🌀-🗿]|[🚀-🛿]|[🇀-🇿]')
        emoji_count = len(emoji_pattern.findall(text))
        
        words = text.split()
        word_count = len(words)
        
        # Calculate style metrics
        enthusiasm_score = min(exclamation_count / max(word_count * 0.1, 1), 1.0)
        curiosity_score = min(question_count / max(word_count * 0.1, 1), 1.0)
        expressiveness_score = min(emoji_count / max(word_count * 0.05, 1), 1.0)
        
        # Determine communication style
        if enthusiasm_score > 0.3:
            style = 'enthusiastic'
        elif curiosity_score > 0.2:
            style = 'inquisitive'
        elif expressiveness_score > 0.2:
            style = 'expressive'
        else:
            style = 'casual'
        
        return {
            'style': style,
            'enthusiasm_score': enthusiasm_score,
            'curiosity_score': curiosity_score,
            'expressiveness_score': expressiveness_score,
            'metrics': {
                'exclamation_count': exclamation_count,
                'question_count': question_count,
                'emoji_count': emoji_count
            }
        }
    
    def _calculate_vibe_score(self, sentiment_analysis: Dict, personality_traits: Dict) -> float:
        """Calculate overall vibe score"""
        # Base score from sentiment
        sentiment_score = sentiment_analysis.get('sentiment_score', 0)
        base_score = (sentiment_score + 1) / 2  # Convert from -1,1 to 0,1
        
        # Boost from positive personality traits
        trait_boost = 0
        positive_traits = ['confident', 'creative', 'social', 'intellectual']
        
        for trait in positive_traits:
            if trait in personality_traits.get('trait_scores', {}):
                trait_score = personality_traits['trait_scores'][trait]['score']
                trait_boost += trait_score * 0.1  # Each trait can add up to 0.1
        
        # Combine scores
        vibe_score = min(base_score + trait_boost, 1.0)
        
        return max(vibe_score, 0.1)  # Minimum score of 0.1
    
    def _generate_trait_summary(self, trait_scores: Dict) -> str:
        """Generate a summary of personality traits"""
        dominant_traits = []
        
        for trait, data in trait_scores.items():
            if data['score'] > 0.2:
                dominant_traits.append(trait)
        
        if not dominant_traits:
            return "Balanced personality with no dominant traits"
        elif len(dominant_traits) == 1:
            return f"Primarily {dominant_traits[0]} personality"
        else:
            return f"Mix of {', '.join(dominant_traits[:-1])} and {dominant_traits[-1]} traits"
    
    def _generate_insights(self, sentiment_analysis: Dict, personality_traits: Dict, interests: List) -> List[str]:
        """Generate insights about the person"""
        insights = []
        
        # Sentiment insights
        dominant_sentiment = sentiment_analysis.get('dominant_sentiment', 'neutral')
        if dominant_sentiment == 'positive':
            insights.append("Shows a positive and optimistic outlook")
        elif dominant_sentiment == 'negative':
            insights.append("May be going through challenging times")
        
        # Personality insights
        dominant_traits = personality_traits.get('dominant_traits', [])
        if 'confident' in dominant_traits:
            insights.append("Displays confidence and self-assurance")
        if 'creative' in dominant_traits:
            insights.append("Has a creative and artistic mindset")
        if 'social' in dominant_traits:
            insights.append("Enjoys social interactions and connections")
        if 'intellectual' in dominant_traits:
            insights.append("Values knowledge and intellectual pursuits")
        
        # Interest insights
        if interests:
            top_interest = interests[0]['category']
            insights.append(f"Shows strong interest in {top_interest}")
        
        if not insights:
            insights.append("Has a balanced and well-rounded personality")
        
        return insights
    
    def _calculate_analysis_confidence(self, text_stats: Dict, sentiment_analysis: Dict) -> float:
        """Calculate confidence in the analysis"""
        confidence_factors = []
        
        # Text length factor
        word_count = text_stats.get('word_count', 0)
        if word_count > 50:
            confidence_factors.append(0.9)
        elif word_count > 20:
            confidence_factors.append(0.7)
        elif word_count > 5:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.2)
        
        # Sentiment confidence
        sentiment_confidence = sentiment_analysis.get('confidence', 0.5)
        confidence_factors.append(sentiment_confidence)
        
        # Average confidence
        return sum(confidence_factors) / len(confidence_factors)
