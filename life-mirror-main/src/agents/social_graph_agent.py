import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from src.agents.base_agent import BaseAgent, AgentInput, AgentOutput
from src.db.session import get_db
from src.db.models import Media, Analysis, User
from src.utils.tracing import log_trace
import numpy as np
import logging
from collections import defaultdict

class PeerComparison(BaseModel):
    """Model for peer comparison data"""
    peer_id: str = Field(description="Peer user ID")
    similarity_score: float = Field(description="Similarity to current user (0-1)")
    performance_comparison: Dict[str, float] = Field(description="Performance comparison by metric")
    strengths: List[str] = Field(description="Peer's strengths")
    learning_opportunities: List[str] = Field(description="What can be learned from this peer")

class RankingData(BaseModel):
    """Model for ranking information"""
    metric: str = Field(description="Metric being ranked")
    user_score: float = Field(description="User's score")
    percentile: float = Field(description="User's percentile (0-100)")
    rank: int = Field(description="User's rank")
    total_users: int = Field(description="Total users in comparison")
    category: str = Field(description="Ranking category")

class SocialInsight(BaseModel):
    """Model for social insights"""
    insight_type: str = Field(description="Type of insight")
    title: str = Field(description="Insight title")
    description: str = Field(description="Detailed description")
    confidence: float = Field(description="Confidence in insight (0-1)")
    actionable: bool = Field(description="Whether insight is actionable")
    priority: str = Field(description="Priority level (high/medium/low)")

class NetworkMetrics(BaseModel):
    """Model for network metrics"""
    network_size: int = Field(description="Size of user's network")
    connection_strength: float = Field(description="Average connection strength")
    influence_score: float = Field(description="User's influence in network")
    centrality_score: float = Field(description="Network centrality score")
    clustering_coefficient: float = Field(description="Local clustering coefficient")

class SocialGraphResult(BaseModel):
    """Result model for social graph analysis"""
    peer_comparisons: List[PeerComparison] = Field(description="Peer comparison data")
    rankings: List[RankingData] = Field(description="User rankings across metrics")
    network_metrics: NetworkMetrics = Field(description="Network analysis metrics")
    social_insights: List[SocialInsight] = Field(description="Social insights")
    improvement_opportunities: List[str] = Field(description="Social improvement opportunities")
    peer_learning_suggestions: List[str] = Field(description="Learning suggestions from peers")
    network_growth_potential: float = Field(description="Potential for network growth (0-1)")
    social_influence_trend: str = Field(description="Trend in social influence")
    confidence: float = Field(description="Overall analysis confidence")
    data_completeness: str = Field(description="Completeness of social data")

class SocialGraphAgent(BaseAgent):
    """Agent for social network analysis and peer comparisons"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Analysis parameters
        self.min_peers_for_ranking = 5
        self.similarity_threshold = 0.7
        self.max_peer_comparisons = 10
        
        # Metric weights for peer similarity
        self.similarity_weights = {
            'overall_score': 0.3,
            'fashion_score': 0.2,
            'face_score': 0.2,
            'posture_score': 0.15,
            'vibe_score': 0.15
        }
        
        # Network analysis parameters
        self.connection_decay_days = 30
        self.influence_factors = ['engagement', 'reach', 'quality']
    
    @log_trace
    def run(self, input_data: AgentInput) -> AgentOutput:
        """Analyze social graph and peer relationships"""
        try:
            # Get user and peer data
            user_data = self._get_user_data(input_data.media_id)
            if not user_data:
                return self._create_no_data_response()
            
            peer_data = self._get_peer_data(user_data['user_id'])
            
            # Perform peer comparisons
            peer_comparisons = self._analyze_peer_comparisons(user_data, peer_data)
            
            # Calculate rankings
            rankings = self._calculate_rankings(user_data, peer_data)
            
            # Analyze network metrics
            network_metrics = self._analyze_network_metrics(user_data['user_id'], peer_data)
            
            # Generate social insights
            social_insights = self._generate_social_insights(peer_comparisons, rankings, network_metrics)
            
            # Identify improvement opportunities
            improvement_opportunities = self._identify_improvement_opportunities(peer_comparisons, rankings)
            
            # Generate peer learning suggestions
            peer_learning_suggestions = self._generate_peer_learning_suggestions(peer_comparisons)
            
            # Calculate network growth potential
            network_growth_potential = self._calculate_network_growth_potential(network_metrics, peer_data)
            
            # Analyze social influence trend
            social_influence_trend = self._analyze_influence_trend(user_data['user_id'])
            
            # Assess data completeness and confidence
            data_completeness = self._assess_data_completeness(peer_data)
            confidence = self._calculate_confidence(peer_comparisons, rankings, data_completeness)
            
            result = SocialGraphResult(
                peer_comparisons=peer_comparisons,
                rankings=rankings,
                network_metrics=network_metrics,
                social_insights=social_insights,
                improvement_opportunities=improvement_opportunities,
                peer_learning_suggestions=peer_learning_suggestions,
                network_growth_potential=network_growth_potential,
                social_influence_trend=social_influence_trend,
                confidence=confidence,
                data_completeness=data_completeness
            )
            
            return AgentOutput(
                success=True,
                data=result.dict(),
                confidence=confidence,
                processing_time=0.0,
                agent_name="social_graph"
            )
            
        except Exception as e:
            self.logger.error(f"Social graph analysis failed: {e}")
            return AgentOutput(
                success=False,
                data={},
                confidence=0.0,
                processing_time=0.0,
                agent_name="social_graph",
                error=str(e)
            )
    
    def _get_user_data(self, media_id: str) -> Optional[Dict[str, Any]]:
        """Get user data and latest analysis"""
        try:
            with get_db() as db:
                # Get media and user
                media = db.query(Media).filter(Media.id == media_id).first()
                if not media:
                    return None
                
                # Get latest analysis for this user
                latest_analysis = db.query(Analysis).join(Media).filter(
                    Media.user_id == media.user_id
                ).order_by(Analysis.created_at.desc()).first()
                
                if not latest_analysis or not latest_analysis.result_data:
                    return None
                
                return {
                    'user_id': media.user_id,
                    'media_id': media_id,
                    'analysis_data': latest_analysis.result_data,
                    'created_at': latest_analysis.created_at
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get user data: {e}")
            return None
    
    def _get_peer_data(self, user_id: str) -> List[Dict[str, Any]]:
        """Get peer data for comparison"""
        try:
            with get_db() as db:
                # Get recent analyses from other users
                peer_analyses = db.query(Analysis).join(Media).filter(
                    Media.user_id != user_id,
                    Analysis.created_at >= datetime.now() - timedelta(days=30),
                    Analysis.result_data.isnot(None)
                ).order_by(Analysis.created_at.desc()).limit(100).all()
                
                peer_data = []
                user_analyses = defaultdict(list)
                
                # Group by user and get latest analysis per user
                for analysis in peer_analyses:
                    user_analyses[analysis.media.user_id].append({
                        'user_id': analysis.media.user_id,
                        'analysis_data': analysis.result_data,
                        'created_at': analysis.created_at
                    })
                
                # Get latest analysis per user
                for peer_user_id, analyses in user_analyses.items():
                    latest = max(analyses, key=lambda x: x['created_at'])
                    peer_data.append(latest)
                
                return peer_data
                
        except Exception as e:
            self.logger.error(f"Failed to get peer data: {e}")
            return []
    
    def _analyze_peer_comparisons(self, user_data: Dict[str, Any], peer_data: List[Dict[str, Any]]) -> List[PeerComparison]:
        """Analyze comparisons with peers"""
        comparisons = []
        user_scores = self._extract_scores(user_data['analysis_data'])
        
        for peer in peer_data:
            peer_scores = self._extract_scores(peer['analysis_data'])
            
            # Calculate similarity
            similarity = self._calculate_similarity(user_scores, peer_scores)
            
            if similarity >= self.similarity_threshold:
                # Calculate performance comparison
                performance_comparison = self._calculate_performance_comparison(user_scores, peer_scores)
                
                # Identify peer strengths
                strengths = self._identify_peer_strengths(user_scores, peer_scores)
                
                # Generate learning opportunities
                learning_opportunities = self._generate_learning_opportunities(user_scores, peer_scores, strengths)
                
                comparison = PeerComparison(
                    peer_id=peer['user_id'],
                    similarity_score=similarity,
                    performance_comparison=performance_comparison,
                    strengths=strengths,
                    learning_opportunities=learning_opportunities
                )
                
                comparisons.append(comparison)
        
        # Sort by similarity and limit
        comparisons.sort(key=lambda x: x.similarity_score, reverse=True)
        return comparisons[:self.max_peer_comparisons]
    
    def _extract_scores(self, analysis_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract scores from analysis data"""
        scores = {}
        
        # Overall score
        if 'overall_score' in analysis_data:
            scores['overall_score'] = analysis_data['overall_score']
        
        # Individual agent scores
        if 'individual_results' in analysis_data:
            for agent, result in analysis_data['individual_results'].items():
                if isinstance(result, dict) and 'score' in result:
                    scores[f"{agent}_score"] = result['score']
        
        return scores
    
    def _calculate_similarity(self, user_scores: Dict[str, float], peer_scores: Dict[str, float]) -> float:
        """Calculate similarity between user and peer"""
        common_metrics = set(user_scores.keys()) & set(peer_scores.keys())
        
        if not common_metrics:
            return 0.0
        
        weighted_similarity = 0.0
        total_weight = 0.0
        
        for metric in common_metrics:
            weight = self.similarity_weights.get(metric, 0.1)
            
            # Calculate similarity for this metric (inverse of absolute difference)
            diff = abs(user_scores[metric] - peer_scores[metric])
            similarity = 1.0 - min(diff, 1.0)
            
            weighted_similarity += weight * similarity
            total_weight += weight
        
        return weighted_similarity / total_weight if total_weight > 0 else 0.0
    
    def _calculate_performance_comparison(self, user_scores: Dict[str, float], peer_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance comparison"""
        comparison = {}
        
        for metric in user_scores:
            if metric in peer_scores:
                # Positive value means user is better, negative means peer is better
                comparison[metric] = user_scores[metric] - peer_scores[metric]
        
        return comparison
    
    def _identify_peer_strengths(self, user_scores: Dict[str, float], peer_scores: Dict[str, float]) -> List[str]:
        """Identify areas where peer is stronger"""
        strengths = []
        
        for metric, peer_score in peer_scores.items():
            if metric in user_scores:
                if peer_score > user_scores[metric] + 0.1:  # Significant difference
                    area = metric.replace('_score', '')
                    strengths.append(area)
        
        return strengths
    
    def _generate_learning_opportunities(self, user_scores: Dict[str, float], peer_scores: Dict[str, float], strengths: List[str]) -> List[str]:
        """Generate learning opportunities from peer"""
        opportunities = []
        
        for strength in strengths:
            metric = f"{strength}_score"
            if metric in user_scores and metric in peer_scores:
                improvement_potential = peer_scores[metric] - user_scores[metric]
                if improvement_potential > 0.2:
                    opportunities.append(f"Learn {strength} techniques to improve by {improvement_potential:.1%}")
                elif improvement_potential > 0.1:
                    opportunities.append(f"Study {strength} approaches for moderate improvement")
        
        return opportunities
    
    def _calculate_rankings(self, user_data: Dict[str, Any], peer_data: List[Dict[str, Any]]) -> List[RankingData]:
        """Calculate user rankings across metrics"""
        rankings = []
        user_scores = self._extract_scores(user_data['analysis_data'])
        
        # Collect all peer scores
        all_scores = defaultdict(list)
        for peer in peer_data:
            peer_scores = self._extract_scores(peer['analysis_data'])
            for metric, score in peer_scores.items():
                all_scores[metric].append(score)
        
        # Calculate rankings for each metric
        for metric, user_score in user_scores.items():
            if metric in all_scores and len(all_scores[metric]) >= self.min_peers_for_ranking:
                peer_scores = all_scores[metric]
                all_scores_with_user = peer_scores + [user_score]
                all_scores_with_user.sort(reverse=True)
                
                rank = all_scores_with_user.index(user_score) + 1
                total_users = len(all_scores_with_user)
                percentile = ((total_users - rank) / (total_users - 1)) * 100 if total_users > 1 else 50
                
                category = self._determine_ranking_category(percentile)
                
                ranking = RankingData(
                    metric=metric,
                    user_score=user_score,
                    percentile=percentile,
                    rank=rank,
                    total_users=total_users,
                    category=category
                )
                
                rankings.append(ranking)
        
        return rankings
    
    def _determine_ranking_category(self, percentile: float) -> str:
        """Determine ranking category based on percentile"""
        if percentile >= 90:
            return "top_performer"
        elif percentile >= 75:
            return "above_average"
        elif percentile >= 50:
            return "average"
        elif percentile >= 25:
            return "below_average"
        else:
            return "needs_improvement"
    
    def _analyze_network_metrics(self, user_id: str, peer_data: List[Dict[str, Any]]) -> NetworkMetrics:
        """Analyze network metrics"""
        # Simplified network analysis (in a real implementation, this would use actual social connections)
        network_size = len(peer_data)
        
        # Calculate average connection strength (based on similarity)
        connection_strengths = []
        for peer in peer_data[:20]:  # Sample for performance
            # This would be based on actual interaction data
            strength = np.random.uniform(0.3, 0.9)  # Placeholder
            connection_strengths.append(strength)
        
        avg_connection_strength = np.mean(connection_strengths) if connection_strengths else 0.0
        
        # Calculate influence score (simplified)
        influence_score = min(network_size / 100, 1.0) * avg_connection_strength
        
        # Calculate centrality (simplified)
        centrality_score = min(network_size / 50, 1.0) * 0.8
        
        # Calculate clustering coefficient (simplified)
        clustering_coefficient = np.random.uniform(0.2, 0.8)  # Placeholder
        
        return NetworkMetrics(
            network_size=network_size,
            connection_strength=avg_connection_strength,
            influence_score=influence_score,
            centrality_score=centrality_score,
            clustering_coefficient=clustering_coefficient
        )
    
    def _generate_social_insights(self, peer_comparisons: List[PeerComparison], rankings: List[RankingData], network_metrics: NetworkMetrics) -> List[SocialInsight]:
        """Generate social insights"""
        insights = []
        
        # Ranking insights
        top_rankings = [r for r in rankings if r.category == "top_performer"]
        if top_rankings:
            insights.append(SocialInsight(
                insight_type="top_performance",
                title="Top Performer in Multiple Areas",
                description=f"You rank in the top 10% for {len(top_rankings)} metrics among your peers.",
                confidence=0.9,
                actionable=False,
                priority="high"
            ))
        
        # Improvement insights
        low_rankings = [r for r in rankings if r.category in ["below_average", "needs_improvement"]]
        if low_rankings:
            insights.append(SocialInsight(
                insight_type="improvement_opportunity",
                title="Areas for Peer-Based Improvement",
                description=f"You have improvement opportunities in {len(low_rankings)} areas compared to peers.",
                confidence=0.8,
                actionable=True,
                priority="high"
            ))
        
        # Network insights
        if network_metrics.network_size > 50:
            insights.append(SocialInsight(
                insight_type="network_size",
                title="Strong Network Presence",
                description="You have a substantial peer network for meaningful comparisons.",
                confidence=0.9,
                actionable=False,
                priority="medium"
            ))
        elif network_metrics.network_size < 10:
            insights.append(SocialInsight(
                insight_type="limited_network",
                title="Limited Peer Network",
                description="Building a larger peer network would provide better comparison insights.",
                confidence=0.8,
                actionable=True,
                priority="medium"
            ))
        
        # Peer learning insights
        high_similarity_peers = [p for p in peer_comparisons if p.similarity_score > 0.8]
        if high_similarity_peers:
            insights.append(SocialInsight(
                insight_type="peer_learning",
                title="High-Quality Peer Matches Found",
                description=f"You have {len(high_similarity_peers)} highly similar peers with valuable learning opportunities.",
                confidence=0.85,
                actionable=True,
                priority="medium"
            ))
        
        return insights
    
    def _identify_improvement_opportunities(self, peer_comparisons: List[PeerComparison], rankings: List[RankingData]) -> List[str]:
        """Identify improvement opportunities"""
        opportunities = []
        
        # From rankings
        low_rankings = [r for r in rankings if r.percentile < 50]
        for ranking in low_rankings[:3]:  # Top 3 areas
            area = ranking.metric.replace('_score', '')
            opportunities.append(f"Improve {area} to move from {ranking.percentile:.0f}th percentile")
        
        # From peer comparisons
        common_strengths = defaultdict(int)
        for peer in peer_comparisons:
            for strength in peer.strengths:
                common_strengths[strength] += 1
        
        # Most common peer strengths are opportunities
        for strength, count in sorted(common_strengths.items(), key=lambda x: x[1], reverse=True)[:2]:
            opportunities.append(f"Focus on {strength} - area where {count} similar peers excel")
        
        return opportunities
    
    def _generate_peer_learning_suggestions(self, peer_comparisons: List[PeerComparison]) -> List[str]:
        """Generate peer learning suggestions"""
        suggestions = []
        
        # Collect all learning opportunities
        all_opportunities = []
        for peer in peer_comparisons:
            all_opportunities.extend(peer.learning_opportunities)
        
        # Count frequency and prioritize
        opportunity_counts = defaultdict(int)
        for opp in all_opportunities:
            opportunity_counts[opp] += 1
        
        # Get top suggestions
        sorted_opportunities = sorted(opportunity_counts.items(), key=lambda x: x[1], reverse=True)
        for opp, count in sorted_opportunities[:5]:
            suggestions.append(opp)
        
        return suggestions
    
    def _calculate_network_growth_potential(self, network_metrics: NetworkMetrics, peer_data: List[Dict[str, Any]]) -> float:
        """Calculate network growth potential"""
        # Factors affecting growth potential
        size_factor = 1.0 - min(network_metrics.network_size / 100, 1.0)  # More potential if smaller network
        connection_factor = network_metrics.connection_strength  # Higher quality connections = more potential
        activity_factor = min(len(peer_data) / 50, 1.0)  # More active peers = more potential
        
        growth_potential = (size_factor * 0.4 + connection_factor * 0.4 + activity_factor * 0.2)
        
        return min(max(growth_potential, 0.0), 1.0)
    
    def _analyze_influence_trend(self, user_id: str) -> str:
        """Analyze social influence trend"""
        # Simplified trend analysis (would use historical data in real implementation)
        trends = ["growing", "stable", "declining"]
        weights = [0.4, 0.5, 0.1]  # Bias toward positive/stable
        
        return np.random.choice(trends, p=weights)
    
    def _assess_data_completeness(self, peer_data: List[Dict[str, Any]]) -> str:
        """Assess completeness of peer data"""
        peer_count = len(peer_data)
        
        if peer_count >= 50:
            return "excellent"
        elif peer_count >= 20:
            return "good"
        elif peer_count >= 10:
            return "fair"
        elif peer_count >= 5:
            return "limited"
        else:
            return "insufficient"
    
    def _calculate_confidence(self, peer_comparisons: List[PeerComparison], rankings: List[RankingData], data_completeness: str) -> float:
        """Calculate overall confidence"""
        # Base confidence from data completeness
        completeness_scores = {
            "excellent": 0.9,
            "good": 0.8,
            "fair": 0.6,
            "limited": 0.4,
            "insufficient": 0.2
        }
        base_confidence = completeness_scores.get(data_completeness, 0.2)
        
        # Adjust based on peer comparisons quality
        if peer_comparisons:
            avg_similarity = np.mean([p.similarity_score for p in peer_comparisons])
            peer_factor = avg_similarity
        else:
            peer_factor = 0.5
        
        # Adjust based on ranking data
        ranking_factor = min(len(rankings) / 5, 1.0)  # More rankings = higher confidence
        
        confidence = base_confidence * 0.5 + peer_factor * 0.3 + ranking_factor * 0.2
        
        return min(max(confidence, 0.0), 1.0)
    
    def _create_no_data_response(self) -> AgentOutput:
        """Create response when no data is available"""
        result = SocialGraphResult(
            peer_comparisons=[],
            rankings=[],
            network_metrics=NetworkMetrics(
                network_size=0,
                connection_strength=0.0,
                influence_score=0.0,
                centrality_score=0.0,
                clustering_coefficient=0.0
            ),
            social_insights=[SocialInsight(
                insight_type="no_data",
                title="No Peer Data Available",
                description="Start building your social network by taking more analyses and connecting with peers.",
                confidence=1.0,
                actionable=True,
                priority="medium"
            )],
            improvement_opportunities=["Build peer network for meaningful comparisons"],
            peer_learning_suggestions=["Connect with similar users to unlock peer learning"],
            network_growth_potential=1.0,
            social_influence_trend="emerging",
            confidence=0.0,
            data_completeness="insufficient"
        )
        
        return AgentOutput(
            success=True,
            data=result.dict(),
            confidence=0.0,
            processing_time=0.0,
            agent_name="social_graph"
        )
