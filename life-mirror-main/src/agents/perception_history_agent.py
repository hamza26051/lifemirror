import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from src.agents.base_agent import BaseAgent, AgentInput, AgentOutput
from src.db.session import get_db
from src.db.models import Media, Analysis
from src.utils.tracing import log_trace
import numpy as np
import logging

class PerceptionTrend(BaseModel):
    """Model for perception trend data"""
    metric: str = Field(description="The metric being tracked")
    values: List[float] = Field(description="Historical values")
    timestamps: List[str] = Field(description="Corresponding timestamps")
    trend_direction: str = Field(description="Overall trend direction")
    trend_strength: float = Field(description="Strength of the trend (0-1)")
    change_rate: float = Field(description="Rate of change per time period")

class PerceptionInsight(BaseModel):
    """Model for perception insights"""
    insight_type: str = Field(description="Type of insight")
    title: str = Field(description="Insight title")
    description: str = Field(description="Detailed description")
    confidence: float = Field(description="Confidence in insight (0-1)")
    actionable: bool = Field(description="Whether insight is actionable")
    priority: str = Field(description="Priority level (high/medium/low)")

class PerceptionHistoryResult(BaseModel):
    """Result model for perception history analysis"""
    trends: List[PerceptionTrend] = Field(description="Identified trends")
    insights: List[PerceptionInsight] = Field(description="Generated insights")
    overall_trajectory: str = Field(description="Overall perception trajectory")
    improvement_areas: List[str] = Field(description="Areas showing improvement")
    decline_areas: List[str] = Field(description="Areas showing decline")
    stability_score: float = Field(description="Perception stability score (0-1)")
    growth_potential: float = Field(description="Potential for growth (0-1)")
    recommendations: List[str] = Field(description="Trend-based recommendations")
    confidence: float = Field(description="Overall analysis confidence")
    data_quality: str = Field(description="Quality of historical data")

class PerceptionHistoryAgent(BaseAgent):
    """Agent for analyzing perception trends and historical patterns"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Trend analysis parameters
        self.min_data_points = 3
        self.trend_window_days = 30
        self.significance_threshold = 0.1
        
        # Metric weights for overall analysis
        self.metric_weights = {
            'overall_score': 0.3,
            'confidence': 0.2,
            'fashion_score': 0.15,
            'face_score': 0.15,
            'posture_score': 0.1,
            'vibe_score': 0.1
        }
    
    @log_trace
    def run(self, input_data: AgentInput) -> AgentOutput:
        """Analyze perception history and trends"""
        try:
            # Get historical data
            historical_data = self._get_historical_data(input_data.media_id)
            
            if not historical_data:
                return self._create_no_data_response()
            
            # Analyze trends
            trends = self._analyze_trends(historical_data)
            
            # Generate insights
            insights = self._generate_insights(trends, historical_data)
            
            # Calculate trajectory and scores
            trajectory = self._calculate_trajectory(trends)
            improvement_areas = self._identify_improvement_areas(trends)
            decline_areas = self._identify_decline_areas(trends)
            stability_score = self._calculate_stability_score(trends)
            growth_potential = self._calculate_growth_potential(trends, historical_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(trends, insights)
            
            # Assess data quality and confidence
            data_quality = self._assess_data_quality(historical_data)
            confidence = self._calculate_confidence(trends, data_quality)
            
            result = PerceptionHistoryResult(
                trends=trends,
                insights=insights,
                overall_trajectory=trajectory,
                improvement_areas=improvement_areas,
                decline_areas=decline_areas,
                stability_score=stability_score,
                growth_potential=growth_potential,
                recommendations=recommendations,
                confidence=confidence,
                data_quality=data_quality
            )
            
            return AgentOutput(
                success=True,
                data=result.dict(),
                confidence=confidence,
                processing_time=0.0,
                agent_name="perception_history"
            )
            
        except Exception as e:
            self.logger.error(f"Perception history analysis failed: {e}")
            return AgentOutput(
                success=False,
                data={},
                confidence=0.0,
                processing_time=0.0,
                agent_name="perception_history",
                error=str(e)
            )
    
    def _get_historical_data(self, media_id: str) -> List[Dict[str, Any]]:
        """Get historical analysis data for the user"""
        try:
            with get_db() as db:
                # Get user from current media
                current_media = db.query(Media).filter(Media.id == media_id).first()
                if not current_media:
                    return []
                
                # Get all analyses for this user
                analyses = db.query(Analysis).join(Media).filter(
                    Media.user_id == current_media.user_id,
                    Analysis.created_at >= datetime.now() - timedelta(days=90)
                ).order_by(Analysis.created_at).all()
                
                historical_data = []
                for analysis in analyses:
                    if analysis.result_data:
                        data_point = {
                            'timestamp': analysis.created_at.isoformat(),
                            'analysis_id': analysis.id,
                            'media_id': analysis.media_id,
                            'data': analysis.result_data
                        }
                        historical_data.append(data_point)
                
                return historical_data
                
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            return []
    
    def _analyze_trends(self, historical_data: List[Dict[str, Any]]) -> List[PerceptionTrend]:
        """Analyze trends in historical data"""
        trends = []
        
        # Extract metrics from historical data
        metrics = self._extract_metrics(historical_data)
        
        for metric_name, values_data in metrics.items():
            if len(values_data) >= self.min_data_points:
                trend = self._calculate_trend(metric_name, values_data)
                if trend:
                    trends.append(trend)
        
        return trends
    
    def _extract_metrics(self, historical_data: List[Dict[str, Any]]) -> Dict[str, List[Tuple[str, float]]]:
        """Extract metrics from historical data"""
        metrics = {}
        
        for data_point in historical_data:
            timestamp = data_point['timestamp']
            data = data_point['data']
            
            # Extract various scores
            if 'overall_score' in data:
                if 'overall_score' not in metrics:
                    metrics['overall_score'] = []
                metrics['overall_score'].append((timestamp, data['overall_score']))
            
            if 'confidence' in data:
                if 'confidence' not in metrics:
                    metrics['confidence'] = []
                metrics['confidence'].append((timestamp, data['confidence']))
            
            # Extract agent-specific scores
            if 'individual_results' in data:
                for agent, result in data['individual_results'].items():
                    if isinstance(result, dict) and 'score' in result:
                        metric_key = f"{agent}_score"
                        if metric_key not in metrics:
                            metrics[metric_key] = []
                        metrics[metric_key].append((timestamp, result['score']))
        
        return metrics
    
    def _calculate_trend(self, metric_name: str, values_data: List[Tuple[str, float]]) -> Optional[PerceptionTrend]:
        """Calculate trend for a specific metric"""
        try:
            # Sort by timestamp
            values_data.sort(key=lambda x: x[0])
            
            timestamps = [item[0] for item in values_data]
            values = [item[1] for item in values_data]
            
            # Calculate trend direction and strength
            if len(values) < 2:
                return None
            
            # Simple linear regression for trend
            x = np.arange(len(values))
            y = np.array(values)
            
            # Calculate slope
            slope = np.polyfit(x, y, 1)[0]
            
            # Determine trend direction
            if abs(slope) < self.significance_threshold:
                direction = "stable"
            elif slope > 0:
                direction = "improving"
            else:
                direction = "declining"
            
            # Calculate trend strength
            correlation = np.corrcoef(x, y)[0, 1] if len(values) > 1 else 0
            strength = abs(correlation) if not np.isnan(correlation) else 0
            
            # Calculate change rate
            change_rate = slope * len(values) if len(values) > 1 else 0
            
            return PerceptionTrend(
                metric=metric_name,
                values=values,
                timestamps=timestamps,
                trend_direction=direction,
                trend_strength=strength,
                change_rate=change_rate
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate trend for {metric_name}: {e}")
            return None
    
    def _generate_insights(self, trends: List[PerceptionTrend], historical_data: List[Dict[str, Any]]) -> List[PerceptionInsight]:
        """Generate insights from trends"""
        insights = []
        
        # Analyze overall performance trends
        overall_trends = [t for t in trends if t.metric == 'overall_score']
        if overall_trends:
            trend = overall_trends[0]
            if trend.trend_direction == "improving" and trend.trend_strength > 0.5:
                insights.append(PerceptionInsight(
                    insight_type="positive_trend",
                    title="Strong Improvement Trajectory",
                    description=f"Your overall perception score has been consistently improving with a {trend.trend_strength:.1%} correlation.",
                    confidence=trend.trend_strength,
                    actionable=True,
                    priority="high"
                ))
            elif trend.trend_direction == "declining" and trend.trend_strength > 0.5:
                insights.append(PerceptionInsight(
                    insight_type="negative_trend",
                    title="Declining Performance Pattern",
                    description=f"Your overall perception score shows a declining trend that needs attention.",
                    confidence=trend.trend_strength,
                    actionable=True,
                    priority="high"
                ))
        
        # Analyze consistency
        confidence_trends = [t for t in trends if t.metric == 'confidence']
        if confidence_trends:
            trend = confidence_trends[0]
            if trend.trend_direction == "stable" and np.std(trend.values) < 0.1:
                insights.append(PerceptionInsight(
                    insight_type="consistency",
                    title="Consistent Performance",
                    description="You maintain consistent results across different analyses.",
                    confidence=0.8,
                    actionable=False,
                    priority="medium"
                ))
        
        # Analyze specific areas
        for trend in trends:
            if trend.metric.endswith('_score') and trend.trend_strength > 0.6:
                area = trend.metric.replace('_score', '')
                if trend.trend_direction == "improving":
                    insights.append(PerceptionInsight(
                        insight_type="area_improvement",
                        title=f"{area.title()} Showing Strong Progress",
                        description=f"Your {area} analysis shows significant improvement over time.",
                        confidence=trend.trend_strength,
                        actionable=True,
                        priority="medium"
                    ))
                elif trend.trend_direction == "declining":
                    insights.append(PerceptionInsight(
                        insight_type="area_decline",
                        title=f"{area.title()} Needs Attention",
                        description=f"Your {area} scores have been declining and may need focused improvement.",
                        confidence=trend.trend_strength,
                        actionable=True,
                        priority="high"
                    ))
        
        return insights
    
    def _calculate_trajectory(self, trends: List[PerceptionTrend]) -> str:
        """Calculate overall trajectory"""
        if not trends:
            return "insufficient_data"
        
        # Weight trends by importance
        weighted_score = 0
        total_weight = 0
        
        for trend in trends:
            weight = self.metric_weights.get(trend.metric, 0.05)
            if trend.trend_direction == "improving":
                weighted_score += weight * trend.trend_strength
            elif trend.trend_direction == "declining":
                weighted_score -= weight * trend.trend_strength
            total_weight += weight
        
        if total_weight == 0:
            return "stable"
        
        avg_score = weighted_score / total_weight
        
        if avg_score > 0.2:
            return "strong_improvement"
        elif avg_score > 0.05:
            return "gradual_improvement"
        elif avg_score > -0.05:
            return "stable"
        elif avg_score > -0.2:
            return "gradual_decline"
        else:
            return "concerning_decline"
    
    def _identify_improvement_areas(self, trends: List[PerceptionTrend]) -> List[str]:
        """Identify areas showing improvement"""
        improvement_areas = []
        
        for trend in trends:
            if (trend.trend_direction == "improving" and 
                trend.trend_strength > 0.4 and 
                trend.metric.endswith('_score')):
                area = trend.metric.replace('_score', '')
                improvement_areas.append(area)
        
        return improvement_areas
    
    def _identify_decline_areas(self, trends: List[PerceptionTrend]) -> List[str]:
        """Identify areas showing decline"""
        decline_areas = []
        
        for trend in trends:
            if (trend.trend_direction == "declining" and 
                trend.trend_strength > 0.4 and 
                trend.metric.endswith('_score')):
                area = trend.metric.replace('_score', '')
                decline_areas.append(area)
        
        return decline_areas
    
    def _calculate_stability_score(self, trends: List[PerceptionTrend]) -> float:
        """Calculate perception stability score"""
        if not trends:
            return 0.0
        
        stability_scores = []
        
        for trend in trends:
            # Stability is inverse of trend strength for non-stable trends
            if trend.trend_direction == "stable":
                stability_scores.append(1.0)
            else:
                stability_scores.append(1.0 - trend.trend_strength)
        
        return np.mean(stability_scores) if stability_scores else 0.0
    
    def _calculate_growth_potential(self, trends: List[PerceptionTrend], historical_data: List[Dict[str, Any]]) -> float:
        """Calculate growth potential based on trends and data"""
        if not trends or not historical_data:
            return 0.5  # Neutral potential
        
        # Factors affecting growth potential
        improving_trends = sum(1 for t in trends if t.trend_direction == "improving")
        total_trends = len(trends)
        
        # Data recency and frequency
        data_recency = self._calculate_data_recency(historical_data)
        data_frequency = len(historical_data) / 30  # Analyses per month
        
        # Calculate potential
        trend_factor = improving_trends / total_trends if total_trends > 0 else 0.5
        recency_factor = min(data_recency, 1.0)
        frequency_factor = min(data_frequency / 4, 1.0)  # Normalize to 4 analyses per month
        
        growth_potential = (trend_factor * 0.5 + recency_factor * 0.3 + frequency_factor * 0.2)
        
        return min(max(growth_potential, 0.0), 1.0)
    
    def _calculate_data_recency(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate how recent the data is"""
        if not historical_data:
            return 0.0
        
        latest_timestamp = max(data['timestamp'] for data in historical_data)
        latest_date = datetime.fromisoformat(latest_timestamp.replace('Z', '+00:00'))
        days_ago = (datetime.now() - latest_date.replace(tzinfo=None)).days
        
        # Recency score decreases with age
        return max(0.0, 1.0 - (days_ago / 30))  # Full score if within last day, 0 after 30 days
    
    def _generate_recommendations(self, trends: List[PerceptionTrend], insights: List[PerceptionInsight]) -> List[str]:
        """Generate trend-based recommendations"""
        recommendations = []
        
        # Recommendations based on trends
        declining_trends = [t for t in trends if t.trend_direction == "declining" and t.trend_strength > 0.5]
        if declining_trends:
            recommendations.append("Focus on areas showing decline to prevent further deterioration")
            for trend in declining_trends[:2]:  # Top 2 declining areas
                area = trend.metric.replace('_score', '')
                recommendations.append(f"Prioritize improvement in {area} analysis")
        
        # Recommendations based on insights
        high_priority_insights = [i for i in insights if i.priority == "high" and i.actionable]
        if high_priority_insights:
            recommendations.append("Address high-priority areas identified in your analysis history")
        
        # General recommendations
        if len(trends) < 3:
            recommendations.append("Continue regular analyses to build a more comprehensive trend history")
        
        stable_trends = [t for t in trends if t.trend_direction == "stable"]
        if len(stable_trends) > len(trends) * 0.7:
            recommendations.append("Consider trying new approaches to break through performance plateaus")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _assess_data_quality(self, historical_data: List[Dict[str, Any]]) -> str:
        """Assess the quality of historical data"""
        if not historical_data:
            return "no_data"
        
        data_count = len(historical_data)
        
        if data_count >= 10:
            return "excellent"
        elif data_count >= 5:
            return "good"
        elif data_count >= 3:
            return "fair"
        else:
            return "limited"
    
    def _calculate_confidence(self, trends: List[PerceptionTrend], data_quality: str) -> float:
        """Calculate overall confidence in the analysis"""
        if not trends:
            return 0.0
        
        # Base confidence from trend strengths
        trend_confidences = [t.trend_strength for t in trends]
        avg_trend_confidence = np.mean(trend_confidences) if trend_confidences else 0.0
        
        # Data quality factor
        quality_factors = {
            "excellent": 1.0,
            "good": 0.8,
            "fair": 0.6,
            "limited": 0.4,
            "no_data": 0.0
        }
        quality_factor = quality_factors.get(data_quality, 0.0)
        
        # Combine factors
        confidence = avg_trend_confidence * 0.7 + quality_factor * 0.3
        
        return min(max(confidence, 0.0), 1.0)
    
    def _create_no_data_response(self) -> AgentOutput:
        """Create response when no historical data is available"""
        result = PerceptionHistoryResult(
            trends=[],
            insights=[PerceptionInsight(
                insight_type="no_data",
                title="No Historical Data Available",
                description="Start building your perception history by taking more analyses.",
                confidence=1.0,
                actionable=True,
                priority="medium"
            )],
            overall_trajectory="insufficient_data",
            improvement_areas=[],
            decline_areas=[],
            stability_score=0.0,
            growth_potential=0.5,
            recommendations=["Take regular analyses to start building your perception history"],
            confidence=0.0,
            data_quality="no_data"
        )
        
        return AgentOutput(
            success=True,
            data=result.dict(),
            confidence=0.0,
            processing_time=0.0,
            agent_name="perception_history"
        )
