import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from src.agents.base_agent import BaseAgent, AgentInput, AgentOutput
from src.agents.memory_agent import MemoryAgent
from src.utils.tracing import log_trace


class ComparisonResult(BaseModel):
    comparison_type: str = Field(..., description="Type of comparison: 'celebrity', 'past_self', 'peer'")
    target_id: Optional[str] = Field(None, description="ID of comparison target")
    target_name: Optional[str] = Field(None, description="Name of comparison target")
    
    # Comparison scores
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Overall similarity")
    style_similarity: float = Field(..., ge=0.0, le=1.0, description="Style similarity")
    presence_similarity: float = Field(..., ge=0.0, le=1.0, description="Presence similarity")
    
    # Analysis
    similarities: List[str] = Field(default_factory=list, description="What's similar")
    differences: List[str] = Field(default_factory=list, description="What's different")
    insights: List[str] = Field(default_factory=list, description="Key insights from comparison")
    
    # Recommendations based on comparison
    recommendations: List[str] = Field(default_factory=list, description="Suggestions based on comparison")
    
    # Metadata
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in comparison")
    comparison_metadata: Dict[str, Any] = Field(default_factory=dict)


class CompareAgent(BaseAgent):
    """
    CompareAgent: performs comparisons with celebrities, past self, or peers.
    
    Supports:
    - Celebrity comparison (using reference database)
    - Past self comparison (using user's historical data)  
    - Peer comparison (using anonymized peer data)
    """
    name = "compare_agent"
    output_schema = ComparisonResult

    def run(self, input: AgentInput) -> AgentOutput:
        """
        Perform comparison analysis.
        
        Expected input.context:
        - comparison_type: str - 'celebrity', 'past_self', 'peer'
        - target_id: Optional[str] - specific target to compare against
        - user_id: str - user making the comparison
        - current_analysis: Dict - current analysis to compare from
        - time_range: Optional[str] - for past_self comparisons ('1_month', '3_months', '1_year')
        """
        
        comparison_type = input.context.get("comparison_type")
        if not comparison_type:
            return AgentOutput(
                success=False,
                data={},
                error="comparison_type required ('celebrity', 'past_self', 'peer')"
            )
        
        if comparison_type not in ["celebrity", "past_self", "peer"]:
            return AgentOutput(
                success=False,
                data={},
                error="Invalid comparison_type. Must be 'celebrity', 'past_self', or 'peer'"
            )
        
        user_id = input.context.get("user_id")
        current_analysis = input.context.get("current_analysis", {})
        target_id = input.context.get("target_id")
        
        if not user_id:
            return AgentOutput(
                success=False,
                data={},
                error="user_id required for comparison"
            )
        
        mode = os.getenv("LIFEMIRROR_MODE", "mock")
        
        if mode == "mock":
            # Return mock comparison based on type
            if comparison_type == "celebrity":
                mock_result = self._create_mock_celebrity_comparison(target_id)
            elif comparison_type == "past_self":
                mock_result = self._create_mock_past_self_comparison()
            else:  # peer
                mock_result = self._create_mock_peer_comparison()
                
            result = AgentOutput(success=True, data=mock_result.dict())
            self._trace(input.dict(), result.dict())
            return result
        
        # Production mode
        try:
            if comparison_type == "celebrity":
                comparison = self._compare_with_celebrity(
                    current_analysis, target_id
                )
            elif comparison_type == "past_self":
                time_range = input.context.get("time_range", "3_months")
                comparison = self._compare_with_past_self(
                    user_id, current_analysis, time_range
                )
            else:  # peer
                comparison = self._compare_with_peers(
                    current_analysis, user_id
                )
            
            result = AgentOutput(success=True, data=comparison.dict())
            self._trace(input.dict(), result.dict())
            return result
            
        except Exception as e:
            result = AgentOutput(
                success=False,
                data={},
                error=f"Comparison failed: {str(e)}"
            )
            self._trace(input.dict(), result.dict())
            return result

    def _compare_with_celebrity(self, current_analysis: Dict, target_id: Optional[str]) -> ComparisonResult:
        """Compare current analysis with celebrity reference"""
        
        # In a real implementation, this would:
        # 1. Load celebrity reference data from database
        # 2. Compare embeddings and analysis results
        # 3. Use LLM to generate insights
        
        # For now, use a simple implementation with mock celebrity data
        celebrity_name = self._get_celebrity_name(target_id)
        celebrity_profile = self._get_celebrity_profile(target_id)
        
        # Calculate similarity scores (simplified)
        overall_score = current_analysis.get("overall_score", 5.0)
        celebrity_score = celebrity_profile.get("overall_score", 8.0)
        
        similarity_score = 1.0 - abs(overall_score - celebrity_score) / 10.0
        style_similarity = max(0.0, similarity_score + 0.1)  # Slight variation
        presence_similarity = max(0.0, similarity_score - 0.1)
        
        # Generate comparison insights using LLM
        insights = self._generate_celebrity_insights(
            current_analysis, celebrity_profile, celebrity_name
        )
        
        return ComparisonResult(
            comparison_type="celebrity",
            target_id=target_id,
            target_name=celebrity_name,
            similarity_score=similarity_score,
            style_similarity=style_similarity,
            presence_similarity=presence_similarity,
            similarities=insights.get("similarities", []),
            differences=insights.get("differences", []),
            insights=insights.get("insights", []),
            recommendations=insights.get("recommendations", []),
            confidence=0.7,
            comparison_metadata={
                "celebrity_profile": celebrity_profile,
                "method": "embedding_similarity"
            }
        )

    def _compare_with_past_self(self, user_id: str, current_analysis: Dict, time_range: str) -> ComparisonResult:
        """Compare current analysis with user's past analyses"""
        
        # Use MemoryAgent to retrieve past analyses
        memory_agent = MemoryAgent()
        
        # Define date range based on time_range parameter
        from datetime import datetime, timedelta
        end_date = datetime.utcnow()
        
        if time_range == "1_month":
            start_date = end_date - timedelta(days=30)
        elif time_range == "3_months":
            start_date = end_date - timedelta(days=90)
        elif time_range == "1_year":
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=90)  # Default to 3 months
        
        # Search for past analyses
        memory_input = AgentInput(
            media_id="comparison_search",
            url="",
            context={
                "user_id": user_id,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "limit": 10
            }
        )
        
        memory_result = memory_agent.run(memory_input)
        
        if not memory_result.success or not memory_result.data.get("results"):
            # No past data available
            return ComparisonResult(
                comparison_type="past_self",
                similarity_score=0.5,
                style_similarity=0.5,
                presence_similarity=0.5,
                insights=["No sufficient historical data for comparison"],
                recommendations=["Continue using the app to build comparison history"],
                confidence=0.1,
                comparison_metadata={"time_range": time_range, "past_analyses_found": 0}
            )
        
        past_results = memory_result.data["results"]
        
        # Calculate average past performance
        past_scores = []
        for result in past_results:
            if result.get("analysis_summary"):
                score = result["analysis_summary"].get("overall_score")
                if score is not None:
                    past_scores.append(score)
        
        if not past_scores:
            avg_past_score = 5.0
        else:
            avg_past_score = sum(past_scores) / len(past_scores)
        
        current_score = current_analysis.get("overall_score", 5.0)
        
        # Calculate improvement/decline
        score_change = current_score - avg_past_score
        similarity_score = max(0.0, 1.0 - abs(score_change) / 10.0)
        
        # Generate insights about progression
        insights = self._generate_past_self_insights(
            current_analysis, past_results, score_change, time_range
        )
        
        return ComparisonResult(
            comparison_type="past_self",
            similarity_score=similarity_score,
            style_similarity=similarity_score + 0.05,
            presence_similarity=similarity_score - 0.05,
            similarities=insights.get("similarities", []),
            differences=insights.get("differences", []),
            insights=insights.get("insights", []),
            recommendations=insights.get("recommendations", []),
            confidence=min(0.9, len(past_scores) / 10.0 + 0.3),
            comparison_metadata={
                "time_range": time_range,
                "past_analyses_count": len(past_results),
                "score_change": score_change,
                "avg_past_score": avg_past_score
            }
        )

    def _compare_with_peers(self, current_analysis: Dict, user_id: str) -> ComparisonResult:
        """Compare with anonymized peer data"""
        
        # In a real implementation, this would query anonymized peer statistics
        # For now, use mock peer data
        
        current_score = current_analysis.get("overall_score", 5.0)
        
        # Mock peer statistics (in reality, these would come from database)
        peer_stats = {
            "avg_score": 6.2,
            "median_score": 6.0,
            "percentile_75": 7.5,
            "percentile_25": 5.0
        }
        
        # Calculate where user stands relative to peers
        if current_score >= peer_stats["percentile_75"]:
            percentile = "top 25%"
        elif current_score >= peer_stats["median_score"]:
            percentile = "above average"
        elif current_score >= peer_stats["percentile_25"]:
            percentile = "below average"
        else:
            percentile = "bottom 25%"
        
        similarity_to_avg = 1.0 - abs(current_score - peer_stats["avg_score"]) / 10.0
        
        insights = [
            f"You score in the {percentile} compared to peers",
            f"Your score is {abs(current_score - peer_stats['avg_score']):.1f} points {'above' if current_score > peer_stats['avg_score'] else 'below'} average"
        ]
        
        recommendations = []
        if current_score < peer_stats["median_score"]:
            recommendations.extend([
                "Focus on areas where peers typically excel",
                "Consider studying high-performing examples"
            ])
        else:
            recommendations.extend([
                "You're performing well compared to peers",
                "Consider sharing your approach with others"
            ])
        
        return ComparisonResult(
            comparison_type="peer",
            similarity_score=similarity_to_avg,
            style_similarity=similarity_to_avg + 0.1,
            presence_similarity=similarity_to_avg - 0.1,
            similarities=["Similar presentation patterns to peer group"],
            differences=["Unique personal style elements"],
            insights=insights,
            recommendations=recommendations,
            confidence=0.6,
            comparison_metadata={
                "peer_stats": peer_stats,
                "user_percentile": percentile
            }
        )

    # Helper methods for mock data and celebrity profiles
    def _get_celebrity_name(self, target_id: Optional[str]) -> str:
        celebrity_map = {
            "celeb_1": "Taylor Swift",
            "celeb_2": "Ryan Gosling", 
            "celeb_3": "Emma Stone",
            "celeb_4": "Michael B. Jordan"
        }
        return celebrity_map.get(target_id, "Celebrity Reference")

    def _get_celebrity_profile(self, target_id: Optional[str]) -> Dict[str, Any]:
        # Mock celebrity profiles
        profiles = {
            "celeb_1": {"overall_score": 9.2, "style_score": 9.5, "presence_score": 8.8},
            "celeb_2": {"overall_score": 8.7, "style_score": 8.2, "presence_score": 9.1},
            "celeb_3": {"overall_score": 8.9, "style_score": 8.8, "presence_score": 8.6},
            "celeb_4": {"overall_score": 9.0, "style_score": 8.5, "presence_score": 9.2}
        }
        return profiles.get(target_id, {"overall_score": 8.5, "style_score": 8.0, "presence_score": 8.5})

    def _generate_celebrity_insights(self, current: Dict, celebrity: Dict, name: str) -> Dict[str, List[str]]:
        """Generate celebrity comparison insights (simplified)"""
        return {
            "similarities": [f"Similar confidence level to {name}", "Comparable presentation style"],
            "differences": [f"{name} has more polished styling", "Different color preferences"],
            "insights": [f"You share {name}'s approachable demeanor", "Strong potential for similar impact"],
            "recommendations": [f"Study {name}'s styling choices", "Work on posture refinement"]
        }

    def _generate_past_self_insights(self, current: Dict, past_results: List, score_change: float, time_range: str) -> Dict[str, List[str]]:
        """Generate past self comparison insights"""
        if score_change > 0.5:
            trend = "improving"
            insights = [f"You've improved significantly over the past {time_range}", "Clear upward trend in presentation"]
        elif score_change < -0.5:
            trend = "declining"
            insights = [f"Some decline noted over the past {time_range}", "Opportunity to refocus on key areas"]
        else:
            trend = "stable"
            insights = [f"Consistent performance over the past {time_range}", "Maintaining your presentation level"]
        
        return {
            "similarities": ["Consistent personal style", "Similar confidence patterns"],
            "differences": ["Evolved styling choices", "Refined presentation approach"],
            "insights": insights,
            "recommendations": ["Continue current approach" if trend == "improving" else "Review past successful strategies"]
        }

    # Mock methods for different comparison types
    def _create_mock_celebrity_comparison(self, target_id: Optional[str]) -> ComparisonResult:
        name = self._get_celebrity_name(target_id)
        return ComparisonResult(
            comparison_type="celebrity",
            target_id=target_id,
            target_name=name,
            similarity_score=0.73,
            style_similarity=0.68,
            presence_similarity=0.78,
            similarities=[f"Similar confident energy to {name}", "Comparable styling approach"],
            differences=[f"{name} has more refined posture", "Different color palette preferences"],
            insights=[f"You share {name}'s natural charisma", "Strong foundation for similar impact"],
            recommendations=[f"Study {name}'s posture techniques", "Experiment with bolder color choices"],
            confidence=0.8,
            comparison_metadata={"mock": True, "celebrity": name}
        )

    def _create_mock_past_self_comparison(self) -> ComparisonResult:
        return ComparisonResult(
            comparison_type="past_self",
            similarity_score=0.82,
            style_similarity=0.75,
            presence_similarity=0.89,
            similarities=["Consistent personal style", "Similar confidence level"],
            differences=["More refined styling choices", "Improved posture alignment"],
            insights=["You've shown steady improvement over time", "Your confidence has grown notably"],
            recommendations=["Continue current styling approach", "Maintain focus on posture work"],
            confidence=0.9,
            comparison_metadata={"mock": True, "time_range": "3_months", "improvement": True}
        )

    def _create_mock_peer_comparison(self) -> ComparisonResult:
        return ComparisonResult(
            comparison_type="peer",
            similarity_score=0.67,
            style_similarity=0.72,
            presence_similarity=0.64,
            similarities=["Similar presentation patterns to peer group", "Comparable confidence levels"],
            differences=["Unique personal style elements", "Different approach to color choices"],
            insights=["You perform above average compared to peers", "Your style choices are distinctive"],
            recommendations=["Continue developing your unique style", "Consider mentoring others in your strengths"],
            confidence=0.7,
            comparison_metadata={"mock": True, "peer_percentile": "top_40%"}
        )
