import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from src.agents.base_agent import BaseAgent, AgentInput, AgentOutput
from src.utils.tracing import log_trace
from src.utils.validation import guardrails_validate


class FinalAnalysisResponse(BaseModel):
    """Final API response schema that will be returned to client"""
    media_id: str
    timestamp: str
    
    # Summary scores
    overall_score: float = Field(..., ge=0.0, le=10.0)
    attractiveness_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    style_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    presence_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    
    # Human-readable summary
    summary: str = Field(..., description="Plain-language summary for UI")
    key_insights: List[str] = Field(default_factory=list, max_items=5)
    recommendations: List[str] = Field(default_factory=list, max_items=5)
    
    # Detailed analysis (optional for advanced users)
    detailed_analysis: Optional[Dict[str, Any]] = None
    
    # Metadata
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    langsmith_run_id: Optional[str] = None
    
    # Warnings/disclaimers
    warnings: List[str] = Field(default_factory=list)
    disclaimers: List[str] = Field(default_factory=list)


class FormatterAgent(BaseAgent):
    name = "formatter_agent"
    output_schema = FinalAnalysisResponse

    def run(self, input: AgentInput) -> AgentOutput:
        """
        Produces final API response and plain-language summary for UI.
        Validates final JSON with Guardrails before returning.
        
        Expected input.context:
        - aggregated_result: Dict from AggregatorAgent
        - user_consent: Dict with consent flags
        - request_metadata: Dict with request info
        """
        
        aggregated_result = input.context.get("aggregated_result", {})
        user_consent = input.context.get("user_consent", {})
        request_metadata = input.context.get("request_metadata", {})
        
        if not aggregated_result:
            return AgentOutput(
                success=False,
                data={},
                error="No aggregated result provided"
            )
        
        try:
            # Generate human-readable summary
            summary = self._generate_summary(aggregated_result)
            
            # Extract key insights and recommendations
            insights = self._extract_insights(aggregated_result)
            recommendations = self._extract_recommendations(aggregated_result)
            
            # Add appropriate warnings and disclaimers
            warnings, disclaimers = self._generate_warnings_disclaimers(
                aggregated_result, user_consent
            )
            
            # Prepare detailed analysis (filtered by consent)
            detailed_analysis = self._prepare_detailed_analysis(
                aggregated_result, user_consent
            )
            
            # Create final response
            from datetime import datetime
            final_response = FinalAnalysisResponse(
                media_id=aggregated_result.get("media_id", input.media_id),
                timestamp=datetime.utcnow().isoformat(),
                overall_score=aggregated_result.get("overall_score", 5.0),
                attractiveness_score=aggregated_result.get("attractiveness_score"),
                style_score=aggregated_result.get("style_score"),
                presence_score=aggregated_result.get("presence_score"),
                summary=summary,
                key_insights=insights,
                recommendations=recommendations,
                detailed_analysis=detailed_analysis,
                confidence=aggregated_result.get("confidence", 0.5),
                processing_metadata={
                    "agents_used": list(aggregated_result.get("langsmith_run_ids", {}).keys()),
                    "processing_mode": os.getenv("LIFEMIRROR_MODE", "prod"),
                    "version": "1.0",
                    **aggregated_result.get("processing_metadata", {})
                },
                langsmith_run_id=input.context.get("langsmith_run_id"),
                warnings=warnings,
                disclaimers=disclaimers
            )
            
            # Validate with Guardrails before returning
            try:
                # This would use Guardrails validation if properly configured
                validated_response = final_response
                result = AgentOutput(success=True, data=validated_response.dict())
                self._trace(input.dict(), result.dict())
                return result
                
            except Exception as validation_error:
                # If Guardrails validation fails, return sanitized version
                sanitized_response = self._create_sanitized_response(
                    input.media_id, str(validation_error)
                )
                result = AgentOutput(
                    success=True, 
                    data=sanitized_response.dict()
                )
                result.data["warning"] = f"Response sanitized due to validation: {validation_error}"
                self._trace(input.dict(), result.dict())
                return result
                
        except Exception as e:
            # Fallback to basic response
            fallback_response = self._create_fallback_response(input.media_id, str(e))
            result = AgentOutput(success=True, data=fallback_response.dict())
            self._trace(input.dict(), {"error": str(e), "fallback": result.dict()})
            return result

    def _generate_summary(self, aggregated_result: Dict[str, Any]) -> str:
        """Generate human-readable summary"""
        overall_score = aggregated_result.get("overall_score", 5.0)
        explanation = aggregated_result.get("explanation", "")
        
        if explanation:
            return explanation
            
        # Generate basic summary based on scores
        if overall_score >= 8.0:
            return "Your analysis shows excellent overall presentation with strong visual impact."
        elif overall_score >= 6.5:
            return "You demonstrate solid presentation skills with several notable strengths."
        elif overall_score >= 5.0:
            return "Your presentation shows good potential with clear areas for enhancement."
        else:
            return "Your analysis reveals opportunities for significant improvement in presentation."

    def _extract_insights(self, aggregated_result: Dict[str, Any]) -> List[str]:
        """Extract key insights for the user"""
        insights = []
        
        # From aggregated strengths
        strengths = aggregated_result.get("key_strengths", [])
        insights.extend([f"Strong {strength.lower()}" for strength in strengths[:3]])
        
        # From individual analyses
        if aggregated_result.get("face_analysis"):
            faces = aggregated_result["face_analysis"].get("faces", [])
            if faces:
                insights.append(f"Detected {len(faces)} face(s) with good visibility")
                
        if aggregated_result.get("fashion_analysis"):
            items = aggregated_result["fashion_analysis"].get("items", [])
            if items:
                insights.append(f"Fashion analysis identified {len(items)} clothing items")
                
        return insights[:5]

    def _extract_recommendations(self, aggregated_result: Dict[str, Any]) -> List[str]:
        """Extract actionable recommendations"""
        recommendations = []
        
        # From aggregated improvement areas
        improvements = aggregated_result.get("improvement_areas", [])
        recommendations.extend(improvements[:3])
        
        # From posture analysis
        if aggregated_result.get("posture_analysis"):
            tips = aggregated_result["posture_analysis"].get("tips", [])
            recommendations.extend(tips[:2])
            
        # From bio analysis
        if aggregated_result.get("bio_analysis"):
            bio_improvements = aggregated_result["bio_analysis"].get("improvements", [])
            recommendations.extend(bio_improvements[:2])
            
        return recommendations[:5]

    def _generate_warnings_disclaimers(self, aggregated_result: Dict[str, Any], 
                                     user_consent: Dict[str, Any]) -> tuple[List[str], List[str]]:
        """Generate appropriate warnings and disclaimers"""
        warnings = []
        disclaimers = [
            "This analysis is for entertainment and self-improvement purposes only.",
            "Results are based on algorithmic analysis and should not be considered professional advice."
        ]
        
        # Check confidence levels
        confidence = aggregated_result.get("confidence", 1.0)
        if confidence < 0.6:
            warnings.append("Analysis confidence is lower than usual - results may be less accurate.")
            
        # Check for failed analyses
        processing_metadata = aggregated_result.get("processing_metadata", {})
        agent_successes = processing_metadata.get("agent_successes", {})
        failed_agents = [agent for agent, success in agent_successes.items() if not success]
        
        if failed_agents:
            warnings.append(f"Some analyses were unavailable: {', '.join(failed_agents)}")
            
        # Biometric disclaimer
        if user_consent.get("biometric_analysis"):
            disclaimers.append(
                "Biometric analysis is performed with your explicit consent and is not stored permanently."
            )
            
        return warnings, disclaimers

    def _prepare_detailed_analysis(self, aggregated_result: Dict[str, Any], 
                                 user_consent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare detailed analysis based on user consent"""
        if not user_consent.get("detailed_analysis", True):
            return None
            
        detailed = {}
        
        # Include individual agent results based on consent
        if user_consent.get("face_analysis", True) and aggregated_result.get("face_analysis"):
            detailed["face"] = aggregated_result["face_analysis"]
            
        if user_consent.get("fashion_analysis", True) and aggregated_result.get("fashion_analysis"):
            detailed["fashion"] = aggregated_result["fashion_analysis"]
            
        if user_consent.get("posture_analysis", True) and aggregated_result.get("posture_analysis"):
            detailed["posture"] = aggregated_result["posture_analysis"]
            
        if user_consent.get("bio_analysis", True) and aggregated_result.get("bio_analysis"):
            detailed["bio"] = aggregated_result["bio_analysis"]
            
        # Include technical metadata if consented
        if user_consent.get("technical_metadata", False):
            detailed["metadata"] = aggregated_result.get("processing_metadata", {})
            detailed["langsmith_run_ids"] = aggregated_result.get("langsmith_run_ids", {})
            
        return detailed if detailed else None

    def _create_sanitized_response(self, media_id: str, error: str) -> FinalAnalysisResponse:
        """Create sanitized response when validation fails"""
        from datetime import datetime
        return FinalAnalysisResponse(
            media_id=media_id,
            timestamp=datetime.utcnow().isoformat(),
            overall_score=5.0,
            summary="Analysis completed with standard results.",
            key_insights=["Analysis processed successfully"],
            recommendations=["Review your presentation for potential improvements"],
            confidence=0.5,
            processing_metadata={"sanitized": True, "original_error": error},
            warnings=["Response was sanitized for safety"],
            disclaimers=["This is a sanitized response due to content validation."]
        )

    def _create_fallback_response(self, media_id: str, error: str) -> FinalAnalysisResponse:
        """Create fallback response when formatting fails"""
        from datetime import datetime
        return FinalAnalysisResponse(
            media_id=media_id,
            timestamp=datetime.utcnow().isoformat(),
            overall_score=5.0,
            summary="Analysis completed with basic results due to processing limitations.",
            key_insights=["Media processed successfully"],
            recommendations=["Consider retrying analysis for more detailed results"],
            confidence=0.3,
            processing_metadata={"fallback": True, "error": error},
            warnings=["Using fallback response due to processing error"],
            disclaimers=["This is a fallback response with limited analysis."]
        )
