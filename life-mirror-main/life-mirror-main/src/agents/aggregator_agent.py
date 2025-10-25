import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from src.agents.base_agent import BaseAgent, AgentInput, AgentOutput
from src.utils.tracing import log_trace


class AggregatedAnalysis(BaseModel):
    media_id: str
    overall_score: float = Field(..., ge=0.0, le=10.0, description="Composite score 0-10")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    
    # Individual agent results
    face_analysis: Optional[Dict[str, Any]] = None
    fashion_analysis: Optional[Dict[str, Any]] = None
    posture_analysis: Optional[Dict[str, Any]] = None
    bio_analysis: Optional[Dict[str, Any]] = None
    embedding_analysis: Optional[Dict[str, Any]] = None
    
    # Computed composite scores
    attractiveness_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    style_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    presence_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    
    # Metadata for traceability
    langsmith_run_ids: Dict[str, str] = Field(default_factory=dict)
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Summary insights
    key_strengths: List[str] = Field(default_factory=list)
    improvement_areas: List[str] = Field(default_factory=list)
    explanation: str = Field(default="", description="Brief explanation of the scores")


class AggregatorAgent(BaseAgent):
    name = "aggregator_agent"
    output_schema = AggregatedAnalysis

    def run(self, input: AgentInput) -> AgentOutput:
        """
        Combines outputs from FaceAgent, FashionAgent, PostureAgent, BioAgent 
        and computes final composite scores with explanation.
        
        Expected input.context should contain:
        - face_result: Dict from FaceAgent
        - fashion_result: Dict from FashionAgent  
        - posture_result: Dict from PostureAgent
        - bio_result: Dict from BioAgent (optional)
        - embedding_result: Dict from EmbedderAgent
        - langsmith_run_ids: Dict mapping agent names to run IDs
        """
        
        # Extract individual agent results
        face_result = input.context.get("face_result", {})
        fashion_result = input.context.get("fashion_result", {})
        posture_result = input.context.get("posture_result", {})
        bio_result = input.context.get("bio_result", {})
        embedding_result = input.context.get("embedding_result", {})
        langsmith_run_ids = input.context.get("langsmith_run_ids", {})
        
        # Compute composite scores
        scores = self._compute_composite_scores(
            face_result, fashion_result, posture_result, bio_result
        )
        
        # Extract key insights
        strengths, improvements = self._extract_insights(
            face_result, fashion_result, posture_result, bio_result
        )
        
        # Generate explanation
        explanation = self._generate_explanation(scores, strengths, improvements)
        
        # Compute overall confidence
        confidences = []
        for result in [face_result, fashion_result, posture_result, bio_result]:
            if result.get("success") and "confidence" in result.get("data", {}):
                confidences.append(result["data"]["confidence"])
        
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Create aggregated result
        aggregated = AggregatedAnalysis(
            media_id=input.media_id,
            overall_score=scores["overall"],
            confidence=overall_confidence,
            face_analysis=face_result.get("data") if face_result.get("success") else None,
            fashion_analysis=fashion_result.get("data") if fashion_result.get("success") else None,
            posture_analysis=posture_result.get("data") if posture_result.get("success") else None,
            bio_analysis=bio_result.get("data") if bio_result.get("success") else None,
            embedding_analysis=embedding_result.get("data") if embedding_result.get("success") else None,
            attractiveness_score=scores["attractiveness"],
            style_score=scores["style"],
            presence_score=scores["presence"],
            langsmith_run_ids=langsmith_run_ids,
            processing_metadata={
                "agent_successes": {
                    "face": face_result.get("success", False),
                    "fashion": fashion_result.get("success", False),
                    "posture": posture_result.get("success", False),
                    "bio": bio_result.get("success", False),
                    "embedding": embedding_result.get("success", False)
                },
                "processing_time": input.context.get("processing_time"),
                "mode": os.getenv("LIFEMIRROR_MODE", "prod")
            },
            key_strengths=strengths,
            improvement_areas=improvements,
            explanation=explanation
        )
        
        result = AgentOutput(success=True, data=aggregated.dict())
        self._trace(input.dict(), result.dict())
        return result

    def _compute_composite_scores(self, face_result: Dict, fashion_result: Dict, 
                                posture_result: Dict, bio_result: Dict) -> Dict[str, float]:
        """Compute composite scores from individual agent outputs"""
        
        # Extract individual scores with defaults
        face_score = 5.0  # Default neutral
        if face_result.get("success") and face_result.get("data", {}).get("faces"):
            # Use number of faces and basic heuristics
            num_faces = len(face_result["data"]["faces"])
            face_score = min(8.0, 4.0 + num_faces * 2.0)  # Basic scoring
        
        fashion_score = 5.0
        if fashion_result.get("success"):
            fashion_data = fashion_result.get("data", {})
            if "overall_rating" in fashion_data:
                fashion_score = min(10.0, fashion_data["overall_rating"])
            elif "items" in fashion_data:
                # Score based on number and quality of items
                items = fashion_data["items"]
                fashion_score = min(8.0, 3.0 + len(items) * 1.5)
        
        posture_score = 5.0
        if posture_result.get("success"):
            posture_data = posture_result.get("data", {})
            if "alignment_score" in posture_data and posture_data["alignment_score"] is not None:
                posture_score = posture_data["alignment_score"]
        
        bio_score = 5.0
        if bio_result.get("success"):
            bio_data = bio_result.get("data", {})
            if "confidence" in bio_data:
                # Convert confidence to 0-10 scale
                bio_score = bio_data["confidence"] * 10.0
        
        # Compute composite scores with weights
        attractiveness_score = (face_score * 0.6 + posture_score * 0.4)
        style_score = (fashion_score * 0.8 + posture_score * 0.2)
        presence_score = (posture_score * 0.4 + bio_score * 0.3 + face_score * 0.3)
        
        # Overall score is weighted average
        overall_score = (
            attractiveness_score * 0.35 + 
            style_score * 0.35 + 
            presence_score * 0.30
        )
        
        return {
            "overall": round(overall_score, 2),
            "attractiveness": round(attractiveness_score, 2),
            "style": round(style_score, 2),
            "presence": round(presence_score, 2)
        }

    def _extract_insights(self, face_result: Dict, fashion_result: Dict,
                         posture_result: Dict, bio_result: Dict) -> tuple[List[str], List[str]]:
        """Extract key strengths and improvement areas from agent results"""
        
        strengths = []
        improvements = []
        
        # From face analysis
        if face_result.get("success"):
            face_data = face_result.get("data", {})
            if face_data.get("num_faces", 0) > 0:
                strengths.append("Clear facial visibility")
            
        # From fashion analysis  
        if fashion_result.get("success"):
            fashion_data = fashion_result.get("data", {})
            if fashion_data.get("overall_rating", 0) > 7:
                strengths.append("Strong fashion sense")
            elif fashion_data.get("overall_rating", 0) < 5:
                improvements.append("Consider fashion improvements")
                
        # From posture analysis
        if posture_result.get("success"):
            posture_data = posture_result.get("data", {})
            alignment = posture_data.get("alignment_score")
            if alignment and alignment > 7:
                strengths.append("Good posture alignment")
            elif alignment and alignment < 5:
                improvements.append("Work on posture alignment")
            
            # Add specific tips if available
            tips = posture_data.get("tips", [])
            improvements.extend(tips[:2])  # Limit to top 2 tips
            
        # From bio analysis
        if bio_result.get("success"):
            bio_data = bio_result.get("data", {})
            strengths.extend(bio_data.get("strengths", [])[:2])  # Top 2
            improvements.extend(bio_data.get("improvements", [])[:2])  # Top 2
            
        return strengths[:5], improvements[:5]  # Limit total items

    def _generate_explanation(self, scores: Dict[str, float], strengths: List[str], 
                            improvements: List[str]) -> str:
        """Generate a brief explanation of the analysis"""
        
        overall = scores["overall"]
        
        if overall >= 8.0:
            tone = "excellent"
        elif overall >= 6.5:
            tone = "strong"
        elif overall >= 5.0:
            tone = "solid"
        else:
            tone = "developing"
            
        explanation = f"Your overall analysis shows a {tone} presentation"
        
        if strengths:
            explanation += f" with key strengths in {', '.join(strengths[:2])}"
            
        if improvements:
            explanation += f". Focus areas for improvement include {', '.join(improvements[:2])}"
            
        return explanation + "."
