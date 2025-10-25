import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from src.agents.base_agent import BaseAgent, AgentInput, AgentOutput
from src.utils.tracing import log_trace


class BioOutput(BaseModel):
    vibe_summary: str = Field(..., description="Short summary of the vibe from bio text")
    strengths: List[str] = Field(..., description="List of identified strengths")
    weaknesses: List[str] = Field(..., description="List of areas for improvement")
    improvements: List[str] = Field(..., description="List of specific improvement suggestions")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the analysis")


class BioAgent(BaseAgent):
    name = "bio_agent"
    output_schema = BioOutput

    def run(self, input: AgentInput) -> AgentOutput:
        """
        Analyzes user-written bio or conversation text to derive vibe categories,
        suggested improvements, and a short friendly summary.
        
        Expected input.context:
        - text: str - the bio/profile text to analyze
        - past_analyses: Optional[List] - previous analyses for context
        """
        mode = os.getenv("LIFEMIRROR_MODE", "mock")
        
        # Extract text from data (since context doesn't exist in AgentInput)
        bio_text = ""
        past_analyses = []
        
        if input.data:
            bio_text = input.data.get("text", "")
            past_analyses = input.data.get("past_analyses", [])
        
        # For now, use a default bio text if none provided (for testing)
        if not bio_text:
            bio_text = "A creative and passionate individual with diverse interests and a positive outlook on life."

        if mode == "mock":
            result = AgentOutput(success=True, data={
                "vibe_summary": "Confident and approachable with creative interests",
                "strengths": ["Clear communication", "Positive tone", "Shows personality"],
                "weaknesses": ["Could be more specific about achievements", "Lacks call-to-action"],
                "improvements": [
                    "Add specific examples of your work",
                    "Include what you're looking for",
                    "Mention your unique value proposition"
                ],
                "confidence": 0.85
            })
            self._trace(input.dict(), result.dict())
            return result

        # Production mode: use LLM for analysis
        try:
            # Prepare context for LLM
            context_str = ""
            if past_analyses:
                context_str = f"\n\nPast analyses context: {past_analyses[:3]}"  # Limit context
            
            prompt = f"""
            You are an empathetic bio and text analysis expert. Analyze the following text and provide insights.
            
            IMPORTANT: You must respond with valid JSON only, following this exact schema:
            {{
                "vibe_summary": "string - brief summary of overall vibe/personality",
                "strengths": ["list of current strengths in the text"],
                "weaknesses": ["list of areas that could be improved"],
                "improvements": ["list of specific, actionable improvement suggestions"],
                "confidence": 0.0-1.0 (confidence in your analysis)
            }}
            
            Guidelines:
            - Be empathetic and constructive
            - Avoid identity-based judgments
            - Focus on communication effectiveness
            - Provide actionable suggestions
            - Keep responses concise but helpful
            
            Text to analyze:
            "{bio_text}"
            {context_str}
            
            Respond with JSON only:
            """

            from openai import OpenAI
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL")
            )
            
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are an expert bio and text analyst. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent output
                response_format={"type": "json_object"}
            )

            raw_json = response.choices[0].message.content
            
            try:
                parsed = BioOutput.model_validate_json(raw_json)
                result = AgentOutput(success=True, data=parsed.dict())
                self._trace(input.dict(), result.dict())
                return result
                
            except ValidationError as ve:
                # Fallback with deterministic response
                result = AgentOutput(success=True, data={
                    "vibe_summary": "Unable to fully analyze - text appears to be personal/creative",
                    "strengths": ["Shows personality", "Communicates intent"],
                    "weaknesses": ["Could be more structured"],
                    "improvements": ["Consider adding specific examples", "Clarify your goals"],
                    "confidence": 0.75
                })
                self._trace(input.dict(), {"error": f"Validation failed: {ve}", "fallback": result.dict()})
                return result

        except Exception as e:
            # Deterministic fallback
            result = AgentOutput(success=True, data={
                "vibe_summary": "Analysis unavailable - using basic assessment",
                "strengths": ["Text provided shows engagement"],
                "weaknesses": ["Unable to analyze in detail"],
                "improvements": ["Consider professional review of content"],
                "confidence": 0.65
            })
            self._trace(input.dict(), {"error": str(e), "fallback": result.dict()})
            return result
