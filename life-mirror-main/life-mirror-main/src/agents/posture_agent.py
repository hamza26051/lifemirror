import os
from .base_agent import BaseAgent, AgentInput, AgentOutput
from src.tools.posture_tool import PostureTool, ToolInput

class PostureAgent(BaseAgent):
    name = "posture_agent"
    output_schema = AgentOutput

    def run(self, input: AgentInput) -> AgentOutput:
        mode = os.getenv("LIFEMIRROR_MODE", "mock")

        if mode == "mock":
            result = AgentOutput(
                success=True,
                data={
                    "alignment_score": 8.5,
                    "crop_url": input.url,
                    "keypoints": [],
                    "tips": ["Relax shoulders", "Straighten spine"],
                    "confidence": 0.85  # High confidence for mock data
                }
            )
            self._trace(input.dict(), result.dict())
            return result

        try:
            tool_res = PostureTool().run(ToolInput(media_id=input.media_id, url=input.url))
            if not tool_res.success:
                result = AgentOutput(success=False, data={}, error=tool_res.error)
                self._trace(input.dict(), result.dict())
                return result

            # Ensure fields exist
            alignment = tool_res.data.get("alignment_score")
            crop_url = tool_res.data.get("crop_url")
            keypoints = tool_res.data.get("keypoints", [])
            tips = tool_res.data.get("tips", [])
            
            # Calculate confidence based on alignment score and keypoints
            confidence = 0.6  # Base confidence
            if alignment is not None:
                confidence = min(0.95, 0.5 + (alignment / 10.0) * 0.4)
            if len(keypoints) > 0:
                confidence = min(0.95, confidence + 0.1)

            result = AgentOutput(
                success=True,
                data={
                    "alignment_score": alignment,
                    "crop_url": crop_url,
                    "keypoints": keypoints,
                    "tips": tips,
                    "confidence": confidence
                }
            )
            self._trace(input.dict(), result.dict())
            return result

        except Exception as e:
            result = AgentOutput(success=False, data={}, error=str(e))
            self._trace(input.dict(), result.dict())
            return result
