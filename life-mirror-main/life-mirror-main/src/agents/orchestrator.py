from .face_agent import FaceAgent
from .fashion_agent import FashionAgent
from .posture_agent import PostureAgent
from .embedder_agent import EmbedderAgent
from .bio_agent import BioAgent
from .aggregator_agent import AggregatorAgent
from .formatter_agent import FormatterAgent
from .memory_agent import MemoryAgent
from .compare_agent import CompareAgent
from .base_agent import AgentInput

# Import existing specialized agents
from .fixit_agent import FixitAgent
from .vibe_analysis_agent import VibeAnalysisAgent
from .reverse_analysis_agent import ReverseAnalysisAgent
from .vibe_compare_agent import VibeComparisonAgent
from .perception_history_agent import PerceptionHistoryAgent
from .social_agent import SocialAgent
from .social_graph_agent import SocialGraphAgent
from .notification_agent import NotificationAgent

from typing import Dict, Any, Optional
import uuid

class Orchestrator:
    def __init__(self):
        # Core analysis agents
        self.face_agent = FaceAgent()
        self.fashion_agent = FashionAgent()
        self.posture_agent = PostureAgent()
        self.embedder_agent = EmbedderAgent()
        self.bio_agent = BioAgent()
        
        # Processing agents
        self.aggregator_agent = AggregatorAgent()
        self.formatter_agent = FormatterAgent()
        
        # Utility agents
        self.memory_agent = MemoryAgent()
        self.compare_agent = CompareAgent()

    def analyze_media(self, media_id: str, url: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete media analysis pipeline using all agents.
        Returns formatted final response ready for API client.
        """
        context = context or {}
        user_id = context.get("user_id")
        
        # Generate LangSmith run ID for traceability
        langsmith_run_id = str(uuid.uuid4())
        
        # Run core analysis agents in parallel (simplified sequential for now)
        agent_input = AgentInput(media_id=media_id, url=url, context=context)
        
        # Core CV and embedding analysis
        embed_result = self.embedder_agent.run(agent_input)
        face_result = self.face_agent.run(agent_input)
        fashion_result = self.fashion_agent.run(agent_input)
        posture_result = self.posture_agent.run(agent_input)
        
        # Bio analysis if text provided
        bio_result = None
        if context.get("bio_text"):
            bio_context = {**context, "text": context["bio_text"]}
            bio_input = AgentInput(media_id=media_id, url=url, context=bio_context)
            bio_result = self.bio_agent.run(bio_input)
        
        # Aggregate all results
        aggregator_context = {
            "face_result": face_result.dict(),
            "fashion_result": fashion_result.dict(),
            "posture_result": posture_result.dict(),
            "bio_result": bio_result.dict() if bio_result else {},
            "embedding_result": embed_result.dict(),
            "langsmith_run_ids": {
                "face": langsmith_run_id + "_face",
                "fashion": langsmith_run_id + "_fashion", 
                "posture": langsmith_run_id + "_posture",
                "bio": langsmith_run_id + "_bio" if bio_result else "",
                "embedding": langsmith_run_id + "_embedding"
            }
        }
        
        aggregator_input = AgentInput(
            media_id=media_id, 
            url=url, 
            context=aggregator_context
        )
        aggregated_result = self.aggregator_agent.run(aggregator_input)
        
        # Format final response
        formatter_context = {
            "aggregated_result": aggregated_result.data,
            "user_consent": context.get("user_consent", {}),
            "request_metadata": context.get("request_metadata", {}),
            "langsmith_run_id": langsmith_run_id
        }
        
        formatter_input = AgentInput(
            media_id=media_id,
            url=url,
            context=formatter_context
        )
        final_result = self.formatter_agent.run(formatter_input)
        
        return final_result.dict()

    def search_memory(self, user_id: str, query_text: Optional[str] = None, 
                     query_vector: Optional[list] = None, **kwargs) -> Dict[str, Any]:
        """Search user's past analyses using MemoryAgent"""
        
        search_context = {
            "user_id": user_id,
            "query_text": query_text,
            "query_vector": query_vector,
            **kwargs
        }
        
        memory_input = AgentInput(
            media_id="search",
            url="",
            context=search_context
        )
        
        return self.memory_agent.run(memory_input).dict()

    def compare_analysis(self, user_id: str, current_analysis: Dict[str, Any],
                        comparison_type: str, **kwargs) -> Dict[str, Any]:
        """Compare current analysis using CompareAgent"""
        
        compare_context = {
            "user_id": user_id,
            "current_analysis": current_analysis,
            "comparison_type": comparison_type,
            **kwargs
        }
        
        compare_input = AgentInput(
            media_id="compare",
            url="", 
            context=compare_context
        )
        
        return self.compare_agent.run(compare_input).dict()

    def analyze_bio_text(self, text: str, user_id: Optional[str] = None, 
                        past_analyses: Optional[list] = None) -> Dict[str, Any]:
        """Analyze bio/profile text using BioAgent"""
        
        bio_context = {
            "text": text,
            "past_analyses": past_analyses or []
        }
        
        bio_input = AgentInput(
            media_id="bio_analysis",
            url="",
            context=bio_context
        )
        
        return self.bio_agent.run(bio_input).dict()


class EnhancedOrchestrator(Orchestrator):
    """
    Enhanced orchestrator that combines core analysis pipeline with specialized agents.
    Provides hybrid integration between new core agents and existing specialized agents.
    """
    
    def __init__(self):
        super().__init__()
        
        # Add existing specialized agents
        self.fixit_agent = FixitAgent()
        self.vibe_analysis_agent = VibeAnalysisAgent()
        self.reverse_analysis_agent = ReverseAnalysisAgent()
        self.vibe_compare_agent = VibeComparisonAgent()
        self.perception_history_agent = PerceptionHistoryAgent()
        self.social_agent = SocialAgent()
        self.social_graph_agent = SocialGraphAgent()
        self.notification_agent = NotificationAgent()

    def full_analysis_with_enhancements(self, media_id: str, url: str, 
                                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete analysis pipeline that combines:
        1. Core analysis (Face, Fashion, Posture, Bio, Embedding -> Aggregator -> Formatter)
        2. Specialized analysis (Fixit, Vibe, History, Social)
        3. Enhanced features (Social graph, comparisons)
        
        Returns comprehensive analysis ready for frontend consumption.
        """
        context = context or {}
        user_id = context.get("user_id")
        
        # Step 1: Run core analysis pipeline
        core_result = self.analyze_media(media_id, url, context)
        
        # Step 2: Run specialized agents in parallel
        specialized_results = {}
        
        if user_id:
            # Fixit suggestions based on recent perception data
            try:
                fixit_input = AgentInput(
                    media_id=media_id,
                    url=url,
                    data={
                        "user_id": user_id,
                        "recent_limit": context.get("recent_limit", 5)
                    }
                )
                fixit_result = self.fixit_agent.run(fixit_input)
                specialized_results["fixit_suggestions"] = fixit_result.data if fixit_result.success else {}
            except Exception as e:
                specialized_results["fixit_suggestions"] = {"error": str(e)}
            
            # Vibe analysis based on recent uploads
            try:
                vibe_input = AgentInput(
                    media_id=media_id,
                    url=url,
                    data={
                        "user_id": user_id,
                        "recent_limit": context.get("recent_limit", 5)
                    }
                )
                vibe_result = self.vibe_analysis_agent.run(vibe_input)
                specialized_results["vibe_analysis"] = vibe_result.data if vibe_result.success else {}
            except Exception as e:
                specialized_results["vibe_analysis"] = {"error": str(e)}
            
            # Perception history trends
            try:
                history_input = AgentInput(
                    media_id=media_id,
                    url=url,
                    data={"user_id": user_id}
                )
                history_result = self.perception_history_agent.run(history_input)
                specialized_results["perception_history"] = history_result.data if history_result.success else {}
            except Exception as e:
                specialized_results["perception_history"] = {"error": str(e)}
            
            # Social graph analysis (if user opted in)
            try:
                social_graph_input = AgentInput(
                    media_id=media_id,
                    url=url,
                    data={"user_id": user_id}
                )
                social_graph_result = self.social_graph_agent.run(social_graph_input)
                specialized_results["social_graph"] = social_graph_result.data if social_graph_result.success else {}
            except Exception as e:
                specialized_results["social_graph"] = {"error": str(e)}
            
            # Social perception summary (from core analysis results)
            try:
                if core_result.get("success") and core_result.get("data"):
                    social_input = AgentInput(
                        media_id=media_id,
                        url=url,
                        data={
                            "perception_data": core_result["data"],
                            "media_id": media_id
                        }
                    )
                    social_result = self.social_agent.run(social_input)
                    specialized_results["social_perception"] = social_result.data if social_result.success else {}
            except Exception as e:
                specialized_results["social_perception"] = {"error": str(e)}
        
        # Step 3: Combine all results
        enhanced_result = {
            "core_analysis": core_result,
            "specialized_analysis": specialized_results,
            "metadata": {
                "analysis_timestamp": context.get("analysis_timestamp"),
                "user_id": user_id,
                "media_id": media_id,
                "processing_mode": "enhanced_hybrid"
            }
        }
        
        return enhanced_result

    def reverse_goal_analysis(self, user_id: str, goal: str, 
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run reverse analysis to determine how to achieve a specific perception goal.
        """
        reverse_input = AgentInput(
            media_id="reverse_analysis",
            url="",
            data={
                "user_id": user_id,
                "goal": goal,
                "recent_limit": context.get("recent_limit", 5) if context else 5
            }
        )
        
        result = self.reverse_analysis_agent.run(reverse_input)
        return result.dict()

    def compare_media_vibes(self, media_id_1: str, media_id_2: str) -> Dict[str, Any]:
        """
        Compare social vibes between two media items.
        """
        compare_input = AgentInput(
            media_id="comparison",
            url="",
            data={
                "media_id_1": int(media_id_1),
                "media_id_2": int(media_id_2)
            }
        )
        
        result = self.vibe_compare_agent.run(compare_input)
        return result.dict()

    def generate_notifications(self, user_id: str) -> Dict[str, Any]:
        """
        Generate personalized notifications for user.
        """
        notification_input = AgentInput(
            media_id="notifications",
            url="",
            data={"user_id": user_id}
        )
        
        result = self.notification_agent.run(notification_input)
        return result.dict()

    def get_comprehensive_user_profile(self, user_id: str, 
                                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive user profile combining all available analysis types.
        """
        context = context or {}
        
        profile = {
            "user_id": user_id,
            "profile_components": {}
        }
        
        # Get perception history
        try:
            history_result = self.perception_history_agent.run(AgentInput(
                media_id="profile",
                url="",
                data={"user_id": user_id}
            ))
            profile["profile_components"]["history"] = history_result.data if history_result.success else {}
        except Exception as e:
            profile["profile_components"]["history"] = {"error": str(e)}
        
        # Get current vibe analysis
        try:
            vibe_result = self.vibe_analysis_agent.run(AgentInput(
                media_id="profile",
                url="",
                data={"user_id": user_id, "recent_limit": 5}
            ))
            profile["profile_components"]["current_vibe"] = vibe_result.data if vibe_result.success else {}
        except Exception as e:
            profile["profile_components"]["current_vibe"] = {"error": str(e)}
        
        # Get social graph position
        try:
            social_graph_result = self.social_graph_agent.run(AgentInput(
                media_id="profile",
                url="",
                data={"user_id": user_id}
            ))
            profile["profile_components"]["social_position"] = social_graph_result.data if social_graph_result.success else {}
        except Exception as e:
            profile["profile_components"]["social_position"] = {"error": str(e)}
        
        # Get improvement suggestions
        try:
            fixit_result = self.fixit_agent.run(AgentInput(
                media_id="profile",
                url="",
                data={"user_id": user_id, "recent_limit": 5}
            ))
            profile["profile_components"]["improvement_suggestions"] = fixit_result.data if fixit_result.success else {}
        except Exception as e:
            profile["profile_components"]["improvement_suggestions"] = {"error": str(e)}
        
        return profile
