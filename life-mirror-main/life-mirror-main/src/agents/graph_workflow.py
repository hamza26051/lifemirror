import os
from langgraph.graph import StateGraph
from .base_agent import AgentInput
from .embedder_agent import EmbedderAgent
from .face_agent import FaceAgent
from .fashion_agent import FashionAgent
from .posture_agent import PostureAgent
from .bio_agent import BioAgent
from .aggregator_agent import AggregatorAgent
from .formatter_agent import FormatterAgent

# Import existing specialized agents
from .fixit_agent import FixitAgent
from .vibe_analysis_agent import VibeAnalysisAgent
from .social_agent import SocialAgent

from typing import Dict, Any, Optional, TypedDict, Annotated
import uuid

def merge_dicts(left, right):
    """Merge two values, with right taking precedence for non-None values"""
    if left is None:
        return right
    if right is None:
        return left
    # For dictionaries, merge them
    if isinstance(left, dict) and isinstance(right, dict):
        return {**left, **right}
    # For other types, right takes precedence
    return right

class GraphState(TypedDict):
    """State schema for the analysis workflow graph"""
    media_id: Annotated[str, merge_dicts]
    url: Annotated[str, merge_dicts]
    context: Annotated[Dict[str, Any], merge_dicts]
    
    # Agent results - using Annotated with merge_dicts to handle concurrent updates
    embedding: Annotated[Optional[Dict[str, Any]], merge_dicts]
    face: Annotated[Optional[Dict[str, Any]], merge_dicts]
    fashion: Annotated[Optional[Dict[str, Any]], merge_dicts]
    posture: Annotated[Optional[Dict[str, Any]], merge_dicts]
    bio: Annotated[Optional[Dict[str, Any]], merge_dicts]
    aggregated: Annotated[Optional[Dict[str, Any]], merge_dicts]
    final_result: Annotated[Optional[Dict[str, Any]], merge_dicts]
    
    # Enhancement results
    social_analysis_result: Annotated[Optional[Dict[str, Any]], merge_dicts]
    enhancement_results: Annotated[Optional[Dict[str, Any]], merge_dicts]
    
    # Tracing
    langsmith_run_id: Annotated[Optional[str], merge_dicts]
    langsmith_run_ids: Annotated[Optional[Dict[str, str]], merge_dicts]

class GraphExecutor:
    def __init__(self):
        self.graph = StateGraph(GraphState)

        # Create agents
        self.embedder_agent = EmbedderAgent()
        self.face_agent = FaceAgent()
        self.fashion_agent = FashionAgent()
        self.posture_agent = PostureAgent()
        self.bio_agent = BioAgent()
        self.aggregator_agent = AggregatorAgent()
        self.formatter_agent = FormatterAgent()

        # Add nodes for parallel analysis
        self.graph.add_node("embedding", self.run_embedder)
        self.graph.add_node("face", self.run_face)
        self.graph.add_node("fashion", self.run_fashion)
        self.graph.add_node("posture", self.run_posture)
        self.graph.add_node("bio", self.run_bio)
        
        # Add processing nodes
        self.graph.add_node("aggregate", self.run_aggregator)
        self.graph.add_node("format", self.run_formatter)

        # Connect edges for parallel processing
        # All core agents can run in parallel after embedding
        self.graph.add_edge("embedding", "face")
        self.graph.add_edge("embedding", "fashion")
        self.graph.add_edge("embedding", "posture")
        self.graph.add_edge("embedding", "bio")
        
        # ALL agents feed into aggregator to prevent hanging
        # (aggregator handles missing/failed agents gracefully)
        self.graph.add_edge("face", "aggregate")
        self.graph.add_edge("fashion", "aggregate")
        self.graph.add_edge("posture", "aggregate")
        self.graph.add_edge("bio", "aggregate")
        
        # Aggregator feeds into formatter
        self.graph.add_edge("aggregate", "format")

        # Mark entry point
        self.graph.set_entry_point("embedding")
        
        # Add END node for proper graph termination
        from langgraph.graph import END
        self.graph.add_edge("format", END)

    def run_embedder(self, state):
        """Run embedding agent and add LangSmith run ID"""
        # Convert state dict to proper format for AgentInput
        state_dict = state if isinstance(state, dict) else state.dict()
        input_data = AgentInput(
            media_id=state_dict["media_id"],
            url=state_dict["url"],
            context=state_dict.get("context", {})
        )
        res = self.embedder_agent.run(input_data)
        
        # Generate LangSmith run ID for this execution
        langsmith_run_id = str(uuid.uuid4())
        
        # Update state with new values
        updated_state = state_dict.copy()
        updated_state.update({
            "embedding": res.dict(), 
            "langsmith_run_id": langsmith_run_id,
            "langsmith_run_ids": {"embedding": langsmith_run_id + "_embedding"}
        })
        return updated_state

    def run_face(self, state):
        """Run face agent with error handling"""
        try:
            state_dict = state if isinstance(state, dict) else state.dict()
            input_data = AgentInput(
                media_id=state_dict["media_id"],
                url=state_dict["url"],
                context=state_dict.get("context", {})
            )
            res = self.face_agent.run(input_data)
            
            # Update run IDs
            run_ids = state_dict.get("langsmith_run_ids", {})
            run_ids["face"] = state_dict.get("langsmith_run_id", "unknown") + "_face"
            
            updated_state = state_dict.copy()
            updated_state.update({
                "face": res.dict(), 
                "langsmith_run_ids": run_ids
            })
            return updated_state
        except Exception as e:
            # Handle errors gracefully
            state_dict = state if isinstance(state, dict) else state.dict()
            error_result = {"success": False, "data": {}, "error": str(e)}
            run_ids = state_dict.get("langsmith_run_ids", {})
            run_ids["face"] = state_dict.get("langsmith_run_id", "unknown") + "_face_error"
            
            updated_state = state_dict.copy()
            updated_state.update({
                "face": error_result,
                "langsmith_run_ids": run_ids
            })
            return updated_state

    def run_fashion(self, state):
        """Run fashion agent with error handling"""
        try:
            state_dict = state if isinstance(state, dict) else state.dict()
            input_data = AgentInput(
                media_id=state_dict.get("media_id", ""),
                url=state_dict.get("url", ""),
                context=state_dict.get("context", {})
            )
            res = self.fashion_agent.run(input_data)
            
            run_ids = state_dict.get("langsmith_run_ids", {})
            run_ids["fashion"] = state_dict.get("langsmith_run_id", "unknown") + "_fashion"
            
            return {
                "fashion": res.dict(),
                "langsmith_run_ids": run_ids,
                **state_dict
            }
        except Exception as e:
            error_result = {"success": False, "data": {}, "error": str(e)}
            state_dict = state if isinstance(state, dict) else state.dict()
            run_ids = state_dict.get("langsmith_run_ids", {})
            run_ids["fashion"] = state_dict.get("langsmith_run_id", "unknown") + "_fashion_error"
            
            return {
                "fashion": error_result,
                "langsmith_run_ids": run_ids,
                **state_dict
            }

    def run_posture(self, state):
        """Run posture agent with error handling"""
        try:
            state_dict = state if isinstance(state, dict) else state.dict()
            input_data = AgentInput(
                media_id=state_dict.get("media_id", ""),
                url=state_dict.get("url", ""),
                context=state_dict.get("context", {})
            )
            res = self.posture_agent.run(input_data)
            
            run_ids = state_dict.get("langsmith_run_ids", {})
            run_ids["posture"] = state_dict.get("langsmith_run_id", "unknown") + "_posture"
            
            return {
                "posture": res.dict(),
                "langsmith_run_ids": run_ids,
                **state_dict
            }
        except Exception as e:
            error_result = {"success": False, "data": {}, "error": str(e)}
            state_dict = state if isinstance(state, dict) else state.dict()
            run_ids = state_dict.get("langsmith_run_ids", {})
            run_ids["posture"] = state_dict.get("langsmith_run_id", "unknown") + "_posture_error"
            
            return {
                "posture": error_result,
                "langsmith_run_ids": run_ids,
                **state_dict
            }

    def run_bio(self, state):
        """Run bio agent if bio text is provided"""
        state_dict = state if isinstance(state, dict) else state.dict()
        context = state_dict.get("context", {})
        
        # Only run bio agent if bio text is provided
        if not context.get("bio_text"):
            return {
                "bio": {"success": True, "data": {}, "skipped": True},
                **state_dict
            }
        
        try:
            # Create bio-specific context
            bio_context = {**context, "text": context["bio_text"]}
            input_data = AgentInput(
                media_id=state_dict.get("media_id", ""),
                url=state_dict.get("url", ""),
                context=bio_context
            )
            res = self.bio_agent.run(input_data)
            
            run_ids = state_dict.get("langsmith_run_ids", {})
            run_ids["bio"] = state_dict.get("langsmith_run_id", "unknown") + "_bio"
            
            return {
                "bio": res.dict(),
                "langsmith_run_ids": run_ids,
                **state_dict
            }
        except Exception as e:
            error_result = {"success": False, "data": {}, "error": str(e)}
            run_ids = state_dict.get("langsmith_run_ids", {})
            run_ids["bio"] = state_dict.get("langsmith_run_id", "unknown") + "_bio_error"
            
            return {
                "bio": error_result,
                "langsmith_run_ids": run_ids,
                **state_dict
            }

    def run_aggregator(self, state):
        """Aggregate all agent results"""
        try:
            state_dict = state if isinstance(state, dict) else state.dict()
            # Prepare aggregator context with safe defaults for None values
            aggregator_context = {
                "face_result": state_dict.get("face") or {"success": False, "data": {}, "error": "Agent not executed"},
                "fashion_result": state_dict.get("fashion") or {"success": False, "data": {}, "error": "Agent not executed"},
                "posture_result": state_dict.get("posture") or {"success": False, "data": {}, "error": "Agent not executed"},
                "bio_result": state_dict.get("bio") or {"success": False, "data": {}, "error": "Agent not executed"},
                "embedding_result": state_dict.get("embedding") or {"success": False, "data": {}, "error": "Agent not executed"},
                "langsmith_run_ids": state_dict.get("langsmith_run_ids", {}),
                "processing_time": state_dict.get("context", {}).get("processing_time")
            }
            
            aggregator_input = AgentInput(
                media_id=state_dict.get("media_id", ""),
                url=state_dict.get("url", ""),
                context=aggregator_context
            )
            
            res = self.aggregator_agent.run(aggregator_input)
            
            updated_state = state_dict.copy()
            updated_state.update({
                "aggregated": res.dict()
            })
            return updated_state
        except Exception as e:
            state_dict = state if isinstance(state, dict) else state.dict()
            error_result = {"success": False, "data": {}, "error": str(e)}
            updated_state = state_dict.copy()
            updated_state.update({
                "aggregated": error_result
            })
            return updated_state

    def run_formatter(self, state):
        """Format final response"""
        print("🔧 DEBUG: run_formatter called!")
        print(f"   - State keys: {list(state.keys())}")
        print(f"   - Aggregated data available: {'aggregated' in state}")
        try:
            aggregated_data = state.get("aggregated", {}).get("data", {})
            context = state.get("context", {})
            print(f"   - Aggregated data: {bool(aggregated_data)}")
            print(f"   - Context: {bool(context)}")
            
            formatter_context = {
                "aggregated_result": aggregated_data,
                "user_consent": context.get("user_consent", {}),
                "request_metadata": context.get("request_metadata", {}),
                "langsmith_run_id": state.get("langsmith_run_id")
            }
            
            formatter_input = AgentInput(
                media_id=state.get("media_id", ""),
                url=state.get("url", ""),
                context=formatter_context
            )
            
            res = self.formatter_agent.run(formatter_input)
            
            # Ensure proper state update for LangGraph
            updated_state = state.copy() if isinstance(state, dict) else state
            updated_state["final_result"] = res.dict()
            return updated_state
        except Exception as e:
            # Create fallback response
            from datetime import datetime
            fallback_response = {
                "success": True,
                "data": {
                    "media_id": state.get("media_id", ""),
                    "timestamp": datetime.utcnow().isoformat(),
                    "overall_score": 5.0,
                    "summary": "Analysis completed with basic results due to processing error.",
                    "key_insights": ["Media processed successfully"],
                    "recommendations": ["Consider retrying analysis"],
                    "confidence": 0.3,
                    "warnings": ["Using fallback response due to processing error"],
                    "disclaimers": ["This is a fallback response with limited analysis."],
                    "processing_metadata": {"fallback": True, "error": str(e)}
                }
            }
            
            return {
                "final_result": fallback_response,
                **state
            }

    def execute(self, media_id: str, url: str, context: dict = None) -> Dict[str, Any]:
        """Execute the full analysis workflow"""
        context = context or {}
        initial_state = {
            "media_id": media_id,
            "url": url,
            "context": context,
            "embedding": None,
            "face": None,
            "fashion": None,
            "posture": None,
            "bio": None,
            "aggregated": None,
            "final_result": None,
            "social_analysis_result": None,
            "enhancement_results": None,
            "langsmith_run_id": str(uuid.uuid4()),
            "langsmith_run_ids": None
        }
        
        try:
            # Compile the graph and then invoke it
            compiled_graph = self.graph.compile()
            final_state = compiled_graph.invoke(initial_state)
            final_result = final_state.get("final_result")
            
            if final_result is None:
                # Return fallback response if final_result is None
                from datetime import datetime
                return {
                    "success": False,
                    "data": {},
                    "error": "Graph execution completed but returned no result",
                    "fallback_response": {
                        "media_id": media_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "overall_score": 0.0,
                        "summary": "Analysis completed but no result was generated.",
                        "confidence": 0.0,
                        "warnings": ["No analysis result generated"],
                        "processing_metadata": {"execution_error": "final_result is None"}
                    }
                }
            
            # Extract the actual data from the formatter agent output
            # final_result is an AgentOutput dict with 'success', 'data', 'error' keys
            # We need to return the 'data' field which contains the FinalAnalysisResponse
            if isinstance(final_result, dict) and "data" in final_result:
                actual_data = final_result["data"]
            else:
                actual_data = final_result
            
            # Return success response with proper structure
            return {
                "success": True,
                "data": actual_data,
                "error": None
            }
            
        except Exception as e:
            # Return error response
            from datetime import datetime
            return {
                "success": False,
                "data": {},
                "error": f"Graph execution failed: {str(e)}",
                "fallback_response": {
                    "media_id": media_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "overall_score": 0.0,
                    "summary": "Analysis failed due to system error.",
                    "confidence": 0.0,
                    "warnings": ["System error occurred during analysis"],
                    "processing_metadata": {"execution_error": str(e)}
                }
            }


class EnhancedGraphExecutor(GraphExecutor):
    """
    Enhanced graph executor that extends the core LangGraph workflow 
    with specialized agents for comprehensive analysis.
    """
    
    def __init__(self):
        super().__init__()
        
        # Add specialized agents
        self.fixit_agent = FixitAgent()
        self.vibe_analysis_agent = VibeAnalysisAgent()
        self.social_agent = SocialAgent()
        
        # Add specialized nodes to the graph
        self.graph.add_node("social_analysis", self.run_social_analysis)
        self.graph.add_node("enhancement", self.run_enhancements)
        
        # Connect specialized nodes to the workflow
        # Social analysis runs after formatting
        self.graph.add_edge("format", "social_analysis")
        # Enhancement runs after social analysis
        self.graph.add_edge("social_analysis", "enhancement")

    def run_social_analysis(self, state):
        """
        Run social perception analysis using the formatted results.
        """
        try:
            from src.utils.tracing import AgentTraceContext
            
            with AgentTraceContext("social_analysis") as trace_ctx:
                # Extract formatted results for social analysis
                formatted_result = state.get("final_result", {})
                
                if formatted_result.get("success") and formatted_result.get("data"):
                    social_input = AgentInput(
                        media_id=state.get("media_id", ""),
                        url=state.get("url", ""),
                        data={
                            "perception_data": formatted_result["data"],
                            "media_id": state.get("media_id")
                        }
                    )
                    
                    social_result = self.social_agent.run(social_input)
                    
                    if social_result.success:
                        state["social_analysis_result"] = social_result.data
                        trace_ctx.log_success("Social analysis completed", social_result.data)
                    else:
                        state["social_analysis_result"] = {"error": social_result.error}
                        trace_ctx.log_error("Social analysis failed", social_result.error)
                else:
                    error_msg = "No valid formatted result available for social analysis"
                    state["social_analysis_result"] = {"error": error_msg}
                    trace_ctx.log_error("Social analysis skipped", error_msg)
                    
        except Exception as e:
            error_msg = f"Social analysis execution failed: {str(e)}"
            state["social_analysis_result"] = {"error": error_msg}
            
        return state

    def run_enhancements(self, state):
        """
        Run enhancement agents (fixit suggestions, vibe analysis) based on user context.
        """
        try:
            from src.utils.tracing import AgentTraceContext
            
            with AgentTraceContext("enhancements") as trace_ctx:
                user_id = state.get("context", {}).get("user_id")
                enhancement_results = {}
                
                if user_id:
                    # Run fixit suggestions
                    try:
                        fixit_input = AgentInput(
                            media_id=state.get("media_id", ""),
                            url=state.get("url", ""),
                            data={
                                "user_id": user_id,
                                "recent_limit": 5
                            }
                        )
                        fixit_result = self.fixit_agent.run(fixit_input)
                        enhancement_results["fixit_suggestions"] = fixit_result.data if fixit_result.success else {"error": fixit_result.error}
                    except Exception as e:
                        enhancement_results["fixit_suggestions"] = {"error": str(e)}
                    
                    # Run vibe analysis
                    try:
                        vibe_input = AgentInput(
                            media_id=state.get("media_id", ""),
                            url=state.get("url", ""),
                            data={
                                "user_id": user_id,
                                "recent_limit": 5
                            }
                        )
                        vibe_result = self.vibe_analysis_agent.run(vibe_input)
                        enhancement_results["vibe_analysis"] = vibe_result.data if vibe_result.success else {"error": vibe_result.error}
                    except Exception as e:
                        enhancement_results["vibe_analysis"] = {"error": str(e)}
                    
                    trace_ctx.log_success("Enhancement analysis completed", enhancement_results)
                else:
                    enhancement_results["info"] = "No user_id provided, skipping personalized enhancements"
                    trace_ctx.log_info("Enhancements skipped", "No user context")
                
                state["enhancement_results"] = enhancement_results
                
        except Exception as e:
            error_msg = f"Enhancement execution failed: {str(e)}"
            state["enhancement_results"] = {"error": error_msg}
            
        return state

    def execute_enhanced(self, media_id: str, url: str, context: dict = None) -> Dict[str, Any]:
        """
        Execute the enhanced analysis workflow that includes specialized agents.
        """
        context = context or {}
        initial_state = {
            "media_id": media_id,
            "url": url,
            "context": context,
            "embedding": None,
            "face": None,
            "fashion": None,
            "posture": None,
            "bio": None,
            "aggregated": None,
            "final_result": None,
            "social_analysis_result": None,
            "enhancement_results": None,
            "langsmith_run_id": str(uuid.uuid4()),
            "langsmith_run_ids": None
        }
        
        try:
            # Compile the graph and then invoke it
            compiled_graph = self.graph.compile()
            final_state = compiled_graph.invoke(initial_state)
            
            # Combine all results
            core_result = final_state.get("final_result", {})
            social_result = final_state.get("social_analysis_result", {})
            enhancement_results = final_state.get("enhancement_results", {})
            
            enhanced_response = {
                "success": True,
                "core_analysis": core_result,
                "social_analysis": social_result,
                "enhancements": enhancement_results,
                "metadata": {
                    "processing_mode": "enhanced_graph",
                    "media_id": media_id,
                    "user_id": context.get("user_id"),
                    "execution_timestamp": context.get("analysis_timestamp")
                }
            }
            
            return enhanced_response
            
        except Exception as e:
            # Return error response with fallback
            from datetime import datetime
            return {
                "success": False,
                "error": f"Enhanced graph execution failed: {str(e)}",
                "core_analysis": {},
                "social_analysis": {},
                "enhancements": {},
                "fallback_response": {
                    "media_id": media_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "overall_score": 0.0,
                    "summary": "Enhanced analysis failed due to system error.",
                    "confidence": 0.0,
                    "warnings": ["System error occurred during enhanced analysis"],
                    "processing_metadata": {"execution_error": str(e)}
                }
            }
