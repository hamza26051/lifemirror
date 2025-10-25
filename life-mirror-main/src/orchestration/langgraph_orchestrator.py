import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
import json
import time

from src.agents.base_agent import BaseAgent, AgentInput, AgentOutput
from src.agents.face_agent import FaceAgent
from src.agents.fashion_agent import FashionAgent
from src.agents.posture_agent import PostureAgent
from src.agents.bio_agent import BioAgent
from src.agents.social_agent import SocialAgent
from src.agents.vibe_analysis_agent import VibeAnalysisAgent
from src.agents.vibe_compare_agent import VibeCompareAgent
from src.agents.reverse_analysis_agent import ReverseAnalysisAgent
from src.agents.compare_agent import CompareAgent
from src.agents.fixit_agent import FixitAgent
from src.agents.formatter_agent import FormatterAgent
from src.agents.memory_agent import MemoryAgent
from src.agents.embedder_agent import EmbedderAgent
from src.agents.aggregator_agent import AggregatorAgent
from src.agents.perception_history_agent import PerceptionHistoryAgent
from src.agents.social_graph_agent import SocialGraphAgent
from src.agents.notification_agent import NotificationAgent
from src.utils.tracing import log_trace

class WorkflowStage(str, Enum):
    """Workflow execution stages"""
    INITIALIZATION = "initialization"
    PRIMARY_ANALYSIS = "primary_analysis"
    SECONDARY_ANALYSIS = "secondary_analysis"
    COMPARISON_ANALYSIS = "comparison_analysis"
    AGGREGATION = "aggregation"
    ENHANCEMENT = "enhancement"
    FINALIZATION = "finalization"

class AgentPriority(str, Enum):
    """Agent execution priorities"""
    CRITICAL = "critical"  # Must complete before others can start
    HIGH = "high"         # Important for core functionality
    MEDIUM = "medium"     # Standard processing
    LOW = "low"          # Enhancement and optional features

@dataclass
class AgentConfig:
    """Configuration for agent execution"""
    agent_class: type
    priority: AgentPriority
    stage: WorkflowStage
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 30
    retry_count: int = 2
    parallel_group: Optional[str] = None

@dataclass
class ExecutionResult:
    """Result of agent execution"""
    agent_name: str
    success: bool
    data: Dict[str, Any]
    confidence: float
    processing_time: float
    error: Optional[str] = None
    stage: WorkflowStage = WorkflowStage.PRIMARY_ANALYSIS
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class WorkflowContext:
    """Context shared across workflow execution"""
    media_id: str
    user_id: Optional[str] = None
    analysis_type: str = "comprehensive"
    results: Dict[str, ExecutionResult] = field(default_factory=dict)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)

class LangGraphOrchestrator:
    """LangGraph-based orchestration system for managing 18-agent workflows"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Agent configurations with dependencies and priorities
        self.agent_configs = {
            # Stage 1: Critical Foundation Agents
            "face": AgentConfig(
                agent_class=FaceAgent,
                priority=AgentPriority.CRITICAL,
                stage=WorkflowStage.PRIMARY_ANALYSIS,
                parallel_group="visual_analysis"
            ),
            "embedder": AgentConfig(
                agent_class=EmbedderAgent,
                priority=AgentPriority.CRITICAL,
                stage=WorkflowStage.INITIALIZATION,
                parallel_group="data_processing"
            ),
            "memory": AgentConfig(
                agent_class=MemoryAgent,
                priority=AgentPriority.CRITICAL,
                stage=WorkflowStage.INITIALIZATION,
                parallel_group="data_processing"
            ),
            
            # Stage 2: Primary Analysis Agents
            "fashion": AgentConfig(
                agent_class=FashionAgent,
                priority=AgentPriority.HIGH,
                stage=WorkflowStage.PRIMARY_ANALYSIS,
                dependencies=["face"],
                parallel_group="visual_analysis"
            ),
            "posture": AgentConfig(
                agent_class=PostureAgent,
                priority=AgentPriority.HIGH,
                stage=WorkflowStage.PRIMARY_ANALYSIS,
                dependencies=["face"],
                parallel_group="visual_analysis"
            ),
            "bio": AgentConfig(
                agent_class=BioAgent,
                priority=AgentPriority.HIGH,
                stage=WorkflowStage.PRIMARY_ANALYSIS,
                dependencies=["embedder"],
                parallel_group="text_analysis"
            ),
            "social": AgentConfig(
                agent_class=SocialAgent,
                priority=AgentPriority.HIGH,
                stage=WorkflowStage.PRIMARY_ANALYSIS,
                dependencies=["embedder"],
                parallel_group="text_analysis"
            ),
            "vibe_analysis": AgentConfig(
                agent_class=VibeAnalysisAgent,
                priority=AgentPriority.HIGH,
                stage=WorkflowStage.PRIMARY_ANALYSIS,
                dependencies=["embedder"],
                parallel_group="text_analysis"
            ),
            
            # Stage 3: Secondary Analysis Agents
            "vibe_compare": AgentConfig(
                agent_class=VibeCompareAgent,
                priority=AgentPriority.MEDIUM,
                stage=WorkflowStage.SECONDARY_ANALYSIS,
                dependencies=["vibe_analysis", "memory"]
            ),
            "reverse_analysis": AgentConfig(
                agent_class=ReverseAnalysisAgent,
                priority=AgentPriority.MEDIUM,
                stage=WorkflowStage.SECONDARY_ANALYSIS,
                dependencies=["face", "fashion", "social"]
            ),
            "perception_history": AgentConfig(
                agent_class=PerceptionHistoryAgent,
                priority=AgentPriority.MEDIUM,
                stage=WorkflowStage.SECONDARY_ANALYSIS,
                dependencies=["memory"]
            ),
            "social_graph": AgentConfig(
                agent_class=SocialGraphAgent,
                priority=AgentPriority.MEDIUM,
                stage=WorkflowStage.SECONDARY_ANALYSIS,
                dependencies=["social", "memory"]
            ),
            
            # Stage 4: Comparison and Enhancement
            "compare": AgentConfig(
                agent_class=CompareAgent,
                priority=AgentPriority.MEDIUM,
                stage=WorkflowStage.COMPARISON_ANALYSIS,
                dependencies=["face", "fashion", "posture", "bio", "social", "memory"]
            ),
            "fixit": AgentConfig(
                agent_class=FixitAgent,
                priority=AgentPriority.MEDIUM,
                stage=WorkflowStage.ENHANCEMENT,
                dependencies=["face", "fashion", "posture", "bio", "social"]
            ),
            
            # Stage 5: Aggregation
            "aggregator": AgentConfig(
                agent_class=AggregatorAgent,
                priority=AgentPriority.HIGH,
                stage=WorkflowStage.AGGREGATION,
                dependencies=["face", "fashion", "posture", "bio", "social", "vibe_analysis"]
            ),
            
            # Stage 6: Final Processing
            "formatter": AgentConfig(
                agent_class=FormatterAgent,
                priority=AgentPriority.MEDIUM,
                stage=WorkflowStage.FINALIZATION,
                dependencies=["aggregator"]
            ),
            "notification": AgentConfig(
                agent_class=NotificationAgent,
                priority=AgentPriority.LOW,
                stage=WorkflowStage.FINALIZATION,
                dependencies=["aggregator"]
            )
        }
        
        # Initialize agent instances
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all agent instances"""
        for agent_name, config in self.agent_configs.items():
            try:
                self.agents[agent_name] = config.agent_class()
                self.logger.info(f"Initialized {agent_name} agent")
            except Exception as e:
                self.logger.error(f"Failed to initialize {agent_name} agent: {e}")
    
    @log_trace
    async def execute_workflow(self, 
                             media_id: str, 
                             user_id: Optional[str] = None,
                             analysis_type: str = "comprehensive",
                             selected_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute the complete workflow with all agents"""
        
        context = WorkflowContext(
            media_id=media_id,
            user_id=user_id,
            analysis_type=analysis_type
        )
        
        try:
            # Determine which agents to run
            agents_to_run = selected_agents or list(self.agent_configs.keys())
            
            # Execute workflow stages in order
            for stage in WorkflowStage:
                stage_agents = [
                    name for name in agents_to_run 
                    if self.agent_configs[name].stage == stage
                ]
                
                if stage_agents:
                    self.logger.info(f"Executing stage: {stage.value} with agents: {stage_agents}")
                    await self._execute_stage(context, stage_agents, stage)
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(context)
            
            # Create final result
            result = {
                "success": True,
                "media_id": media_id,
                "user_id": user_id,
                "analysis_type": analysis_type,
                "execution_time": (datetime.now() - context.start_time).total_seconds(),
                "agent_results": {name: result.data for name, result in context.results.items()},
                "individual_results": {name: result.data for name, result in context.results.items()},
                "overall_metrics": overall_metrics,
                "metadata": context.execution_metadata,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Workflow completed successfully in {result['execution_time']:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "media_id": media_id,
                "partial_results": {name: result.data for name, result in context.results.items()},
                "execution_time": (datetime.now() - context.start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _execute_stage(self, context: WorkflowContext, agent_names: List[str], stage: WorkflowStage):
        """Execute a specific workflow stage"""
        stage_start = time.time()
        
        # Group agents by parallel execution groups
        parallel_groups = self._group_agents_for_parallel_execution(agent_names)
        
        # Execute each parallel group
        for group_name, group_agents in parallel_groups.items():
            if len(group_agents) > 1:
                # Execute in parallel
                await self._execute_parallel_group(context, group_agents)
            else:
                # Execute single agent
                await self._execute_single_agent(context, group_agents[0])
        
        stage_time = time.time() - stage_start
        context.execution_metadata[f"{stage.value}_time"] = stage_time
        self.logger.info(f"Stage {stage.value} completed in {stage_time:.2f}s")
    
    def _group_agents_for_parallel_execution(self, agent_names: List[str]) -> Dict[str, List[str]]:
        """Group agents by their parallel execution groups"""
        groups = {}
        
        for agent_name in agent_names:
            config = self.agent_configs[agent_name]
            group_name = config.parallel_group or f"single_{agent_name}"
            
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(agent_name)
        
        return groups
    
    async def _execute_parallel_group(self, context: WorkflowContext, agent_names: List[str]):
        """Execute a group of agents in parallel"""
        self.logger.info(f"Executing parallel group: {agent_names}")
        
        # Check dependencies for all agents in group
        for agent_name in agent_names:
            if not self._check_dependencies(context, agent_name):
                self.logger.warning(f"Dependencies not met for {agent_name}, skipping parallel execution")
                return
        
        # Create tasks for parallel execution
        tasks = []
        for agent_name in agent_names:
            task = asyncio.create_task(self._execute_single_agent(context, agent_name))
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_single_agent(self, context: WorkflowContext, agent_name: str):
        """Execute a single agent"""
        config = self.agent_configs[agent_name]
        agent = self.agents.get(agent_name)
        
        if not agent:
            self.logger.error(f"Agent {agent_name} not found")
            return
        
        # Check dependencies
        if not self._check_dependencies(context, agent_name):
            self.logger.warning(f"Dependencies not met for {agent_name}, skipping")
            return
        
        start_time = time.time()
        
        try:
            # Prepare agent input
            agent_input = self._prepare_agent_input(context, agent_name)
            
            # Execute agent with timeout and retry
            result = await self._execute_with_retry(agent, agent_input, config)
            
            # Store result
            execution_result = ExecutionResult(
                agent_name=agent_name,
                success=result.success,
                data=result.data,
                confidence=result.confidence,
                processing_time=time.time() - start_time,
                error=result.error,
                stage=config.stage
            )
            
            context.results[agent_name] = execution_result
            
            # Update shared data for dependent agents
            if result.success:
                context.shared_data[agent_name] = result.data
            
            self.logger.info(f"Agent {agent_name} completed in {execution_result.processing_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Agent {agent_name} execution failed: {e}")
            context.results[agent_name] = ExecutionResult(
                agent_name=agent_name,
                success=False,
                data={},
                confidence=0.0,
                processing_time=time.time() - start_time,
                error=str(e),
                stage=config.stage
            )
    
    def _check_dependencies(self, context: WorkflowContext, agent_name: str) -> bool:
        """Check if agent dependencies are satisfied"""
        config = self.agent_configs[agent_name]
        
        for dependency in config.dependencies:
            if dependency not in context.results:
                return False
            if not context.results[dependency].success:
                return False
        
        return True
    
    def _prepare_agent_input(self, context: WorkflowContext, agent_name: str) -> AgentInput:
        """Prepare input for agent execution"""
        # Get dependency results
        dependency_data = {}
        config = self.agent_configs[agent_name]
        
        for dependency in config.dependencies:
            if dependency in context.results and context.results[dependency].success:
                dependency_data[dependency] = context.results[dependency].data
        
        # Prepare context with all available data
        agent_context = {
            "analysis_type": context.analysis_type,
            "user_id": context.user_id,
            "shared_data": context.shared_data,
            "dependency_results": dependency_data,
            "execution_metadata": context.execution_metadata
        }
        
        return AgentInput(
            media_id=context.media_id,
            context=agent_context
        )
    
    async def _execute_with_retry(self, agent: BaseAgent, agent_input: AgentInput, config: AgentConfig) -> AgentOutput:
        """Execute agent with retry logic"""
        last_error = None
        
        for attempt in range(config.retry_count + 1):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    asyncio.to_thread(agent.run, agent_input),
                    timeout=config.timeout_seconds
                )
                
                if result.success:
                    return result
                else:
                    last_error = result.error or "Agent execution failed"
                    if attempt < config.retry_count:
                        self.logger.warning(f"Agent {agent.__class__.__name__} failed (attempt {attempt + 1}), retrying...")
                        await asyncio.sleep(1)  # Brief delay before retry
                    
            except asyncio.TimeoutError:
                last_error = f"Agent execution timed out after {config.timeout_seconds}s"
                if attempt < config.retry_count:
                    self.logger.warning(f"Agent {agent.__class__.__name__} timed out (attempt {attempt + 1}), retrying...")
                    await asyncio.sleep(1)
            except Exception as e:
                last_error = str(e)
                if attempt < config.retry_count:
                    self.logger.warning(f"Agent {agent.__class__.__name__} error (attempt {attempt + 1}): {e}, retrying...")
                    await asyncio.sleep(1)
        
        # All retries failed
        return AgentOutput(
            success=False,
            data={},
            confidence=0.0,
            processing_time=0.0,
            agent_name=agent.__class__.__name__.lower().replace('agent', ''),
            error=last_error
        )
    
    def _calculate_overall_metrics(self, context: WorkflowContext) -> Dict[str, Any]:
        """Calculate overall workflow metrics"""
        successful_agents = [r for r in context.results.values() if r.success]
        failed_agents = [r for r in context.results.values() if not r.success]
        
        if not successful_agents:
            return {
                "overall_score": 0.0,
                "confidence": 0.0,
                "success_rate": 0.0,
                "total_agents": len(context.results),
                "successful_agents": 0,
                "failed_agents": len(failed_agents)
            }
        
        # Calculate weighted average score
        total_weight = 0
        weighted_score = 0
        weighted_confidence = 0
        
        for result in successful_agents:
            # Weight by agent priority
            config = self.agent_configs[result.agent_name]
            weight = {
                AgentPriority.CRITICAL: 1.0,
                AgentPriority.HIGH: 0.8,
                AgentPriority.MEDIUM: 0.6,
                AgentPriority.LOW: 0.4
            }.get(config.priority, 0.5)
            
            # Extract score from agent data
            agent_score = self._extract_agent_score(result.data)
            
            weighted_score += agent_score * weight
            weighted_confidence += result.confidence * weight
            total_weight += weight
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        overall_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
        
        return {
            "overall_score": overall_score,
            "confidence": overall_confidence,
            "success_rate": len(successful_agents) / len(context.results),
            "total_agents": len(context.results),
            "successful_agents": len(successful_agents),
            "failed_agents": len(failed_agents),
            "execution_time": (datetime.now() - context.start_time).total_seconds(),
            "stage_times": {k: v for k, v in context.execution_metadata.items() if k.endswith('_time')}
        }
    
    def _extract_agent_score(self, agent_data: Dict[str, Any]) -> float:
        """Extract score from agent data"""
        # Try different score field names
        score_fields = ['score', 'overall_score', 'confidence', 'rating']
        
        for field in score_fields:
            if field in agent_data:
                score = agent_data[field]
                if isinstance(score, (int, float)):
                    return float(score)
        
        # If no direct score, try to calculate from sub-scores
        if 'scores' in agent_data and isinstance(agent_data['scores'], dict):
            scores = list(agent_data['scores'].values())
            numeric_scores = [s for s in scores if isinstance(s, (int, float))]
            if numeric_scores:
                return sum(numeric_scores) / len(numeric_scores)
        
        # Default score
        return 0.5
    
    async def execute_partial_workflow(self, 
                                     media_id: str,
                                     agent_names: List[str],
                                     user_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute a partial workflow with selected agents"""
        return await self.execute_workflow(
            media_id=media_id,
            user_id=user_id,
            analysis_type="partial",
            selected_agents=agent_names
        )
    
    def get_agent_dependencies(self, agent_name: str) -> List[str]:
        """Get dependencies for a specific agent"""
        config = self.agent_configs.get(agent_name)
        return config.dependencies if config else []
    
    def get_workflow_graph(self) -> Dict[str, Any]:
        """Get the workflow dependency graph"""
        graph = {
            "agents": {},
            "stages": {},
            "dependencies": {}
        }
        
        for agent_name, config in self.agent_configs.items():
            graph["agents"][agent_name] = {
                "priority": config.priority.value,
                "stage": config.stage.value,
                "parallel_group": config.parallel_group,
                "timeout": config.timeout_seconds
            }
            
            graph["dependencies"][agent_name] = config.dependencies
        
        # Group by stages
        for stage in WorkflowStage:
            stage_agents = [
                name for name, config in self.agent_configs.items()
                if config.stage == stage
            ]
            graph["stages"][stage.value] = stage_agents
        
        return graph
    
    def validate_workflow(self) -> Dict[str, Any]:
        """Validate the workflow configuration"""
        issues = []
        warnings = []
        
        # Check for circular dependencies
        for agent_name in self.agent_configs:
            if self._has_circular_dependency(agent_name, set()):
                issues.append(f"Circular dependency detected for agent: {agent_name}")
        
        # Check if all dependencies exist
        for agent_name, config in self.agent_configs.items():
            for dependency in config.dependencies:
                if dependency not in self.agent_configs:
                    issues.append(f"Agent {agent_name} depends on non-existent agent: {dependency}")
        
        # Check for agents without dependencies in later stages
        for agent_name, config in self.agent_configs.items():
            if config.stage != WorkflowStage.INITIALIZATION and not config.dependencies:
                warnings.append(f"Agent {agent_name} in stage {config.stage.value} has no dependencies")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "total_agents": len(self.agent_configs),
            "stages": len(WorkflowStage)
        }
    
    def _has_circular_dependency(self, agent_name: str, visited: set) -> bool:
        """Check for circular dependencies"""
        if agent_name in visited:
            return True
        
        visited.add(agent_name)
        config = self.agent_configs.get(agent_name)
        
        if config:
            for dependency in config.dependencies:
                if self._has_circular_dependency(dependency, visited.copy()):
                    return True
        
        return False
    
    def get_execution_plan(self, selected_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get the execution plan for the workflow"""
        agents_to_run = selected_agents or list(self.agent_configs.keys())
        
        plan = {
            "stages": [],
            "total_agents": len(agents_to_run),
            "estimated_time": 0
        }
        
        for stage in WorkflowStage:
            stage_agents = [
                name for name in agents_to_run
                if self.agent_configs[name].stage == stage
            ]
            
            if stage_agents:
                # Group by parallel execution
                parallel_groups = self._group_agents_for_parallel_execution(stage_agents)
                
                stage_info = {
                    "stage": stage.value,
                    "agents": stage_agents,
                    "parallel_groups": parallel_groups,
                    "estimated_time": max(
                        self.agent_configs[agent].timeout_seconds
                        for agent in stage_agents
                    )
                }
                
                plan["stages"].append(stage_info)
                plan["estimated_time"] += stage_info["estimated_time"]
        
        return plan

# Global orchestrator instance
orchestrator = LangGraphOrchestrator()