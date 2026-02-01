"""
Graph Construction for PSC_Agents Workflow.

Builds the LangGraph workflow connecting all agents in a cyclic
research automation pipeline.

Author: PSC_Agents Team
"""

import sys
from pathlib import Path
from typing import Any, Literal, Dict

# -------------------------------------------------------------------------
# Path Setup (Consider using PYTHONPATH in production instead of this hack)
# -------------------------------------------------------------------------
sys.path.append("src")

from langgraph.graph import StateGraph, END

# Import shared state definition
from workflow.state import AgentState, create_initial_state

# Import Core Definitions
from core.config import Settings, MCPConfig

# Import Agents
from agent.meta_agent import MetaAgent
from agent.data_agent import DataAgent
from agent.design_agent import DesignAgent
from agent.fab_agent import FabAgent
from agent.analysis_agent import AnalysisAgent
from agent.memory_agent import MemoryAgent


# Default maximum iterations to prevent infinite loops
DEFAULT_MAX_ITERATIONS = 10


def check_termination(
    state: AgentState,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> Literal["continue", "end"]:
    """
    Conditional logic to determine workflow termination.
    """
    # 1. Goal Achieved Check
    if state.get("is_finished", False):
        return "end"

    # 2. Safety Limit Check
    if state.get("current_iteration", 0) >= max_iterations:
        return "end"

    return "continue"


def build_agent_settings(
    base_settings: Settings | None,
    mcp_servers: Dict[str, Any] | None,
) -> Settings:
    """
    Helper to merge global settings with agent-specific MCP configs.
    Creates a new Settings instance to ensure isolation.
    """
    # Create MCP Config from the dictionary
    mcp_config = MCPConfig.from_dict(mcp_servers) if mcp_servers else MCPConfig()

    if base_settings:
        # Clone LLM/Project settings, inject new MCP config
        return Settings(
            llm=base_settings.llm,
            mcp=mcp_config,
            project=base_settings.project,
        )
    else:
        return Settings(mcp=mcp_config)


class ResearchWorkflow:
    """
    Orchestrator for the multi-agent research workflow.
    
    Acts as the Dependency Injection container: it takes the raw configuration
    and injects it into the appropriate agents during initialization.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        mcp_configs: Dict[str, Dict[str, Any]] | None = None,
        design_tool_mode: str = "mock",
        papers_dir: str | None = None,
    ) -> None:
        self.settings = settings
        self.max_iterations = max_iterations
        # 允许 mcp_configs 为空，保证鲁棒性
        self.mcp_configs = mcp_configs or {}
        # DesignAgent 工具模式: "mock" LLM辅助生成(全自动), "interactive" 与服务器交互(手动)
        self.design_tool_mode = design_tool_mode
        # DataAgent 论文保存目录
        self.papers_dir = papers_dir
        
        self.agents: Dict[str, Any] = {}
        self.graph = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize agents and compile the graph."""
        if self._initialized:
            return

        # --- 1. Agent Instantiation (Dependency Injection) ---
        # We explicitly map config sections to agents here.
        self.agents = {
            "meta": MetaAgent(
                settings=build_agent_settings(
                    self.settings, self.mcp_configs.get("meta")
                )
            ),
            "data": DataAgent(
                settings=build_agent_settings(
                    self.settings, self.mcp_configs.get("data")
                ),
                local_papers_dir=self.papers_dir,
            ),
            "design": DesignAgent(
                settings=build_agent_settings(
                    self.settings, self.mcp_configs.get("design")
                ),
                tool_mode=self.design_tool_mode,
            ),
            "fab": FabAgent(
                settings=build_agent_settings(
                    self.settings, self.mcp_configs.get("fab")
                )
            ),
            "analysis": AnalysisAgent(
                settings=build_agent_settings(
                    self.settings, self.mcp_configs.get("analysis")
                )
            ),
            "memory": MemoryAgent(
                settings=build_agent_settings(
                    self.settings, self.mcp_configs.get("memory")
                )
            ),
        }

        # --- 2. Async Connection (MCP Handshake) ---
        print("🔌 Connecting agents to tool servers...")
        for name, agent in self.agents.items():
            await agent._initialize()

        # --- 3. Graph Compilation ---
        self.graph = self._build_graph()
        self._initialized = True
        print("✅ Workflow Graph built successfully.")

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph topology."""
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("meta", self.agents["meta"].run)
        workflow.add_node("data", self.agents["data"].run)
        workflow.add_node("design", self.agents["design"].run)
        workflow.add_node("fab", self.agents["fab"].run)
        workflow.add_node("analysis", self.agents["analysis"].run)
        workflow.add_node("memory", self.agents["memory"].run)

        # Edges
        workflow.set_entry_point("meta")

        # Conditional Edge (Meta -> [End, Data])
        workflow.add_conditional_edges(
            "meta",
            lambda state: check_termination(state, self.max_iterations),
            {
                "continue": "data",
                "end": END,
            },
        )

        # Sequential Loop
        workflow.add_edge("data", "design")
        workflow.add_edge("design", "fab")
        workflow.add_edge("fab", "analysis")
        workflow.add_edge("analysis", "memory")
        workflow.add_edge("memory", "meta")

        return workflow.compile()

    async def run(self, goal: str) -> AgentState:
        """Run the workflow for a specific research goal."""
        if not self._initialized:
            await self.initialize()

        # Use shared factory for consistent state initialization
        initial_state = create_initial_state(goal)

        # Run Graph
        # Note: ainvoke input is just the dict, output is the final state dict
        final_state = await self.graph.ainvoke(initial_state)
        
        # === CRITICAL: Ensure final conclusion is always generated ===
        # Whether finished normally or hit max iterations, we need a conclusion
        if not final_state.get("final_conclusion"):
            print(f"\n{'='*60}")
            print(f"🏁 Workflow ended without conclusion. Generating final summary...")
            print(f"{'='*60}")
            
            # Generate conclusion using MetaAgent
            conclusion = await self._generate_final_conclusion(final_state)
            final_state["final_conclusion"] = conclusion
        
        return final_state

    async def _generate_final_conclusion(self, state: dict) -> str:
        """
        Generate a final conclusion when the workflow ends without one.
        This happens when max_iterations is reached.
        """
        meta_agent = self.agents.get("meta")
        if not meta_agent:
            return "Failed to generate conclusion: MetaAgent not available."
        
        # Call MetaAgent's conclusion generation method
        try:
            conclusion = await meta_agent._generate_final_conclusion(
                goal=state.get("goal", ""),
                memory_log=state.get("memory_log", []),
                structured_memory=state.get("structured_memory", []),
                current_iteration=state.get("current_iteration", 0),
            )
            return conclusion
        except Exception as e:
            return f"Failed to generate conclusion: {e}"

    async def shutdown(self) -> None:
        """Gracefully close all MCP connections."""
        print("🛑 Shutting down agents...")
        for agent in self.agents.values():
            await agent._shutdown()
        self._initialized = False


def build_research_graph(
    settings: Settings | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    mcp_configs: Dict[str, Dict[str, Any]] | None = None,
    design_tool_mode: str = "mock",
) -> ResearchWorkflow:
    """Factory function for easy instantiation."""
    return ResearchWorkflow(
        settings=settings,
        max_iterations=max_iterations,
        mcp_configs=mcp_configs,
        design_tool_mode=design_tool_mode,
    )