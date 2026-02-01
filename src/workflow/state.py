"""
State Definition for PSC_Agents Workflow.

Defines the shared context (AgentState) that flows between all agents
in the multi-agent research system.

Author: PSC_Agents Team
"""

import operator
from typing import TypedDict, Annotated, List, Dict, Any, Union, Optional


# =============================================================================
# Agent State: Central Shared Memory
# =============================================================================

class AgentState(TypedDict):
    """
    Shared state dictionary for all agents in the workflow.
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         STATE FLOW DIAGRAM                              │
    │                                                                         │
    │   User Goal ──► MetaAgent ──► DataAgent ──► DesignAgent ──► FabAgent   │
    │                    │              │              │              │       │
    │                    ▼              ▼              ▼              ▼       │
    │                  plan       data_context   exp_params     fab_results  │
    │                    │              │              │              │       │
    │                    └──────────────┴──────────────┴──────────────┘       │
    │                                      │                                  │
    │                                      ▼                                  │
    │                               AnalysisAgent                             │
    │                                      │                                  │
    │                                      ▼                                  │
    │                              analysis_report                            │
    │                                      │                                  │
    │                                      ▼                                  │
    │                               MemoryAgent                               │
    │                                      │                                  │
    │                                      ▼                                  │
    │                               memory_log ──► Next Iteration             │
    └─────────────────────────────────────────────────────────────────────────┘
    
    AGENT INPUT/OUTPUT CONTRACT:
    
    [MetaAgent]
      Reads:  goal, memory_log, analysis_report (from previous iteration)
      Writes: plan (JSON with agent_tasks, hypothesis, constraints)
    
    [DataAgent]
      Reads:  goal, plan (agent_tasks.DataAgent)
      Writes: data_context (JSON with literature findings, extracted data)
    
    [DesignAgent]
      Reads:  goal, plan, data_context
      Writes: experimental_params (JSON with composition, process, precursors)
    
    [FabAgent]
      Reads:  goal, plan, data_context, experimental_params
      Writes: fab_results (JSON with predicted metrics, analysis)
    
    [AnalysisAgent]
      Reads:  goal, plan, data_context, experimental_params, fab_results
      Writes: analysis_report (JSON with diagnosis, gap analysis, suggestions)
    
    [MemoryAgent]
      Reads:  ALL fields from current iteration
      Writes: memory_log (append knowledge capsule)
    """

    # =========================================================================
    # Global Context (Set by User, Read by All)
    # =========================================================================
    goal: str  # User's research objective
    
    # =========================================================================
    # MetaAgent Output
    # =========================================================================
    # JSON format:
    # {
    #   "analysis": "previous iteration analysis",
    #   "hypothesis": "scientific hypothesis for this iteration",
    #   "strategy": "continue/pivot/refine",
    #   "constraints": ["constraint1", "constraint2"],
    #   "agent_tasks": {
    #     "DataAgent": "specific task or SKIP",
    #     "DesignAgent": "specific task",
    #     "FabAgent": "specific task",
    #     "AnalysisAgent": "specific task"
    #   },
    #   "success_criteria": "how to judge if goal is met"
    # }
    plan: Optional[Union[str, Dict[str, Any]]]
    
    # =========================================================================
    # DataAgent Output
    # =========================================================================
    # JSON format:
    # {
    #   "status": "success/partial/failed",
    #   "papers_processed": 3,
    #   "key_findings": {
    #     "performance_benchmarks": [...],
    #     "synthesis_methods": [...],
    #     "material_compositions": [...]
    #   },
    #   "extracted_data": [...],
    #   "summary": "brief summary of literature findings"
    # }
    data_context: Optional[str]
    
    # =========================================================================
    # DesignAgent Output
    # =========================================================================
    # JSON format:
    # {
    #   "task_type": "material_design/synthesizability/full_recipe",
    #   "composition": {
    #     "formula": "FA0.9Cs0.1PbI3",
    #     "structure_type": "3D",
    #     "synthesizability": {"result": true, "confidence": 0.85}
    #   },
    #   "process": {
    #     "method": "solution",
    #     "precursors": [...],
    #     "solvents": {...},
    #     "parameters": {...}
    #   },
    #   "candidate_screening": {
    #     "total_generated": 5,
    #     "total_passed_filter": 3,
    #     "selected_formula": "...",
    #     "selection_reasoning": "..."
    #   },
    #   "analysis": "scientific rationale",
    #   "status": "success/partial/failed"
    # }
    experimental_params: Optional[Dict[str, Any]]
    
    # =========================================================================
    # FabAgent Output
    # =========================================================================
    # JSON format:
    # {
    #   "composition": "FA0.9Cs0.1PbI3",
    #   "predicted_metrics": {
    #     "PCE_percent": 22.5,
    #     "Voc_V": 1.12,
    #     "Jsc_mA_cm2": 25.3,
    #     "FF_percent": 79.5,
    #     "BandGap_eV": 1.52,
    #     "E_hull_eV": 0.02
    #   },
    #   "prediction_confidence": {...},
    #   "analysis": "scientific interpretation",
    #   "visualizations": ["path/to/chart1.png"],
    #   "status": "success/failed"
    # }
    fab_results: Optional[Union[str, Dict[str, Any]]]
    
    # =========================================================================
    # AnalysisAgent Output
    # =========================================================================
    # JSON format:
    # {
    #   "is_goal_met": false,
    #   "performance_gap": {
    #     "target_PCE": 25.0,
    #     "achieved_PCE": 22.5,
    #     "shortfall_percent": 10.0
    #   },
    #   "scientific_diagnosis": {
    #     "root_cause": "Voc loss due to non-radiative recombination",
    #     "contributing_factors": ["grain boundary defects", "interface traps"],
    #     "structural_insights": "..."
    #   },
    #   "iteration_feedback": {
    #     "status": "PROCEED/REVISE/ABORT",
    #     "suggested_adjustment": "Add KI passivation to reduce defect density",
    #     "priority_changes": ["increase annealing time", "add surface treatment"]
    #   },
    #   "status": "success"
    # }
    analysis_report: Optional[str]
    
    # =========================================================================
    # Long-term Memory (Append-only, managed by MemoryAgent)
    # =========================================================================
    # Each entry is a JSON string:
    # {
    #   "iteration_id": 1,
    #   "knowledge_triplet": {
    #     "formula": "FA0.9Cs0.1PbI3",
    #     "predicted_pce": 22.5,
    #     "verdict": "PARTIAL",
    #     "reason": "Voc lower than expected"
    #   },
    #   "critical_learning": "...",
    #   "meta_agent_feedback": "..."
    # }
    memory_log: Annotated[List[str], operator.add]
    
    # =========================================================================
    # MemoryAgent Output (Structured)
    # =========================================================================
    # List of structured dict records for programmatic access:
    # [
    #   {
    #     "iteration": 0,
    #     "formula": "CsPbI3",
    #     "method": "Solution processing",
    #     "synthesis_protocol": "Full protocol text...",
    #     "precursors": "CsI, PbI2",
    #     "pce": "18.5%",
    #     "verdict": "SUCCESS",
    #     "aligned_with_goal": true,
    #     "learning": "...",
    #     "advice": "..."
    #   },
    #   ...
    # ]
    structured_memory: Annotated[List[Dict[str, Any]], operator.add]
    
    # =========================================================================
    # Final Conclusion (MetaAgent output when finished)
    # =========================================================================
    # Comprehensive research conclusion generated when workflow finishes.
    # Includes: goal summary, best solution, synthesis protocol, recommendations.
    final_conclusion: Optional[str]
    
    # =========================================================================
    # MetaAgent History (accumulated across iterations)
    # =========================================================================
    # List of MetaAgent outputs for each iteration, for tracking full reasoning process.
    # Each entry: {"iteration": N, "response": "full response text", "plan": {...}}
    meta_agent_history: Annotated[List[Dict[str, Any]], operator.add]
    
    # =========================================================================
    # Control Flow
    # =========================================================================
    current_iteration: int
    is_finished: bool


# =============================================================================
# State Factory
# =============================================================================

def create_initial_state(goal: str) -> AgentState:
    """
    Create an initial state with default values.

    Args:
        goal: The user's research goal or objective.

    Returns:
        AgentState with initialized default values.
    """
    return {
        "goal": goal,
        "plan": None,             
        "data_context": None,
        "experimental_params": None,
        "fab_results": None,
        "analysis_report": None,
        "memory_log": [],
        "structured_memory": [],   # Structured memory for MetaAgent        
        "final_conclusion": None,  # Final conclusion when workflow finishes
        "meta_agent_history": [],  # Full history of MetaAgent outputs per iteration
        "current_iteration": 0,
        "is_finished": False,
    }


# =============================================================================
# State Utilities
# =============================================================================

def get_upstream_context(state: AgentState, agent_name: str) -> dict[str, Any]:
    """
    Get relevant upstream context for a specific agent.
    
    This utility helps agents access their required inputs consistently.
    
    Args:
        state: Current workflow state
        agent_name: Name of the agent requesting context
        
    Returns:
        Dict with relevant upstream outputs
    """
    base_context = {
        "goal": state.get("goal", ""),
        "plan": state.get("plan"),
        "current_iteration": state.get("current_iteration", 0),
    }
    
    if agent_name == "MetaAgent":
        return {
            **base_context,
            "memory_log": state.get("memory_log", []),
            "structured_memory": state.get("structured_memory", []),  # Structured memory for insights
            "analysis_report": state.get("analysis_report"),
        }
    
    elif agent_name == "DataAgent":
        return base_context
    
    elif agent_name == "DesignAgent":
        return {
            **base_context,
            "data_context": state.get("data_context"),
        }
    
    elif agent_name == "FabAgent":
        return {
            **base_context,
            "data_context": state.get("data_context"),
            "experimental_params": state.get("experimental_params"),
        }
    
    elif agent_name == "AnalysisAgent":
        return {
            **base_context,
            "data_context": state.get("data_context"),
            "experimental_params": state.get("experimental_params"),
            "fab_results": state.get("fab_results"),
        }
    
    elif agent_name == "MemoryAgent":
        # Memory agent needs everything
        return {
            **base_context,
            "data_context": state.get("data_context"),
            "experimental_params": state.get("experimental_params"),
            "fab_results": state.get("fab_results"),
            "analysis_report": state.get("analysis_report"),
            "memory_log": state.get("memory_log", []),
        }
    
    return base_context


def format_context_summary(state: AgentState) -> str:
    """
    Format a human-readable summary of current state for debugging.
    
    Returns:
        Formatted string showing state contents
    """
    lines = [
        "┌─────────────── State Summary ───────────────┐",
        f"│ Goal: {state.get('goal', 'N/A')[:40]}...",
        f"│ Iteration: {state.get('current_iteration', 0)}",
        f"│ Finished: {state.get('is_finished', False)}",
        "├─────────────── Agent Outputs ───────────────┤",
    ]
    
    # Plan
    plan = state.get("plan")
    if plan:
        lines.append(f"│ 📋 Plan: {len(str(plan))} chars")
    else:
        lines.append("│ 📋 Plan: None")
    
    # Data context
    data = state.get("data_context")
    if data:
        lines.append(f"│ 📚 Data: {len(str(data))} chars")
    else:
        lines.append("│ 📚 Data: None")
    
    # Experimental params
    params = state.get("experimental_params")
    if params:
        formula = params.get("composition", {}).get("formula", "N/A")
        lines.append(f"│ 🧪 Design: {formula}")
    else:
        lines.append("│ 🧪 Design: None")
    
    # Fab results
    fab = state.get("fab_results")
    if fab and isinstance(fab, dict):
        metrics = fab.get("predicted_metrics", {})
        pce = metrics.get("PCE_percent", "N/A")
        lines.append(f"│ 🏭 Fab: PCE={pce}")
    else:
        lines.append("│ 🏭 Fab: None")
    
    # Analysis
    analysis = state.get("analysis_report")
    if analysis:
        lines.append(f"│ 📊 Analysis: {len(str(analysis))} chars")
    else:
        lines.append("│ 📊 Analysis: None")
    
    # Memory (text log)
    memory = state.get("memory_log", [])
    lines.append(f"│ 💾 Memory Log: {len(memory)} entries")
    
    # Structured Memory
    structured = state.get("structured_memory", [])
    lines.append(f"│ 📦 Structured Memory: {len(structured)} records")
    
    lines.append("└─────────────────────────────────────────────┘")
    
    return "\n".join(lines)