#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ablation_experiment.py
PeroMAS 消融实验脚本

实验目标：
  逐个移除功能性 Agent，验证每个 Agent 对整体工作流的贡献。
  MetaAgent 和 MemoryAgent 始终保留，分别负责规划和记忆。

消融方案（共 4 种，FULL 对照组在 workflow test 中完成）：
  NO_DATA       - 移除 DataAgent     （无文献检索）
  NO_DESIGN     - 移除 DesignAgent   （无材料设计）
  NO_FAB        - 移除 FabAgent      （无性能预测）
  NO_ANALYSIS   - 移除 AnalysisAgent （无差距分析）

移除策略：
  被移除的 Agent 节点替换为一个 pass-through 函数，该函数：
  1. 不调用任何 LLM 或工具
  2. 将其对应的状态字段设为一个明确的占位值（标注 "ABLATED"）
  3. 后续 Agent 读到 ABLATED 占位值后正常运行，只是缺少该信息

用法：
    cd d:\\PeroMAS\\PeroMAS
    python experiment/ablation/ablation_experiment.py                          # 运行全部 5 种
    python experiment/ablation/ablation_experiment.py --mode NO_DATA           # 只跑一种
    python experiment/ablation/ablation_experiment.py --mode NO_DATA,NO_FAB    # 跑指定几种
    python experiment/ablation/ablation_experiment.py --list                   # 列出所有模式
    python experiment/ablation/ablation_experiment.py --iterations 2           # 设置迭代数

Author: PSC_Agents Team
"""

import os
import sys
import json
import csv
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any

# =============================================================================
# 路径配置
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "core"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from core.config import Settings, LLMConfig, MCPConfig, MCPServerConfig, ProviderConfig
from core.llm import LLMClient
from workflow.state import AgentState, create_initial_state

# Agent imports
from agent.meta_agent import MetaAgent
from agent.data_agent import DataAgent
from agent.design_agent import DesignAgent
from agent.fab_agent import FabAgent
from agent.analysis_agent import AnalysisAgent
from agent.memory_agent import MemoryAgent

from langgraph.graph import StateGraph, END

# =============================================================================
# 配置
# =============================================================================
DEFAULT_MAX_ITERATIONS = 3
OUTPUT_DIR = Path(__file__).parent / "ablation_output"
PAPERS_DIR = Path(__file__).parent / "papers"

ARXIV_MCP_SERVER_URL = "https://seuyishu-arxiv-mcp-server.hf.space/sse"

# =============================================================================
# 消融实验 Query 列表（从 Query.md 的 20 个中精选 10 个，每组选最典型的）
# 选择原则：覆盖全部 6 个研究方向，侧重对 4 个 Agent 依赖度各不同的任务
# =============================================================================
ABLATION_QUERIES = [
    # Group 1: Phase Stability — B-site 掺杂（重度依赖文献 + 机理分析）
    {
        "id": "Q001",
        "type": "phase_stability",
        "query": "We are trying to synthesize all-inorganic CsPbI3, but the black phase is extremely unstable in air. I want to try B-site doping with small divalent metals (like Zn2+ or Mn2+) to stabilize the lattice.\n\nFirst, search the literature for the most effective B-site dopants for CsPbI3 in the last two years.\nBased on this, design a specific CsPb(1-x)MxI3 recipe and outline the annealing protocol.\nPredict if the alpha-phase formation energy is lowered enough to be stable at room temperature, and if the PCE exceeds 18%.\nFinally, analyze the mechanism: does the dopant stabilize the structure by relaxing lattice strain (Goldschmidt tolerance factor) or by increasing bond strength?",
    },
    # Group 1: Phase Stability — 宽带隙卤化物相分离（重度依赖 FabAgent 预测）
    {
        "id": "Q003",
        "type": "phase_stability",
        "query": "For our tandem top cell (1.75 eV), the mixed halide (I/Br) perovskite suffers from phase segregation under illumination.\n\nSearch for 'triple cation' (Cs/MA/FA) strategies specifically reported to suppress this segregation.\nDesign a wide-bandgap recipe utilizing these cations.\nPredict the bandgap stability under continuous 1-sun illumination and the steady-state Voc.\nAnalyze the root cause: use your tools to determine if the suppression comes from immobilizing halide ions or modifying the crystal lattice stiffness.",
    },
    # Group 2: Passivation — 2D/3D 界面（重度依赖 DesignAgent 工艺）
    {
        "id": "Q004",
        "type": "passivation",
        "query": "Humidity stability is our main bottleneck. I want to build a 2D/3D heterojunction using hydrophobic large organic cations.\n\nSearch for fluorinated organic ammonium salts (like F-PEA) used for surface passivation recently.\nDesign a process to spin-coat this 2D layer on top of a FAPbI3 bulk film.\nPredict the T80 lifetime under 60% relative humidity and any change in Series Resistance (Rs).\nAnalyze the interface electronics: does the 2D layer create a transport barrier for holes, or does it effectively block moisture ingress?",
    },
    # Group 2: Passivation — 埋底界面修饰（重度依赖 AnalysisAgent 诊断）
    {
        "id": "Q005",
        "type": "passivation",
        "query": "We suspect the interface between the SnO2 ETL and the perovskite has high defect density. I need a molecular bridge to modify this buried interface.\n\nSearch for molecules with bifunctional groups (e.g., carboxyl + amine) that bind to both SnO2 and Pb.\nDesign a modification protocol for the SnO2 substrate before perovskite deposition.\nPredict the improvement in Open-Circuit Voltage (Voc) and Fill Factor.\nAnalyze the chemical mechanism: confirm if the molecule creates a dipole that shifts the work function to better align the energy levels.",
    },
    # Group 3: Eco-friendly — Sn-Pb 混合（全链路均衡依赖）
    {
        "id": "Q007",
        "type": "eco_friendly",
        "query": "We are developing a bottom cell with 1.25 eV bandgap using Sn-Pb perovskite, but Sn oxidation is killing the efficiency.\n\nSearch for novel 'scavenger' additives (like metallic Sn powder or specific reductants) used in high-efficiency Sn-Pb cells.\nDesign a mixed Sn-Pb recipe incorporating the most promising antioxidant strategy.\nPredict the Jsc and the stability in ambient air (T50).\nAnalyze the mechanism: does the additive actively reduce Sn4+ back to Sn2+, or does it form a protective shell around the grains?",
    },
    # Group 3: Eco-friendly — 绿色溶剂（重度依赖 DataAgent 文献）
    {
        "id": "Q009",
        "type": "eco_friendly",
        "query": "Toxicity of DMF/DMSO is a major issue for industrialization. We need a green solvent system for processing MAPbI3.\n\nSearch for solvent systems based on TEP (Triethyl phosphate) or other non-toxic alternatives.\nDesign a fully green ink formulation and the corresponding quenching method.\nPredict the film morphology (roughness) and the final PCE compared to the DMF control.\nAnalyze the colloid chemistry: compare the coordination number (Gutmann Donor Number) of your green solvent vs DMSO to explain the crystallization kinetics.",
    },
    # Group 4: Processing — 碳电极 HTL-free（重度依赖 DesignAgent + FabAgent）
    {
        "id": "Q011",
        "type": "processing",
        "query": "To cut costs, we want to make HTL-free devices using carbon electrodes. The solvent in the carbon paste often destroys the perovskite.\n\nSearch for recent 'solvent-proof' perovskite compositions or protective interlayers for carbon-based PSCs.\nDesign a robust device stack (FTO/ETL/Perovskite/Carbon).\nPredict the stability (T80) and the efficiency potential.\nAnalyze the charge extraction: calculate the energy barrier for hole transfer directly from Perovskite to Carbon without an HTL.",
    },
    # Group 5: Frontier — 自修复（重度依赖 AnalysisAgent 热力学分析）
    {
        "id": "Q013",
        "type": "frontier",
        "query": "I heard some perovskites can self-heal after moisture degradation. I want to exploit this for long-life solar cells.\n\nSearch for additives (like dynamic polymers or methylamine-complexing agents) that promote self-healing/recrystallization.\nDesign a material system with this self-healing capability clearly defined.\nPredict the PCE recovery percentage after a degradation-healing cycle.\nAnalyze the chemical reversibility: explain the thermodynamics of the hydration and dehydration reaction involved.",
    },
    # Group 5: Frontier — 量子点配体交换（重度依赖 DataAgent + FabAgent）
    {
        "id": "Q016",
        "type": "frontier",
        "query": "We want to use CsPbI3 quantum dots (QDs) for a solar cell, but ligand exchange is tricky.\n\nSearch for short-chain ligands aimed at replacing oleic acid to improve conductivity.\nDesign a layer-by-layer deposition process with a specific ligand exchange solvent.\nPredict the carrier mobility and the final PCE.\nAnalyze the surface defects: how does the new ligand passivation reduce the trap density on the QD surface?",
    },
    # Group 6: Special Environment — 沙漠热循环（全链路均衡 + 机械分析）
    {
        "id": "Q019",
        "type": "special_environment",
        "query": "Our modules will be deployed in desert environments with extreme day/night temperature swings (thermal cycling from -10C to 85C).\n\n1. Search for 'elastic' grain boundary additives (e.g., cross-linkable polymers or elastomers) that can buffer thermal stress.\n2. Design a grain boundary modification strategy to prevent crack propagation during cooling.\n3. Predict the T80 lifetime under IEC 61215 thermal cycling standards.\n4. Analyze the mechanical failure: Discuss the role of Coefficient of Thermal Expansion (CTE) mismatch between the perovskite and the transport layers.",
    },
    {
        "id": "Q021",
        "type": "special_environment",
        "query": """We need to commercialize a perovskite solar cell that simultaneously achieves PCE > 20%, T80 > 1000h, and minimizes lead content — the classic efficiency-stability-toxicity trilemma.

    1. Search for compositions or device architectures from the last 2–3 years that have made genuine progress on at least two of the three objectives simultaneously (not just incremental improvements on one axis).
    2. Design a portfolio of 5 perovskite compositions spanning the performance-toxicity space (low-Pb to fully lead-free). For each, state the stoichiometry and the key physical principle that justifies its inclusion.
    3. For the most promising candidate, outline a fabrication protocol focused on the critical decisions: solvent coordination chemistry, anti-solvent timing window, and passivation mechanism.
    4. Analyze the fundamental trade-offs: for each of the 5 compositions, identify the root-cause performance ceiling (e.g., Sn²⁺ oxidation kinetics, indirect bandgap penalty, phase segregation thermodynamics) and estimate the theoretical PCE and T80 limits.""",
    },
]

# 默认使用的 Query（可通过 --query-id 选择）
DEFAULT_QUERY_ID = "Q001"


# =============================================================================
# 消融模式定义（4 种消融，FULL 对照组在 workflow test 中完成）
# =============================================================================

ABLATION_MODES = {
    "NO_DATA": {
        "label": "移除 DataAgent",
        "remove": "data",
        "description": "No literature retrieval; DesignAgent works without data_context",
    },
    "NO_DESIGN": {
        "label": "移除 DesignAgent",
        "remove": "design",
        "description": "No material design; FabAgent receives no experimental_params",
    },
    "NO_FAB": {
        "label": "移除 FabAgent",
        "remove": "fab",
        "description": "No performance prediction; AnalysisAgent receives no fab_results",
    },
    "NO_ANALYSIS": {
        "label": "移除 AnalysisAgent",
        "remove": "analysis",
        "description": "No gap analysis; MemoryAgent receives no analysis_report",
    },
}


# =============================================================================
# Pass-through（占位）节点 —— 代替被移除的 Agent
# =============================================================================

async def ablated_data_agent(state: dict[str, Any]) -> dict[str, Any]:
    """DataAgent 被移除：不检索文献，返回占位值。"""
    print(f"\n  [ABLATED] DataAgent skipped — no literature retrieval")
    return {
        "data_context": json.dumps({
            "status": "ABLATED",
            "message": "DataAgent was removed in this ablation experiment.",
            "papers_analyzed": 0,
            "extracted_data": [],
        }),
    }


async def ablated_design_agent(state: dict[str, Any]) -> dict[str, Any]:
    """DesignAgent 被移除：不设计材料，返回占位值。"""
    print(f"\n  [ABLATED] DesignAgent skipped — no material design")
    return {
        "experimental_params": {
            "status": "ABLATED",
            "message": "DesignAgent was removed in this ablation experiment.",
            "composition": {"formula": "N/A"},
            "process": {},
        },
    }


async def ablated_fab_agent(state: dict[str, Any]) -> dict[str, Any]:
    """FabAgent 被移除：不做性能预测，返回占位值。"""
    print(f"\n  [ABLATED] FabAgent skipped — no performance prediction")
    return {
        "fab_results": {
            "status": "ABLATED",
            "message": "FabAgent was removed in this ablation experiment.",
            "composition": "N/A",
            "predicted_metrics": {},
        },
    }


async def ablated_analysis_agent(state: dict[str, Any]) -> dict[str, Any]:
    """AnalysisAgent 被移除：不做差距分析，返回占位值。"""
    print(f"\n  [ABLATED] AnalysisAgent skipped — no gap analysis")
    return {
        "analysis_report": json.dumps({
            "status": "ABLATED",
            "message": "AnalysisAgent was removed in this ablation experiment.",
            "is_goal_met": False,
        }),
    }


# =============================================================================
# 消融工作流构建
# =============================================================================

def get_settings() -> Settings:
    return Settings(llm=LLMConfig())


def get_mcp_configs() -> dict:
    return {
        "data": {
            "arxiv": {
                "command": "",
                "args": [],
                "url": ARXIV_MCP_SERVER_URL,
                "enabled": True,
            }
        },
    }


def build_agent_settings(base_settings: Settings, mcp_servers: dict | None) -> Settings:
    """从 graph.py 复制的辅助函数"""
    from core.config import MCPConfig as _MCPConfig
    mcp_config = _MCPConfig.from_dict(mcp_servers) if mcp_servers else _MCPConfig()
    return Settings(llm=base_settings.llm, mcp=mcp_config, project=base_settings.project)


class AblationWorkflow:
    """支持消融的工作流。"""

    def __init__(
        self,
        mode: str,
        settings: Settings | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ):
        self.mode = mode
        self.remove_target = ABLATION_MODES[mode]["remove"]
        self.settings = settings or get_settings()
        self.max_iterations = max_iterations
        self.mcp_configs = get_mcp_configs()
        self.agents: dict[str, Any] = {}
        self.graph = None
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return

        # --- 实例化所有 Agent（被移除的不实例化）---
        self.agents["meta"] = MetaAgent(
            settings=build_agent_settings(self.settings, self.mcp_configs.get("meta"))
        )
        self.agents["memory"] = MemoryAgent(
            settings=build_agent_settings(self.settings, self.mcp_configs.get("memory"))
        )

        if self.remove_target != "data":
            self.agents["data"] = DataAgent(
                settings=build_agent_settings(self.settings, self.mcp_configs.get("data")),
                local_papers_dir=str(PAPERS_DIR),
            )
        if self.remove_target != "design":
            self.agents["design"] = DesignAgent(
                settings=build_agent_settings(self.settings, self.mcp_configs.get("design")),
                tool_mode="mock",
            )
        if self.remove_target != "fab":
            self.agents["fab"] = FabAgent(
                settings=build_agent_settings(self.settings, self.mcp_configs.get("fab"))
            )
        if self.remove_target != "analysis":
            self.agents["analysis"] = AnalysisAgent(
                settings=build_agent_settings(self.settings, self.mcp_configs.get("analysis"))
            )

        # --- 初始化连接 ---
        print(f"  Connecting agents...")
        for name, agent in self.agents.items():
            await agent._initialize()

        # --- 构建图 ---
        self.graph = self._build_graph()
        self._initialized = True

    def _build_graph(self) -> Any:
        """构建 LangGraph 图，被移除的节点用 pass-through 替代。"""
        workflow = StateGraph(AgentState)

        # --- 注册节点 ---
        workflow.add_node("meta", self.agents["meta"].run)

        # Data
        if self.remove_target == "data":
            workflow.add_node("data", ablated_data_agent)
        else:
            workflow.add_node("data", self.agents["data"].run)

        # Design
        if self.remove_target == "design":
            workflow.add_node("design", ablated_design_agent)
        else:
            workflow.add_node("design", self.agents["design"].run)

        # Fab
        if self.remove_target == "fab":
            workflow.add_node("fab", ablated_fab_agent)
        else:
            workflow.add_node("fab", self.agents["fab"].run)

        # Analysis
        if self.remove_target == "analysis":
            workflow.add_node("analysis", ablated_analysis_agent)
        else:
            workflow.add_node("analysis", self.agents["analysis"].run)

        workflow.add_node("memory", self.agents["memory"].run)

        # --- 边（与原工作流完全相同的拓扑）---
        workflow.set_entry_point("meta")
        workflow.add_conditional_edges(
            "meta",
            lambda state: "end" if (
                state.get("is_finished", False)
                or state.get("current_iteration", 0) >= self.max_iterations
            ) else "continue",
            {"continue": "data", "end": END},
        )
        workflow.add_edge("data", "design")
        workflow.add_edge("design", "fab")
        workflow.add_edge("fab", "analysis")
        workflow.add_edge("analysis", "memory")
        workflow.add_edge("memory", "meta")

        return workflow.compile()

    async def run(self, goal: str) -> dict:
        if not self._initialized:
            await self.initialize()

        initial_state = create_initial_state(goal)
        final_state = await self.graph.ainvoke(initial_state)

        # 确保生成结论
        if not final_state.get("final_conclusion"):
            meta = self.agents.get("meta")
            if meta:
                try:
                    conclusion = await meta._generate_final_conclusion(
                        goal=final_state.get("goal", ""),
                        memory_log=final_state.get("memory_log", []),
                        structured_memory=final_state.get("structured_memory", []),
                        current_iteration=final_state.get("current_iteration", 0),
                    )
                    final_state["final_conclusion"] = conclusion
                except Exception as e:
                    final_state["final_conclusion"] = f"Failed: {e}"

        return final_state

    def collect_token_usage(self) -> dict[str, Any]:
        """聚合所有 Agent 的 LLM token 使用量。"""
        per_agent = {}
        total_input = 0
        total_output = 0
        total_calls = 0

        for name, agent in self.agents.items():
            llm = getattr(agent, "llm", None)
            if llm and hasattr(llm, "get_statistics"):
                stats = llm.get_statistics()
                inp = stats.get("total_input_tokens", 0)
                out = stats.get("total_output_tokens", 0)
                calls = stats.get("total_calls", 0)
                per_agent[name] = {
                    "input_tokens": inp,
                    "output_tokens": out,
                    "total_tokens": inp + out,
                    "llm_calls": calls,
                }
                total_input += inp
                total_output += out
                total_calls += calls

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_llm_calls": total_calls,
            "per_agent": per_agent,
        }

    async def shutdown(self):
        for agent in self.agents.values():
            await agent._shutdown()
        self._initialized = False


# =============================================================================
# 从 final_state 提取各 Agent 输出（与 psc_agent_workflow_test.py 一致）
# =============================================================================

def extract_agent_outputs(final_state: dict) -> dict:
    """从最终状态中提取各智能体的输出 — 与 workflow test 格式完全一致。"""
    outputs = {}

    # MetaAgent
    plan = final_state.get("plan", {})
    final_conclusion = final_state.get("final_conclusion", None)
    meta_history = final_state.get("meta_agent_history", [])
    outputs["MetaAgent"] = {
        "history": meta_history,
        "final_conclusion": final_conclusion if final_conclusion else "N/A",
        "has_conclusion": bool(final_conclusion),
    }
    if isinstance(plan, dict):
        outputs["MetaAgent"]["last_plan"] = plan
    else:
        outputs["MetaAgent"]["last_plan"] = {"raw": str(plan) if plan else "N/A"}

    # DataAgent
    data_context = final_state.get("data_context", "")
    data_agent_output = {
        "data_context": data_context if data_context else "No data collected",
        "papers_analyzed": 0,
        "paper_list": [],
    }
    if data_context:
        try:
            data_json = json.loads(data_context)
            extracted_data = data_json.get("extracted_data", [])
            data_agent_output["papers_analyzed"] = len(extracted_data)
            data_agent_output["paper_list"] = [
                {
                    "paper_id": p.get("arxiv_id", p.get("paper_id", "Unknown")),
                    "title": p.get("title", "Unknown"),
                    "key_findings": p.get("key_findings", []),
                    "performance_metrics": p.get("performance_metrics", {}),
                }
                for p in extracted_data
            ]
        except (json.JSONDecodeError, TypeError):
            pass
    outputs["DataAgent"] = data_agent_output

    # DesignAgent
    exp_params = final_state.get("experimental_params", {})
    if exp_params:
        composition = exp_params.get("composition", {})
        process = exp_params.get("process", {})
        outputs["DesignAgent"] = {
            "formula": composition.get("formula", "N/A"),
            "structure_type": composition.get("structure_type", "N/A"),
            "synthesizability": composition.get("synthesizability", {}),
            "method": process.get("method", "N/A"),
            "synthesis_protocol": process.get("synthesis_protocol", "N/A"),
            "precursors": process.get("precursors", []),
            "full_params": exp_params,
        }
    else:
        outputs["DesignAgent"] = {"status": "No design generated"}

    # FabAgent
    fab_results = final_state.get("fab_results") or {}
    if fab_results and isinstance(fab_results, dict):
        raw_metrics = fab_results.get("predicted_metrics") or fab_results.get("metrics")
        metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
        outputs["FabAgent"] = {
            "composition": fab_results.get("composition") or "N/A",
            "PCE_percent": metrics.get("PCE_percent", "N/A") if metrics else "N/A",
            "Voc_V": metrics.get("Voc_V", "N/A") if metrics else "N/A",
            "Jsc_mA_cm2": metrics.get("Jsc_mA_cm2", "N/A") if metrics else "N/A",
            "FF_percent": metrics.get("FF_percent", "N/A") if metrics else "N/A",
            "full_results": fab_results,
        }
    else:
        outputs["FabAgent"] = {"status": "No prediction available"}

    # AnalysisAgent
    analysis = final_state.get("analysis_report", "")
    outputs["AnalysisAgent"] = {
        "analysis_report": analysis if analysis else "No analysis available"
    }

    # MemoryAgent
    memory_log = final_state.get("memory_log", [])
    structured_memory = final_state.get("structured_memory", [])
    outputs["MemoryAgent"] = {
        "entries_count": len(memory_log),
        "memory_log": memory_log,
        "structured_memory": structured_memory,
    }

    return outputs


# =============================================================================
# LLM Judge（与 psc_agent_workflow_test.py 的 WorkflowJudge 一致）
# =============================================================================

class AblationJudge:
    """评估消融实验的输出质量 — 评分体系与 WorkflowJudge 完全一致。"""

    JUDGE_SYSTEM_PROMPT = """你是一个专业的钙钛矿太阳能电池研究方案评估专家。
你的任务是评估AI多智能体系统生成的研究方案质量。

## 重要背景
这可能是一个消融实验，其中某个Agent被移除。
- 如果某些信息因被移除的Agent而缺失，请基于实际输出内容公平评分。
- 被移除一个Agent的系统**预期会表现更差** — 请如实评判实际产出质量。

## 评分系统 (0-100分):

### 维度1: 科学合理性 (0-30分)
- 提出的材料组成是否符合钙钛矿化学原理？
- 合成方法是否可行？
- 预测的性能指标是否合理？

### 维度2: 目标达成度 (0-30分)
- 是否满足用户提出的效率/稳定性/毒性要求？
- 设计方案是否针对性解决了用户的核心问题？
- 最终结论是否明确回应了研究目标？

### 维度3: 方案完整性 (0-20分)
- 是否包含完整的材料组成设计？
- 是否提供了合成路线和前驱体信息？
- 是否有性能预测和分析？

### 维度4: 实用指导价值 (0-20分)
- 方案是否可以直接指导实验？
- 是否给出了具体的参数建议？
- 对潜在问题是否有预警和建议？

## 评分等级:
- 90-100: 优秀 - 方案科学完整，可直接指导实验
- 75-89: 良好 - 方案合理，需少量补充
- 60-74: 及格 - 基本可用，但有明显不足
- 40-59: 较差 - 方案不完整或有科学错误
- 0-39: 失败 - 无法使用或严重错误

请仅以JSON格式回复:
{
    "score": 0-100,
    "scientific_validity": 0-30,
    "goal_achievement": 0-30,
    "completeness": 0-20,
    "practical_value": 0-20,
    "reasoning": "详细评价说明",
    "key_strengths": ["优点1", "优点2"],
    "key_weaknesses": ["不足1", "不足2"]
}
"""

    def __init__(self):
        judge_config = LLMConfig(provider="openai", temperature=0.3, max_tokens=2000, timeout=120.0)
        judge_config.openai = ProviderConfig(
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("LLM_MODEL_ID", "gpt-4o"),
        )
        self.llm = LLMClient(judge_config)

    async def evaluate(
        self, goal: str, mode: str, agent_outputs: dict, final_state: dict,
    ) -> dict:
        """评估工作流结果质量 — Prompt 结构与 WorkflowJudge.evaluate_workflow_result 一致。"""
        info = ABLATION_MODES[mode]

        meta_output = agent_outputs.get("MetaAgent", {})
        data_output = agent_outputs.get("DataAgent", {})
        design_output = agent_outputs.get("DesignAgent", {})
        fab_output = agent_outputs.get("FabAgent", {})
        analysis_output = agent_outputs.get("AnalysisAgent", {})
        memory_output = agent_outputs.get("MemoryAgent", {})

        # === MetaAgent 完整历史 ===
        meta_history = meta_output.get("history", [])
        meta_history_text = ""
        for entry in meta_history:
            iter_num = entry.get("iteration", "?")
            response = entry.get("response", "N/A")
            if len(response) > 2000:
                response = response[:2000] + "...[截断]"
            meta_history_text += f"\n--- 第{iter_num}轮 ---\n{response}\n"

        # === 文献信息 ===
        papers_analyzed = data_output.get("papers_analyzed", 0)
        paper_list = data_output.get("paper_list", [])
        literature_text = f"分析论文数: {papers_analyzed}\n"
        for p in paper_list[:10]:
            literature_text += f"- [{p.get('paper_id', 'N/A')}] {p.get('title', 'Unknown')[:80]}\n"
            findings = p.get('key_findings', [])
            if findings:
                if isinstance(findings, list):
                    literature_text += f"  Key Findings: {'; '.join(str(f)[:100] for f in findings[:3])}\n"
                else:
                    literature_text += f"  Key Findings: {str(findings)[:200]}\n"

        # === Memory 迭代学习 ===
        structured_memory = memory_output.get("structured_memory", [])
        memory_summary = ""
        for m in structured_memory:
            iter_n = m.get("iteration", "?")
            formula = m.get("formula", "N/A")
            pce = m.get("pce", "N/A")
            verdict = m.get("verdict", "N/A")
            learning = m.get("learning", "N/A")
            lit_summary = m.get("literature_summary", "")
            memory_summary += f"- Iter{iter_n}: {formula}, PCE={pce}, {verdict}\n"
            memory_summary += f"  Learning: {str(learning)[:150]}...\n"
            if lit_summary:
                memory_summary += f"  {lit_summary[:300]}...\n"

        eval_prompt = f"""## 消融模式
{mode} — {info['label']}
{info['description']}

## 用户研究目标
{goal}

## 工作流概览
- 总迭代次数: {final_state.get("current_iteration", 0)}
- 是否完成: {final_state.get("is_finished", False)}

---

## 一、MetaAgent 完整迭代历史 (Chief Scientist 规划)
{meta_history_text if meta_history_text else "无历史记录"}

### MetaAgent 最终结论:
{meta_output.get("final_conclusion", "未生成最终结论")}

---

## 二、DataAgent 文献检索结果
{literature_text if papers_analyzed > 0 else "未收集到文献数据"}

---

## 三、DesignAgent 材料设计
{json.dumps(design_output, indent=2, ensure_ascii=False)}

---

## 四、FabAgent 性能预测
{json.dumps(fab_output, indent=2, ensure_ascii=False)}

---

## 五、AnalysisAgent 分析报告
{json.dumps(analysis_output, indent=2, ensure_ascii=False)[:2000]}

---

## 六、MemoryAgent 迭代学习记录
{memory_summary if memory_summary else "无学习记录"}

---

请全面评估这个**完整的多轮迭代研究过程**的质量，给出100分制评分。
评估要点：
1. 是否有文献支撑设计决策？
2. 各轮迭代是否有学习和改进？
3. 最终方案是否回应了用户目标？
"""
        try:
            response = await self.llm.ainvoke_simple(
                prompt=eval_prompt, system_message=self.JUDGE_SYSTEM_PROMPT,
            )
            return self._parse_response(response)
        except Exception as e:
            return {
                "score": 0, "scientific_validity": 0, "goal_achievement": 0,
                "completeness": 0, "practical_value": 0,
                "reasoning": f"Judge 错误: {e}",
                "key_strengths": [], "key_weaknesses": ["评估失败"],
            }

    def _parse_response(self, response: str) -> dict:
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            result = json.loads(json_str)
            return {
                "score": result.get("score", 0),
                "scientific_validity": result.get("scientific_validity", 0),
                "goal_achievement": result.get("goal_achievement", 0),
                "completeness": result.get("completeness", 0),
                "practical_value": result.get("practical_value", 0),
                "reasoning": result.get("reasoning", "无说明"),
                "key_strengths": result.get("key_strengths", []),
                "key_weaknesses": result.get("key_weaknesses", []),
            }
        except Exception as e:
            return {
                "score": 0, "scientific_validity": 0, "goal_achievement": 0,
                "completeness": 0, "practical_value": 0,
                "reasoning": f"解析错误: {e}",
                "key_strengths": [], "key_weaknesses": ["解析失败"],
            }


# =============================================================================
# 结果记录器
# =============================================================================

class AblationLogger:
    """消融实验记录器。

    输出结构（每个 Query 一个文件夹，方便对比同一 Query 下的 5 种模式）：
        ablation_output/
        ├── ablation_results.json          # 全量 JSON（所有 Query × 所有模式）
        ├── Q001/
        │   ├── ablation_Q001.csv          # Q001 的 5 种模式对比表
        │   └── meta_agent_Q001.csv        # Q001 的 MetaAgent 每轮输出
        ├── Q003/
        │   ├── ablation_Q003.csv
        │   └── meta_agent_Q003.csv
        └── ...
    """

    CSV_FIELDS = [
        "mode", "label", "query_id", "query_type", "iterations", "is_finished",
        "score", "scientific_validity", "goal_achievement",
        "completeness", "practical_value",
        "judge_reasoning", "judge_strengths", "judge_weaknesses",
        # 性能与成本
        "elapsed_seconds", "total_llm_calls",
        "total_input_tokens", "total_output_tokens", "total_tokens",
        # 状态
        "has_data_context", "has_experimental_params",
        "has_fab_results", "has_analysis_report",
        "memory_entries",
        # 完整内容
        "query", "final_conclusion",
        "timestamp", "error",
    ]

    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.json_path = OUTPUT_DIR / "ablation_results.json"
        self.results: list[dict] = []
        self._load()

    def _load(self):
        if self.json_path.exists():
            try:
                data = json.load(open(self.json_path, encoding="utf-8"))
                self.results = data.get("experiments", [])
                print(f"  Loaded {len(self.results)} existing results")
            except Exception:
                pass

    def get_completed_modes(self) -> set[str]:
        """返回已完成的 mode 集合（向后兼容）。"""
        return {r["mode"] for r in self.results}

    def add(self, result: dict):
        self.results.append(result)
        self._save()
        print(f"  Saved: {result.get('query_id', '?')} x {result.get('mode', '?')}")

    def _make_row(self, r: dict) -> dict:
        """从一条 result 构造 CSV 行。"""
        j = r.get("judge_result", {})
        meta = r.get("agent_outputs", {}).get("MetaAgent", {})
        return {
            "mode": r.get("mode", ""),
            "label": r.get("label", ""),
            "query_id": r.get("query_id", ""),
            "query_type": r.get("query_type", ""),
            "iterations": r.get("iterations", 0),
            "is_finished": r.get("is_finished", False),
            "score": j.get("score", 0),
            "scientific_validity": j.get("scientific_validity", 0),
            "goal_achievement": j.get("goal_achievement", 0),
            "completeness": j.get("completeness", 0),
            "practical_value": j.get("practical_value", 0),
            "judge_reasoning": j.get("reasoning", ""),
            "judge_strengths": json.dumps(j.get("key_strengths", []), ensure_ascii=False),
            "judge_weaknesses": json.dumps(j.get("key_weaknesses", []), ensure_ascii=False),
            # 性能与成本
            "elapsed_seconds": r.get("elapsed_seconds", 0),
            "total_llm_calls": r.get("total_llm_calls", 0),
            "total_input_tokens": r.get("total_input_tokens", 0),
            "total_output_tokens": r.get("total_output_tokens", 0),
            "total_tokens": r.get("total_tokens", 0),
            # 状态
            "has_data_context": r.get("has_data_context"),
            "has_experimental_params": r.get("has_experimental_params"),
            "has_fab_results": r.get("has_fab_results"),
            "has_analysis_report": r.get("has_analysis_report"),
            "memory_entries": r.get("memory_entries"),
            # 完整内容
            "query": r.get("query", ""),
            "final_conclusion": meta.get("final_conclusion", "N/A"),
            "timestamp": r.get("timestamp", ""),
            "error": r.get("error", ""),
        }

    def _save(self):
        # === 1. 全量 JSON ===
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "experiment_info": {
                        "timestamp": datetime.now().isoformat(),
                        "max_iterations": DEFAULT_MAX_ITERATIONS,
                        "total": len(self.results),
                    },
                    "experiments": self.results,
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  JSON save failed: {e}")

        if not self.results:
            return

        # === 2. 按 query_id 分组，每个 Query 一个文件夹 ===
        from collections import defaultdict
        by_query: dict[str, list[dict]] = defaultdict(list)
        for r in self.results:
            qid = r.get("query_id", "unknown")
            by_query[qid].append(r)

        for qid, runs in by_query.items():
            query_dir = OUTPUT_DIR / qid
            query_dir.mkdir(parents=True, exist_ok=True)

            # CSV — 该 Query 的所有模式对比表
            csv_path = query_dir / f"ablation_{qid}.csv"
            try:
                with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
                    writer.writeheader()
                    for r in runs:
                        writer.writerow(self._make_row(r))
            except Exception as e:
                print(f"  CSV save failed for {qid}: {e}")

            # MetaAgent 每轮输出
            meta_csv_path = query_dir / f"meta_agent_{qid}.csv"
            try:
                with open(meta_csv_path, "w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["mode", "iteration", "meta_agent_response"])
                    writer.writeheader()
                    for r in runs:
                        mode = r.get("mode", "")
                        meta = r.get("agent_outputs", {}).get("MetaAgent", {})
                        for entry in meta.get("history", []):
                            writer.writerow({
                                "mode": mode,
                                "iteration": entry.get("iteration", "?"),
                                "meta_agent_response": entry.get("response", "N/A"),
                            })
            except Exception as e:
                print(f"  MetaAgent CSV save failed for {qid}: {e}")

    def print_summary(self):
        if not self.results:
            print("No results.")
            return

        from collections import defaultdict
        by_query: dict[str, list[dict]] = defaultdict(list)
        for r in self.results:
            by_query[r.get("query_id", "?")].append(r)

        print(f"\n{'='*70}")
        print(f"  消融实验摘要报告")
        print(f"{'='*70}")

        # 逐 Query 打印对比表
        for qid, runs in sorted(by_query.items()):
            print(f"\n  ── {qid} ──")
            print(f"  {'Mode':<15} {'Score':>6} {'SciVal':>7} {'GoalAch':>8} {'Compl':>6} {'Pract':>6} {'Iter':>5}")
            print(f"  {'─'*15} {'─'*6} {'─'*7} {'─'*8} {'─'*6} {'─'*6} {'─'*5}")

            for r in runs:
                j = r.get("judge_result", {})
                mode = r.get("mode", "?")
                score = j.get("score", 0)
                print(
                    f"  {mode:<15} {score:>6} "
                    f"{j.get('scientific_validity',0):>7} "
                    f"{j.get('goal_achievement',0):>8} "
                    f"{j.get('completeness',0):>6} "
                    f"{j.get('practical_value',0):>6} "
                    f"{r.get('iterations',0):>5}"
                )

        # 跨 Query 汇总：每种模式的平均分
        by_mode: dict[str, list[int]] = defaultdict(list)
        for r in self.results:
            by_mode[r.get("mode", "?")].append(r.get("judge_result", {}).get("score", 0))

        print(f"\n  ── Cross-query average ──")
        for mode in ["NO_DATA", "NO_DESIGN", "NO_FAB", "NO_ANALYSIS"]:
            scores = by_mode.get(mode, [])
            if not scores:
                continue
            avg = sum(scores) / len(scores)
            print(f"  {mode:<15} {avg:>6.1f}/100  ({len(scores)} queries)")
        print(f"\n  Note: Compare with FULL baseline from workflow test results.")

        print(f"\n  Output: {OUTPUT_DIR}")
        print(f"  JSON:   {self.json_path}")
        for qid in sorted(by_query.keys()):
            print(f"  CSV:    {OUTPUT_DIR / qid / f'ablation_{qid}.csv'}")
        print(f"{'='*70}")


# =============================================================================
# 论文目录清理
# =============================================================================

def cleanup_papers():
    if PAPERS_DIR.exists():
        for f in PAPERS_DIR.glob("*.md"):
            try:
                f.unlink()
            except Exception:
                pass


# =============================================================================
# 单次消融实验运行
# =============================================================================

async def run_ablation(
    mode: str,
    query: str,
    max_iterations: int,
    judge: AblationJudge,
) -> dict:
    """运行单个消融实验 — 结果结构与 psc_agent_workflow_test.run_single_experiment 对齐。"""
    info = ABLATION_MODES[mode]
    print(f"\n{'='*70}")
    print(f"  Mode: {mode} — {info['label']}")
    print(f"  {info['description']}")
    print(f"  Iterations: {max_iterations}")
    print(f"{'='*70}")

    cleanup_papers()
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    start = datetime.now()
    wf = AblationWorkflow(mode=mode, max_iterations=max_iterations)

    try:
        final_state = await wf.run(query)
        elapsed = (datetime.now() - start).total_seconds()

        # === 提取各 Agent 输出（与 workflow test 一致）===
        agent_outputs = extract_agent_outputs(final_state)

        # 状态摘要
        has_data = bool(final_state.get("data_context"))
        has_design = bool(final_state.get("experimental_params"))
        has_fab = bool(final_state.get("fab_results"))
        has_analysis = bool(final_state.get("analysis_report"))
        mem_count = len(final_state.get("memory_log", []))

        # === Token 统计 ===
        token_usage = wf.collect_token_usage()

        print(f"\n  State summary:")
        print(f"    data_context:        {'YES' if has_data else 'NO'}")
        print(f"    experimental_params: {'YES' if has_design else 'NO'}")
        print(f"    fab_results:         {'YES' if has_fab else 'NO'}")
        print(f"    analysis_report:     {'YES' if has_analysis else 'NO'}")
        print(f"    memory_log entries:  {mem_count}")
        print(f"    iterations:          {final_state.get('current_iteration', 0)}")
        print(f"    elapsed:             {elapsed:.1f}s")
        print(f"    LLM calls:           {token_usage['total_llm_calls']}")
        print(f"    input tokens:        {token_usage['total_input_tokens']:,}")
        print(f"    output tokens:       {token_usage['total_output_tokens']:,}")
        print(f"    total tokens:        {token_usage['total_tokens']:,}")

        # === LLM Judge 评估（传入完整 agent_outputs）===
        print(f"\n  Evaluating with LLM Judge...")
        judge_result = await judge.evaluate(
            goal=query,
            mode=mode,
            agent_outputs=agent_outputs,
            final_state=final_state,
        )

        score = judge_result.get("score", 0)
        print(f"\n  Score: {score}/100")
        print(f"  Scientific: {judge_result.get('scientific_validity', 0)}/30 | "
              f"Goal: {judge_result.get('goal_achievement', 0)}/30 | "
              f"Complete: {judge_result.get('completeness', 0)}/20 | "
              f"Practical: {judge_result.get('practical_value', 0)}/20")
        reasoning = judge_result.get("reasoning", "N/A")
        print(f"  Reasoning: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")

        return {
            "mode": mode,
            "label": info["label"],
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "iterations": final_state.get("current_iteration", 0),
            "is_finished": final_state.get("is_finished", False),
            # 与 workflow test 一致的字段
            "agent_outputs": agent_outputs,
            "judge_result": judge_result,
            # token 与性能统计
            "token_usage": token_usage,
            "total_tokens": token_usage["total_tokens"],
            "total_input_tokens": token_usage["total_input_tokens"],
            "total_output_tokens": token_usage["total_output_tokens"],
            "total_llm_calls": token_usage["total_llm_calls"],
            # 消融特有的布尔字段（方便 CSV 对比）
            "has_data_context": has_data,
            "has_experimental_params": has_design,
            "has_fab_results": has_fab,
            "has_analysis_report": has_analysis,
            "memory_entries": mem_count,
            "error": "",
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        elapsed = (datetime.now() - start).total_seconds()
        return {
            "mode": mode,
            "label": info["label"],
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "iterations": 0,
            "is_finished": False,
            "agent_outputs": {},
            "judge_result": {
                "score": 0, "scientific_validity": 0, "goal_achievement": 0,
                "completeness": 0, "practical_value": 0,
                "reasoning": f"执行错误: {e}",
                "key_strengths": [], "key_weaknesses": ["执行失败"],
            },
            "token_usage": {},
            "total_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_llm_calls": 0,
            "has_data_context": False,
            "has_experimental_params": False,
            "has_fab_results": False,
            "has_analysis_report": False,
            "memory_entries": 0,
            "error": str(e),
        }
    finally:
        await wf.shutdown()
        cleanup_papers()


# =============================================================================
# 入口
# =============================================================================

def list_queries():
    """列出所有可用的 Query。"""
    print(f"\n  Available queries ({len(ABLATION_QUERIES)}):")
    for q in ABLATION_QUERIES:
        print(f"    [{q['id']}] ({q.get('type', 'N/A')}) {q['query'][:70]}...")


async def main():
    parser = argparse.ArgumentParser(
        description="PeroMAS 消融实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python experiment/ablation/ablation_experiment.py                              # 默认 Q001, 全部模式
  python experiment/ablation/ablation_experiment.py --query-id Q005              # 指定 Query
  python experiment/ablation/ablation_experiment.py --query-id Q001,Q005         # 多个 Query
  python experiment/ablation/ablation_experiment.py --mode NO_DATA               # 只跑一种模式
  python experiment/ablation/ablation_experiment.py --mode NO_DATA,NO_FAB        # 指定几种模式
  python experiment/ablation/ablation_experiment.py --list                       # 列出模式
  python experiment/ablation/ablation_experiment.py --list-queries               # 列出 Query
  python experiment/ablation/ablation_experiment.py --iterations 2               # 设置迭代数
        """,
    )
    parser.add_argument("--mode", "-m", type=str, default=None,
                        help="消融模式，逗号分隔 (如: NO_DATA,NO_FAB)")
    parser.add_argument("--query-id", type=str, default=None,
                        help=f"Query ID，逗号分隔 (如: Q001,Q005)。默认: {DEFAULT_QUERY_ID}")
    parser.add_argument("--iterations", "-n", type=int, default=DEFAULT_MAX_ITERATIONS,
                        help=f"最大迭代次数 (默认: {DEFAULT_MAX_ITERATIONS})")
    parser.add_argument("--list", action="store_true", help="列出所有消融模式")
    parser.add_argument("--list-queries", action="store_true", help="列出所有可用 Query")
    parser.add_argument("--no-skip", action="store_true", help="不跳过已完成的组合")
    args = parser.parse_args()

    if args.list:
        print(f"\n  Available ablation modes:")
        for name, info in ABLATION_MODES.items():
            print(f"    {name:<15} {info['label']}")
            print(f"    {'':15} {info['description']}")
        return

    if args.list_queries:
        list_queries()
        return

    # 确定要运行的模式
    if args.mode:
        modes = [m.strip() for m in args.mode.split(",")]
        for m in modes:
            if m not in ABLATION_MODES:
                print(f"  Unknown mode: {m}")
                print(f"  Available: {', '.join(ABLATION_MODES.keys())}")
                return
    else:
        modes = list(ABLATION_MODES.keys())

    # 确定要运行的 Query（不指定则跑全部）
    if args.query_id:
        query_ids = [qid.strip() for qid in args.query_id.split(",")]
        queries_to_run = []
        for qid in query_ids:
            match = next((q for q in ABLATION_QUERIES if q["id"] == qid), None)
            if not match:
                print(f"  Unknown query ID: {qid}")
                print(f"  Available: {[q['id'] for q in ABLATION_QUERIES]}")
                return
            queries_to_run.append(match)
    else:
        queries_to_run = ABLATION_QUERIES

    # 打印配置信息（含模型）
    settings = get_settings()
    llm_config = settings.llm
    judge_model = os.getenv("LLM_MODEL_ID", "gpt-4o")

    print(f"\n{'='*70}")
    print(f"  PeroMAS Ablation Experiment")
    print(f"{'='*70}")
    print(f"  Agent LLM:    {llm_config.model_name} ({llm_config.provider})")
    print(f"  Base URL:     {llm_config.base_url[:60]}...")
    print(f"  Temperature:  {llm_config.temperature}")
    print(f"  Judge LLM:    {judge_model}")
    print(f"{'─'*70}")
    print(f"  Queries:      {[q['id'] for q in queries_to_run]}")
    print(f"  Modes:        {modes}")
    print(f"  Max iters:    {args.iterations}")
    print(f"  Total runs:   {len(queries_to_run)} queries x {len(modes)} modes = {len(queries_to_run) * len(modes)}")
    print(f"{'='*70}")

    logger = AblationLogger()
    judge = AblationJudge()

    # 跳过已完成的组合（key = "mode::query_id"）
    completed_keys = set()
    if not args.no_skip:
        for r in logger.results:
            key = f"{r.get('mode', '')}::{r.get('query_id', '')}"
            completed_keys.add(key)
        if completed_keys:
            print(f"  Skipping {len(completed_keys)} completed runs")

    run_idx = 0
    total_runs = len(queries_to_run) * len(modes)

    for query_info in queries_to_run:
        qid = query_info["id"]
        query_text = query_info["query"]

        for mode in modes:
            run_idx += 1
            combo_key = f"{mode}::{qid}"

            if combo_key in completed_keys:
                print(f"\n  [{run_idx}/{total_runs}] {mode} x {qid} — already completed, skipping")
                continue

            print(f"\n  [{run_idx}/{total_runs}] Running {mode} x {qid}...")
            result = await run_ablation(
                mode=mode,
                query=query_text,
                max_iterations=args.iterations,
                judge=judge,
            )
            # 在结果中记录 query_id 方便追溯
            result["query_id"] = qid
            result["query_type"] = query_info.get("type", "")
            logger.add(result)

            # 等待避免限流
            if run_idx < total_runs:
                print(f"\n  Waiting 5s before next run...")
                await asyncio.sleep(5)

    logger.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
