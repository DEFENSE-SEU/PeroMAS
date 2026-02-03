#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
psc_agent_workflow_test.py
PSC_Agents 全流程实验测试脚本

实验目标：
1. 测试完整的多智能体工作流程
2. 记录每个智能体的输出
3. 使用 LLM Judge 评估最终研究方案质量
4. 保存完整的实验记录

测试场景：
- 高效率钙钛矿设计 (High Efficiency)
- 高稳定性钙钛矿设计 (High Stability)
- 低毒性/无铅钙钛矿设计 (Lead-Free/Low Toxicity)
- 综合优化设计 (Multi-Objective)

Usage:
    cd f:\\PSC_Agents\\experiment\\workflow
    python psc_agent_workflow_test.py
    python psc_agent_workflow_test.py --query Q001   # 运行单个查询
    python psc_agent_workflow_test.py --query Q001,Q005  # 运行多个查询
    python psc_agent_workflow_test.py --list         # 列出所有查询
    python psc_agent_workflow_test.py --no-skip      # 不跳过已完成的
    python psc_agent_workflow_test.py --iterations 3 # 设置最大迭代次数
    
Author: PSC_Agents Team
Date: 2026-01-31
"""

import os
import sys
import json
import csv
import asyncio
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

# =============================================================================
# 路径配置
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "core"))

# DataAgent 论文保存目录（每个 Query 结束后清理）
PAPERS_DIR = Path(__file__).parent / "papers"

# 加载 .env 文件（override=True 覆盖系统环境变量）
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from core.config import Settings, LLMConfig, ProviderConfig
from core.llm import LLMClient
from workflow.graph import ResearchWorkflow
from workflow.state import create_initial_state



# =============================================================================
# 实验参数配置
# =============================================================================
DEFAULT_MAX_ITERATIONS = 3   # 默认最大迭代次数（节省API调用）
OUTPUT_DIR = Path(__file__).parent / "experiment_output"

# MCP Server URL
ARXIV_MCP_SERVER_URL = "https://seuyishu-arxiv-mcp-server.hf.space/sse"

# =============================================================================
# 测试 Query 列表 (20个 - 六大前沿研究方向)
# =============================================================================
TEST_QUERIES = [
    # === Group 1: Phase Stability & Composition (Q001-Q003) ===
    {
        "id": "Q021",
        "type": "wet-experiment",
        "query":"""
I am currently researching novel perovskite material formulations and need to design experimental protocols that balance three critical conflicting objectives: High Efficiency (PCE > 20%), High Stability (T80 > 1000h), and Reduced Toxicity (Low-Lead or Lead-Free).Please act as a Senior Materials Scientist and execute the following 4-step workflow:Step 1: Literature Review & Trend AnalysisSearch for the most recent high-impact papers (last 3 years) that successfully tackle the 'efficiency-toxicity-stability' trade-off.Identify emerging strategies such as Sn-Pb alloying, Double Perovskites, 2D/3D heterostructures, or Green Solvent Engineering.Step 2: Formula Design (5 Distinct Candidates)Based on Step 1, design 5 distinct perovskite compositions ranging from 'High Performance/Low Lead' to 'Pure Lead-Free'.For each candidate, explicitly state the stoichiometry (e.g., $FA_{0.7}MA_{0.3}Pb_{0.5}Sn_{0.5}I_3$) and the rationale behind its selection.Step 3: Detailed Experimental ProtocolsFor the most promising candidate among the five, provide a step-by-step fabrication protocol covering:Precursor Solution Preparation: Exact molar ratios, solvent choice (prioritize non-toxic solvents like DMSO/Anisole), and mixing conditions.Deposition Method: Detailed spin-coating parameters, anti-solvent quenching timing, and annealing profile.Passivation Strategy: Suggest a specific surface passivation layer (e.g., BAI, PMMA, or molecular additives) to minimize defects.Step 4: Critical Analysis & Trade-off AssessmentCritically analyze each of the 5 designs from Step 2.If a design sacrifices Efficiency for Toxicity (or vice versa), explicitly explain why (e.g., 'Sn-based perovskites have lower Voc due to oxidation' or 'Double perovskites suffer from indirect bandgaps').Provide a theoretical estimation of the expected PCE and Stability limits for each.""" }
]


# =============================================================================
# 工具函数
# =============================================================================
def get_test_settings() -> Settings:
    """创建测试用的 Settings 配置"""
    return Settings(
        llm=LLMConfig(),  # 从 .env 读取模型配置
    )


def get_judge_llm_config() -> LLMConfig:
    """创建 Judge 专用的 LLM 配置"""
    judge_model = os.getenv("LLM_MODEL_ID", "gpt-4o")
    
    config = LLMConfig(
        provider="openai",
        temperature=0.3,
        max_tokens=2000,
        timeout=120.0,
    )
    config.openai = ProviderConfig(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        model=judge_model,
    )
    return config


def get_mcp_configs(enable_mcp: bool = True) -> dict:
    """获取MCP配置"""
    if not enable_mcp:
        return {}
    
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


def print_config_info(settings: Settings, max_iterations: int):
    """打印当前配置信息"""
    llm_config = settings.llm
    judge_model = os.getenv("LLM_MODEL_ID", "gpt-4o")
    print(f"\n{'='*70}")
    print(f"📌 实验配置信息")
    print(f"{'='*70}")
    print(f"   🤖 Agent 模型: {llm_config.model_name}")
    print(f"   🌡️  Temperature: {llm_config.temperature}")
    print(f"   🔗 Base URL: {llm_config.base_url[:50]}...")
    print(f"   ⚖️  Judge 模型: {judge_model}")
    print(f"   🔄 最大迭代次数: {max_iterations}")
    print(f"   📁 输出目录: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


def cleanup_papers_directory() -> None:
    """清理 DataAgent 下载的论文目录"""
    if PAPERS_DIR.exists():
        try:
            # 统计文件数量
            files = list(PAPERS_DIR.glob("*.md"))
            if files:
                print(f"   🧹 清理论文目录: {len(files)} 个文件")
            # 删除目录及其内容
            shutil.rmtree(PAPERS_DIR)
            print(f"   ✅ 论文目录已清理: {PAPERS_DIR}")
        except Exception as e:
            print(f"   ⚠️ 清理论文目录失败: {e}")


def extract_agent_outputs(final_state: dict) -> dict:
    """从最终状态中提取各智能体的输出 - 完整内容，不截取"""
    outputs = {}
    
    # MetaAgent - 完整历史、plan、final_conclusion
    plan = final_state.get("plan", {})
    final_conclusion = final_state.get("final_conclusion", None)
    meta_history = final_state.get("meta_agent_history", [])
    
    outputs["MetaAgent"] = {
        "history": meta_history,  # 每轮完整输出
        "final_conclusion": final_conclusion if final_conclusion else "N/A",
        "has_conclusion": bool(final_conclusion),
    }
    
    # 添加最后一轮的 plan 信息
    if isinstance(plan, dict):
        outputs["MetaAgent"]["last_plan"] = plan
    else:
        outputs["MetaAgent"]["last_plan"] = {"raw": str(plan) if plan else "N/A"}
    
    # DataAgent - data_context (完整) + 解析文献信息
    data_context = final_state.get("data_context", "")
    data_agent_output = {
        "data_context": data_context if data_context else "No data collected",
        "papers_analyzed": 0,
        "paper_list": [],
    }
    
    # 解析 data_context 提取文献信息
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
    
    # DesignAgent - experimental_params (完整)
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
            "full_params": exp_params,  # 完整参数
        }
    else:
        outputs["DesignAgent"] = {"status": "No design generated"}
    
    # FabAgent - fab_results (完整)
    fab_results = final_state.get("fab_results") or {}
    if fab_results and isinstance(fab_results, dict):
        # Handle None values: get() returns None if key exists but value is None
        # Also handle cases where metrics might be non-dict types
        raw_metrics = fab_results.get("predicted_metrics") or fab_results.get("metrics")
        metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
        outputs["FabAgent"] = {
            "composition": fab_results.get("composition") or "N/A",
            "PCE_percent": metrics.get("PCE_percent", "N/A") if metrics else "N/A",
            "Voc_V": metrics.get("Voc_V", "N/A") if metrics else "N/A",
            "Jsc_mA_cm2": metrics.get("Jsc_mA_cm2", "N/A") if metrics else "N/A",
            "FF_percent": metrics.get("FF_percent", "N/A") if metrics else "N/A",
            "full_results": fab_results,  # 完整结果
        }
    else:
        outputs["FabAgent"] = {"status": "No prediction available"}
    
    # AnalysisAgent - analysis_report (完整)
    analysis = final_state.get("analysis_report", "")
    outputs["AnalysisAgent"] = {
        "analysis_report": analysis if analysis else "No analysis available"
    }
    
    # MemoryAgent - memory_log (完整)
    memory_log = final_state.get("memory_log", [])
    structured_memory = final_state.get("structured_memory", [])
    outputs["MemoryAgent"] = {
        "entries_count": len(memory_log),
        "memory_log": memory_log,  # 完整日志
        "structured_memory": structured_memory,  # 结构化记忆
    }
    
    return outputs


def format_agent_outputs_for_display(outputs: dict) -> str:
    """格式化智能体输出用于显示 - 重点显示 MetaAgent 完整输出"""
    lines = []
    
    # === MetaAgent 完整输出（最重要）===
    meta_output = outputs.get("MetaAgent", {})
    lines.append(f"\n{'='*70}")
    lines.append(f"🧠 MetaAgent 完整输出历史")
    lines.append(f"{'='*70}")
    
    # 显示每轮的完整输出
    history = meta_output.get("history", [])
    for entry in history:
        iteration = entry.get("iteration", "?")
        response = entry.get("response", "N/A")
        lines.append(f"\n{'─'*50}")
        lines.append(f"📋 第 {iteration} 轮 MetaAgent 思考与输出:")
        lines.append(f"{'─'*50}")
        lines.append(response)  # 完整输出，不截取
    
    # 显示最终结论
    if meta_output.get("has_conclusion"):
        lines.append(f"\n{'='*70}")
        lines.append(f"🏁 MetaAgent 最终研究结论")
        lines.append(f"{'='*70}")
        lines.append(meta_output.get("final_conclusion", "N/A"))
    
    # === 其他 Agent 简要信息 ===
    lines.append(f"\n{'='*70}")
    lines.append(f"📊 其他智能体输出摘要")
    lines.append(f"{'='*70}")
    
    for agent_name, output in outputs.items():
        if agent_name == "MetaAgent":
            continue  # 已经显示过
        
        lines.append(f"\n{'─'*40}")
        lines.append(f"📍 {agent_name}:")
        
        if agent_name == "DesignAgent":
            lines.append(f"   Formula: {output.get('formula', 'N/A')}")
            lines.append(f"   Method: {output.get('method', 'N/A')}")
        elif agent_name == "FabAgent":
            lines.append(f"   PCE: {output.get('PCE_percent', 'N/A')}")
            lines.append(f"   Voc: {output.get('Voc_V', 'N/A')}")
        elif agent_name == "MemoryAgent":
            lines.append(f"   Entries: {output.get('entries_count', 0)}")
        else:
            # 简要显示
            for key, value in list(output.items())[:3]:
                if isinstance(value, str) and len(value) > 200:
                    lines.append(f"   {key}: {value[:200]}...")
                else:
                    lines.append(f"   {key}: {value}")
    
    return "\n".join(lines)


# =============================================================================
# LLM Judge - 评估研究方案质量 (100分制)
# =============================================================================
class WorkflowJudge:
    """LLM 裁判：评估 PSC_Agents 全流程输出质量"""
    
    JUDGE_SYSTEM_PROMPT = """你是一个专业的钙钛矿太阳能电池研究方案评估专家。
你的任务是评估AI多智能体系统生成的研究方案质量。

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

## 任务类型说明:
- high_efficiency: 高效率设计 - 关注PCE、带隙、载流子传输
- high_stability: 高稳定性设计 - 关注热稳定性、湿度稳定性、抗离子迁移
- lead_free: 无铅/低毒性设计 - 关注Sn/Bi替代、双钙钛矿
- multi_objective: 综合优化 - 平衡效率、稳定性、环保性

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
        judge_config = get_judge_llm_config()
        self.llm = LLMClient(judge_config)
        self.model_name = judge_config.model_name
    
    async def evaluate_workflow_result(
        self, 
        user_query: str, 
        query_type: str,
        agent_outputs: dict,
        final_state: dict,
    ) -> dict[str, Any]:
        """评估工作流结果质量"""
        
        # 任务类型描述
        type_desc_map = {
            "high_efficiency": "🚀 高效率设计 - 目标: 最大化PCE",
            "high_stability": "🛡️ 高稳定性设计 - 目标: 长期稳定运行",
            "lead_free": "🌱 无铅/低毒性设计 - 目标: 环保无铅材料",
            "multi_objective": "⚖️ 综合优化设计 - 目标: 多目标平衡",
        }
        task_desc = type_desc_map.get(query_type, f"❓ 其他类型: {query_type}")
        
        # 提取关键结果
        meta_output = agent_outputs.get("MetaAgent", {})
        data_output = agent_outputs.get("DataAgent", {})
        design_output = agent_outputs.get("DesignAgent", {})
        fab_output = agent_outputs.get("FabAgent", {})
        analysis_output = agent_outputs.get("AnalysisAgent", {})
        memory_output = agent_outputs.get("MemoryAgent", {})
        
        # === 格式化 MetaAgent 完整历史 ===
        meta_history = meta_output.get("history", [])
        meta_history_text = ""
        for entry in meta_history:
            iter_num = entry.get("iteration", "?")
            response = entry.get("response", "N/A")
            # 截取每轮的关键部分（不超过2000字符）
            if len(response) > 2000:
                response = response[:2000] + "...[截断]"
            meta_history_text += f"\n--- 第{iter_num}轮 ---\n{response}\n"
        
        # === 格式化文献信息 ===
        papers_analyzed = data_output.get("papers_analyzed", 0)
        paper_list = data_output.get("paper_list", [])
        literature_text = f"分析论文数: {papers_analyzed}\n"
        for p in paper_list[:10]:  # 最多显示10篇
            literature_text += f"- [{p.get('paper_id', 'N/A')}] {p.get('title', 'Unknown')[:80]}\n"
            findings = p.get('key_findings', [])
            if findings:
                if isinstance(findings, list):
                    literature_text += f"  Key Findings: {'; '.join(str(f)[:100] for f in findings[:3])}\n"
                else:
                    literature_text += f"  Key Findings: {str(findings)[:200]}\n"
        
        # === 格式化 Memory 中的迭代学习（含文献引用）===
        structured_memory = memory_output.get("structured_memory", [])
        memory_summary = ""
        for m in structured_memory:
            iter_n = m.get("iteration", "?")
            formula = m.get("formula", "N/A")
            pce = m.get("pce", "N/A")
            verdict = m.get("verdict", "N/A")
            learning = m.get("learning", "N/A")
            lit_refs = m.get("literature_refs", [])
            lit_summary = m.get("literature_summary", "")
            
            memory_summary += f"- Iter{iter_n}: {formula}, PCE={pce}, {verdict}\n"
            memory_summary += f"  Learning: {str(learning)[:150]}...\n"
            
            # 显示该轮引用的文献（使用格式化的摘要）
            if lit_summary:
                # 使用精简的文献摘要
                memory_summary += f"  {lit_summary[:300]}...\n"
            elif lit_refs:
                # Fallback: 仅显示论文ID
                memory_summary += f"  Literature ({len(lit_refs)} papers): "
                ref_ids = [r.get("paper_id", "?") for r in lit_refs[:5]]
                memory_summary += ", ".join(ref_ids) + "\n"
        
        # 构建评估 Prompt - 完整展示所有迭代
        eval_prompt = f"""## 任务类型
{task_desc}

## 用户研究目标
{user_query}

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
                prompt=eval_prompt,
                system_message=self.JUDGE_SYSTEM_PROMPT
            )
            return self._parse_response(response)
        except Exception as e:
            return {
                "score": 0, 
                "scientific_validity": 0, 
                "goal_achievement": 0, 
                "completeness": 0, 
                "practical_value": 0, 
                "reasoning": f"Judge 错误: {e}",
                "key_strengths": [],
                "key_weaknesses": ["评估失败"]
            }
    
    def _parse_response(self, response: str) -> dict[str, Any]:
        """解析 Judge 响应"""
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
                "key_weaknesses": result.get("key_weaknesses", [])
            }
        except Exception as e:
            return {
                "score": 0, 
                "scientific_validity": 0, 
                "goal_achievement": 0, 
                "completeness": 0, 
                "practical_value": 0, 
                "reasoning": f"解析错误: {e}",
                "key_strengths": [],
                "key_weaknesses": ["解析失败"]
            }


# =============================================================================
# 实验结果记录器
# =============================================================================
class ExperimentLogger:
    """实验结果记录器 - 支持断点续跑，完整保存 MetaAgent 输出"""
    
    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.json_path = OUTPUT_DIR / "workflow_experiment_results.json"
        self.csv_path = OUTPUT_DIR / "workflow_experiment_results.csv"
        self.meta_csv_path = OUTPUT_DIR / "meta_agent_outputs.csv"  # 新增：MetaAgent 每轮输出
        self.results: list[dict[str, Any]] = []
        self._load_existing()
    
    def _load_existing(self) -> None:
        """加载已有结果"""
        if self.json_path.exists():
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.results = data.get("experiments", [])
                    print(f"📂 已加载 {len(self.results)} 条历史记录")
            except Exception as e:
                print(f"⚠️ 加载历史记录失败: {e}")
    
    def get_completed_ids(self) -> set[str]:
        """获取已完成的查询ID"""
        return {r["query_id"] for r in self.results}
    
    def add_result(self, result: dict[str, Any]) -> None:
        """添加结果并保存"""
        self.results.append(result)
        self._save()
        print(f"💾 已保存: {result['query_id']}")
    
    def _save(self) -> None:
        """保存到 JSON 和 CSV - 完整内容不截取"""
        # JSON - 完整保存所有内容
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "experiment_info": {
                        "timestamp": datetime.now().isoformat(),
                        "max_iterations": DEFAULT_MAX_ITERATIONS,
                        "total": len(self.results)
                    },
                    "experiments": self.results
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ JSON 保存失败: {e}")
        
        # CSV 1 - 主表：Query + 最终结论 + Judge 评价（完整）
        if self.results:
            try:
                with open(self.csv_path, "w", encoding="utf-8-sig", newline="") as f:
                    fieldnames = [
                        "query_id", "type", "iterations", "score",
                        "query",  # 完整 Query
                        "final_conclusion",  # 完整最终结论
                        "judge_reasoning",  # 完整 Judge 评价
                        "scientific_validity", "goal_achievement", 
                        "completeness", "practical_value",
                        "judge_strengths", "judge_weaknesses",
                        "timestamp"
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for r in self.results:
                        judge = r.get("judge_result", {})
                        meta = r.get("agent_outputs", {}).get("MetaAgent", {})
                        writer.writerow({
                            "query_id": r.get("query_id", ""),
                            "type": r.get("type", ""),
                            "iterations": r.get("iterations", 0),
                            "score": judge.get("score", 0),
                            "query": r.get("query", ""),  # 完整
                            "final_conclusion": meta.get("final_conclusion", "N/A"),  # 完整
                            "judge_reasoning": judge.get("reasoning", ""),  # 完整
                            "scientific_validity": judge.get("scientific_validity", 0),
                            "goal_achievement": judge.get("goal_achievement", 0),
                            "completeness": judge.get("completeness", 0),
                            "practical_value": judge.get("practical_value", 0),
                            "judge_strengths": json.dumps(judge.get("key_strengths", []), ensure_ascii=False),
                            "judge_weaknesses": json.dumps(judge.get("key_weaknesses", []), ensure_ascii=False),
                            "timestamp": r.get("timestamp", "")
                        })
            except Exception as e:
                print(f"⚠️ 主 CSV 保存失败: {e}")
        
        # CSV 2 - MetaAgent 每轮输出详情（完整）
        if self.results:
            try:
                with open(self.meta_csv_path, "w", encoding="utf-8-sig", newline="") as f:
                    fieldnames = [
                        "query_id", "iteration", "meta_agent_response"
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for r in self.results:
                        query_id = r.get("query_id", "")
                        meta = r.get("agent_outputs", {}).get("MetaAgent", {})
                        history = meta.get("history", [])
                        for entry in history:
                            writer.writerow({
                                "query_id": query_id,
                                "iteration": entry.get("iteration", "?"),
                                "meta_agent_response": entry.get("response", "N/A")  # 完整输出
                            })
            except Exception as e:
                print(f"⚠️ MetaAgent CSV 保存失败: {e}")
    
    def print_summary(self) -> None:
        """打印实验摘要"""
        if not self.results:
            print("📊 暂无实验结果")
            return
        
        scores = [r.get("judge_result", {}).get("score", 0) for r in self.results]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # 按类型统计
        type_scores = {}
        for r in self.results:
            t = r.get("type", "unknown")
            s = r.get("judge_result", {}).get("score", 0)
            if t not in type_scores:
                type_scores[t] = []
            type_scores[t].append(s)
        
        print(f"\n{'='*70}")
        print(f"📊 实验摘要报告")
        print(f"{'='*70}")
        print(f"   总实验数: {len(self.results)}")
        print(f"   平均分数: {avg_score:.1f}/100")
        print(f"\n   按类型统计:")
        for t, ss in type_scores.items():
            avg = sum(ss) / len(ss) if ss else 0
            print(f"     - {t}: {len(ss)}个查询, 平均{avg:.1f}分")
        print(f"{'='*70}")


# =============================================================================
# 主实验运行器
# =============================================================================
async def run_single_experiment(
    query_info: dict,
    max_iterations: int,
    logger: ExperimentLogger,
    judge: WorkflowJudge,
    enable_mcp: bool = True,
) -> dict[str, Any]:
    """运行单个实验"""
    
    query_id = query_info["id"]
    query_type = query_info["type"]
    query = query_info["query"]
    
    print(f"\n{'='*70}")
    print(f"🧪 开始实验: {query_id} ({query_type})")
    print(f"{'='*70}")
    print(f"📋 查询: {query}")
    print(f"{'─'*70}")
    
    # 开始前先清理论文目录，确保每个 Query 从干净状态开始
    cleanup_papers_directory()
    # 确保目录存在
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now()
    
    # 创建工作流
    workflow = ResearchWorkflow(
        settings=get_test_settings(),
        max_iterations=max_iterations,
        mcp_configs=get_mcp_configs(enable_mcp),
        papers_dir=str(PAPERS_DIR),  # 传递论文目录给 DataAgent
    )
    
    try:
        # 运行工作流
        print(f"\n🚀 启动多智能体工作流...")
        final_state = await workflow.run(query)
        
        # 提取各智能体输出
        agent_outputs = extract_agent_outputs(final_state)
        
        # 显示各智能体输出
        print(f"\n{'='*70}")
        print(f"📊 各智能体输出摘要")
        print(f"{'='*70}")
        print(format_agent_outputs_for_display(agent_outputs))
        
        # LLM Judge 评估
        print(f"\n{'='*70}")
        print(f"⚖️  LLM Judge 评估中...")
        print(f"{'='*70}")
        
        judge_result = await judge.evaluate_workflow_result(
            user_query=query,
            query_type=query_type,
            agent_outputs=agent_outputs,
            final_state=final_state,
        )
        
        # 显示评分结果
        score = judge_result.get("score", 0)
        if score >= 90:
            grade = "🌟 优秀"
        elif score >= 75:
            grade = "✅ 良好"
        elif score >= 60:
            grade = "⚠️ 及格"
        elif score >= 40:
            grade = "❌ 较差"
        else:
            grade = "💀 失败"
        
        print(f"\n📊 评分结果: {score}/100 {grade}")
        print(f"   科学合理性: {judge_result.get('scientific_validity', 0)}/30")
        print(f"   目标达成度: {judge_result.get('goal_achievement', 0)}/30")
        print(f"   方案完整性: {judge_result.get('completeness', 0)}/20")
        print(f"   实用指导价值: {judge_result.get('practical_value', 0)}/20")
        print(f"\n📝 评价: {judge_result.get('reasoning', 'N/A')}")
        
        if judge_result.get("key_strengths"):
            print(f"\n✅ 优点: {', '.join(judge_result['key_strengths'])}")
        if judge_result.get("key_weaknesses"):
            print(f"❌ 不足: {', '.join(judge_result['key_weaknesses'])}")
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # 构建结果记录
        result = {
            "query_id": query_id,
            "type": query_type,
            "query": query,
            "query_en": query_info.get("query_en", ""),
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed_time,
            "iterations": final_state.get("current_iteration", 0),
            "is_finished": final_state.get("is_finished", False),
            "agent_outputs": agent_outputs,
            "judge_result": judge_result,
        }
        
        # 保存结果
        logger.add_result(result)
        
        print(f"\n✅ 实验完成: {query_id} (耗时: {elapsed_time:.1f}s)")
        
        return result
        
    except Exception as e:
        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        
        result = {
            "query_id": query_id,
            "type": query_type,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed_time,
            "error": str(e),
            "agent_outputs": {},
            "judge_result": {"score": 0, "reasoning": f"执行错误: {e}"}
        }
        logger.add_result(result)
        return result
        
    finally:
        # 关闭工作流
        await workflow.shutdown()
        # 清理 DataAgent 下载的论文目录
        cleanup_papers_directory()


async def run_experiments(
    query_ids: list[str] | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    skip_completed: bool = True,
    enable_mcp: bool = True,
):
    """运行批量实验"""
    
    # 初始化
    settings = get_test_settings()
    print_config_info(settings, max_iterations)
    
    logger = ExperimentLogger()
    judge = WorkflowJudge()
    
    # 确定要运行的查询
    if query_ids:
        queries_to_run = [q for q in TEST_QUERIES if q["id"] in query_ids]
    else:
        queries_to_run = TEST_QUERIES
    
    # 跳过已完成
    if skip_completed:
        completed = logger.get_completed_ids()
        queries_to_run = [q for q in queries_to_run if q["id"] not in completed]
        if completed:
            print(f"⏭️  跳过已完成: {len(completed)} 个")
    
    if not queries_to_run:
        print("✅ 所有查询已完成!")
        logger.print_summary()
        return
    
    print(f"📋 待运行查询: {len(queries_to_run)} 个")
    print(f"   ID列表: {[q['id'] for q in queries_to_run]}")
    print(f"{'='*70}\n")
    
    # 逐个运行
    for i, query_info in enumerate(queries_to_run, 1):
        print(f"\n{'🔬'*35}")
        print(f"   进度: {i}/{len(queries_to_run)}")
        print(f"{'🔬'*35}")
        
        await run_single_experiment(
            query_info=query_info,
            max_iterations=max_iterations,
            logger=logger,
            judge=judge,
            enable_mcp=enable_mcp,
        )
        
        # 每个实验后暂停一下，避免API限流
        if i < len(queries_to_run):
            print(f"\n⏳ 等待5秒后继续下一个实验...")
            await asyncio.sleep(5)
    
    # 打印最终摘要
    logger.print_summary()


def list_queries():
    """列出所有测试查询"""
    print(f"\n{'='*70}")
    print(f"📋 PSC_Agents 工作流测试查询列表")
    print(f"{'='*70}")
    
    current_type = None
    for q in TEST_QUERIES:
        if q["type"] != current_type:
            current_type = q["type"]
            type_names = {
                "high_efficiency": "🚀 高效率设计",
                "high_stability": "🛡️ 高稳定性设计",
                "lead_free": "🌱 无铅/低毒性设计",
                "multi_objective": "⚖️ 综合优化设计",
            }
            print(f"\n{type_names.get(current_type, current_type)}")
            print(f"{'─'*50}")
        
        print(f"  [{q['id']}] {q['query'][:60]}...")
    
    print(f"\n{'='*70}")
    print(f"总计: {len(TEST_QUERIES)} 个测试查询")
    print(f"{'='*70}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PSC_Agents 全流程工作流实验测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python psc_agent_workflow_test.py                    # 运行所有查询
  python psc_agent_workflow_test.py --query Q001      # 运行单个查询
  python psc_agent_workflow_test.py --query Q001,Q005 # 运行多个查询
  python psc_agent_workflow_test.py --list            # 列出所有查询
  python psc_agent_workflow_test.py --no-skip         # 不跳过已完成的
  python psc_agent_workflow_test.py --iterations 3    # 设置最大迭代次数
  python psc_agent_workflow_test.py --no-mcp          # 禁用MCP工具
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="要运行的查询ID，多个用逗号分隔 (如: Q001,Q005)"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有测试查询"
    )
    
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="不跳过已完成的查询"
    )
    
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"最大迭代次数 (默认: {DEFAULT_MAX_ITERATIONS})"
    )
    
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="禁用MCP工具 (使用Mock模式)"
    )
    
    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_args()
    
    # 列出查询
    if args.list:
        list_queries()
        return
    
    # 打印 Banner
    print(f"\n{'🔬'*35}")
    print(f"   PSC_Agents 全流程工作流实验测试")
    print(f"{'🔬'*35}")
    print(f"\n   应用场景: 高效率 / 高稳定性 / 低毒性 钙钛矿设计")
    print(f"   评估方式: LLM Judge (100分制)")
    
    # 解析查询ID
    query_ids = None
    if args.query:
        query_ids = [q.strip() for q in args.query.split(",")]
    
    # 运行实验
    await run_experiments(
        query_ids=query_ids,
        max_iterations=args.iterations,
        skip_completed=not args.no_skip,
        enable_mcp=not args.no_mcp,
    )


if __name__ == "__main__":
    asyncio.run(main())
