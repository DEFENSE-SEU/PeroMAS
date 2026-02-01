#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_agent_experiment.py
DataAgent 批量实验脚本

实验目标：
1. 测试工具调用是否正常（打印调用的工具列表）
2. 使用 LLM Judge 判定输出内容是否合理

实验类型：
- Type A: 纯文献搜索（不下载本地与抽取）
- Type B: 文献搜索 + 内容抽取

Usage:
    cd f:\PSC_Agents\experiment\single_point
    python data_agent_experiment.py
    python data_agent_experiment.py --query Q001   # 运行单个查询
    python data_agent_experiment.py --list         # 列出所有查询
    
Author: PSC_Agents Team
Date: 2026-01-27
"""

import os
import sys
import json
import csv
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Any

# =============================================================================
# 路径配置
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "core"))  # 让 llm.py 能找到 config

from core.config import Settings, LLMConfig, MCPConfig, MCPServerConfig, ProviderConfig
from core.llm import LLMClient

# 直接从模块导入，避免触发 agent/__init__.py 的全部导入
import importlib.util
spec = importlib.util.spec_from_file_location(
    "data_agent", 
    PROJECT_ROOT / "src" / "agent" / "data_agent.py"
)
data_agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_agent_module)
DataAgent = data_agent_module.DataAgent

# =============================================================================
# 实验参数配置
# =============================================================================
MAX_PAPERS = 5          # 单次最多搜索/下载篇数
MAX_ITERATIONS = 30     # 智能体最大迭代次数 (extract 任务需要更多轮次)
OUTPUT_DIR = Path(__file__).parent / "experiment_output"

# MCP 服务器配置
ARXIV_MCP_SERVER_URL = "https://seuyishu-arxiv-mcp-server.hf.space/sse"

# =============================================================================
# 测试 Query 列表
# 格式: {"id": str, "type": "search" | "extract", "query": str, "extraction_goal": str | None}
# - type="search": 纯文献搜索，不下载和抽取
# - type="extract": 文献搜索 + 下载 + 内容抽取
# =============================================================================
TEST_QUERIES = [
    # === Type A: 纯文献搜索 (search) ===
    {
        "id": "Q001",
        "type": "search",
        "query": "Search for recent review papers on perovskite solar cells.",
        "extraction_goal": None
    },
    {
        "id": "Q002",
        "type": "search",
        "query": "Find literature regarding the synthesis of CsPbI3",
        "extraction_goal": None
    },
    {
        "id": "Q003",
        "type": "search",
        "query": "Find literature about lead-free perovskite materials.",
        "extraction_goal": None
    },
    {
        "id": "Q004",
        "type": "search",
        "query": "Search for papers on improving the stability of perovskite solar cells.",
        "extraction_goal": None
    },
    {
        "id": "Q005",
        "type": "search",
        "query": "Find literature about lead-free perovskite materials for photovoltaic applications",
        "extraction_goal": None
    },
    {
        "id": "Q006",
        "type": "search",
        "query": "Retrieve articles on high-efficiency perovskite solar cell fabrication.",
        "extraction_goal": None
    },
    {
        "id": "Q007",
        "type": "search",
        "query": "Search for high-efficiency perovskite solar cells (>24%).",
        "extraction_goal": None
    },
    {
        "id": "Q008",
        "type": "search",
        "query": "Search for machine learning papers in perovskite material discovery.",
        "extraction_goal": None
    },
    {
        "id": "Q009",
        "type": "search",
        "query": "Find research on all-inorganic perovskite solar cells.",
        "extraction_goal": None
    },
    {
        "id": "Q010",
        "type": "search",
        "query": "Find research on tin-based perovskite solar cells.",
        "extraction_goal": None
    },
    {
        "id": "Q011",
        "type": "search",
        "query": "Search for studies on double perovskite materials.",
        "extraction_goal": None
    },
    {
        "id": "Q012",
        "type": "search",
        "query": "Find literature on interface engineering in perovskite solar cells.",
        "extraction_goal": None
    },
    {
        "id": "Q013",
        "type": "search",
        "query": "Search for papers on the toxicity of perovskite materials.",
        "extraction_goal": None
    },
    {
        "id": "Q014",
        "type": "search",
        "query": "Retrieve articles about flexible perovskite solar cells.",
        "extraction_goal": None
    },
    {
        "id": "Q015",
        "type": "search",
        "query": "Search for papers on transport layer materials for PSCs.",
        "extraction_goal": None
    },
    {
        "id": "Q016",
        "type": "search",
        "query": "Find literature on defect passivation techniques in perovskites.",
        "extraction_goal": None
    },
    {
        "id": "Q017",
        "type": "search",
        "query": "Find recent progress in perovskite synthesis methods.",
        "extraction_goal": None
    },
    {
        "id": "Q018",
        "type": "search",
        "query": "Search for research on 2D perovskite materials.",
        "extraction_goal": None
    },
    {
        "id": "Q019",
        "type": "search",
        "query": "Search for literature regarding large-area perovskite modules.",
        "extraction_goal": None
    },
    {
        "id": "Q020",
        "type": "search",
        "query": "Search for literature on the environmental impact of perovskite solar cells.",
        "extraction_goal": None
    },
    # === Type B: 文献搜索 + 内容抽取 (extract) ===
    {
        "id": "Q021",
        "type": "extract",
        "query": "Search for recent papers on perovskite solar cell stability and extract the T80 lifetime values. ",
    },
    {
        "id": "Q022",
        "type": "extract",
        "query": "Find literature about lead-free perovskite materials and extract the bandgap energy (Eg).",
    },
    {
        "id": "Q023",
        "type": "extract",
        "query": "Search for high-efficiency perovskite solar cells and list the maximum efficiency (PCE) reported.",
    },
    {
        "id": "Q024",
        "type": "extract",
        "query": "Retrieve articles on large-area perovskite modules and extract the active area size.",
    },
    {
        "id": "Q025",
        "type": "extract",
        "query": "Find studies on perovskite synthesis methods and summarize the annealing temperature used.",
    },
    {
        "id": "Q026",
        "type": "extract",
        "query": "Search for papers on inverted perovskite solar cells and identify the hole transport layer (HTL) materials.",
    },
    {
        "id": "Q027",
        "type": "extract",
        "query": "Retrieve literature on flexible perovskite solar cells and extract the substrate type (e.g., PET, PEN).",
    },
    {
        "id": "Q028",
        "type": "extract",
        "query": "Retrieve literature on flexible perovskite solar cells and extract the substrate type (e.g., PET, PEN).",
    },
    {
        "id": "Q029",
        "type": "extract",
        "query": "Search for all-inorganic perovskite research and extract the phase transition temperature.",
    },
    {
        "id": "Q030",
        "type": "extract",
        "query": "Search for literature regarding defect passivation in perovskites and list the passivation agents used.",
    },
    {
        "id": "Q031",
        "type": "extract",
        "query": "Retrieve studies on tin-based perovskites and extract the short-circuit current density (Jsc).",
    },
    {
        "id": "Q032",
        "type": "extract",
        "query": "Find literature about perovskite moisture stability and summarize the encapsulation method.",
    },
    {
        "id": "Q033",
        "type": "extract",
        "query": "Find papers regarding the toxicity of perovskites and summarize the environmental impact conclusion.",
    },
    {
        "id": "Q034",
        "type": "extract",
        "query": "Search for literature on slot-die coating of perovskites and extract the coating speed. ",
    },
    {
        "id": "Q035",
        "type": "extract",
        "query": "Retrieve studies on carbon-based perovskite solar cells and extract the fill factor (FF).",
    },
    {
        "id": "Q036",
        "type": "extract",
        "query": "Search for papers on double perovskite materials and extract the crystal structure type.",
    },
    {
        "id": "Q037",
        "type": "extract",
        "query": "Find literature on solvent engineering in perovskites and list the antisolvents used.",
    },
    {
        "id": "Q038",
        "type": "extract",
        "query": "Search for research on carrier dynamics in perovskites and extract the carrier lifetime.",
    },
    {
        "id": "Q039",
        "type": "extract",
        "query": "Retrieve articles about perovskites additive engineering and identify the specific additives added.",
    },
    {
        "id": "Q040",
        "type": "extract",
        "query": "Search for literature on perovskite interface engineering and summarize the interface modification strategy.",
    },
    
]


# =============================================================================
# 工具函数
# =============================================================================
def get_test_settings() -> Settings:
    """创建测试用的 Settings 配置（DataAgent 使用）"""
    mcp_config = MCPConfig(
        servers={
            "arxiv": MCPServerConfig(
                command="",
                args=[],
                url=ARXIV_MCP_SERVER_URL,
                enabled=True,
            )
        }
    )
    return Settings(
        llm=LLMConfig(),  # 从 .env 读取模型配置
        mcp=mcp_config,
    )


def get_judge_llm_config() -> LLMConfig:
    """创建 Judge 专用的 LLM 配置（使用通用 LLM 配置）"""
    # Judge 使用通用的 LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_ID
    judge_model = os.getenv("LLM_MODEL_ID", "gpt-5.2")
    
    config = LLMConfig(
        provider="openai",  # 通过代理都用 openai provider
        temperature=0.3,    # Judge 用较低温度确保评分稳定
        max_tokens=1000,
        timeout=60.0,
    )
    # 使用通用的 LLM 配置
    config.openai = ProviderConfig(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        model=judge_model,
    )
    return config


def print_model_info(settings: Settings):
    """打印当前使用的模型信息"""
    llm_config = settings.llm
    judge_model = os.getenv("LLM_MODEL_ID", "gpt-4o-mini")
    print(f"\n{'='*60}")
    print(f"📌 模型配置信息")
    print(f"{'='*60}")
    print(f"   🤖 DataAgent 模型: {llm_config.model_name}")
    print(f"   🌡️  Temperature: {llm_config.temperature}")
    print(f"   🔗 Base URL: {llm_config.base_url[:50]}...")
    print(f"   ⚖️  Judge 模型: {judge_model}")
    print(f"{'='*60}\n")


# =============================================================================
# LLM Judge - 评估输出质量 (100分制)
# =============================================================================
class LLMJudge:
    """LLM 裁判：评估 DataAgent 输出质量"""
    
    JUDGE_SYSTEM_PROMPT = """You are a fair evaluator for an AI research assistant agent.
Your job is to score the agent's output based on how well it addresses the user's query.

## Scoring System (0-100 points):

### Dimension 1: Task Completion (0-40 points)
- Did the agent attempt the requested task?
- Did the agent use appropriate tools?
- Did the agent provide a final answer/output?

### Dimension 2: Output Relevance (0-30 points)
- Is the output relevant to the user's query?
- Does the output address the core question?
- Is the information on-topic?

### Dimension 3: Output Quality (0-30 points)
- Is the output well-organized and clear?
- Is the information useful and actionable?
- Did the agent provide good explanations?

## Scoring Guidelines:
- 90-100: Excellent - Fully addresses the query with high-quality output
- 70-89: Good - Addresses the query well with minor gaps
- 50-69: Acceptable - Partially addresses the query, useful but incomplete
- 30-49: Poor - Significant gaps, limited usefulness
- 0-29: Failed - Does not address the query or completely wrong

## Important Notes:
- This agent only has access to arXiv preprints, not journals or other databases
- Judge based on what the agent CAN do, not what would be ideal
- If agent honestly explains limitations, that's positive behavior

Respond in JSON format ONLY:
{
    "score": 0-100,
    "task_completion": 0-40,
    "relevance": 0-30,
    "quality": 0-30,
    "reasoning": "Brief explanation"
}
"""

    def __init__(self):
        # Judge 固定使用 GPT-5.2
        judge_config = get_judge_llm_config()
        self.llm = LLMClient(judge_config)
        self.model_name = judge_config.model_name
    
    async def check_result_quality(
        self, 
        user_query: str, 
        query_type: str,
        agent_output: str,
        tools_called: list[str],
        tool_results: list[dict] | None = None
    ) -> dict[str, Any]:
        """评估 Agent 输出质量"""
        
        # 统计工具调用
        tool_summary = {}
        for t in tools_called:
            tool_summary[t] = tool_summary.get(t, 0) + 1
        
        # 任务类型描述
        if query_type == "search":
            task_desc = "🔍 Literature Search - Find relevant papers on arXiv"
        else:
            task_desc = "📊 Data Extraction - Search papers AND extract specific data"
        
        # 构建评估 Prompt
        eval_prompt = f"""## Task Type
{task_desc}

## User Query
{user_query}

## Agent Tool Usage
{json.dumps(tool_summary, indent=2)}
Total tool calls: {len(tools_called)}

## Agent Final Output
{agent_output[:5000] if agent_output else "(No output provided)"}

---
Please evaluate the agent's performance and provide a score out of 100.
"""
        try:
            response = await self.llm.ainvoke_simple(
                prompt=eval_prompt,
                system_message=self.JUDGE_SYSTEM_PROMPT
            )
            return self._parse_response(response)
        except Exception as e:
            return {"score": 0, "task_completion": 0, "relevance": 0, "quality": 0, "reasoning": f"Judge error: {e}"}
    
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
                "task_completion": result.get("task_completion", 0),
                "relevance": result.get("relevance", 0),
                "quality": result.get("quality", 0),
                "reasoning": result.get("reasoning", "No reasoning")
            }
        except Exception as e:
            return {"score": 0, "task_completion": 0, "relevance": 0, "quality": 0, "reasoning": f"Parse error: {e}"}


# =============================================================================
# 实验结果记录器
# =============================================================================
class ExperimentLogger:
    """实验结果记录器 - 支持断点续跑"""
    
    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.json_path = OUTPUT_DIR / "experiment_results.json"
        self.csv_path = OUTPUT_DIR / "experiment_results.csv"
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
        """保存到 JSON 和 CSV（带重试和错误处理）"""
        import time
        
        # JSON - 通常不会有权限问题
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "experiment_info": {
                        "timestamp": datetime.now().isoformat(),
                        "max_papers": MAX_PAPERS,
                        "max_iterations": MAX_ITERATIONS,
                        "total": len(self.results)
                    },
                    "experiments": self.results
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ JSON 保存失败: {e}")
        
        # CSV - 可能被 Excel 等程序占用，添加重试逻辑
        if self.results:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with open(self.csv_path, "w", encoding="utf-8-sig", newline="") as f:
                        # 完整的字段列表，方便人类专家审核
                        fieldnames = [
                            "query_id", 
                            "query_type", 
                            "query",  # 完整查询
                            "success", 
                            "iterations",
                            "tool_count",
                            "tools_called",
                            # 论文信息
                            "papers_found",  # 找到的论文数量
                            "paper_titles",  # 论文标题列表
                            "paper_ids",     # 论文ID列表
                            # Agent 输出
                            "agent_output",  # Agent 完整输出
                            # 抽取结果（仅 extract 类型）
                            "extraction_result",  # 抽取的数据
                            # Judge 评价
                            "judge_score",
                            "judge_task_completion",
                            "judge_relevance", 
                            "judge_quality",
                            "judge_reasoning",  # 完整评价理由
                            # 错误信息
                            "error"
                        ]
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        
                        for r in self.results:
                            judge = r.get("judge_result", {})
                            papers_info = r.get("papers_info", [])
                            extraction = r.get("extraction_result", {})
                            
                            # 提取论文标题和ID
                            paper_titles = [p.get("title", "")[:100] for p in papers_info]
                            paper_ids = [p.get("paper_id", "") for p in papers_info]
                            
                            # 格式化抽取结果
                            extraction_str = ""
                            if extraction:
                                try:
                                    extraction_str = json.dumps(extraction.get("extracted_data", []), ensure_ascii=False, indent=None)
                                except:
                                    extraction_str = str(extraction)
                            
                            writer.writerow({
                                "query_id": r.get("query_id"),
                                "query_type": r.get("query_type"),
                                "query": r.get("query", ""),  # 完整查询
                                "success": r.get("success"),
                                "iterations": r.get("iterations", 0),
                                "tool_count": len(r.get("tools_called", [])),
                                "tools_called": " | ".join(r.get("tools_called", [])),
                                # 论文信息
                                "papers_found": len(papers_info),
                                "paper_titles": " || ".join(paper_titles),
                                "paper_ids": " | ".join(paper_ids),
                                # Agent 输出
                                "agent_output": r.get("agent_output", ""),
                                # 抽取结果
                                "extraction_result": extraction_str,
                                # Judge 评价
                                "judge_score": judge.get("score", 0),
                                "judge_task_completion": judge.get("task_completion", 0),
                                "judge_relevance": judge.get("relevance", 0),
                                "judge_quality": judge.get("quality", 0),
                                "judge_reasoning": judge.get("reasoning", ""),  # 完整理由
                                # 错误
                                "error": r.get("error", "")
                            })
                    break  # 成功就退出重试循环
                except PermissionError:
                    if attempt < max_retries - 1:
                        print(f"⚠️ CSV 文件被占用，{attempt + 1}/{max_retries} 次重试中... (请关闭 Excel)")
                        time.sleep(2)
                    else:
                        print(f"⚠️ CSV 保存失败: 文件被占用。结果已保存到 JSON: {self.json_path}")
                except Exception as e:
                    print(f"⚠️ CSV 保存失败: {e}")
    
    def print_summary(self) -> None:
        """打印总结"""
        total = len(self.results)
        if total == 0:
            print("无实验结果")
            return
        
        successful = sum(1 for r in self.results if r.get("success"))
        avg_score = sum(r.get("judge_result", {}).get("score", 0) for r in self.results) / total
        
        # 分类统计
        search_results = [r for r in self.results if r.get("query_type") == "search"]
        extract_results = [r for r in self.results if r.get("query_type") == "extract"]
        
        print(f"\n{'='*60}")
        print("📊 实验总结")
        print(f"{'='*60}")
        print(f"   总计: {total} 个实验")
        print(f"   成功率: {successful}/{total} ({100*successful/total:.1f}%)")
        print(f"   平均分: {avg_score:.1f}/100")
        if search_results:
            search_avg = sum(r.get("judge_result", {}).get("score", 0) for r in search_results) / len(search_results)
            print(f"   搜索任务: {len(search_results)} 个, 平均分 {search_avg:.1f}/100")
        if extract_results:
            extract_avg = sum(r.get("judge_result", {}).get("score", 0) for r in extract_results) / len(extract_results)
            print(f"   抽取任务: {len(extract_results)} 个, 平均分 {extract_avg:.1f}/100")
        print(f"   结果文件: {self.json_path}")
        print(f"   CSV文件: {self.csv_path}")
        print(f"{'='*60}")


# =============================================================================
# 工具函数 - 清理 papers 目录
# =============================================================================
def _cleanup_papers_dir(papers_dir: Path) -> None:
    """清理 papers 目录下的所有 .md 文件"""
    import time
    import gc
    
    if not papers_dir.exists():
        return
    
    md_files = list(papers_dir.glob("*.md"))
    if not md_files:
        return
    
    # 强制垃圾回收，释放可能的文件句柄
    gc.collect()
    time.sleep(0.1)
    
    deleted = 0
    failed = []
    
    for f in md_files:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if f.exists():
                    f.unlink()
                    time.sleep(0.05)  # 给系统一点时间
                    # 验证是否真的删除了
                    if not f.exists():
                        deleted += 1
                        break
                    else:
                        if attempt < max_retries - 1:
                            time.sleep(0.5)
                        else:
                            failed.append(f"{f.name} (仍存在)")
                else:
                    deleted += 1  # 文件已不存在
                    break
            except PermissionError:
                if attempt < max_retries - 1:
                    gc.collect()  # 尝试释放句柄
                    time.sleep(1)
                else:
                    failed.append(f"{f.name} (权限拒绝)")
            except Exception as e:
                failed.append(f"{f.name}: {e}")
                break
    
    if deleted > 0:
        print(f"🧹 已清理 {deleted} 个 .md 文件")
    if failed:
        print(f"⚠️ 清理失败: {', '.join(failed)}")
    
    # 最终检查
    remaining = list(papers_dir.glob("*.md"))
    if remaining:
        print(f"⚠️ 目录中仍有 {len(remaining)} 个 .md 文件: {[f.name for f in remaining]}")


# =============================================================================
# 辅助函数 - 从工具调用中提取论文信息
# =============================================================================
def _extract_papers_info(tool_calls: list[dict]) -> list[dict]:
    """从工具调用结果中提取论文信息（标题、摘要等）"""
    papers = []
    seen_ids = set()
    
    for tc in tool_calls:
        name = tc.get("name", "")
        result = tc.get("result", "")
        
        # 从 search_papers 结果中提取
        if name == "search_papers" and result:
            try:
                if isinstance(result, str):
                    # 尝试解析 JSON
                    if "papers" in result:
                        import re
                        # 提取 papers 数组
                        data = json.loads(result) if result.startswith("{") else None
                        if data and "papers" in data:
                            for p in data["papers"]:
                                pid = p.get("id", "")
                                if pid and pid not in seen_ids:
                                    seen_ids.add(pid)
                                    papers.append({
                                        "paper_id": pid,
                                        "title": p.get("title", "")[:200],
                                        "abstract": p.get("abstract", "")[:500] if p.get("abstract") else "",
                                        "authors": p.get("authors", [])[:5],  # 最多5个作者
                                        "published": p.get("published", ""),
                                        "categories": p.get("categories", []),
                                    })
            except Exception:
                pass
        
        # 从 read_paper 结果中提取（包含更详细信息）
        elif name == "read_paper" and result:
            try:
                if isinstance(result, str) and result.startswith("{"):
                    data = json.loads(result)
                    pid = data.get("paper_id", "")
                    if pid and pid not in seen_ids:
                        seen_ids.add(pid)
                        # read_paper 的内容可能很长，只取摘要部分
                        content = data.get("content_preview", "") or ""
                        papers.append({
                            "paper_id": pid,
                            "title": data.get("title", "")[:200],
                            "content_preview": content[:1000],  # 截取预览
                        })
            except Exception:
                pass
    
    return papers


def _extract_extraction_result(tool_calls: list[dict]) -> dict | None:
    """从工具调用结果中提取数据抽取结果"""
    for tc in tool_calls:
        name = tc.get("name", "")
        result = tc.get("result", "")
        
        if name == "extract_data_from_papers" and result:
            try:
                if isinstance(result, str) and result.startswith("{"):
                    data = json.loads(result)
                    # 返回完整的抽取结果
                    return {
                        "status": data.get("status", ""),
                        "total_papers": data.get("total_papers", 0),
                        "extracted_data": data.get("extracted_data", []),
                    }
            except Exception:
                pass
    return None


# =============================================================================
# 主实验逻辑
# =============================================================================
async def run_single_query(
    query_info: dict[str, Any],
    settings: Settings,
    papers_dir: Path
) -> dict[str, Any]:
    """运行单个查询实验"""
    query_id = query_info["id"]
    query_type = query_info["type"]
    query = query_info["query"]
    
    print(f"\n{'='*60}")
    print(f"🧪 实验: {query_id} [{query_type.upper()}]")
    print(f"❓ Query: {query[:70]}...")
    print(f"📁 Papers目录: {papers_dir.resolve()}")
    print(f"{'='*60}")
    
    # 清空论文目录（实验开始前）
    _cleanup_papers_dir(papers_dir)
    
    # 构建 state - 包含 papers_dir 路径
    papers_dir_str = str(papers_dir.resolve())  # 使用绝对路径
    
    if query_type == "search":
        state = {
            "goal": query,
            "plan": f"搜索相关论文，不下载和抽取。最多搜索 {MAX_PAPERS} 篇。"
        }
    else:
        state = {
            "goal": query,
            "plan": f"搜索、下载并抽取论文数据。抽取目标根据query确定。最多处理 {MAX_PAPERS} 篇。",
            "papers_dir": papers_dir_str  # 传递论文目录
        }
    
    try:
        async with DataAgent(
            settings=settings,
            max_papers_per_topic=MAX_PAPERS,
            local_papers_dir=papers_dir_str,
        ) as agent:
            # 运行 Agent（使用 autonomous_thinking 获取工具调用信息）
            if query_type == "search":
                prompt = f"""# 研究任务
目标: {state['goal']}
计划: {state['plan']}

请根据任务类型执行相应操作，并返回结果。
"""
            else:
                # extract 类型需要明确告知 papers_dir，使用绝对路径
                prompt = f"""# 研究任务
目标: {state['goal']}
计划: {state['plan']}

## 重要配置 (必须使用以下绝对路径)
- 论文保存目录: {papers_dir_str}

## 执行流程
1. search_papers: 搜索相关论文
2. download_paper: 下载找到的论文
3. read_paper: 读取论文内容
4. save_markdown_locally: 保存到本地，save_path 必须使用绝对路径: {papers_dir_str}/{{paper_id}}.md
5. extract_data_from_papers: 从本地论文提取数据，papers_dir="{papers_dir_str}"

⚠️ 重要：save_path 必须使用完整绝对路径，例如: {papers_dir_str}/2301.12345v1.md
请严格按照上述流程执行，最后必须调用 extract_data_from_papers 工具完成数据抽取。
"""
            result = await agent.autonomous_thinking(
                prompt=prompt,
                state=state,
                max_iterations=MAX_ITERATIONS
            )
            
            # 提取工具调用
            tool_calls = result.get("tool_calls", [])
            tool_names = [tc.get("name", "unknown") for tc in tool_calls]
            
            # 打印工具调用
            print(f"\n📊 工具调用 ({len(tool_calls)} 次):")
            tool_counts = {}
            for name in tool_names:
                tool_counts[name] = tool_counts.get(name, 0) + 1
            for name, count in sorted(tool_counts.items()):
                tool_type = "🌐MCP" if name in {"search_papers", "download_paper", "read_paper"} else "📍Local"
                print(f"   [{tool_type}] {name}: {count}x")
            
            # 提取论文信息和抽取结果
            papers_info = _extract_papers_info(tool_calls)
            extraction_result = _extract_extraction_result(tool_calls)
            
            return {
                "query_id": query_id,
                "query": query,
                "query_type": query_type,
                "tools_called": tool_names,
                "tool_call_details": tool_calls,
                "agent_output": result.get("response", ""),
                "papers_info": papers_info,  # 论文标题、摘要等
                "extraction_result": extraction_result,  # 抽取结果
                "iterations": result.get("iterations", 0),
                "success": True,
                "error": None
            }
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "query_id": query_id,
            "query": query,
            "query_type": query_type,
            "tools_called": [],
            "tool_call_details": [],
            "agent_output": "",
            "papers_info": [],  # 空列表
            "extraction_result": None,  # 无抽取结果
            "iterations": 0,
            "success": False,
            "error": str(e)
        }


async def run_all_experiments(skip_completed: bool = True) -> None:
    """运行所有实验"""
    # 初始化
    settings = get_test_settings()
    if not settings.llm.is_valid():
        print("❌ LLM 未配置！请设置 LLM_API_KEY")
        return
    
    # 打印模型信息
    print_model_info(settings)
    
    print(f"{'='*60}")
    print("🚀 DataAgent 批量实验")
    print(f"{'='*60}")
    print(f"   查询数量: {len(TEST_QUERIES)}")
    print(f"   最大论文数: {MAX_PAPERS}")
    print(f"   最大迭代数: {MAX_ITERATIONS}")
    print(f"{'='*60}")
    
    logger = ExperimentLogger()
    judge = LLMJudge()
    papers_dir = OUTPUT_DIR / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    
    completed_ids = logger.get_completed_ids() if skip_completed else set()
    if completed_ids:
        print(f"⏭️ 跳过 {len(completed_ids)} 个已完成的查询")
    
    # 运行实验
    for i, q in enumerate(TEST_QUERIES, 1):
        query_id = q["id"]
        
        if query_id in completed_ids:
            print(f"\n⏭️ [{i}/{len(TEST_QUERIES)}] {query_id} - 已完成，跳过")
            continue
        
        print(f"\n\n{'#'*60}")
        print(f"# [{i}/{len(TEST_QUERIES)}] 运行: {query_id}")
        print(f"{'#'*60}")
        
        try:
            # 运行实验
            result = await run_single_query(q, settings, papers_dir)
            
            # LLM Judge 评估
            if result["success"]:
                print(f"\n🔍 LLM Judge 评估中...")
                judge_result = await judge.check_result_quality(
                    user_query=q["query"],
                    query_type=q["type"],
                    agent_output=result["agent_output"],
                    tools_called=result["tools_called"],
                    tool_results=result.get("tool_results")
                )
                result["judge_result"] = judge_result
                print(f"   📊 总分: {judge_result['score']}/100")
                print(f"   📋 任务完成: {judge_result['task_completion']}/40 | 相关性: {judge_result['relevance']}/30 | 质量: {judge_result['quality']}/30")
                print(f"   💬 理由: {judge_result['reasoning'][:100]}..." if len(judge_result['reasoning']) > 100 else f"   💬 理由: {judge_result['reasoning']}")
            else:
                result["judge_result"] = {
                    "score": 0,
                    "task_completion": 0,
                    "relevance": 0,
                    "quality": 0,
                    "reasoning": f"实验失败: {result['error']}"
                }
            
            # 保存结果
            logger.add_result(result)
            
        except Exception as e:
            import traceback
            print(f"❌ 意外错误: {e}")
            traceback.print_exc()
            logger.add_result({
                "query_id": query_id,
                "query": q["query"],
                "query_type": q["type"],
                "tools_called": [],
                "agent_output": "",
                "success": False,
                "error": str(e),
                "judge_result": {"is_reasonable": False, "score": 0, "reasoning": f"崩溃: {e}"}
            })
        
        # 清理 papers 目录（每个 query 运行后）
        _cleanup_papers_dir(papers_dir)
        
        # 延迟避免限流
        await asyncio.sleep(1)
    
    # 打印总结
    logger.print_summary()


# =============================================================================
# 命令行入口
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="DataAgent 批量实验")
    parser.add_argument("--query", "-q", type=str, help="运行单个查询 (如 Q001)")
    parser.add_argument("--no-skip", action="store_true", help="不跳过已完成的查询")
    parser.add_argument("--list", action="store_true", help="列出所有测试查询")
    args = parser.parse_args()
    
    if args.list:
        print("\n📋 测试查询列表:")
        for q in TEST_QUERIES:
            print(f"   {q['id']} [{q['type']}]: {q['query'][:50]}...")
        return
    
    if args.query:
        query = next((q for q in TEST_QUERIES if q["id"] == args.query), None)
        if not query:
            print(f"❌ 未找到查询: {args.query}")
            return
        
        async def run_one():
            settings = get_test_settings()
            
            # 打印模型信息
            print_model_info(settings)
            
            papers_dir = OUTPUT_DIR / "papers"
            papers_dir.mkdir(parents=True, exist_ok=True)
            logger = ExperimentLogger()
            judge = LLMJudge()
            
            result = await run_single_query(query, settings, papers_dir)
            if result["success"]:
                judge_result = await judge.check_result_quality(
                    user_query=query["query"], 
                    query_type=query["type"], 
                    agent_output=result["agent_output"], 
                    tools_called=result["tools_called"],
                    tool_results=result.get("tool_results")
                )
                result["judge_result"] = judge_result
                print(f"\n📊 Judge 评分: {judge_result['score']}/100")
                print(f"   📋 任务完成: {judge_result['task_completion']}/40 | 相关性: {judge_result['relevance']}/30 | 质量: {judge_result['quality']}/30")
            logger.add_result(result)
            
            # 清理 papers 目录
            _cleanup_papers_dir(papers_dir)
        
        asyncio.run(run_one())
    else:
        asyncio.run(run_all_experiments(skip_completed=not args.no_skip))


if __name__ == "__main__":
    main()
