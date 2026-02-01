#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fab_agent_experiment.py
FabAgent 批量实验脚本

实验目标：
1. 测试工具调用是否正常（打印调用的工具列表）
2. 使用 LLM Judge 判定输出内容是否合理

实验类型：
- Type A: 单组分预测（简单的 ABX3 钙钛矿）
- Type B: 混合组分预测（复杂的混合阳离子/卤素钙钛矿）
- Type C: 多材料比较与趋势分析

Usage:
    cd f:\PSC_Agents\experiment\single_point\fab_agent
    python fab_agent_experiment.py
    python fab_agent_experiment.py --query Q001   # 运行单个查询
    python fab_agent_experiment.py --list         # 列出所有查询
    
Author: PSC_Agents Team
Date: 2026-01-30
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
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "core"))

# 加载 .env 文件
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from core.config import Settings, LLMConfig, ProviderConfig
from core.llm import LLMClient

# 直接从模块导入，避免触发 agent/__init__.py 的全部导入
import importlib.util
spec = importlib.util.spec_from_file_location(
    "fab_agent", 
    PROJECT_ROOT / "src" / "agent" / "fab_agent.py"
)
fab_agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fab_agent_module)
FabAgent = fab_agent_module.FabAgent

# =============================================================================
# 实验参数配置
# =============================================================================
MAX_ITERATIONS = 15     # 智能体最大迭代次数（多材料比较需要更多轮次）
OUTPUT_DIR = Path(__file__).parent / "experiment_output"

# =============================================================================
# 测试 Query 列表 (35个)
# 格式: {"id": str, "type": str, "query": str}
# =============================================================================
TEST_QUERIES = [
    # === Type A: 单组分基准预测 (Q001-Q006) ===
    {
        "id": "Q001",
        "type": "single",
        "query": "请预测标杆材料 MAPbI3 的光电转换效率 (PCE) 和带隙。"
    },
    {
        "id": "Q002",
        "type": "single",
        "query": "评估纯 FAPbI3 的热力学稳定性 (E_hull)，判断其是否容易发生相变。"
    },
    {
        "id": "Q003",
        "type": "single",
        "query": "帮我计算全无机钙钛矿 CsPbI3 的理论开路电压 (Voc) 和填充因子 (FF)。"
    },
    {
        "id": "Q004",
        "type": "single",
        "query": "预测锡基钙钛矿 MASnI3 的短路电流密度 (Jsc) 是多少？"
    },
    {
        "id": "Q005",
        "type": "single",
        "query": "CsPbBr3 是一种宽带隙材料，请预测它的具体带隙数值。"
    },
    {
        "id": "Q006",
        "type": "single",
        "query": "预测双钙钛矿材料 Cs2AgBiBr6 的光伏性能指标。"
    },
    
    # === Type B: 混合阳离子预测 (Q007-Q014) ===
    {
        "id": "Q007",
        "type": "mixed_cation",
        "query": "预测混合阳离子配方 FA0.8Cs0.2PbI3 的 PCE 能否超过 23%？"
    },
    {
        "id": "Q008",
        "type": "mixed_cation",
        "query": "比较 FA0.9Cs0.1PbI3 的 E_hull 值，看掺杂 10% Cs 是否降低了形成能（提高稳定性）。"
    },
    {
        "id": "Q009",
        "type": "mixed_cation",
        "query": "请计算经典双阳离子配方 FA0.85MA0.15PbI3 的所有 6 项性能参数。"
    },
    {
        "id": "Q010",
        "type": "mixed_cation",
        "query": "预测三阳离子配方 Cs0.05FA0.79MA0.16PbI3 的开路电压和填充因子。"
    },
    {
        "id": "Q011",
        "type": "mixed_cation",
        "query": "引入微量铷离子：预测 Rb0.05Cs0.05FA0.9PbI3 的效率变化。"
    },
    {
        "id": "Q012",
        "type": "mixed_cation",
        "query": "评估富铯配方 Cs0.4FA0.6PbI3 的带隙是否适合用于室内光伏（宽带隙）。"
    },
    {
        "id": "Q013",
        "type": "mixed_cation",
        "query": "预测无甲胺 (MA-free) 配方 Cs0.3FA0.7PbI3 的短路电流密度。"
    },
    {
        "id": "Q014",
        "type": "mixed_cation",
        "query": "分析 FA0.98Cs0.02PbI3 这种微掺杂配方与纯 FAPbI3 的性能差异。"
    },
    
    # === Type C: 混合卤素预测 (Q015-Q020) ===
    {
        "id": "Q015",
        "type": "mixed_halide",
        "query": "为了匹配叠层顶电池，请预测混合卤素 MAPb(I0.8Br0.2)3 的带隙是多少？"
    },
    {
        "id": "Q016",
        "type": "mixed_halide",
        "query": "预测高溴含量 CsPb(I0.6Br0.4)3 的开路电压 (Voc)，它是否超过了 1.3V?"
    },
    {
        "id": "Q017",
        "type": "mixed_halide",
        "query": "添加氯能否改善性能？请预测 MAPbI2.9Cl0.1 的效率和稳定性。"
    },
    {
        "id": "Q018",
        "type": "mixed_halide",
        "query": "预测 FAPb(Br0.1I0.9)3 的带隙和短路电流密度。"
    },
    {
        "id": "Q019",
        "type": "mixed_halide",
        "query": "评估全无机混合卤素 CsPb(I0.5Br0.5)3 的热力学稳定性 (E_hull)。"
    },
    {
        "id": "Q020",
        "type": "mixed_halide",
        "query": "预测 FA0.8Cs0.2Pb(I0.7Br0.3)3 这种复杂双位掺杂配方的 PCE。"
    },
    
    # === Type D: 目标验证与筛选 (Q021-Q022) ===
    {
        "id": "Q021",
        "type": "target_verify",
        "query": "我需要 1.7eV 带隙的材料，请验证 CsPb(I0.6Br0.4)3 是否符合要求。"
    },
    {
        "id": "Q022",
        "type": "target_verify",
        "query": "预测锡铅混合窄带隙材料 MASn0.5Pb0.5I3 的 Jsc 和带隙。"
    },
    
    # === Type E: 多材料比较 (Q023-Q028) ===
    {
        "id": "Q023",
        "type": "comparison",
        "query": "比较以下三种材料的 PCE，告诉我哪个最高：MAPbI3, FAPbI3, CsPbI3。"
    },
    {
        "id": "Q024",
        "type": "comparison",
        "query": "在 FA0.8Cs0.2PbI3 和 FA0.9Cs0.1PbI3 之间，哪个具有更低的 E_hull（理论上更稳定）？"
    },
    {
        "id": "Q025",
        "type": "comparison",
        "query": "请列出 CsPb(IxBr1-x)3 系列中，随着 Br 含量增加，带隙的变化趋势。"
    },
    {
        "id": "Q026",
        "type": "comparison",
        "query": "筛选高电压材料：在 CsPbBr3, MAPbBr3, FAPbBr3 中，谁的 Voc 预测值最高？"
    },
    {
        "id": "Q027",
        "type": "comparison",
        "query": "批量预测：请一次性给出 MA0.5FA0.5PbI3 和 MA0.7FA0.3PbI3 的所有性能指标。"
    },
    {
        "id": "Q028",
        "type": "comparison",
        "query": "寻找带隙最宽的组合：比较 CsPbI3, CsPbBr3, CsPbCl3 的带隙预测值。"
    },
    
    # === Type F: 综合评估与可视化 (Q029-Q035) ===
    {
        "id": "Q029",
        "type": "evaluation",
        "query": "综合评估：在 MAPbI3 和 MASnI3 中，考虑到 PCE 和稳定性平衡，推荐哪一个？"
    },
    {
        "id": "Q030",
        "type": "visualization",
        "query": "请可视化 FAxCs1-xPbI3 系列 (x=0.8, 0.9, 1.0) 的 PCE 预测结果图表。"
    },
    {
        "id": "Q031",
        "type": "visualization",
        "query": "模拟 FA(1-x)Cs(x)PbI3 系列 (x=0, 0.1, 0.2, 0.3) 的性能，并绘制带隙 (Bandgap) 随 Cs 含量变化的趋势图。"
    },
    {
        "id": "Q032",
        "type": "visualization",
        "query": "预测 MAPb(I(1-y)Br(y))3 混合卤素体系 (y 从 0 到 1) 的 Voc，并生成折线图以观察开路电压是否随 Br 含量线性增加。"
    },
    {
        "id": "Q033",
        "type": "visualization",
        "query": "随着 Sn 含量在 MASn(x)Pb(1-x)I3 中增加，预测短路电流密度 (Jsc) 的变化，并可视化该趋势。"
    },
    {
        "id": "Q034",
        "type": "visualization",
        "query": "研究 Rb 掺杂对 Cs0.05FA0.95PbI3 效率的影响 (0% 到 5%)，并画出 PCE 的变化曲线。"
    },
    {
        "id": "Q035",
        "type": "visualization",
        "query": "预测 FAPbI3 在引入少量 Cl 掺杂后的结晶能变化，并用图表展示稳定性是否提升。"
    },
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
    judge_model = os.getenv("LLM_MODEL_ID", "gpt-5.2")
    
    config = LLMConfig(
        provider="openai",
        temperature=0.3,
        max_tokens=1000,
        timeout=60.0,
    )
    config.openai = ProviderConfig(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        model=judge_model,
    )
    return config


def print_model_info(settings: Settings):
    """打印当前使用的模型信息"""
    llm_config = settings.llm
    judge_model = os.getenv("LLM_MODEL_ID", "gpt-5.2")
    print(f"\n{'='*60}")
    print(f"📌 模型配置信息")
    print(f"{'='*60}")
    print(f"   🤖 FabAgent 模型: {llm_config.model_name}")
    print(f"   🌡️  Temperature: {llm_config.temperature}")
    print(f"   🔗 Base URL: {llm_config.base_url[:50]}...")
    print(f"   ⚖️  Judge 模型: {judge_model}")
    print(f"{'='*60}\n")


# =============================================================================
# LLM Judge - 评估输出质量 (100分制)
# =============================================================================
class LLMJudge:
    """LLM 裁判：评估 FabAgent 输出质量"""
    
    JUDGE_SYSTEM_PROMPT = """你是一个公正的AI虚拟制造智能体评估者。
你的任务是根据智能体对用户查询的响应质量进行评分。

## 评分系统 (0-100分):

### 维度1: 任务完成度 (0-40分)
- 智能体是否调用了预测工具？
- 智能体是否提供了关键指标的数值预测？
- 如果请求可视化，智能体是否尝试生成图表？

### 维度2: 输出相关性 (0-30分)
- 预测的指标是否与查询相关？
- 智能体是否正确使用了化学式？
- 预测值是否在钙钛矿材料的合理范围内？
  * PCE: 通常 5-25%（优化组分可更高）
  * Voc: 通常 0.6-1.2 V
  * Jsc: 通常 15-26 mA/cm²
  * FF: 通常 50-85%
  * 带隙: 通常 1.2-3.0 eV

### 维度3: 输出质量 (0-30分)
- 预测结果是否清晰展示并带有单位？
- 智能体是否提供了科学解读？
- 如果是比较任务，是否给出了明确结论？

## 评分指南:
- 90-100: 优秀 - 完整预测，分析清晰，单位正确
- 70-89: 良好 - 提供了预测，展示有小瑕疵
- 50-69: 及格 - 基本预测但分析不足或不清晰
- 30-49: 较差 - 预测不完整或数值不合理
- 0-29: 失败 - 无预测或完全错误

## 重要说明:
- 该智能体使用训练好的随机森林模型进行钙钛矿性能预测
- 预测可能有不确定性，合理范围内的值都可接受
- 零幻觉很重要 - 如果模型失败，诚实报错是好的行为

请仅以JSON格式回复:
{
    "score": 0-100,
    "task_completion": 0-40,
    "relevance": 0-30,
    "quality": 0-30,
    "reasoning": "简要说明"
}
"""

    def __init__(self):
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
        
        # 任务类型中文描述
        type_desc_map = {
            "single": "🔬 单组分预测 - 简单 ABX3 钙钛矿",
            "mixed_cation": "🧬 混合阳离子预测 - A位掺杂钙钛矿",
            "mixed_halide": "🧪 混合卤素预测 - X位掺杂钙钛矿",
            "target_verify": "🎯 目标验证 - 验证是否满足特定指标",
            "comparison": "📊 多材料比较 - 多种组分对比分析",
            "evaluation": "⚖️ 综合评估 - 多维度权衡推荐",
            "visualization": "📈 可视化任务 - 趋势图表生成",
        }
        task_desc = type_desc_map.get(query_type, f"❓ 其他类型: {query_type}")
        
        # 构建评估 Prompt
        eval_prompt = f"""## 任务类型
{task_desc}

## 用户查询
{user_query}

## 智能体工具调用
{json.dumps(tool_summary, indent=2, ensure_ascii=False)}
总调用次数: {len(tools_called)}

## 智能体最终输出
{agent_output[:5000] if agent_output else "(无输出)"}

---
请评估智能体的表现并给出100分制评分。
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
                "task_completion": 0, 
                "relevance": 0, 
                "quality": 0, 
                "reasoning": f"Judge 错误: {e}"
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
                "task_completion": result.get("task_completion", 0),
                "relevance": result.get("relevance", 0),
                "quality": result.get("quality", 0),
                "reasoning": result.get("reasoning", "无说明")
            }
        except Exception as e:
            return {
                "score": 0, 
                "task_completion": 0, 
                "relevance": 0, 
                "quality": 0, 
                "reasoning": f"解析错误: {e}"
            }


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
        """保存到 JSON 和 CSV"""
        import time
        
        # JSON
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "experiment_info": {
                        "timestamp": datetime.now().isoformat(),
                        "max_iterations": MAX_ITERATIONS,
                        "total": len(self.results)
                    },
                    "experiments": self.results
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ JSON 保存失败: {e}")
        
        # CSV - 方便人类专家查看
        if self.results:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with open(self.csv_path, "w", encoding="utf-8-sig", newline="") as f:
                        fieldnames = [
                            "查询ID", 
                            "任务类型", 
                            "查询内容",
                            "是否成功", 
                            "迭代次数",
                            "工具调用次数",
                            "调用的工具",
                            # 智能体输出
                            "智能体输出",
                            # Judge 评价
                            "总分",
                            "任务完成分",
                            "相关性分", 
                            "质量分",
                            "评价理由",
                            # 错误信息
                            "错误信息"
                        ]
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        
                        for r in self.results:
                            judge = r.get("judge_result", {})
                            
                            writer.writerow({
                                "查询ID": r.get("query_id"),
                                "任务类型": r.get("query_type"),
                                "查询内容": r.get("query", ""),
                                "是否成功": "✅" if r.get("success") else "❌",
                                "迭代次数": r.get("iterations", 0),
                                "工具调用次数": len(r.get("tools_called", [])),
                                "调用的工具": " | ".join(r.get("tools_called", [])),
                                # 智能体输出（截取前2000字符）
                                "智能体输出": r.get("agent_output", "")[:2000],
                                # Judge 评价
                                "总分": judge.get("score", 0),
                                "任务完成分": judge.get("task_completion", 0),
                                "相关性分": judge.get("relevance", 0),
                                "质量分": judge.get("quality", 0),
                                "评价理由": judge.get("reasoning", ""),
                                # 错误
                                "错误信息": r.get("error", "")
                            })
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        print(f"⚠️ CSV 文件被占用，第 {attempt + 1}/{max_retries} 次重试...")
                        time.sleep(2)
                    else:
                        print(f"⚠️ CSV 保存失败: 文件被占用。结果已保存到 JSON")
                except Exception as e:
                    print(f"⚠️ CSV 保存失败: {e}")
                    break
    
    def print_summary(self) -> None:
        """打印总结"""
        total = len(self.results)
        if total == 0:
            print("无实验结果")
            return
        
        successful = sum(1 for r in self.results if r.get("success"))
        avg_score = sum(r.get("judge_result", {}).get("score", 0) for r in self.results) / total
        
        # 按类型统计
        type_stats = {}
        for r in self.results:
            qtype = r.get("query_type", "unknown")
            if qtype not in type_stats:
                type_stats[qtype] = {"count": 0, "total_score": 0}
            type_stats[qtype]["count"] += 1
            type_stats[qtype]["total_score"] += r.get("judge_result", {}).get("score", 0)
        
        # 类型中文名映射
        type_names = {
            "single": "单组分预测",
            "mixed_cation": "混合阳离子",
            "mixed_halide": "混合卤素",
            "target_verify": "目标验证",
            "comparison": "多材料比较",
            "evaluation": "综合评估",
            "visualization": "可视化任务",
        }
        
        print(f"\n{'='*60}")
        print("📊 FabAgent 实验总结")
        print(f"{'='*60}")
        print(f"   总计: {total} 个实验")
        print(f"   成功率: {successful}/{total} ({100*successful/total:.1f}%)")
        print(f"   平均分: {avg_score:.1f}/100")
        print(f"\n   📋 分类统计:")
        for qtype, stats in sorted(type_stats.items()):
            type_name = type_names.get(qtype, qtype)
            avg = stats["total_score"] / stats["count"] if stats["count"] > 0 else 0
            print(f"      {type_name}: {stats['count']} 个, 平均分 {avg:.1f}/100")
        
        print(f"\n   📁 结果文件: {self.json_path}")
        print(f"   📁 CSV文件: {self.csv_path}")
        print(f"{'='*60}")


# =============================================================================
# 辅助函数 - 从工具调用中提取预测结果
# =============================================================================
def _extract_predictions(tool_calls: list[dict], tool_results: list[dict]) -> dict[str, Any]:
    """从工具调用结果中提取预测值"""
    predictions = {}
    
    for i, tc in enumerate(tool_calls):
        name = tc.get("name", "")
        
        if name == "predict_perovskite" and i < len(tool_results):
            result_str = tool_results[i].get("result", "")
            try:
                if isinstance(result_str, str) and result_str.startswith("{"):
                    data = json.loads(result_str)
                    if "predictions" in data:
                        predictions = data["predictions"]
                        break
            except Exception:
                pass
    
    return predictions


# =============================================================================
# 主实验逻辑
# =============================================================================
async def run_single_query(
    query_info: dict[str, Any],
    settings: Settings
) -> dict[str, Any]:
    """运行单个查询实验"""
    query_id = query_info["id"]
    query_type = query_info["type"]
    query = query_info["query"]
    
    print(f"\n{'='*60}")
    print(f"🧪 实验: {query_id} [{query_type}]")
    print(f"❓ 查询: {query}")
    print(f"{'='*60}")
    
    # 设置输出目录 - 保存在当前实验目录下
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建 state
    state = {
        "goal": query,
        "plan": "根据用户查询预测钙钛矿材料性能",
        "experimental_params": {}
    }
    
    try:
        # 使用 async with 正确初始化 Agent（包括 LLM client）
        async with FabAgent(settings=settings, output_dir=str(output_dir)) as agent:
            # 设置 query_id 用于文件命名
            agent.set_query_id(query_id)
            
            # 构建 prompt - 直接使用用户查询
            prompt = f"""# 预测任务

{query}

请根据用户的需求灵活使用工具完成任务。
预测完成后，请给出科学解读和结论。
"""
            
            # 运行 Agent
            result = await agent.autonomous_thinking(
                prompt=prompt,
                state=state,
                max_iterations=MAX_ITERATIONS
            )
            
            # 提取工具调用
            tool_calls = result.get("tool_calls", [])
            tool_results = result.get("tool_results", [])
            tool_names = [tc.get("name", "unknown") for tc in tool_calls]
            
            # 打印工具调用
            print(f"\n📊 工具调用 ({len(tool_calls)} 次):")
            tool_counts = {}
            for name in tool_names:
                tool_counts[name] = tool_counts.get(name, 0) + 1
            for name, count in sorted(tool_counts.items()):
                print(f"   [📍Local] {name}: {count}x")
            
            # 提取预测结果
            predictions = _extract_predictions(tool_calls, tool_results)
            
            return {
                "query_id": query_id,
                "query": query,
                "query_type": query_type,
                "tools_called": tool_names,
                "tool_call_details": tool_calls,
                "agent_output": result.get("response", ""),
                "predictions": predictions,
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
            "predictions": {},
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
    print("🚀 FabAgent 批量实验")
    print(f"{'='*60}")
    print(f"   查询数量: {len(TEST_QUERIES)}")
    print(f"   最大迭代数: {MAX_ITERATIONS}")
    print(f"{'='*60}")
    
    logger = ExperimentLogger()
    judge = LLMJudge()
    
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
            result = await run_single_query(q, settings)
            
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
                reasoning = judge_result['reasoning']
                print(f"   💬 理由: {reasoning[:100]}..." if len(reasoning) > 100 else f"   💬 理由: {reasoning}")
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
                "predictions": {},
                "success": False,
                "error": str(e),
                "judge_result": {
                    "score": 0, 
                    "task_completion": 0,
                    "relevance": 0,
                    "quality": 0,
                    "reasoning": f"崩溃: {e}"
                }
            })
        
        # 延迟避免限流
        await asyncio.sleep(1)
    
    # 打印总结
    logger.print_summary()


# =============================================================================
# 命令行入口
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="FabAgent 批量实验")
    parser.add_argument("--query", "-q", type=str, help="运行单个查询 (如 Q001)")
    parser.add_argument("--no-skip", action="store_true", help="不跳过已完成的查询")
    parser.add_argument("--list", action="store_true", help="列出所有测试查询")
    args = parser.parse_args()
    
    if args.list:
        print("\n📋 测试查询列表 (35个):")
        print("-" * 80)
        current_type = None
        for q in TEST_QUERIES:
            if q["type"] != current_type:
                current_type = q["type"]
                type_names = {
                    "single": "📦 单组分预测",
                    "mixed_cation": "🧬 混合阳离子",
                    "mixed_halide": "🧪 混合卤素",
                    "target_verify": "🎯 目标验证",
                    "comparison": "📊 多材料比较",
                    "evaluation": "⚖️ 综合评估",
                    "visualization": "📈 可视化任务",
                }
                print(f"\n{type_names.get(current_type, current_type)}:")
            print(f"   {q['id']}: {q['query'][:60]}...")
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
            
            logger = ExperimentLogger()
            judge = LLMJudge()
            
            result = await run_single_query(query, settings)
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
        
        asyncio.run(run_one())
    else:
        asyncio.run(run_all_experiments(skip_completed=not args.no_skip))


if __name__ == "__main__":
    main()
