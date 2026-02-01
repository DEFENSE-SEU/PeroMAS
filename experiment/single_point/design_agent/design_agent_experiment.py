#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
design_agent_experiment.py
DesignAgent 批量实验脚本

实验目标：
1. 测试工具调用是否正常（打印调用的工具列表）
2. 使用 LLM Judge 判定输出内容是否合理
3. 保存完整的工具调用和输出记录

执行模式：
- mock: 使用模拟结果进行快速本地测试
- interactive: 等待服务器结果（通过终端输入）

实验类型：
- material_design: 材料设计
- synthesizability: 可合成性检验
- synthesis_method: 合成方法预测
- precursor_design: 前驱体设计
- full_recipe: 完整配方设计
- optimization: 优化任务
- complex: 复杂场景

Usage:
    cd f:\\PSC_Agents\\experiment\\single_point\\design_agent
    python design_agent_experiment.py
    python design_agent_experiment.py --mode interactive  # 与服务器交互
    python design_agent_experiment.py --query Q001       # 运行单个查询
    python design_agent_experiment.py --query Q001,Q006  # 运行多个查询
    python design_agent_experiment.py --list             # 列出所有查询
    python design_agent_experiment.py --no-skip          # 不跳过已完成的
    
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
sys.path.insert(0, str(PROJECT_ROOT / "mcp" / "design_agent"))

# 加载 .env 文件
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from core.config import Settings, LLMConfig, ProviderConfig
from core.llm import LLMClient

# 直接从模块导入，避免触发 agent/__init__.py 的全部导入
import importlib.util
spec = importlib.util.spec_from_file_location(
    "design_agent", 
    PROJECT_ROOT / "src" / "agent" / "design_agent.py"
)
design_agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(design_agent_module)
DesignAgent = design_agent_module.DesignAgent

# =============================================================================
# 实验参数配置
# =============================================================================
MAX_ITERATIONS = 5     # 智能体最大迭代次数
OUTPUT_DIR = Path(__file__).parent / "experiment_output"

# =============================================================================
# 测试 Query 列表 (40个) - 基于实际服务器工具能力设计
# 工具: generate_material_structure (MatterGen), check_synthesizability, 
#       predict_synthesis_method, predict_precursors (CSLLM)
# =============================================================================
TEST_QUERIES = [
    # =========================================================================
    # Type A: 材料生成 + 合成性检验 (Q001-Q012)
    # 预期工具链: generate_material_structure → check_synthesizability
    # =========================================================================
    {"id": "Q001", "type": "generate_and_check", 
     "query": "请生成一种 PCE > 24% 的钙钛矿材料，并立即判断其是否可以合成。"},
    {"id": "Q002", "type": "generate_and_check",
     "query": "设计一种带隙为 1.7 eV 的宽带隙材料，并验证其合成可行性。"},
    {"id": "Q003", "type": "generate_and_check",
     "query": "我需要一种高 Voc (>1.2V) 的钙钛矿候选物，请生成多个结构并筛选出能够合成的那些。"},
    {"id": "Q004", "type": "generate_and_check",
     "query": "生成 Jsc > 25 mA/cm² 的钙钛矿晶体结构，并确认该结构在实验上是否可合成。"},
    {"id": "Q005", "type": "generate_and_check",
     "query": "设计一种无铅双钙钛矿结构，并专门运行合成性检查工具以排除不可合成的假想材料。"},
    {"id": "Q006", "type": "generate_and_check",
     "query": "生成 Hull Energy < 0.02 eV/atom 的稳定钙钛矿，并给出明确结论：它能被合成吗？"},
    {"id": "Q007", "type": "generate_and_check",
     "query": "针对室内光伏应用设计带隙 1.9 eV 的钙钛矿吸收层，如果被判断为不可合成，请自动忽略。"},
    {"id": "Q008", "type": "generate_and_check",
     "query": "探索基于 Sn-Ge 混合的窄带隙钙钛矿，并检查该化学计量比是否具备合成可行性。"},
    {"id": "Q009", "type": "generate_and_check",
     "query": "生成填充因子 FF > 80% 的钙钛矿，并判断其合成性。"},
    {"id": "Q010", "type": "generate_and_check",
     "query": "查找一种理论 PCE > 26% 的极限效率钙钛矿，并告诉我它在现实中是'可合成'还是'不可合成'。"},
    {"id": "Q011", "type": "generate_and_check",
     "query": "设计一种适合叠层电池底部的 1.2 eV 锡铅钙钛矿，并验证其是否存在合成障碍。"},
    {"id": "Q012", "type": "generate_and_check",
     "query": "评估一下 Cs0.9Rb0.1PbI3 这种掺杂组分在热力学上是否稳定，我只需要知道它在实验中能不能被合成出来。"},
    
    # =========================================================================
    # Type B: 单独材料生成 (Q013-Q017)
    # 预期工具链: generate_material_structure (无需检验合成性)
    # =========================================================================
    {"id": "Q013", "type": "material_generation",
     "query": "帮我设计一种带隙在 1.4eV 左右的锡铅混合钙钛矿。"},
    {"id": "Q014", "type": "material_generation",
     "query": "生成含有 FA/Cs 混合阳离子的钙钛矿结构，并检查其是否通过了合成性验证。"},
    {"id": "Q015", "type": "material_generation",
     "query": "我想要一种高稳定性的全无机钙钛矿，请生成结构并输出 Yes/No 的合成判断结果。"},
    {"id": "Q016", "type": "material_generation",
     "query": "尝试生成一种全新的高熵钙钛矿，并判定其是否属于可合成的材料范畴。"},
    {"id": "Q017", "type": "material_generation",
     "query": "对比生成两种 PCE=22% 的钙钛矿（I基和Br基），并检查这两种材料是否都能合成。"},
    
    # =========================================================================
    # Type C: 合成方法预测 (Q018-Q022)
    # 预期工具链: predict_synthesis_method
    # =========================================================================
    {"id": "Q018", "type": "synthesis_method",
     "query": "请为 CsPbI3 钙钛矿推荐一套标准合成工艺参数。"},
    {"id": "Q019", "type": "synthesis_method",
     "query": "如果我想制备 Cs2AgBiBr6 双钙钛矿，请预测最佳方法。"},
    {"id": "Q020", "type": "synthesis_method",
     "query": "针对大面积制备场景，请预测适合 MAPbI3 钙钛矿的工业级合成路线。"},
    {"id": "Q021", "type": "synthesis_method",
     "query": "给定目标产物 CsPbBr3 钙钛矿，请反向推导其反应前驱体和溶剂体系。"},
    {"id": "Q022", "type": "synthesis_method",
     "query": "为合成 FAPbI3 钙钛矿，请预测最佳合成方法。"},
    
    # =========================================================================
    # Type D: 前驱体预测 (Q023-Q027)
    # 预期工具链: predict_precursors
    # =========================================================================
    {"id": "Q023", "type": "precursor_prediction",
     "query": "预测合成 MAPbI3 钙钛矿所需的所有前驱体化学品清单。"},
    {"id": "Q024", "type": "precursor_prediction",
     "query": "请推荐合成混合卤素 CsPb(I0.5Br0.5)3 钙钛矿的卤化物前驱体比例。"},
    {"id": "Q025", "type": "precursor_prediction",
     "query": "预测制备 Sn 基钙钛矿 CsSnI3 所需的特殊前驱体。"},
    {"id": "Q026", "type": "precursor_prediction",
     "query": "为合成 FA0.9Cs0.1PbI3 钙钛矿，请预测所需前驱体。"},
    {"id": "Q027", "type": "precursor_prediction",
     "query": "预测合成 (FAPbI3)0.95(MAPbBr3)0.05 混合钙钛矿需要哪些前驱体。"},
    
    # =========================================================================
    # Type E: 完整链路 - 生成→检验→方法→前驱体 (Q028-Q040)
    # 预期工具链: generate → check → method → precursors (多工具组合)
    # =========================================================================
    {"id": "Q028", "type": "full_pipeline",
     "query": "请设计一种 PCE > 23% 的钙钛矿，判断其能否合成，如果答案是 Yes，请预测其合成方法。"},
    {"id": "Q029", "type": "full_pipeline",
     "query": "生成一种带隙 1.5 eV 的钙钛矿材料，如果合成检查通过，请给出合成所需的前驱体列表。"},
    {"id": "Q030", "type": "full_pipeline",
     "query": "寻找一种高稳定性的无铅钙钛矿，确认其可合成后，推荐一步旋涂法的前驱体配方。"},
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
        max_tokens=1500,
        timeout=60.0,
    )
    config.openai = ProviderConfig(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        model=judge_model,
    )
    return config


def print_model_info(settings: Settings, mode: str):
    """打印当前使用的模型信息"""
    llm_config = settings.llm
    judge_model = os.getenv("LLM_MODEL_ID", "gpt-5.2")
    print(f"\n{'='*60}")
    print(f"📌 模型配置信息")
    print(f"{'='*60}")
    print(f"   🤖 DesignAgent 模型: {llm_config.model_name}")
    print(f"   🌡️  Temperature: {llm_config.temperature}")
    print(f"   🔗 Base URL: {llm_config.base_url[:50]}...")
    print(f"   ⚖️  Judge 模型: {judge_model}")
    print(f"   🔧 执行模式: {mode.upper()}")
    print(f"{'='*60}\n")


# =============================================================================
# LLM Judge - 评估输出质量 (100分制)
# =============================================================================
class LLMJudge:
        """LLM 裁判：评估 DesignAgent 输出质量"""
    
        JUDGE_SYSTEM_PROMPT = """你是一个公正的AI钙钛矿材料设计智能体评估者。
你的任务是根据智能体对用户查询的响应质量进行评分。

## 可用工具说明 (共4个服务器工具):
1. **generate_material_structure** (MatterGen): 根据目标性能生成钙钛矿晶体结构
   - 输入: target_pce, target_bandgap, target_voc, target_jsc, target_ff, energy_above_hull 等
   - 输出: 候选材料的化学式、晶体结构、预测性能
   
2. **check_synthesizability** (CSLLM): 检验材料是否可以合成
   - 输入: formula (化学式)
   - 输出: Yes/No 合成判断，置信度，理由
   
3. **predict_synthesis_method** (CSLLM): 预测最佳合成方法
   - 输入: formula
   - 输出: 合成步骤 (前驱体溶液配置→旋涂→退火→成膜)
   
4. **predict_precursors** (CSLLM): 预测所需前驱体清单
   - 输入: formula, synthesis_method (可选)
   - 输出: 前驱体化学品列表、摩尔比、溶剂体系

## 任务类型与预期工具链:
- **generate_and_check**: generate → check (2个工具)
- **material_generation**: generate (→ check 可选) (1-2个工具)
- **synthesis_method**: predict_synthesis_method (1个工具)
- **precursor_prediction**: predict_precursors (1个工具)
- **full_pipeline**: generate → check → method → precursors (2-4个工具，按需)

## 评分系统 (0-100分):

### 维度1: 科学准确性 (0-35分)
- 材料化学式是否正确？(ABX3格式，A=MA/FA/Cs，B=Pb/Sn，X=I/Br/Cl)
- 性能参数是否在合理范围内？(PCE: 5-26%, 带隙: 1.2-3.0 eV)
- 合成方法是否可行？前驱体选择是否合理？

### 维度2: 任务完成度 (0-35分)
- **关键**: 工具调用是否匹配任务类型？
  * 单一任务 (合成性/方法/前驱体) 只需 1 个工具
  * 组合任务 (生成+检验) 需要 2 个工具
  * 完整链路需要 2-4 个工具，按条件调用
- 是否回答了用户的所有子问题？
- 条件逻辑是否正确？(如"若可合成则预测方法")

### 维度3: 实用价值 (0-30分)
- 输出是否清晰、结构化？
- 是否给出明确结论？(Yes/No, 可合成/不可合成)
- 是否提供了可操作的建议？

## 评分指南:
- 90-100: 优秀 - 工具链完全正确，输出完整准确
- 70-89: 良好 - 工具基本正确，有小瑕疵
- 50-69: 及格 - 完成任务但工具调用不够精准
- 30-49: 较差 - 工具调用错误或遗漏关键步骤
- 0-29: 失败 - 无法完成任务

请仅以JSON格式回复:
{
    "score": 0-100,
    "scientific_accuracy": 0-35,
    "task_completion": 0-35,
    "practical_value": 0-30,
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
                "generate_and_check": "🧬+🔬 材料生成+合成检验 - 预期工具: generate → check (2个)",
                "material_generation": "🧬 材料生成 - 预期工具: generate (→ check 可选) (1-2个)",
                "synthesis_method": "🧪 合成方法预测 - 预期工具: predict_synthesis_method (1个)",
                "precursor_prediction": "⚗️ 前驱体预测 - 预期工具: predict_precursors (1个)",
                "full_pipeline": "📋 完整链路 - 预期工具: generate → check → method → precursors (2-4个)",
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
    {agent_output[:6000] if agent_output else "(无输出)"}

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
                    "scientific_accuracy": 0, 
                    "task_completion": 0, 
                    "practical_value": 0, 
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
                    "scientific_accuracy": result.get("scientific_accuracy", 0),
                    "task_completion": result.get("task_completion", 0),
                    "practical_value": result.get("practical_value", 0),
                    "reasoning": result.get("reasoning", "无说明")
                }
            except Exception as e:
                return {
                    "score": 0, 
                    "scientific_accuracy": 0, 
                    "task_completion": 0, 
                    "practical_value": 0, 
                    "reasoning": f"解析错误: {e}"
                }


# =============================================================================
# 实验结果记录器 - 支持断点续跑
# =============================================================================
class ExperimentLogger:
    """实验结果记录器 - 支持断点续跑"""
    
    def __init__(self, mode: str = "mock"):
        self.mode = mode
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.json_path = OUTPUT_DIR / f"experiment_results_{mode}.json"
        self.csv_path = OUTPUT_DIR / f"experiment_results_{mode}.csv"
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
                        "mode": self.mode,
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
                            "科学准确性",
                            "任务完成度", 
                            "实用价值",
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
                                "科学准确性": judge.get("scientific_accuracy", 0),
                                "任务完成度": judge.get("task_completion", 0),
                                "实用价值": judge.get("practical_value", 0),
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
            "material_design": "材料设计",
            "synthesizability": "可合成性检验",
            "synthesis_method": "合成方法预测",
            "precursor_design": "前驱体设计",
            "full_recipe": "完整配方设计",
            "optimization": "优化任务",
            "complex": "复杂场景",
        }
        
        print(f"\n{'='*60}")
        print("📊 DesignAgent 实验总结")
        print(f"{'='*60}")
        print(f"   模式: {self.mode.upper()}")
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
# 主实验逻辑
# =============================================================================
async def run_single_query(
    query_info: dict[str, Any],
    settings: Settings,
    mode: str = "mock"
) -> dict[str, Any]:
    """运行单个查询实验"""
    query_id = query_info["id"]
    query_type = query_info["type"]
    query = query_info["query"]
    
    print(f"\n{'='*60}")
    print(f"🧪 实验: {query_id} [{query_type}]")
    print(f"❓ 查询: {query}")
    print(f"🔧 模式: {mode.upper()}")
    print(f"{'='*60}")
    
    if mode == "interactive":
        print(f"\n{'─'*60}")
        print(f"⚠️  INTERACTIVE MODE (与服务器交互):")
        print(f"   当调用服务器工具时，你将看到:")
        print(f"   1. 需要在服务器上执行的命令 (MatterGen/CSLLM)")
        print(f"   2. 提示你粘贴执行结果")
        print(f"   3. 输入 'END' (单独一行) 表示输入完成")
        print(f"   4. 输入 'SKIP' 使用模拟数据")
        print(f"{'─'*60}")
    
    # 设置输出目录
    output_dir = OUTPUT_DIR / "single_queries"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建 state
    state = {
        "goal": query,
        "plan": "根据用户查询完成材料设计任务",
        "data_context": ""
    }
    
    try:
        # 使用 async with 正确初始化 Agent
        async with DesignAgent(settings=settings, tool_mode=mode) as agent:
            
            # 构建 prompt
            prompt = f"""# 设计任务

{query}

请根据任务需求选择合适的工具完成设计。
设计完成后，请给出科学解读和结论。
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
                # 判断是服务器工具还是本地工具
                server_tools = ["generate_material_structure", "check_synthesizability", 
                               "predict_synthesis_method", "predict_precursors"]
                if name in server_tools:
                    print(f"   [📍Server] {name}: {count}x")
                else:
                    print(f"   [🏠Local] {name}: {count}x")
            
            # 保存单个结果
            single_result_file = output_dir / f"{query_id}_{mode}.json"
            with open(single_result_file, "w", encoding="utf-8") as f:
                json.dump({
                    "query_id": query_id,
                    "query_type": query_type,
                    "query": query,
                    "mode": mode,
                    "tool_calls": tool_calls,
                    "tool_results": tool_results,
                    "agent_output": result.get("response", ""),
                    "iterations": result.get("iterations", 0),
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            print(f"   📁 详细结果: {single_result_file}")
            
            return {
                "query_id": query_id,
                "query": query,
                "query_type": query_type,
                "mode": mode,
                "tools_called": tool_names,
                "tool_call_details": tool_calls,
                "tool_results": tool_results,
                "agent_output": result.get("response", ""),
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
            "mode": mode,
            "tools_called": [],
            "tool_call_details": [],
            "tool_results": [],
            "agent_output": "",
            "iterations": 0,
            "success": False,
            "error": str(e)
        }


async def run_all_experiments(
    mode: str = "mock",
    skip_completed: bool = True,
    query_ids: list[str] | None = None
) -> None:
    """运行所有实验"""
    # 初始化
    settings = get_test_settings()
    if not settings.llm.is_valid():
        print("❌ LLM 未配置！请设置 LLM_API_KEY")
        return
    
    # 打印模型信息
    print_model_info(settings, mode)
    
    # 筛选要运行的查询
    queries_to_run = TEST_QUERIES
    if query_ids:
        queries_to_run = [q for q in TEST_QUERIES if q["id"] in query_ids]
    
    print(f"{'='*60}")
    print("🚀 DesignAgent 批量实验")
    print(f"{'='*60}")
    print(f"   查询数量: {len(queries_to_run)}")
    print(f"   执行模式: {mode.upper()}")
    print(f"   最大迭代数: {MAX_ITERATIONS}")
    print(f"{'='*60}")
    
    logger = ExperimentLogger(mode=mode)
    judge = LLMJudge()
    
    completed_ids = logger.get_completed_ids() if skip_completed else set()
    if completed_ids:
        print(f"⏭️ 跳过 {len(completed_ids)} 个已完成的查询")
    
    # 运行实验
    for i, q in enumerate(queries_to_run, 1):
        query_id = q["id"]
        
        if query_id in completed_ids:
            print(f"\n⏭️ [{i}/{len(queries_to_run)}] {query_id} - 已完成，跳过")
            continue
        
        print(f"\n\n{'#'*60}")
        print(f"# [{i}/{len(queries_to_run)}] 运行: {query_id}")
        print(f"{'#'*60}")
        
        try:
            # 运行实验
            result = await run_single_query(q, settings, mode=mode)
            
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
                print(f"   📋 科学准确性: {judge_result['scientific_accuracy']}/35 | 任务完成度: {judge_result['task_completion']}/35 | 实用价值: {judge_result['practical_value']}/30")
                reasoning = judge_result['reasoning']
                print(f"   💬 理由: {reasoning[:100]}..." if len(reasoning) > 100 else f"   💬 理由: {reasoning}")
            else:
                result["judge_result"] = {
                    "score": 0,
                    "scientific_accuracy": 0,
                    "task_completion": 0,
                    "practical_value": 0,
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
                "mode": mode,
                "tools_called": [],
                "agent_output": "",
                "iterations": 0,
                "success": False,
                "error": str(e),
                "judge_result": {
                    "score": 0, 
                    "scientific_accuracy": 0,
                    "task_completion": 0,
                    "practical_value": 0,
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
    parser = argparse.ArgumentParser(description="DesignAgent 批量实验")
    parser.add_argument("--mode", "-m", choices=["mock", "interactive"], default="mock",
                       help="执行模式: mock (模拟) 或 interactive (服务器交互)")
    parser.add_argument("--query", "-q", type=str, 
                       help="运行单个或多个查询 (如 Q001 或 Q001,Q006)")
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
                    "material_design": "🧬 材料设计",
                    "synthesizability": "🔬 可合成性检验",
                    "synthesis_method": "🧪 合成方法预测",
                    "precursor_design": "⚗️ 前驱体设计",
                    "full_recipe": "📋 完整配方设计",
                    "optimization": "⚙️ 优化任务",
                    "complex": "🎯 复杂场景",
                }
                print(f"\n{type_names.get(current_type, current_type)}:")
            print(f"   {q['id']}: {q['query'][:55]}...")
        return
    
    # 解析查询 ID
    query_ids = None
    if args.query:
        query_ids = [q.strip() for q in args.query.split(",")]
    
    # 运行实验
    asyncio.run(run_all_experiments(
        mode=args.mode,
        skip_completed=not args.no_skip,
        query_ids=query_ids
    ))


if __name__ == "__main__":
    main()
