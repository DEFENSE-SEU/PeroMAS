#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analysis_agent_experiment.py
AnalysisAgent 批量实验脚本

实验目标：
1. 测试工具调用是否正常（打印调用的工具列表）
2. 使用 LLM Judge 判定输出内容是否合理
3. 保存完整的工具调用和输出记录

实验类型：
- Type A: 化学式分析（电荷平衡、质量百分比、氧化态）
- Type B: 有机阳离子分析（LogP、TPSA、疏水性）
- Type C: 机理诊断（降解机制、性能损失）
- Type D: SHAP分析（特征重要性、相关性）
- Type E: 综合分析（多工具链式调用）

Usage:
    cd f:\\PSC_Agents\\experiment\\single_point\\analysis_agent
    python analysis_agent_experiment.py
    python analysis_agent_experiment.py --query Q001   # 运行单个查询
    python analysis_agent_experiment.py --query Q001,Q005,Q010  # 运行多个查询
    python analysis_agent_experiment.py --list         # 列出所有查询
    python analysis_agent_experiment.py --no-skip      # 不跳过已完成的
    
Author: PSC_Agents Team
Date: 2026-01-31
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

# 加载 .env 文件（override=True 覆盖系统环境变量）
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from core.config import Settings, LLMConfig, ProviderConfig
from core.llm import LLMClient

# 直接从模块导入，避免触发 agent/__init__.py 的全部导入
import importlib.util
spec = importlib.util.spec_from_file_location(
    "analysis_agent", 
    PROJECT_ROOT / "src" / "agent" / "analysis_agent.py"
)
analysis_agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis_agent_module)
AnalysisAgent = analysis_agent_module.AnalysisAgent

# =============================================================================
# 实验参数配置
# =============================================================================
MAX_ITERATIONS = 15     # 智能体最大迭代次数
OUTPUT_DIR = Path(__file__).parent / "experiment_output"

# =============================================================================
# 测试 Query 列表 (20个)
# 格式: {"id": str, "type": str, "query": str, "query_en": str}
# =============================================================================
TEST_QUERIES = [
    # === Type A: 化学式分析 (Q001-Q004) ===
    {
        "id": "Q001",
        "type": "stoichiometry",
        "query": "检查化学式CsPbI3和CsPbI2.5的电荷平衡情况，判断哪个结构在化学上是合理的？",
        "query_en": "Check the charge balance of the chemical formulas CsPbI3 and CsPbI2.5 to determine which structure is chemically reasonable."
    },
    {
        "id": "Q002",
        "type": "stoichiometry",
        "query": "计算混合阳离子钙钛矿FA0.8Cs0.2PbI3中铅元素(Pb)的质量百分比是多少？",
        "query_en": "Calculate the mass percentage of Lead (Pb) in the mixed-cation perovskite FA0.8Cs0.2PbI3."
    },
    {
        "id": "Q003",
        "type": "stoichiometry",
        "query": "验证无铅双钙钛矿Cs2AgBiBr6的书写规范性及氧化态猜测。",
        "query_en": "Verify the standard notation and oxidation state guesses for the lead-free double perovskite Cs2AgBiBr6."
    },
    {
        "id": "Q004",
        "type": "stoichiometry",
        "query": "验证无铅双钙钛矿Cs2AgBiBr6的化学式书写是否规范，并列出各元素的原子占比。",
        "query_en": "Verify if the chemical formula of the lead-free double perovskite Cs2AgBiBr6 is standard and list the atomic fraction of each element."
    },
    
    # === Type B: 有机阳离子分析 (Q005-Q007) ===
    {
        "id": "Q005",
        "type": "organic_cation",
        "query": "分析常用钝化分子苯乙基铵(PEA, SMILES: NCCc1ccccc1)的LogP值，评估其对钙钛矿表面的疏水改性能力。",
        "query_en": "Analyze the LogP value of the common passivation molecule Phenethylammonium (PEA, SMILES: NCCc1ccccc1) to evaluate its hydrophobic modification capability on the perovskite surface."
    },
    {
        "id": "Q006",
        "type": "organic_cation",
        "query": "对比3D组分甲胺(MA, CN)和2D间隔层丁胺(BA, CCCCN)的拓扑极性表面积(TPSA)，解释为何BA能提升抗湿性。",
        "query_en": "Compare the Topological Polar Surface Area (TPSA) of the 3D component Methylammonium (MA, CN) and the 2D spacer Butylammonium (BA, CCCCN) to explain why BA improves moisture resistance."
    },
    {
        "id": "Q007",
        "type": "organic_cation",
        "query": "评估长链分子辛胺(OA, CCCCCCCCN)的疏水性，判断其是否适合用于制备准二维(Quasi-2D)钙钛矿。",
        "query_en": "Evaluate the hydrophobicity of the long-chain molecule Octylamine (OA, CCCCCCCCN) to determine if it is suitable for preparing Quasi-2D perovskites."
    },
    
    # === Type C: 机理诊断 (Q008-Q011) ===
    {
        "id": "Q008",
        "type": "mechanism",
        "query": "诊断MAPbI3钙钛矿在高温(>85°C)环境下的主要降解机制是什么？",
        "query_en": "Diagnose the main degradation mechanism of MAPbI3 perovskite under high-temperature (>85°C) environments."
    },
    {
        "id": "Q009",
        "type": "mechanism",
        "query": "我的宽带隙钙钛矿电池Voc只有1.05V(理论值应为1.3V)，请分析导致这0.25V电压损失的主要缺陷复合机制。",
        "query_en": "My wide-bandgap perovskite cell has a Voc of only 1.05V (theoretical 1.3V); please analyze the main defect recombination mechanisms causing this 0.25V voltage loss."
    },
    {
        "id": "Q010",
        "type": "mechanism",
        "query": "分析光照浸润(Light Soaking)现象对混合卤素钙钛矿(I/Br混合)相分离的影响机制。",
        "query_en": "Analyze the mechanism of the Light Soaking effect on phase segregation in mixed-halide perovskites (I/Br mixed)."
    },
    {
        "id": "Q011",
        "type": "mechanism",
        "query": "解释为什么在3D钙钛矿表面引入2D覆盖层可以阻隔水分子入侵。",
        "query_en": "Explain why introducing a 2D capping layer on the 3D perovskite surface can block water molecule intrusion."
    },
    
    # === Type D: SHAP/统计分析 (Q012-Q014) ===
    {
        "id": "Q012",
        "type": "shap_analysis",
        "query": '基于我提供的特征重要性字典{"bandgap": 0.35, "tolerance_factor": 0.25, "annealing": 0.1}，请计算并列出对PCE影响最大的前3个特征。',
        "query_en": 'Based on the provided feature importance dictionary {"bandgap": 0.35, "tolerance_factor": 0.25, "annealing": 0.1}, please calculate and list the top 3 features affecting PCE.'
    },
    {
        "id": "Q013",
        "type": "shap_analysis",
        "query": '我有T80寿命模型的特征重要性数据{"humidity": 0.4, "encapsulation": 0.3, "grain_size": 0.2}，请生成一张全局SHAP柱状摘要图(Bar Summary Plot)。',
        "query_en": 'I have feature importance data for the T80 lifetime model {"humidity": 0.4, "encapsulation": 0.3, "grain_size": 0.2}, please generate a global SHAP Bar Summary Plot.'
    },
    {
        "id": "Q014",
        "type": "correlation",
        "query": '基于我提供的这组退火实验数据：[{"Temp": 100, "PCE": 18.5}, {"Temp": 120, "PCE": 20.1}, {"Temp": 140, "PCE": 19.8}]，计算退火温度(Temp)与PCE之间的皮尔逊相关系数。',
        "query_en": 'Based on the annealing experimental data I provided: [{"Temp": 100, "PCE": 18.5}, {"Temp": 120, "PCE": 20.1}, ...], calculate the Pearson correlation coefficient between Annealing Temperature (Temp) and PCE.'
    },
    
    # === Type E: 综合分析 (Q015-Q020) ===
    {
        "id": "Q015",
        "type": "comprehensive",
        "query": "首先检查化学式MAPbI3是否电荷平衡，然后计算其有机组分MA(CN)的LogP值，最后综合评估其在潮湿环境下的本质不稳定性。",
        "query_en": "First check if the chemical formula MAPbI3 is charge balanced, then calculate the LogP value of its organic component MA (CN), and finally comprehensively evaluate its intrinsic instability in humid environments."
    },
    {
        "id": "Q016",
        "type": "comprehensive",
        "query": "计算FAPbI3的分子量，并分析FA([NH2+]=CN)的TPSA，解释为什么FA基钙钛矿比MA基更吸湿，但检索知识库后发现其热稳定性更好？",
        "query_en": "Calculate the molecular weight of FAPbI3, analyze the TPSA of FA ([NH2+]=CN), and explain why FA-based perovskites are more hygroscopic than MA-based ones, yet retrieving knowledge base shows they are more thermally stable?"
    },
    {
        "id": "Q017",
        "type": "comprehensive",
        "query": "对比2D钙钛矿间隔层：分别分析PEA(NCCc1ccccc1)和BA(CCCCN)的LogP值，基于'LogP越大抗湿性越好'的规律，推断谁能提供更好的降解防护。",
        "query_en": 'Compare 2D perovskite spacers: Analyze the LogP values of PEA (NCCc1ccccc1) and BA (CCCCN) respectively, and based on the rule "higher LogP means better moisture resistance", infer which one provides better degradation protection.'
    },
    {
        "id": "Q018",
        "type": "correlation",
        "query": '验证"晶粒越大效率越高"的假设。数据：[{"GrainSize": 200, "PCE": 15.1}, {"GrainSize": 500, "PCE": 18.2}, {"GrainSize": 1000, "PCE": 21.5}]。计算相关性。',
        "query_en": 'Verify the hypothesis "larger grains mean higher efficiency". Data: [{"GrainSize": 200, "PCE": 15.1}, {"GrainSize": 500, "PCE": 18.2}, {"GrainSize": 1000, "PCE": 21.5}]. Calculate correlation.'
    },
    {
        "id": "Q019",
        "type": "shap_comparison",
        "query": '对比CsPbI3(预测14%)和MAPbI3(预测19%)。贡献数据：[{"name": "CsPbI3", "predicted_value": 14, "contributions": [{"feature": "stability", "contribution": 1.0}]}, {"name": "MAPbI3", "predicted_value": 19, "contributions": [{"feature": "bandgap", "contribution": 1.5}]}]。',
        "query_en": 'Compare CsPbI3 (14%) and MAPbI3 (19%). Data: [{"name": "CsPbI3", "predicted_value": 14, "contributions": [{"feature": "stability", "contribution": 1.0}]}, {"name": "MAPbI3", "predicted_value": 19, "contributions": [{"feature": "bandgap", "contribution": 1.5}]}]'
    },
    {
        "id": "Q020",
        "type": "shap_comparison",
        "query": '对比高温退火和低温退火样品。数据：[{"name": "HighT", "predicted_value": 18, "contributions": [{"feature": "crystallinity", "contribution": 2.0}]}, {"name": "LowT", "predicted_value": 15, "contributions": [{"feature": "crystallinity", "contribution": -1.0}]}]。',
        "query_en": 'Compare High-T and Low-T annealing samples. Data: [{"name": "HighT", "predicted_value": 18, "contributions": [{"feature": "crystallinity", "contribution": 2.0}]}, {"name": "LowT", "predicted_value": 15, "contributions": [{"feature": "crystallinity", "contribution": -1.0}]}]'
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
    print(f"   🤖 AnalysisAgent 模型: {llm_config.model_name}")
    print(f"   🌡️  Temperature: {llm_config.temperature}")
    print(f"   🔗 Base URL: {llm_config.base_url[:50]}...")
    print(f"   ⚖️  Judge 模型: {judge_model}")
    print(f"{'='*60}\n")


# =============================================================================
# LLM Judge - 评估输出质量 (100分制)
# =============================================================================
class LLMJudge:
    """LLM 裁判：评估 AnalysisAgent 输出质量"""
    
    JUDGE_SYSTEM_PROMPT = """你是一个公正的AI分析智能体评估者。
你的任务是根据智能体对用户分析查询的响应质量进行评分。

## 评分系统 (0-100分):

### 维度1: 科学准确性 (0-35分)
- 化学式分析是否正确（电荷平衡、分子量、氧化态）？
- 有机分子分析是否准确（LogP、TPSA等描述符）？
- 机理解释是否符合钙钛矿物理化学原理？
- SHAP/统计分析是否数学正确？

### 维度2: 任务完成度 (0-35分)
- 智能体是否调用了正确的分析工具？
- 是否回答了用户的所有问题点？
- 综合任务是否完成了全部步骤？

### 维度3: 实用价值 (0-30分)
- 分析结果是否对研究有指导意义？
- 是否给出了明确的结论或建议？
- 解释是否清晰易懂？

## 评分指南:
- 90-100: 优秀 - 分析准确、完整、有深度
- 70-89: 良好 - 分析正确但深度不足
- 50-69: 及格 - 基本分析但有小错误
- 30-49: 较差 - 分析不完整或有明显错误
- 0-29: 失败 - 无分析或完全错误

## 任务类型说明:
- stoichiometry: 化学式分析（电荷平衡、分子量、原子占比）
- organic_cation: 有机阳离子分析（LogP、TPSA、疏水性）
- mechanism: 机理诊断（降解、性能损失、结构-性质关系）
- shap_analysis: SHAP特征重要性分析
- correlation: 相关性统计分析
- comprehensive: 综合分析（多工具链式调用）

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
            "stoichiometry": "🧪 化学式分析 - 电荷平衡、分子量、氧化态",
            "organic_cation": "🔬 有机阳离子分析 - LogP、TPSA、疏水性",
            "mechanism": "⚙️ 机理诊断 - 降解、性能损失分析",
            "shap_analysis": "📊 SHAP分析 - 特征重要性",
            "shap_comparison": "📈 SHAP对比 - 多样品特征贡献对比",
            "correlation": "📉 相关性分析 - 统计相关系数",
            "comprehensive": "🔗 综合分析 - 多工具链式调用",
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
            "stoichiometry": "化学式分析",
            "organic_cation": "有机阳离子分析",
            "mechanism": "机理诊断",
            "shap_analysis": "SHAP分析",
            "shap_comparison": "SHAP对比",
            "correlation": "相关性分析",
            "comprehensive": "综合分析",
        }
        
        print(f"\n{'='*60}")
        print("📊 AnalysisAgent 实验总结")
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
    
    # 构建 state
    state = {
        "goal": query,
        "plan": "根据用户查询进行钙钛矿材料分析",
        "experimental_params": {}
    }
    
    try:
        # 使用 async with 正确初始化 Agent（包括 LLM client）
        async with AnalysisAgent(settings=settings) as agent:
            # 构建 prompt - 直接使用用户查询
            prompt = f"""# 分析任务

{query}

请根据用户的需求灵活使用工具完成分析任务。
分析完成后，请给出科学解读和结论。
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
            
            # 保存单个查询结果
            single_output_path = OUTPUT_DIR / "single_queries" / f"{query_id}.json"
            single_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(single_output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "query_id": query_id,
                    "query_type": query_type,
                    "query": query,
                    "tool_calls": tool_calls,
                    "tool_results": tool_results,
                    "agent_output": result.get("response", ""),
                    "iterations": result.get("iterations", 0),
                }, f, indent=2, ensure_ascii=False)
            print(f"   📁 详细结果: {single_output_path}")
            
            return {
                "query_id": query_id,
                "query": query,
                "query_type": query_type,
                "tools_called": tool_names,
                "tool_call_details": tool_calls,
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
            "tools_called": [],
            "tool_call_details": [],
            "agent_output": "",
            "iterations": 0,
            "success": False,
            "error": str(e)
        }


async def run_all_experiments(skip_completed: bool = True, query_ids: list[str] | None = None) -> None:
    """运行所有实验"""
    # 初始化
    settings = get_test_settings()
    if not settings.llm.is_valid():
        print("❌ LLM 未配置！请设置 LLM_API_KEY")
        return
    
    # 打印模型信息
    print_model_info(settings)
    
    # 筛选要运行的查询
    if query_ids:
        queries_to_run = [q for q in TEST_QUERIES if q["id"] in query_ids]
        if not queries_to_run:
            print(f"❌ 未找到指定的查询: {query_ids}")
            return
    else:
        queries_to_run = TEST_QUERIES
    
    print(f"{'='*60}")
    print("🚀 AnalysisAgent 批量实验")
    print(f"{'='*60}")
    print(f"   查询数量: {len(queries_to_run)}")
    print(f"   最大迭代数: {MAX_ITERATIONS}")
    print(f"{'='*60}")
    
    logger = ExperimentLogger()
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
                "tools_called": [],
                "agent_output": "",
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
    parser = argparse.ArgumentParser(description="AnalysisAgent 批量实验")
    parser.add_argument("--query", "-q", type=str, help="运行指定查询 (如 Q001 或 Q001,Q005,Q010)")
    parser.add_argument("--no-skip", action="store_true", help="不跳过已完成的查询")
    parser.add_argument("--list", action="store_true", help="列出所有测试查询")
    args = parser.parse_args()
    
    if args.list:
        print("\n📋 测试查询列表 (20个):")
        print("-" * 80)
        current_type = None
        for q in TEST_QUERIES:
            if q["type"] != current_type:
                current_type = q["type"]
                type_names = {
                    "stoichiometry": "🧪 化学式分析",
                    "organic_cation": "🔬 有机阳离子分析",
                    "mechanism": "⚙️ 机理诊断",
                    "shap_analysis": "📊 SHAP分析",
                    "shap_comparison": "📈 SHAP对比",
                    "correlation": "📉 相关性分析",
                    "comprehensive": "🔗 综合分析",
                }
                print(f"\n{type_names.get(current_type, current_type)}:")
            print(f"   {q['id']}: {q['query'][:60]}...")
        return
    
    if args.query:
        # 支持逗号分隔的多个查询ID
        query_ids = [qid.strip() for qid in args.query.split(",")]
        asyncio.run(run_all_experiments(skip_completed=not args.no_skip, query_ids=query_ids))
    else:
        asyncio.run(run_all_experiments(skip_completed=not args.no_skip))


if __name__ == "__main__":
    main()
