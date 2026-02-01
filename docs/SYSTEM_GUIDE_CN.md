# PSC_Agents 系统说明文档

## 📖 系统概述

PSC_Agents 是一个自主多智能体系统，专为钙钛矿太阳能电池 (Perovskite Solar Cell, PSC) 研究设计。系统协调多个专业化 AI 智能体，自动化完成整个研究流程：

```
文献检索 → 材料设计 → 性能预测 → 结果分析 → 知识归档 → 循环迭代
```

## 🏗️ 系统架构

### 工作流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PSC_Agents 工作流程                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   用户目标 ──► MetaAgent ──► DataAgent ──► DesignAgent ──► FabAgent    │
│   (goal)       (规划)        (文献)        (设计)         (预测)       │
│                  │              │              │              │         │
│                  ▼              ▼              ▼              ▼         │
│                plan       data_context   exp_params     fab_results     │
│                  │              │              │              │         │
│                  └──────────────┴──────────────┴──────────────┘         │
│                                      │                                  │
│                                      ▼                                  │
│                               AnalysisAgent (分析)                      │
│                                      │                                  │
│                                      ▼                                  │
│                              analysis_report                            │
│                                      │                                  │
│                                      ▼                                  │
│                               MemoryAgent (记忆)                        │
│                                      │                                  │
│                                      ▼                                  │
│                               memory_log ───────► 下一迭代              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 智能体详情

| 智能体 | 角色定位 | 工具类型 | 输出字段 |
|--------|----------|----------|----------|
| **MetaAgent** | 首席科学家 - 战略规划与任务分配 | 无（纯推理） | `plan` |
| **DataAgent** | 文献情报官 - 论文搜索与数据提取 | MCP: arXiv服务器 | `data_context` |
| **DesignAgent** | 实验设计师 - 材料与合成设计 | 服务器: MatterGen, CSLLM | `experimental_params` |
| **FabAgent** | 虚拟制造师 - 性能预测 | 本地: RF模型 | `fab_results` |
| **AnalysisAgent** | 首席分析师 - 差距分析与诊断 | 本地: 化学工具, SHAP | `analysis_report` |
| **MemoryAgent** | 知识管理员 - 归档与学习 | 无（纯推理） | `memory_log` |

---

## 🧠 各智能体详细说明

### 1. MetaAgent (首席科学家)

**职责**：纯推理智能体，负责整体研究策略的规划和任务分配。

**输入状态**：
- `goal`: 用户的研究目标
- `memory_log`: 历史迭代的知识记录
- `analysis_report`: 最新的分析诊断报告

**能力**：
1. **目标理解**：解析用户目标，识别约束条件和成功标准
2. **假设生成**：为每次迭代提出可验证的科学假设
3. **策略调整**：根据结果动态调整研究方向
4. **智能体编排**：为下游智能体分配具体任务
5. **关键决策**：评估结果，识别失败，提出修正方案

**自主决策权**：
- `SKIP`: 跳过某个智能体（如后续迭代无需文献检索）
- `REDO`: 要求智能体以调整后的参数重做任务
- `PIVOT`: 切换研究策略
- `TERMINATE`: 提前终止（目标达成或明确不可行）

**输出格式**：
```json
{
  "analysis": "对上一迭代结果的分析",
  "hypothesis": "本次迭代的科学假设",
  "strategy": "continue/pivot/refine",
  "constraints": ["约束条件1", "约束条件2"],
  "agent_tasks": {
    "DataAgent": "具体任务或SKIP",
    "DesignAgent": "具体任务",
    "FabAgent": "具体任务",
    "AnalysisAgent": "具体分析任务"
  },
  "success_criteria": "判断目标是否达成的标准"
}
```

---

### 2. DataAgent (文献情报官)

**职责**：从科学文献中检索、处理和提取信息。

**输入状态**：
- `goal`: 研究目标
- `plan`: MetaAgent的策略（查看`plan.agent_tasks.DataAgent`获取任务）

**工具箱**：

| 工具 | 类型 | 描述 |
|------|------|------|
| `search_papers` | MCP | 搜索arXiv论文 |
| `download_paper` | MCP | 下载指定论文 |
| `read_paper` | MCP | 将PDF转为文本 |
| `save_markdown_locally` | 本地 | 保存论文到本地 |
| `extract_data_from_papers` | 本地 | 提取结构化数据 |

**工具约束**：
1. `extract_data_from_papers` 只能处理已保存的本地文件
2. 搜索查询应包含领域关键词（如 "perovskite solar cell"）
3. 默认处理3-5篇论文

**输出格式**：
```json
{
  "status": "success/partial/failed",
  "papers_processed": 3,
  "key_findings": {
    "performance_benchmarks": [...],
    "synthesis_methods": [...],
    "material_compositions": [...]
  },
  "extracted_data": [...],
  "summary": "文献发现摘要"
}
```

---

### 3. DesignAgent (实验设计师)

**职责**：设计钙钛矿材料并规划合成路线。

**输入状态**：
- `goal`: 研究目标（目标PCE、稳定性、约束等）
- `plan`: MetaAgent的假设和任务
- `data_context`: DataAgent的文献发现（参考组成、合成方法）

**工具箱**：

| 工具 | 类型 | 描述 |
|------|------|------|
| `generate_material_structure` | 服务器(MatterGen) | 生成候选钙钛矿结构 |
| `check_synthesizability` | 服务器(CSLLM) | 验证配方可合成性 |
| `predict_synthesis_method` | 服务器(CSLLM) | 预测最优合成路线 |
| `predict_precursors` | 服务器(CSLLM) | 识别前驱体化学品 |
| `screen_candidates` | 本地 | 筛选和排序候选材料 |

**工具模式**：
- `mock`: 模拟结果（用于测试）
- `interactive`: 通过终端输入服务器结果

**关键约束**：
1. **必须使用实际工具结果**：生成材料后，必须使用返回的精确配方进行后续调用
2. **工具依赖**：`predict_precursors` 需要先知道合成方法
3. **信任服务器结果**：代表实际科学知识

**输出格式**：
```json
{
  "task_type": "material_design/synthesizability/full_recipe",
  "composition": {
    "formula": "FA0.9Cs0.1PbI3",
    "structure_type": "3D",
    "synthesizability": {"result": true, "confidence": 0.85}
  },
  "process": {
    "method": "solution",
    "precursors": [...],
    "solvents": {...}
  },
  "analysis": "科学原理说明",
  "status": "success/partial/failed"
}
```

---

### 4. FabAgent (虚拟制造师)

**职责**：使用训练好的ML模型预测太阳能电池性能。

**输入状态**：
- `goal`: 研究目标（目标PCE等）
- `plan`: MetaAgent的策略
- `data_context`: DataAgent的文献数据
- `experimental_params`: **关键** - DesignAgent的材料设计

**工具箱**：

| 工具 | 描述 |
|------|------|
| `predict_perovskite` | 预测性能：PCE, Voc, Jsc, FF, 带隙, E_hull |
| `visualize_predictions` | 单材料柱状图 |
| `visualize_series_trend` | 趋势线图（组成系列） |
| `visualize_comparison` | 多材料对比图 |

**工具约束**：
1. 预测优先于可视化
2. 仅在用户请求时生成可视化
3. 从 `experimental_params.composition.formula` 获取配方

**输出格式**：
```json
{
  "composition": "FA0.9Cs0.1PbI3",
  "predicted_metrics": {
    "PCE_percent": 22.5,
    "Voc_V": 1.12,
    "Jsc_mA_cm2": 25.3,
    "FF_percent": 79.5,
    "BandGap_eV": 1.52,
    "E_hull_eV": 0.02
  },
  "analysis": "科学解释",
  "status": "success/failed"
}
```

---

### 5. AnalysisAgent (首席分析师)

**职责**：分析材料和实验结果，诊断性能差距和根本原因。

**输入状态**：
- `goal`: 研究目标
- `plan`: MetaAgent的假设（需验证）
- `data_context`: 文献基准
- `experimental_params`: 材料设计
- `fab_results`: **关键** - 预测结果

**工具箱**：

| 工具 | 类别 | 描述 |
|------|------|------|
| `analyze_stoichiometry` | 化学分析 | 验证配方、计算分子量、检查电荷平衡 |
| `analyze_organic_cation` | 化学分析 | 有机阳离子性质（LogP, TPSA） |
| `analyze_mechanism` | 机理诊断 | 降解/性能机理分析 |
| `calculate_correlation` | 统计分析 | Pearson相关系数计算 |
| `shap_feature_importance` | SHAP分析 | 特征重要性排名 |
| `shap_summary_plot` | SHAP分析 | 特征重要性可视化 |
| `visualize_structure` | 可视化 | 3D晶体结构渲染 |

**工具选择原则**：
1. 配方问题 → `analyze_stoichiometry`
2. 有机分子性质 → `analyze_organic_cation`
3. 为什么/机理问题 → `analyze_mechanism`
4. 数据相关性 → `calculate_correlation`
5. 特征重要性 → `shap_feature_importance`

**输出格式**：
```json
{
  "is_goal_met": false,
  "performance_gap": {
    "target_PCE": 25.0,
    "achieved_PCE": 22.5,
    "shortfall_percent": 10.0
  },
  "scientific_diagnosis": {
    "root_cause": "Voc损失由非辐射复合引起",
    "contributing_factors": ["晶界缺陷", "界面陷阱"]
  },
  "iteration_feedback": {
    "status": "PROCEED/REVISE/ABORT",
    "suggested_adjustment": "添加KI钝化以降低缺陷密度"
  }
}
```

---

### 6. MemoryAgent (知识管理员)

**职责**：将工作流结果提炼为高价值"知识胶囊"。

**输入状态**：全部可见
- `goal`, `plan`, `data_context`, `experimental_params`, `fab_results`, `analysis_report`, `memory_log`

**任务**：
1. **提取知识三元组**：配方、PCE、成功/失败原因
2. **洞察提炼**：识别结果背后的"为什么"
3. **趋势检测**：与目标对比，判断系统是否收敛

**归档原则**：
- **成功案例**：归档为"黄金模板"
- **失败案例**：聚焦根本原因诊断
- **科学完整性**：保留精确的指标和单位

**输出格式**：
```
### Iteration N [SUCCESS/FAILURE/PARTIAL]
- **Formula**: FA0.9Cs0.1PbI3
- **PCE**: 22.5%
- **Reason**: Cs掺杂改善了相稳定性
- **Learning**: 最重要的教训
- **Feedback**: 对下一迭代的建议
```

---

## 📊 全局状态定义

```python
class AgentState(TypedDict):
    # 全局上下文
    goal: str                              # 用户研究目标
    
    # 各智能体输出
    plan: Optional[Dict[str, Any]]         # MetaAgent → 策略计划
    data_context: Optional[str]            # DataAgent → 文献数据
    experimental_params: Optional[Dict]    # DesignAgent → 实验参数
    fab_results: Optional[Dict]            # FabAgent → 预测结果
    analysis_report: Optional[str]         # AnalysisAgent → 分析报告
    memory_log: List[str]                  # MemoryAgent → 知识记录（追加）
    
    # 控制流
    current_iteration: int                 # 当前迭代次数
    is_finished: bool                      # 是否完成
```

---

## 🚀 使用方法

### 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量 (.env)
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o

# 运行测试
cd src/test
python workflow_test.py
```

### 测试命令

```bash
# 完整工作流测试
python workflow_test.py

# Mock模式（无外部依赖）
python workflow_test.py --mode mock

# 单智能体链测试
python workflow_test.py --mode chain

# 自定义研究目标
python workflow_test.py --query "设计PCE>25%的钙钛矿"

# 多次迭代
python workflow_test.py --iterations 5

# 查看预设查询
python workflow_test.py --list-queries
```

---

## 🔧 开发指南

### 添加新工具

1. 定义工具Schema：
```python
NEW_TOOL = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "工具描述...",
        "parameters": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
}
```

2. 实现执行函数：
```python
async def _execute_my_tool(self, args: dict) -> dict:
    # 实现逻辑
    return {"result": "..."}
```

3. 注册到执行器：
```python
TOOL_EXECUTORS = {
    "my_tool": _execute_my_tool,
}
```

### 自定义智能体Prompt

每个智能体的行为由其 `SYSTEM_PROMPT` 控制：

```python
SYSTEM_PROMPT = """You are [AgentName] - [角色] of PSC_Agents.

## Role
...

## Your Input (State Access)
- `goal`: ...
- `plan`: ...

## Your Specialized Toolbox
- `tool1`: 描述
- `tool2`: 描述

## Output
Your output -> `state_field`
"""
```

---

## 📋 支持的研究任务

| 任务类型 | 示例查询 |
|----------|----------|
| **组成设计** | "设计PCE>25%的钙钛矿组成" |
| **无铅设计** | "设计Sn基无铅钙钛矿" |
| **稳定性优化** | "优化热稳定性>85°C" |
| **带隙工程** | "设计1.7eV带隙用于叠层电池" |
| **混合阳离子** | "优化FA/MA/Cs比例" |
| **2D钙钛矿** | "设计2D Ruddlesden-Popper结构" |
| **机理分析** | "分析CsPbI3的Voc损失机理" |

---

## 📧 联系方式

如有问题或合作意向，请提交Issue或联系团队。
