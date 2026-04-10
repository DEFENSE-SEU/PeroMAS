# AnalysisAgent 功能文档

## 概述

AnalysisAgent 是 PSC_Agents 系统中的**首席战略分析师**，负责执行"审计与诊断"阶段。其核心职责是：不仅判断设计是否成功，更要分析**为什么**成功或失败，并提供可操作的改进建议。

AnalysisAgent 具备**物理、化学、数学、AI 解释**四个维度的分析能力：

| 维度 | 工具 | 依赖库 | 核心功能 |
|------|------|--------|---------|
| **AI 解释** | `shap_*` 系列 | shap | 解释模型为什么预测这个分数 |
| **物理检查** | `analyze_stoichiometry` | pymatgen | 检查化学式电荷平衡、分子量 |
| **化学性质** | `analyze_organic_cation` | rdkit | 分析有机分子疏水性/抗湿性 |
| **数据挖掘** | `calculate_correlation` | pandas | 挖掘数据中的统计学规律 |
| **机制诊断** | `analyze_mechanism` | - | 降解/性能瓶颈分析 |

---

## 1. 可用工具总览

### 1.1 结构与机制分析
| 工具名称 | 功能描述 |
|---------|---------|
| `visualize_structure` | 将 CIF 晶体结构可视化为交互式 3D HTML |
| `analyze_mechanism` | 分析性能/降解/结构-性能关系机制 |

### 1.2 AI 模型解释 (SHAP)
| 工具名称 | 功能描述 | 必需输入 |
|---------|---------|---------|
| `shap_feature_importance` | 获取 SHAP 特征重要性排名 | `feature_importance` dict |
| `shap_summary_plot` | 生成 SHAP 汇总可视化 (bar/beeswarm) | `feature_importance` dict |
| `shap_analyze_prediction` | 分析单个预测的特征贡献 | `contributions`, `base_value`, `predicted_value` |
| `shap_dependence_plot` | 生成特征依赖性分析图 | `feature_name`, `feature_values`, `shap_values` |

> ⚠️ **重要**：SHAP 工具不再生成模拟数据，需要用户提供真实的分析数据。

### 1.3 化学分析工具 (NEW)
| 工具名称 | 功能描述 | 依赖库 |
|---------|---------|--------|
| `analyze_stoichiometry` | 化学计量分析：电荷平衡、分子量、原子比例 | pymatgen |
| `analyze_organic_cation` | 有机阳离子分析：疏水性(LogP)、抗湿性评估 | rdkit |
| `calculate_correlation` | 相关性分析：计算特征间皮尔逊相关系数 | pandas |

---

## 2. 工具详细说明

### 2.1 `visualize_structure` - 晶体结构可视化

**功能**：将 CIF 格式的晶体结构数据渲染为交互式 3D HTML 视图，支持旋转、缩放、平移。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `cif_content` | string | ✅ | CIF 格式的晶体结构数据 |
| `name` | string | ❌ | 结构名称（用于文件命名） |
| `supercell` | string | ❌ | 超胞尺寸，如 "2,2,2" |
| `theme` | string | ❌ | 主题 "light" 或 "dark" |

**输出示例**：
```json
{
  "success": true,
  "filepath": "analysis_output/CsPbI3_20241201_143025.html",
  "info": {
    "formula": "Cs Pb I3",
    "n_atoms": 5,
    "elements": ["Cs", "Pb", "I"],
    "volume": 248.5,
    "cell": {"a": 6.28, "b": 6.28, "c": 6.28}
  }
}
```

---

### 2.2 `analyze_mechanism` - 机制分析

**功能**：分析钙钛矿材料的降解机制、性能瓶颈或结构-性能关系。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `analysis_type` | string | ✅ | 分析类型：`degradation`, `performance`, `structure_property` |
| `material_info` | string | ✅ | 材料信息（如化学式） |
| `conditions` | string | ❌ | 实验条件（温度、湿度等） |
| `metrics` | string | ❌ | 性能指标 JSON，如 `{"PCE": 18.5, "Voc": 1.05}` |

**分析类型说明**：
- `degradation`：降解机制分析（热稳定性、湿度稳定性、光稳定性）
- `performance`：性能瓶颈诊断（Voc/Jsc/FF 限制因素）
- `structure_property`：结构-性能关系分析

**输出示例（降解分析）**：
```json
{
  "analysis_type": "degradation",
  "material": "MAPbI3",
  "identified_pathways": [
    {
      "pathway": "Organic cation volatilization",
      "trigger": "Thermal stress (>85°C)",
      "mechanism": "MA+ ions escape from the lattice, leaving behind PbI2",
      "severity": "High"
    }
  ],
  "dominant_mechanism": "Organic cation volatilization",
  "recommendations": [
    "Consider mixed-cation approach (Cs/FA) to improve thermal stability"
  ]
}
```

---

### 2.3 `analyze_stoichiometry` - 化学计量分析 (NEW)

**功能**：使用 pymatgen 分析化学式的基本属性，包括分子量、原子占比、氧化态猜测和电荷平衡验证。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `formula` | string | ✅ | 化学式（如 `CsPbI3`, `MAPbI3`, `FA0.8Cs0.2PbI2.4Br0.6`） |

**支持的有机阳离子缩写**：
- `MA` → CH₆N (甲胺)
- `FA` → CH₅N₂ (甲脒)

**输出示例**：
```json
{
  "success": true,
  "original_formula": "CsPbI3",
  "formula_pretty": "CsPbI3",
  "molecular_weight": 720.82,
  "elements": ["Cs", "Pb", "I"],
  "atomic_fractions": {"Cs": 0.2, "Pb": 0.2, "I": 0.6},
  "oxi_state_guesses": [{"Cs": 1.0, "Pb": 2.0, "I": -1.0}],
  "is_charge_balanced": true,
  "charge_balance_note": "Charge balanced"
}
```

**电荷不平衡示例** (CsPbI2.5)：
```json
{
  "is_charge_balanced": false,
  "charge_balance_note": "No valid oxidation state combination found - may indicate unbalanced formula"
}
```

---

### 2.4 `analyze_organic_cation` - 有机阳离子分析 (NEW)

**功能**：使用 RDKit 分析有机阳离子的分子描述符，预测湿度稳定性。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `smiles` | string | ✅ | SMILES 分子结构字符串 |
| `name` | string | ❌ | 阳离子名称（可选） |

**常见钙钛矿有机阳离子 SMILES**：
| 阳离子 | 名称 | SMILES |
|--------|------|--------|
| MA | 甲胺 | `CN` |
| FA | 甲脒 | `[NH2+]=CN` |
| PEA | 苯乙基铵 | `NCCc1ccccc1` |
| BA | 丁胺 | `CCCCN` |
| OA | 辛胺 | `CCCCCCCCN` |
| GA | 胍 | `NC(=N)N` |

**输出示例**：
```json
{
  "success": true,
  "smiles": "NCCc1ccccc1",
  "name": "PEA",
  "molecular_weight": 121.18,
  "LogP": 1.188,
  "TPSA": 26.02,
  "H_bond_donors": 1,
  "rotatable_bonds": 2,
  "hydrophobicity_assessment": "Moderate (good moisture resistance)",
  "stability_prediction": {
    "moisture_resistance": "High",
    "recommendation": "Higher LogP = better humidity stability"
  }
}
```

**疏水性排序参考**：
| 阳离子 | LogP | 抗湿性 |
|--------|------|--------|
| PEA | 1.188 | High |
| BA | 0.745 | Medium |
| MA | -0.425 | Low |

---

### 2.5 `calculate_correlation` - 相关性分析 (NEW)

**功能**：使用 pandas 计算实验数据中特征之间的皮尔逊相关系数。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `data_json` | string | ✅ | JSON 格式的数据（list of dicts） |
| `target_column` | string | ❌ | 目标列（默认 "PCE"） |

**输入示例**：
```json
[
  {"PCE": 18.5, "Voc": 1.10, "Jsc": 22.5, "bandgap": 1.55},
  {"PCE": 20.1, "Voc": 1.15, "Jsc": 23.0, "bandgap": 1.52},
  {"PCE": 16.2, "Voc": 1.05, "Jsc": 21.0, "bandgap": 1.60}
]
```

**输出示例**：
```json
{
  "success": true,
  "n_samples": 5,
  "correlation_matrix": {
    "PCE": {"Voc": 0.986, "Jsc": 0.966, "bandgap": -0.998}
  },
  "target_analysis": {
    "top_correlated_features": {"bandgap": 0.998, "Voc": 0.986},
    "top_positive_correlations": {"Voc": 0.986, "Jsc": 0.966},
    "top_negative_correlations": {"bandgap": -0.998}
  }
}
```

---

### 2.6 `shap_feature_importance` - 特征重要性分析

**功能**：获取 SHAP 特征重要性排名，了解哪些特征对目标属性影响最大。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `feature_importance` | string | ✅ | 特征重要性 JSON，如 `{"tolerance_factor": 0.25, "bandgap_eV": 0.20}` |
| `material_composition` | string | ❌ | 材料组分（可选） |
| `target_property` | string | ❌ | 目标属性：PCE, Voc, Jsc, FF, T80（默认 PCE） |
| `n_top_features` | number | ❌ | 返回前 N 个重要特征（默认 10） |

**输出示例**：
```json
{
  "success": true,
  "target_property": "PCE",
  "top_features": [
    {"rank": 1, "feature": "tolerance_factor", "importance": 0.25},
    {"rank": 2, "feature": "bandgap_eV", "importance": 0.20}
  ],
  "analysis_summary": "[SHAP Analysis] For PCE prediction, the top 3 most important features are: tolerance_factor, bandgap_eV, FA_fraction."
}
```

---

### 2.7 `shap_summary_plot` - SHAP 汇总图

**功能**：生成 SHAP 汇总可视化图表（柱状图或蜂群图）。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `feature_importance` | string | ✅ | 特征重要性 JSON（bar 图必需） |
| `shap_values` | string | ❌ | 2D SHAP 值数组 JSON（beeswarm 图需要） |
| `feature_values` | string | ❌ | 2D 特征值数组 JSON（beeswarm 图需要） |
| `feature_names` | string | ❌ | 特征名称列表 JSON |
| `target_property` | string | ❌ | 目标属性（默认 PCE） |
| `plot_type` | string | ❌ | `bar`（柱状图）或 `beeswarm`（蜂群图） |
| `max_features` | number | ❌ | 显示的最大特征数（默认 15） |

**输出示例**：
```json
{
  "success": true,
  "filepath": "analysis_output/shap/shap_summary_PCE_bar_20250129.png",
  "plot_type": "bar",
  "target_property": "PCE"
}
```

---

### 2.8 `shap_analyze_prediction` - 单样本预测解释

**功能**：分析单个材料预测的特征贡献，提供瀑布式分解。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `contributions` | string | ✅ | 特征贡献列表 JSON，如 `[{"feature": "bandgap_eV", "value": 1.55, "contribution": 0.8}]` |
| `base_value` | number | ✅ | 基准值（平均预测值） |
| `predicted_value` | number | ✅ | 预测值 |
| `target_property` | string | ❌ | 目标属性（默认 PCE） |

**输入示例**：
```json
{
  "contributions": [
    {"feature": "tolerance_factor", "value": 0.95, "contribution": 1.2},
    {"feature": "bandgap_eV", "value": 1.55, "contribution": 0.8},
    {"feature": "defect_density", "value": 1e15, "contribution": -0.5}
  ],
  "base_value": 15.0,
  "predicted_value": 18.5
}
```

**输出示例**：
```json
{
  "success": true,
  "base_value": 15.0,
  "predicted_value": 18.5,
  "top_positive": [
    {"feature": "tolerance_factor", "contribution": 1.2, "direction": "positive"}
  ],
  "top_negative": [
    {"feature": "defect_density", "contribution": -0.5, "direction": "negative"}
  ],
  "interpretation": "[SHAP Waterfall] Starting from base PCE=15.0%, the model predicts 18.5%. Positive contributors: tolerance_factor, bandgap_eV. Negative contributors: defect_density."
}
```

---

### 2.9 `shap_dependence_plot` - 特征依赖性图

**功能**：展示单个特征如何影响预测结果，可选择交互特征进行着色。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `feature_name` | string | ✅ | 要分析的特征名称 |
| `feature_values` | string | ✅ | 特征值列表 JSON，如 `[0.85, 0.90, 0.95, 1.0]` |
| `shap_values` | string | ✅ | SHAP 值列表 JSON，如 `[-0.5, 0.0, 0.5, 1.0]` |
| `interaction_values` | string | ❌ | 交互特征值列表（用于着色） |
| `interaction_feature` | string | ❌ | 交互特征名称 |
| `target_property` | string | ❌ | 目标属性（默认 PCE） |

**输出示例**：
```json
{
  "success": true,
  "feature": "tolerance_factor",
  "correlation": 0.991,
  "trend": "positive",
  "filepath": "analysis_output/shap/shap_dependence_tolerance_factor_20250129.png"
}
```

---

## 3. 支持的特征列表

SHAP 分析支持以下 **18 个特征**：

| 类别 | 特征名称 | 说明 |
|------|---------|------|
| **结构特征** | `A_site_ionic_radius` | A 位离子半径 |
| | `B_site_ionic_radius` | B 位离子半径 |
| | `tolerance_factor` | 容忍因子 |
| **电子特征** | `bandgap_eV` | 带隙 (eV) |
| | `formation_energy` | 形成能 |
| **组分特征** | `Cs_fraction`, `FA_fraction`, `MA_fraction` | A 位组分比例 |
| | `Pb_fraction`, `Sn_fraction` | B 位组分比例 |
| | `I_fraction`, `Br_fraction`, `Cl_fraction` | X 位组分比例 |
| **工艺特征** | `film_thickness_nm` | 薄膜厚度 (nm) |
| | `annealing_temp_C` | 退火温度 (°C) |
| | `humidity_percent` | 湿度 (%) |
| | `light_intensity` | 光强 |
| | `defect_density` | 缺陷密度 |

**支持的目标属性**：PCE, Voc, Jsc, FF, T80

---

## 4. 可完成的任务

| 任务类型 | 描述 | 对应工具 |
|---------|------|---------|
| **晶体结构可视化** | 将 CIF 数据渲染为交互式 3D 视图 | `visualize_structure` |
| **降解机制诊断** | 识别热、湿度、光照引起的降解路径 | `analyze_mechanism` |
| **性能瓶颈分析** | 诊断 Voc/Jsc/FF 哪个是主要限制因素 | `analyze_mechanism` |
| **电荷平衡验证** | 检查化学式是否电荷平衡 | `analyze_stoichiometry` |
| **分子量计算** | 计算材料分子量和原子比例 | `analyze_stoichiometry` |
| **有机阳离子分析** | 分析 MA/FA/PEA 等的疏水性 | `analyze_organic_cation` |
| **抗湿性评估** | 基于 LogP 预测湿度稳定性 | `analyze_organic_cation` |
| **特征相关性** | 计算实验数据中的相关系数 | `calculate_correlation` |
| **全局特征重要性** | 确定对预测影响最大的特征 | `shap_feature_importance` |
| **单样本解释** | 解释为什么某材料预测值是 X | `shap_analyze_prediction` |

---

## 5. 查询示例

### 5.1 化学分析查询 (NEW)
| 查询类型 | 示例问题 |
|---------|---------|
| 电荷平衡 | "检查 CsPbI3 和 CsPbI2.5 的电荷平衡情况，哪个是合理的结构？" |
| 分子量 | "计算 FAPbI3 中 Pb 元素的质量百分比" |
| 疏水性 | "分析 PEA (苯乙基铵) 的 LogP 值，评估其能否提高湿度稳定性" |
| 阳离子对比 | "比较 MA 和 BA 的分子极性差异" |
| 2D 钙钛矿 | "分析长链阳离子 OA (辛胺) 作为 2D 钙钛矿间隔层的潜力" |
| 相关性 | "计算退火温度、时间与结晶度之间的皮尔逊相关系数" |
| 数据挖掘 | "分析当前数据集，找出与缺陷密度最相关的两个物理特征" |

### 5.2 机制分析查询
| 查询类型 | 示例问题 |
|---------|---------|
| 热降解 | "MAPbI3 在高温(>85°C)下的降解机制是什么？" |
| 湿度降解 | "为什么这个钙钛矿在湿度高时会失效？" |
| 性能诊断 | "这个材料 PCE 只有 15%，瓶颈在哪里？" |
| Voc 分析 | "Voc 偏低的原因可能是什么？" |

### 5.3 SHAP 分析查询
| 查询类型 | 示例问题 |
|---------|---------|
| 特征重要性 | "根据这些特征重要性数据生成排名：{tolerance_factor: 0.25, bandgap_eV: 0.20}" |
| 可视化 | "用这些数据生成 PCE 的特征重要性柱状图" |
| 依赖性 | "分析 tolerance_factor 与 SHAP 值的关系，数据为 [0.85, 0.90, 0.95] 对应 [-0.5, 0.0, 0.5]" |
| 单样本解释 | "基准值 15%，预测值 18.5%，解释各特征贡献：bandgap +0.8, defect -0.5" |

> 💡 **提示**：SHAP 工具需要你提供来自 ML 模型的真实数据，而非内部生成模拟数据。

### 5.4 综合分析查询
| 查询类型 | 示例问题 |
|---------|---------|
| 综合 | "分析 MAPbI3 的化学计量比，并计算其有机组分 MA 的疏水性指数" |
| 对比 | "对比 BA 和 PEA 的 TPSA，判断谁更适合做钝化层" |

---

## 6. 输入输出总结

| 项目 | 内容 |
|------|------|
| **输入** | 化学式、CIF 结构、SMILES 分子式、实验数据 JSON、性能指标 |
| **输出** | 电荷平衡报告、分子描述符、相关性矩阵、3D 可视化、SHAP 图表 |
| **核心任务** | 化学验证、稳定性评估、数据挖掘、机制诊断、AI 解释 |

---

## 7. 依赖库

| 库 | 用途 | 安装命令 |
|----|------|---------|
| pymatgen | 化学计量分析 | `pip install pymatgen` |
| rdkit | 有机分子分析 | `pip install rdkit` |
| pandas | 数据分析 | `pip install pandas` |
| shap | 模型解释 | `pip install shap` |
| plotly | 3D 可视化 | `pip install plotly` |

---

*文档更新时间：2025年1月*
*PSC_Agents 项目组*
