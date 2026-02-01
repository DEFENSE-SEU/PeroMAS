"""
Chemistry Analysis Tools for AnalysisAgent

Provides scientific analysis capabilities:
1. Stoichiometry analysis (pymatgen)
2. Organic cation analysis (rdkit)
3. Correlation analysis (sklearn/pandas)

Author: PSC_Agents Team
"""

import json
from typing import Dict, Any, Optional

# ============================================================
# Pymatgen - Stoichiometry Analysis
# ============================================================

try:
    from pymatgen.core import Composition
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False


def analyze_stoichiometry(formula: str) -> Dict[str, Any]:
    """
    分析化学式的基本属性：分子量、原子占比、氧化态猜测、电荷平衡
    
    Args:
        formula: 化学式字符串 (e.g., "CsPbI3", "FA0.8Cs0.2PbI2.4Br0.6")
        
    Returns:
        Dict with molecular weight, atomic fractions, oxidation states, charge balance
    """
    if not HAS_PYMATGEN:
        return {"error": "pymatgen not installed. Run: pip install pymatgen"}
    
    try:
        # 处理常见的有机阳离子缩写
        formula_processed = formula
        # MA = CH3NH3 = CH6N
        formula_processed = formula_processed.replace("MA", "CH6N")
        # FA = CH(NH2)2 = CH5N2
        formula_processed = formula_processed.replace("FA", "CH5N2")
        
        comp = Composition(formula_processed)
        
        # 获取氧化态猜测
        oxi_guesses = []
        try:
            oxi_guesses = comp.oxi_state_guesses()
        except Exception:
            oxi_guesses = []
        
        # 格式化氧化态猜测结果
        oxi_state_formatted = []
        for guess in oxi_guesses[:3]:  # 只取前3个猜测
            oxi_state_formatted.append({str(k): v for k, v in guess.items()})
        
        return {
            "success": True,
            "original_formula": formula,
            "formula_pretty": comp.reduced_formula,
            "formula_hill": comp.hill_formula,
            "molecular_weight": round(comp.weight, 2),
            "num_atoms": comp.num_atoms,
            "elements": [str(e) for e in comp.elements],
            "atomic_fractions": {str(k): round(v, 4) for k, v in comp.fractional_composition.as_dict().items()},
            "element_amounts": {str(k): round(v, 4) for k, v in comp.as_dict().items()},
            "oxi_state_guesses": oxi_state_formatted,
            "is_charge_balanced": len(oxi_guesses) > 0,
            "charge_balance_note": "Charge balanced" if len(oxi_guesses) > 0 else "No valid oxidation state combination found - may indicate unbalanced formula"
        }
    except Exception as e:
        return {"success": False, "error": str(e), "original_formula": formula}


# ============================================================
# RDKit - Organic Cation Analysis  
# ============================================================

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# 常见钙钛矿有机阳离子的 SMILES 字典
COMMON_CATIONS = {
    "MA": {"smiles": "C[NH3+]", "name": "Methylammonium (MA)", "formula": "CH6N+"},
    "FA": {"smiles": "[NH2+]=CN", "name": "Formamidinium (FA)", "formula": "CH5N2+"},
    "PEA": {"smiles": "NCCc1ccccc1", "name": "Phenethylammonium (PEA)", "formula": "C8H12N+"},
    "BA": {"smiles": "CCCCN", "name": "Butylammonium (BA)", "formula": "C4H12N+"},
    "OA": {"smiles": "CCCCCCCCN", "name": "Octylammonium (OA)", "formula": "C8H20N+"},
    "GA": {"smiles": "NC(=N)N", "name": "Guanidinium (GA)", "formula": "CH6N3+"},
    "EA": {"smiles": "CCN", "name": "Ethylammonium (EA)", "formula": "C2H8N+"},
    "DMA": {"smiles": "C[NH2+]C", "name": "Dimethylammonium (DMA)", "formula": "C2H8N+"},
}


def analyze_organic_cation(smiles: str, name: Optional[str] = None) -> Dict[str, Any]:
    """
    分析有机阳离子的性质，预测湿度稳定性
    
    Args:
        smiles: SMILES 字符串
        name: 可选的阳离子名称
        
    Returns:
        Dict with LogP, TPSA, molecular weight, hydrophobicity assessment
    """
    if not HAS_RDKIT:
        return {"error": "rdkit not installed. Run: pip install rdkit"}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {"success": False, "error": f"Invalid SMILES: {smiles}"}
        
        # 计算分子描述符
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        mw = Descriptors.MolWt(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)  # 氢键供体
        hba = rdMolDescriptors.CalcNumHBA(mol)  # 氢键受体
        rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        
        # 疏水性评估
        if logp > 2.0:
            hydrophobicity = "High (excellent moisture resistance)"
        elif logp > 0.5:
            hydrophobicity = "Moderate (good moisture resistance)"
        elif logp > -1.0:
            hydrophobicity = "Low (limited moisture resistance)"
        else:
            hydrophobicity = "Very Low (poor moisture resistance)"
        
        # 2D钙钛矿适用性评估
        if rotatable_bonds >= 3 and mw > 100:
            spacer_suitability = "Good for 2D perovskite spacer layer"
        elif mw > 150:
            spacer_suitability = "Suitable for quasi-2D structure"
        else:
            spacer_suitability = "Better for 3D perovskite A-site"
        
        return {
            "success": True,
            "smiles": smiles,
            "name": name or "Unknown",
            "molecular_weight": round(mw, 2),
            "LogP": round(logp, 3),
            "TPSA": round(tpsa, 2),
            "H_bond_donors": hbd,
            "H_bond_acceptors": hba,
            "rotatable_bonds": rotatable_bonds,
            "hydrophobicity_assessment": hydrophobicity,
            "spacer_suitability": spacer_suitability,
            "stability_prediction": {
                "moisture_resistance": "High" if logp > 1 else "Medium" if logp > -0.5 else "Low",
                "thermal_stability": "Depends on bond strength",
                "recommendation": "Higher LogP = better humidity stability"
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_cation_info(cation_name: str) -> Dict[str, Any]:
    """
    获取常见有机阳离子的信息
    
    Args:
        cation_name: 阳离子缩写 (MA, FA, PEA, BA, OA, GA, EA, DMA)
        
    Returns:
        Dict with cation info and analysis
    """
    cation_upper = cation_name.upper()
    
    if cation_upper in COMMON_CATIONS:
        info = COMMON_CATIONS[cation_upper]
        analysis = analyze_organic_cation(info["smiles"], info["name"])
        analysis["formula"] = info["formula"]
        return analysis
    else:
        return {
            "success": False,
            "error": f"Unknown cation: {cation_name}",
            "available_cations": list(COMMON_CATIONS.keys())
        }


def compare_cations(cation_list: list) -> Dict[str, Any]:
    """
    比较多个有机阳离子的性质
    
    Args:
        cation_list: 阳离子名称列表 (e.g., ["MA", "FA", "PEA"])
        
    Returns:
        Dict with comparison table
    """
    results = []
    for cation in cation_list:
        info = get_cation_info(cation)
        if info.get("success"):
            results.append({
                "name": info.get("name", cation),
                "LogP": info.get("LogP"),
                "TPSA": info.get("TPSA"),
                "MW": info.get("molecular_weight"),
                "moisture_resistance": info.get("stability_prediction", {}).get("moisture_resistance")
            })
    
    # 按 LogP 排序（从高到低，疏水性从强到弱）
    results_sorted = sorted(results, key=lambda x: x.get("LogP", -999), reverse=True)
    
    return {
        "success": True,
        "comparison": results_sorted,
        "best_for_humidity": results_sorted[0]["name"] if results_sorted else None,
        "recommendation": f"{results_sorted[0]['name']} has highest LogP ({results_sorted[0]['LogP']}), best for moisture resistance" if results_sorted else "No valid cations"
    }


# ============================================================
# Sklearn/Pandas - Correlation Analysis
# ============================================================

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def calculate_correlation(data_json: str, target_column: str = "PCE") -> Dict[str, Any]:
    """
    计算特征之间的皮尔逊相关系数
    
    Args:
        data_json: JSON 字符串格式的数据 (list of dicts)
        target_column: 目标列名，用于分析相关性 (default: "PCE")
        
    Returns:
        Dict with correlation matrix and top correlated features
    """
    if not HAS_PANDAS:
        return {"error": "pandas not installed. Run: pip install pandas numpy"}
    
    try:
        # 解析 JSON 数据
        if isinstance(data_json, str):
            data = json.loads(data_json)
        else:
            data = data_json
            
        df = pd.DataFrame(data)
        
        # 只保留数值列
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {"success": False, "error": "No numeric columns found in data"}
        
        # 计算相关系数矩阵
        corr_matrix = numeric_df.corr(method='pearson').round(3)
        
        result = {
            "success": True,
            "n_samples": len(df),
            "n_features": len(numeric_df.columns),
            "features": list(numeric_df.columns),
            "correlation_matrix": corr_matrix.to_dict()
        }
        
        # 如果目标列存在，找出与其最相关的特征
        if target_column in corr_matrix.columns:
            target_corr = corr_matrix[target_column].abs().sort_values(ascending=False)
            # 排除自身
            target_corr = target_corr[target_corr.index != target_column]
            
            top_positive = corr_matrix[target_column].sort_values(ascending=False)
            top_positive = top_positive[top_positive.index != target_column][:3]
            
            top_negative = corr_matrix[target_column].sort_values(ascending=True)
            top_negative = top_negative[top_negative.index != target_column][:3]
            
            result["target_analysis"] = {
                "target_column": target_column,
                "top_correlated_features": target_corr[:5].to_dict(),
                "top_positive_correlations": top_positive.to_dict(),
                "top_negative_correlations": top_negative.to_dict(),
                "interpretation": f"Features most strongly correlated with {target_column}"
            }
        else:
            result["target_analysis"] = {
                "note": f"Target column '{target_column}' not found. Available columns: {list(numeric_df.columns)}"
            }
        
        return result
        
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def calculate_feature_statistics(data_json: str) -> Dict[str, Any]:
    """
    计算数据集的基本统计信息
    
    Args:
        data_json: JSON 字符串格式的数据
        
    Returns:
        Dict with statistics for each feature
    """
    if not HAS_PANDAS:
        return {"error": "pandas not installed. Run: pip install pandas numpy"}
    
    try:
        if isinstance(data_json, str):
            data = json.loads(data_json)
        else:
            data = data_json
            
        df = pd.DataFrame(data)
        numeric_df = df.select_dtypes(include=[np.number])
        
        stats = numeric_df.describe().round(3).to_dict()
        
        return {
            "success": True,
            "n_samples": len(df),
            "statistics": stats
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
