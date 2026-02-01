#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
钙钛矿材料性质预测 API

根据 MatterGen 生成的 CIF 文件和材料化学式，调用预训练的 RF/GBDT/NN 模型进行性质预测。

预测目标 (8 个):
    - pce: 光电转换效率 (%)
    - dft_band_gap: DFT 计算带隙 (eV)
    - energy_above_hull: 热力学稳定性 (eV/atom)
    - stability_retention: 稳定性保持率 (%)
    - stability_t80: T80 衰减时间 (hours)
    - voc: 开路电压 (V)
    - jsc: 短路电流密度 (mA/cm²)
    - ff: 填充因子 (%)

使用方法:
    from predict_api import PerovskitePredictor
    
    # 初始化预测器
    predictor = PerovskitePredictor()
    
    # 从 CIF 文件预测
    results = predictor.predict_from_cif("path/to/structure.cif")
    
    # 批量预测 (从 MatterGen 输出目录)
    results = predictor.predict_from_mattergen_output("generation_results/xxx/cif/")
    
    # 从化学式预测 (仅使用成分特征)
    results = predictor.predict_from_formula("CsPbI3")
    
    # 使用指定模型
    results = predictor.predict_from_cif("structure.cif", model="RF")  # RF, GBDT, NN

Author: PSC_Agents
Date: 2026-01-21
"""

import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

warnings.filterwarnings('ignore')

# 添加当前目录到路径
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from process import cbfv_table, cif_vector_table
from revised_CBFV import composition

# pymatgen
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser


# =============================================================================
# 配置
# =============================================================================

# 模型路径
MODEL_DIR = SCRIPT_DIR / "data" / "model"

# 已训练的模型
AVAILABLE_MODELS = {
    "RF": MODEL_DIR / "model_cif_comp_RF_sp1_oliynyk_zero_r0.pkl",
    "GBDT": MODEL_DIR / "model_cif_comp_GBDT_sp1_oliynyk_zero_r0.pkl",
    "NN": MODEL_DIR / "model_cif_comp_NN_sp1_oliynyk_zero_r0.pkl",
}

# 预测目标
TARGET_PROPERTIES = [
    "pce",              # 光电转换效率 (%)
    "dft_band_gap",     # DFT 带隙 (eV)
    "energy_above_hull",# 热力学稳定性 (eV/atom)
    "stability_retention",  # 稳定性保持率 (%)
    "stability_t80",    # T80 衰减时间 (hours)
    "voc",              # 开路电压 (V)
    "jsc",              # 短路电流密度 (mA/cm²)
    "ff",               # 填充因子 (%)
]

# 特征配置
FEATURE_CONFIG = {
    "use_X": "cif_comp",
    "split_way": 1,
    "per_elem_prop": "oliynyk",
    "fill_way": "zero",
}

# 特征列路径
COLUMNS_PATH = SCRIPT_DIR / "data" / "csr" / "cif_comp_sp1_oliynyk_zero_columns.npy"


# =============================================================================
# 特征提取
# =============================================================================

def extract_composition_features(formula: str, elem_prop: str = "oliynyk") -> np.ndarray:
    """从化学式提取 CBFV 特征
    
    Args:
        formula: 化学式，如 "CsPbI3"
        elem_prop: 元素特征类型 (oliynyk/magpie/mat2vec)
        
    Returns:
        特征向量 (264 维 for oliynyk)
    """
    try:
        df_temp = pd.DataFrame([[formula, 0]], columns=["formula", "target"])
        X, y, formulae, skipped = composition.generate_features(df_temp, elem_prop=elem_prop)
        return np.array(X.fillna(0))[0]
    except Exception as e:
        print(f"Warning: Failed to extract composition features for {formula}: {e}")
        if elem_prop == "oliynyk":
            return np.zeros(264)
        elif elem_prop == "magpie":
            return np.zeros(132)
        elif elem_prop == "mat2vec":
            return np.zeros(1200)
        return np.zeros(264)


def extract_cif_features(structure: Structure) -> np.ndarray:
    """从晶体结构提取特征
    
    Args:
        structure: pymatgen Structure 对象
        
    Returns:
        特征向量 (9 维)
    """
    try:
        feats = [
            structure.density,
            structure.volume,
            structure.lattice.a,
            structure.lattice.b,
            structure.lattice.c,
            structure.lattice.alpha,
            structure.lattice.beta,
            structure.lattice.gamma,
            structure.num_sites
        ]
        return np.array(feats)
    except Exception as e:
        print(f"Warning: Failed to extract CIF features: {e}")
        return np.zeros(9)


def extract_features(
    formula: str,
    structure: Optional[Structure] = None,
    cif_path: Optional[str] = None,
    cif_string: Optional[str] = None,
    elem_prop: str = "oliynyk"
) -> np.ndarray:
    """提取完整特征向量 (成分 + 结构)
    
    Args:
        formula: 化学式
        structure: pymatgen Structure 对象 (可选)
        cif_path: CIF 文件路径 (可选)
        cif_string: CIF 字符串 (可选)
        elem_prop: 元素特征类型
        
    Returns:
        完整特征向量 (273 维 = 264 成分 + 9 结构)
    """
    # 1. 成分特征
    comp_feats = extract_composition_features(formula, elem_prop)
    
    # 2. 结构特征
    if structure is not None:
        cif_feats = extract_cif_features(structure)
    elif cif_path is not None:
        structure = Structure.from_file(cif_path)
        cif_feats = extract_cif_features(structure)
    elif cif_string is not None:
        structure = Structure.from_str(cif_string, fmt="cif")
        cif_feats = extract_cif_features(structure)
    else:
        # 如果没有结构信息，使用零向量
        cif_feats = np.zeros(9)
    
    # 合并特征
    return np.concatenate([comp_feats, cif_feats])


# =============================================================================
# 预测器类
# =============================================================================

class PerovskitePredictor:
    """钙钛矿材料性质预测器
    
    封装 RF/GBDT/NN 三个模型，提供统一的预测接口。
    
    Attributes:
        models: 已加载的模型字典
        columns: 特征列名
        target_names: 预测目标名称列表
    """
    
    def __init__(self, load_all_models: bool = True):
        """初始化预测器
        
        Args:
            load_all_models: 是否一次性加载所有模型
        """
        self.models: Dict[str, Any] = {}
        self.columns: Optional[np.ndarray] = None
        self.target_names = TARGET_PROPERTIES
        
        # 加载特征列名
        if COLUMNS_PATH.exists():
            self.columns = np.load(str(COLUMNS_PATH), allow_pickle=True)
        
        # 加载模型
        if load_all_models:
            for name, path in AVAILABLE_MODELS.items():
                self.load_model(name)
    
    def load_model(self, model_name: str) -> bool:
        """加载指定模型
        
        Args:
            model_name: 模型名称 (RF/GBDT/NN)
            
        Returns:
            是否加载成功
        """
        if model_name in self.models:
            return True
            
        model_path = AVAILABLE_MODELS.get(model_name)
        if model_path is None:
            print(f"Error: Unknown model '{model_name}'. Available: {list(AVAILABLE_MODELS.keys())}")
            return False
            
        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}")
            return False
        
        try:
            self.models[model_name] = joblib.load(str(model_path))
            print(f"Loaded model: {model_name}")
            return True
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return False
    
    def _predict_single(
        self,
        features: np.ndarray,
        model_name: str = "RF"
    ) -> Dict[str, float]:
        """单样本预测
        
        Args:
            features: 特征向量
            model_name: 模型名称
            
        Returns:
            预测结果字典
        """
        if model_name not in self.models:
            if not self.load_model(model_name):
                return {}
        
        model = self.models[model_name]
        features_2d = features.reshape(1, -1)
        
        try:
            predictions = model.predict(features_2d)[0]
            return {name: float(pred) for name, pred in zip(self.target_names, predictions)}
        except Exception as e:
            print(f"Prediction error: {e}")
            return {}
    
    def predict_from_formula(
        self,
        formula: str,
        model: str = "RF"
    ) -> Dict[str, Any]:
        """从化学式预测 (仅使用成分特征)
        
        Args:
            formula: 化学式，如 "CsPbI3"
            model: 模型名称 (RF/GBDT/NN)
            
        Returns:
            预测结果
        """
        features = extract_features(formula, structure=None)
        predictions = self._predict_single(features, model)
        
        return {
            "formula": formula,
            "model": model,
            "predictions": predictions,
            "has_structure": False,
        }
    
    def predict_from_structure(
        self,
        structure: Structure,
        model: str = "RF"
    ) -> Dict[str, Any]:
        """从 pymatgen Structure 预测
        
        Args:
            structure: pymatgen Structure 对象
            model: 模型名称
            
        Returns:
            预测结果
        """
        formula = structure.composition.reduced_formula
        features = extract_features(formula, structure=structure)
        predictions = self._predict_single(features, model)
        
        return {
            "formula": formula,
            "model": model,
            "predictions": predictions,
            "has_structure": True,
            "structure_info": {
                "num_atoms": len(structure),
                "volume": structure.volume,
                "density": structure.density,
            }
        }
    
    def predict_from_cif(
        self,
        cif_path: str,
        model: str = "RF"
    ) -> Dict[str, Any]:
        """从 CIF 文件预测
        
        Args:
            cif_path: CIF 文件路径
            model: 模型名称
            
        Returns:
            预测结果
        """
        try:
            structure = Structure.from_file(cif_path)
            result = self.predict_from_structure(structure, model)
            result["cif_path"] = str(cif_path)
            return result
        except Exception as e:
            print(f"Error reading CIF file {cif_path}: {e}")
            return {"error": str(e), "cif_path": str(cif_path)}
    
    def predict_batch(
        self,
        items: List[Union[str, Structure]],
        model: str = "RF",
        input_type: str = "auto"
    ) -> List[Dict[str, Any]]:
        """批量预测
        
        Args:
            items: 输入列表 (化学式、CIF路径、或Structure对象)
            model: 模型名称
            input_type: 输入类型 ("auto", "formula", "cif", "structure")
            
        Returns:
            预测结果列表
        """
        results = []
        
        for item in items:
            if input_type == "structure" or isinstance(item, Structure):
                result = self.predict_from_structure(item, model)
            elif input_type == "cif" or (isinstance(item, str) and item.endswith(".cif")):
                result = self.predict_from_cif(item, model)
            elif input_type == "formula" or isinstance(item, str):
                # 检查是否为文件路径
                if os.path.isfile(item):
                    result = self.predict_from_cif(item, model)
                else:
                    result = self.predict_from_formula(item, model)
            else:
                result = {"error": f"Unknown input type: {type(item)}"}
            
            results.append(result)
        
        return results
    
    def predict_ensemble(
        self,
        formula: str = None,
        structure: Structure = None,
        cif_path: str = None
    ) -> Dict[str, Any]:
        """使用三个模型集成预测，返回均值和标准差
        
        Args:
            formula: 化学式
            structure: Structure 对象
            cif_path: CIF 文件路径
            
        Returns:
            集成预测结果 (包含均值、标准差和各模型预测)
        """
        # 提取特征
        if cif_path is not None:
            structure = Structure.from_file(cif_path)
            formula = structure.composition.reduced_formula
        elif structure is not None:
            formula = structure.composition.reduced_formula
        
        if formula is None:
            return {"error": "Must provide formula, structure, or cif_path"}
        
        features = extract_features(formula, structure=structure)
        
        # 三个模型预测
        all_predictions = {}
        for model_name in ["RF", "GBDT", "NN"]:
            all_predictions[model_name] = self._predict_single(features, model_name)
        
        # 计算集成结果
        ensemble_mean = {}
        ensemble_std = {}
        for target in self.target_names:
            values = [all_predictions[m].get(target, np.nan) for m in ["RF", "GBDT", "NN"]]
            values = [v for v in values if not np.isnan(v)]
            if values:
                ensemble_mean[target] = np.mean(values)
                ensemble_std[target] = np.std(values)
        
        return {
            "formula": formula,
            "has_structure": structure is not None,
            "ensemble_predictions": ensemble_mean,
            "ensemble_std": ensemble_std,
            "individual_predictions": all_predictions,
        }
    
    def predict_from_mattergen_output(
        self,
        output_dir: str,
        model: str = "ensemble"
    ) -> Dict[str, Any]:
        """从 MatterGen 生成输出目录预测
        
        Args:
            output_dir: MatterGen 输出目录 (包含 cif/ 子目录)
            model: 模型名称 ("RF"/"GBDT"/"NN"/"ensemble")
            
        Returns:
            预测结果汇总
        """
        output_dir = Path(output_dir)
        
        # 查找 CIF 文件
        cif_dir = output_dir / "cif"
        if not cif_dir.exists():
            cif_dir = output_dir  # 可能直接是 CIF 目录
        
        cif_files = list(cif_dir.glob("*.cif"))
        if not cif_files:
            return {"error": f"No CIF files found in {cif_dir}"}
        
        print(f"Found {len(cif_files)} CIF files in {cif_dir}")
        
        # 批量预测
        results = []
        for cif_file in cif_files:
            if model == "ensemble":
                result = self.predict_ensemble(cif_path=str(cif_file))
            else:
                result = self.predict_from_cif(str(cif_file), model)
            result["filename"] = cif_file.name
            results.append(result)
        
        # 汇总统计
        summary = self._summarize_results(results, model)
        
        return {
            "output_dir": str(output_dir),
            "num_structures": len(results),
            "model": model,
            "summary": summary,
            "results": results,
        }
    
    def _summarize_results(
        self,
        results: List[Dict],
        model: str
    ) -> Dict[str, Dict[str, float]]:
        """汇总预测结果统计
        
        Args:
            results: 预测结果列表
            model: 模型名称
            
        Returns:
            各属性的统计信息
        """
        summary = {}
        
        for target in self.target_names:
            values = []
            for r in results:
                if "error" in r:
                    continue
                if model == "ensemble":
                    val = r.get("ensemble_predictions", {}).get(target)
                else:
                    val = r.get("predictions", {}).get(target)
                if val is not None:
                    values.append(val)
            
            if values:
                summary[target] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }
        
        return summary


# =============================================================================
# 便捷函数
# =============================================================================

_predictor: Optional[PerovskitePredictor] = None

def get_predictor() -> PerovskitePredictor:
    """获取全局预测器实例（懒加载）"""
    global _predictor
    if _predictor is None:
        _predictor = PerovskitePredictor(load_all_models=True)
    return _predictor


def predict(
    formula: str = None,
    cif_path: str = None,
    structure: Structure = None,
    model: str = "ensemble"
) -> Dict[str, Any]:
    """便捷预测函数
    
    Args:
        formula: 化学式
        cif_path: CIF 文件路径
        structure: pymatgen Structure
        model: 模型名称 ("RF"/"GBDT"/"NN"/"ensemble")
        
    Returns:
        预测结果
    """
    predictor = get_predictor()
    
    if model == "ensemble":
        return predictor.predict_ensemble(formula, structure, cif_path)
    elif cif_path:
        return predictor.predict_from_cif(cif_path, model)
    elif structure:
        return predictor.predict_from_structure(structure, model)
    elif formula:
        return predictor.predict_from_formula(formula, model)
    else:
        return {"error": "Must provide formula, cif_path, or structure"}


def predict_mattergen_results(output_dir: str, model: str = "ensemble") -> Dict[str, Any]:
    """预测 MatterGen 生成结果
    
    Args:
        output_dir: MatterGen 输出目录
        model: 模型名称
        
    Returns:
        预测结果汇总
    """
    predictor = get_predictor()
    return predictor.predict_from_mattergen_output(output_dir, model)


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="钙钛矿材料性质预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 从化学式预测 (使用集成模型)
    python predict_api.py --formula CsPbI3
    
    # 从 CIF 文件预测
    python predict_api.py --cif structure.cif --model RF
    
    # 批量预测 MatterGen 输出
    python predict_api.py --mattergen-dir generation_results/xxx/
    
    # 预测多个化学式
    python predict_api.py --formulas CsPbI3 CsPbBr3 CsSnI3
        """
    )
    
    parser.add_argument("--formula", type=str, help="单个化学式")
    parser.add_argument("--formulas", nargs="+", help="多个化学式")
    parser.add_argument("--cif", type=str, help="CIF 文件路径")
    parser.add_argument("--cif-dir", type=str, help="CIF 文件目录")
    parser.add_argument("--mattergen-dir", type=str, help="MatterGen 输出目录")
    parser.add_argument("--model", type=str, default="ensemble",
                        choices=["RF", "GBDT", "NN", "ensemble"],
                        help="预测模型 (默认: ensemble)")
    parser.add_argument("--output", type=str, help="输出 JSON 文件路径")
    
    args = parser.parse_args()
    
    predictor = PerovskitePredictor()
    results = None
    
    # 从化学式预测
    if args.formula:
        if args.model == "ensemble":
            results = predictor.predict_ensemble(formula=args.formula)
        else:
            results = predictor.predict_from_formula(args.formula, args.model)
    
    # 批量化学式
    elif args.formulas:
        results = []
        for formula in args.formulas:
            if args.model == "ensemble":
                r = predictor.predict_ensemble(formula=formula)
            else:
                r = predictor.predict_from_formula(formula, args.model)
            results.append(r)
    
    # 从 CIF 文件
    elif args.cif:
        if args.model == "ensemble":
            results = predictor.predict_ensemble(cif_path=args.cif)
        else:
            results = predictor.predict_from_cif(args.cif, args.model)
    
    # 从 CIF 目录
    elif args.cif_dir:
        results = predictor.predict_from_mattergen_output(args.cif_dir, args.model)
    
    # 从 MatterGen 输出
    elif args.mattergen_dir:
        results = predictor.predict_from_mattergen_output(args.mattergen_dir, args.model)
    
    else:
        parser.print_help()
        return
    
    # 打印结果
    print("\n" + "="*60)
    print("预测结果")
    print("="*60)
    
    if isinstance(results, dict) and "results" in results:
        # MatterGen 批量结果
        print(f"\n📊 汇总统计 ({results['num_structures']} 个结构):")
        print("-"*50)
        for prop, stats in results.get("summary", {}).items():
            print(f"  {prop:20}: {stats['mean']:.4f} ± {stats['std']:.4f} (范围: {stats['min']:.4f} ~ {stats['max']:.4f})")
        
        print(f"\n📋 前 5 个结构预测:")
        print("-"*50)
        for r in results["results"][:5]:
            formula = r.get("formula", "N/A")
            if args.model == "ensemble":
                preds = r.get("ensemble_predictions", {})
            else:
                preds = r.get("predictions", {})
            pce = preds.get("pce", 0)
            bg = preds.get("dft_band_gap", 0)
            print(f"  {formula:15} | PCE: {pce:.2f}% | 带隙: {bg:.2f} eV")
    
    elif isinstance(results, list):
        # 批量化学式结果
        for r in results:
            formula = r.get("formula", "N/A")
            if args.model == "ensemble":
                preds = r.get("ensemble_predictions", {})
            else:
                preds = r.get("predictions", {})
            print(f"\n{formula}:")
            for prop, val in preds.items():
                print(f"  {prop:20}: {val:.4f}")
    
    else:
        # 单个结果
        formula = results.get("formula", "N/A")
        print(f"\n化学式: {formula}")
        
        if args.model == "ensemble":
            print(f"\n集成预测 (RF + GBDT + NN 均值):")
            for prop, val in results.get("ensemble_predictions", {}).items():
                std = results.get("ensemble_std", {}).get(prop, 0)
                print(f"  {prop:20}: {val:.4f} ± {std:.4f}")
        else:
            print(f"\n{args.model} 模型预测:")
            for prop, val in results.get("predictions", {}).items():
                print(f"  {prop:20}: {val:.4f}")
    
    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 结果已保存: {args.output}")


if __name__ == "__main__":
    main()
