#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
perovskite_predictor.py
钙钛矿太阳能电池性质预测器 - 用于 FabAgent 集成

提供基于 RF 模型的性质预测功能:
- PCE (光电转换效率)
- Voc (开路电压)
- Jsc (短路电流密度)
- FF (填充因子)
- dft_band_gap (DFT带隙)
- energy_above_hull (凸包上方能量)

Author: PSC_Agents Team
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 设置路径
PREDICTOR_DIR = Path(__file__).parent.absolute()
MODEL_PROJECT_DIR = PREDICTOR_DIR / "Perovskite_PI_Multi"

# 添加模型项目到路径
sys.path.insert(0, str(MODEL_PROJECT_DIR))

# 延迟导入，避免启动时加载大量依赖
_numpy = None
_pandas = None
_joblib = None
_CBFV_AVAILABLE = False
_PYMATGEN_AVAILABLE = False


def _lazy_import():
    """延迟导入依赖"""
    global _numpy, _pandas, _joblib, _CBFV_AVAILABLE, _PYMATGEN_AVAILABLE
    
    if _numpy is None:
        import numpy as np
        import pandas as pd
        import joblib
        _numpy = np
        _pandas = pd
        _joblib = joblib
        
        # 检查 CBFV
        try:
            from revised_CBFV import composition
            _CBFV_AVAILABLE = True
        except ImportError:
            _CBFV_AVAILABLE = False
        
        # 检查 pymatgen
        try:
            from pymatgen.io.cif import CifParser
            from pymatgen.core.structure import Structure
            _PYMATGEN_AVAILABLE = True
        except ImportError:
            _PYMATGEN_AVAILABLE = False


# =============================================================================
# 常量定义
# =============================================================================

MODEL_DIR = MODEL_PROJECT_DIR / "data" / "model" / "single_target"

# 所有支持的目标属性
ALL_TARGETS = ["pce", "voc", "jsc", "ff", "dft_band_gap", "energy_above_hull"]

# 目标属性信息
TARGET_INFO = {
    "pce": {"name": "PCE", "unit": "%", "full_name": "Power Conversion Efficiency"},
    "voc": {"name": "Voc", "unit": "V", "full_name": "Open-Circuit Voltage"},
    "jsc": {"name": "Jsc", "unit": "mA/cm²", "full_name": "Short-Circuit Current Density"},
    "ff": {"name": "FF", "unit": "%", "full_name": "Fill Factor"},
    "dft_band_gap": {"name": "Band Gap", "unit": "eV", "full_name": "DFT Band Gap"},
    "energy_above_hull": {"name": "E_hull", "unit": "eV/atom", "full_name": "Energy Above Hull"}
}

# 默认模型类型
DEFAULT_MODEL_TYPE = "RF"


# =============================================================================
# 特征生成函数
# =============================================================================

def generate_cbfv_features(composition_str: str, elem_prop: str = "oliynyk"):
    """生成基于成分的特征向量 (CBFV)"""
    _lazy_import()
    
    if not _CBFV_AVAILABLE:
        print("Warning: CBFV not available, using zero features")
        return _numpy.zeros(264)
    
    # 清理输入
    composition_str = composition_str.replace("|", "")
    
    # 加载缩写对照表
    corr_path = MODEL_PROJECT_DIR / "revised_CBFV" / "Perovskite_a_ion_correspond_arr.csv"
    if corr_path.exists():
        try:
            corr = _pandas.read_csv(corr_path)
            for i in range(len(corr)):
                composition_str = composition_str.replace(
                    str(corr["Abbreviation"].iloc[i]), 
                    str(corr["Chemical Formula"].iloc[i])
                )
        except Exception:
            pass
    
    try:
        from revised_CBFV import composition
        df_temp = _pandas.DataFrame([[composition_str, 0]], columns=["formula", "target"])
        X, y, formulae, skipped = composition.generate_features(df_temp, elem_prop=elem_prop)
        return X.fillna(0).values[0]
    except Exception as e:
        print(f"CBFV feature generation failed for '{composition_str}': {e}")
        return _numpy.zeros(264)


def generate_cif_features(cif_content: str):
    """从 CIF 文件内容生成晶体结构特征"""
    _lazy_import()
    
    if not _PYMATGEN_AVAILABLE:
        print("Warning: pymatgen not available, using zero features")
        return _numpy.zeros(9)
    
    # 修复转义字符
    if "\\n" in cif_content:
        cif_content = cif_content.replace("\\n", "\n")
    
    default_feats = _numpy.zeros(9)
    
    try:
        from pymatgen.io.cif import CifParser
        from pymatgen.core.structure import Structure
        
        try:
            parser = CifParser.from_string(cif_content)
            structure = parser.get_structures()[0]
        except:
            structure = Structure.from_str(cif_content, fmt="cif")
        
        return _numpy.array([
            structure.density,
            structure.volume,
            structure.lattice.a,
            structure.lattice.b,
            structure.lattice.c,
            structure.lattice.alpha,
            structure.lattice.beta,
            structure.lattice.gamma,
            structure.num_sites
        ])
    except Exception as e:
        print(f"CIF parsing failed: {e}")
        return default_feats


# =============================================================================
# 预测器类
# =============================================================================

class PerovskitePredictor:
    """钙钛矿太阳能电池性质预测器"""
    
    _instance = None  # 单例模式
    
    def __new__(cls, *args, **kwargs):
        """单例模式，避免重复加载模型"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_type: str = DEFAULT_MODEL_TYPE):
        """
        初始化预测器
        
        Args:
            model_type: 模型类型 (RF, GBDT, NN)，默认 RF
        """
        if self._initialized and self.model_type == model_type:
            return
        
        _lazy_import()
        
        self.model_type = model_type
        self.model_dir = MODEL_DIR
        self.models: Dict[str, Dict[str, Any]] = {}
        self._initialized = True
        self._scan_and_load_models()
    
    def _scan_and_load_models(self):
        """扫描并加载模型"""
        print(f"[PerovskitePredictor] Loading {self.model_type} models...")
        
        for input_mode in ["comp_only", "cif_only"]:
            self.models[input_mode] = {}
            for target in ALL_TARGETS:
                model_path = self.model_dir / f"{input_mode}_{self.model_type}_{target}.pkl"
                if model_path.exists():
                    try:
                        self.models[input_mode][target] = _joblib.load(model_path)
                        print(f"  ✓ Loaded: {input_mode}/{target}")
                    except Exception as e:
                        print(f"  ✗ Failed to load {input_mode}/{target}: {e}")
                else:
                    print(f"  - Not found: {model_path.name}")
    
    def get_available_targets(self, input_mode: str = "comp_only") -> List[str]:
        """获取可用的预测目标"""
        return list(self.models.get(input_mode, {}).keys())
    
    def predict_from_composition(
        self, 
        composition: str,
        targets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        根据分子式预测性质
        
        Args:
            composition: 分子式字符串，如 "CsPbI3", "FA0.25MA0.75PbI3"
            targets: 要预测的目标列表，默认预测所有
            
        Returns:
            预测结果字典
        """
        if targets is None:
            targets = ALL_TARGETS
        
        # 生成特征
        features = generate_cbfv_features(composition)
        X = features.reshape(1, -1)
        
        results = {
            "input": composition,
            "input_mode": "composition",
            "model_type": self.model_type,
            "predictions": {}
        }
        
        for target in targets:
            if target in self.models.get("comp_only", {}):
                try:
                    model = self.models["comp_only"][target]
                    pred = model.predict(X)[0]
                    info = TARGET_INFO.get(target, {})
                    results["predictions"][target] = {
                        "value": float(pred),
                        "unit": info.get("unit", ""),
                        "name": info.get("name", target)
                    }
                except Exception as e:
                    results["predictions"][target] = {"error": str(e)}
            else:
                results["predictions"][target] = {"error": "Model not available"}
        
        return results
    
    def predict_from_cif(
        self,
        cif_content: str,
        targets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        根据 CIF 结构预测性质
        
        Args:
            cif_content: CIF 文件内容字符串
            targets: 要预测的目标列表，默认预测所有
            
        Returns:
            预测结果字典
        """
        if targets is None:
            targets = ALL_TARGETS
        
        # 生成特征
        features = generate_cif_features(cif_content)
        X = features.reshape(1, -1)
        
        results = {
            "input": "CIF structure",
            "input_mode": "cif",
            "model_type": self.model_type,
            "predictions": {}
        }
        
        for target in targets:
            if target in self.models.get("cif_only", {}):
                try:
                    model = self.models["cif_only"][target]
                    pred = model.predict(X)[0]
                    info = TARGET_INFO.get(target, {})
                    results["predictions"][target] = {
                        "value": float(pred),
                        "unit": info.get("unit", ""),
                        "name": info.get("name", target)
                    }
                except Exception as e:
                    results["predictions"][target] = {"error": str(e)}
            else:
                results["predictions"][target] = {"error": "Model not available"}
        
        return results
    
    def predict(
        self,
        composition: Optional[str] = None,
        cif_content: Optional[str] = None,
        cif_file: Optional[str] = None,
        targets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        统一预测接口
        
        Args:
            composition: 分子式字符串
            cif_content: CIF 内容字符串
            cif_file: CIF 文件路径
            targets: 预测目标列表
            
        Returns:
            预测结果
        """
        # 优先使用分子式
        if composition:
            return self.predict_from_composition(composition, targets)
        
        # 其次使用 CIF
        if cif_content:
            return self.predict_from_cif(cif_content, targets)
        
        # 从文件读取 CIF
        if cif_file and os.path.exists(cif_file):
            with open(cif_file, 'r') as f:
                cif_content = f.read()
            return self.predict_from_cif(cif_content, targets)
        
        return {"error": "No valid input provided. Use composition, cif_content, or cif_file."}
    
    def format_results_for_visualization(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        将预测结果格式化为可视化工具所需的格式
        
        Returns:
            {PCE_percent, Voc_V, Jsc_mA_cm2, FF_percent, ...}
        """
        predictions = results.get("predictions", {})
        formatted = {}
        
        # 映射到可视化格式
        mapping = {
            "pce": "PCE_percent",
            "voc": "Voc_V",
            "jsc": "Jsc_mA_cm2",
            "ff": "FF_percent",
            "dft_band_gap": "BandGap_eV",
            "energy_above_hull": "E_hull_eV"
        }
        
        for target, viz_name in mapping.items():
            if target in predictions:
                pred = predictions[target]
                if isinstance(pred, dict) and "value" in pred:
                    formatted[viz_name] = pred["value"]
        
        return formatted


# =============================================================================
# 便捷函数
# =============================================================================

_predictor_instance: Optional[PerovskitePredictor] = None


def get_predictor(model_type: str = DEFAULT_MODEL_TYPE) -> PerovskitePredictor:
    """获取预测器实例（单例）"""
    global _predictor_instance
    if _predictor_instance is None or _predictor_instance.model_type != model_type:
        _predictor_instance = PerovskitePredictor(model_type)
    return _predictor_instance


def predict_perovskite_properties(
    composition: Optional[str] = None,
    cif_content: Optional[str] = None,
    cif_file: Optional[str] = None,
    targets: Optional[List[str]] = None,
    model_type: str = DEFAULT_MODEL_TYPE
) -> Dict[str, Any]:
    """
    预测钙钛矿太阳能电池性质 - 主要入口函数
    
    Args:
        composition: 分子式，如 "CsPbI3", "FA0.25MA0.75PbI3", "MAPbI3"
        cif_content: CIF 文件内容字符串
        cif_file: CIF 文件路径
        targets: 预测目标列表，可选值: pce, voc, jsc, ff, dft_band_gap, energy_above_hull
        model_type: 模型类型，默认 "RF"
        
    Returns:
        预测结果字典，包含 predictions 和 formatted_for_viz
        
    Example:
        >>> result = predict_perovskite_properties(composition="CsPbI3")
        >>> print(result["predictions"]["pce"]["value"])  # PCE 预测值
    """
    predictor = get_predictor(model_type)
    results = predictor.predict(
        composition=composition,
        cif_content=cif_content,
        cif_file=cif_file,
        targets=targets
    )
    
    # 添加格式化后的结果和状态
    if "predictions" in results:
        results["formatted_for_viz"] = predictor.format_results_for_visualization(results)
        # 检查是否有有效预测
        has_valid_prediction = any(
            isinstance(p, dict) and "value" in p 
            for p in results["predictions"].values()
        )
        results["status"] = "success" if has_valid_prediction else "partial"
    else:
        results["status"] = "error"
    
    return results


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing PerovskitePredictor")
    print("=" * 60)
    
    # 测试分子式预测
    test_compositions = ["CsPbI3", "MAPbI3", "FA0.25MA0.75PbI3"]
    
    for comp in test_compositions:
        print(f"\n🔹 Testing: {comp}")
        result = predict_perovskite_properties(composition=comp)
        
        if "predictions" in result:
            for target, pred in result["predictions"].items():
                if isinstance(pred, dict) and "value" in pred:
                    print(f"   {pred.get('name', target)}: {pred['value']:.4f} {pred.get('unit', '')}")
                elif isinstance(pred, dict) and "error" in pred:
                    print(f"   {target}: ERROR - {pred['error']}")
        
        print(f"   Viz format: {result.get('formatted_for_viz', {})}")
