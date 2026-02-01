#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_predictor_test.py
钙钛矿太阳能电池性质预测测试脚本

功能:
    1. 支持选择模型类型 (RF, GBDT, NN)
    2. 支持选择输入类型 (comp_only: 仅分子式, cif_only: 晶体结构)
    3. 支持选择预测目标 (pce, voc, jsc, ff, dft_band_gap, energy_above_hull)
    4. 支持单个/批量预测
    5. 支持交互模式

使用方法:
    # 交互模式
    python model_predictor_test.py --interactive
    
    # 命令行单次预测 - 分子式
    python model_predictor_test.py --composition "FA0.25MA0.75PbI3" --model RF --target pce
    
    # 命令行预测所有属性
    python model_predictor_test.py --composition "CsPbI3" --model RF --all-targets
    
    # 批量预测多个分子式
    python model_predictor_test.py --compositions "FA0.25MA0.75PbI3,CsPbI3,MAPbI3" --model RF
    
    # 使用CIF文件预测
    python model_predictor_test.py --cif-file "structure.cif" --model RF --target dft_band_gap
    
    # 比较不同模型
    python model_predictor_test.py --composition "CsPbI3" --compare-models

作者: PSC_Agents
日期: 2025-01
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置路径
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

# 导入特征生成模块
try:
    from revised_CBFV import composition
    CBFV_AVAILABLE = True
except ImportError:
    CBFV_AVAILABLE = False
    print("Warning: revised_CBFV not available, composition features disabled")

try:
    from pymatgen.io.cif import CifParser
    from pymatgen.core.structure import Structure
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    print("Warning: pymatgen not available, CIF features disabled")


# =============================================================================
# 常量定义
# =============================================================================

MODEL_DIR = SCRIPT_DIR / "data" / "model" / "single_target"

# 所有支持的目标属性
ALL_TARGETS = ["pce", "voc", "jsc", "ff", "dft_band_gap", "energy_above_hull"]

# 目标属性中文描述和单位
TARGET_INFO = {
    "pce": {"name": "光电转换效率 (PCE)", "unit": "%", "description": "Power Conversion Efficiency"},
    "voc": {"name": "开路电压 (Voc)", "unit": "V", "description": "Open-Circuit Voltage"},
    "jsc": {"name": "短路电流密度 (Jsc)", "unit": "mA/cm²", "description": "Short-Circuit Current Density"},
    "ff": {"name": "填充因子 (FF)", "unit": "%", "description": "Fill Factor"},
    "dft_band_gap": {"name": "DFT带隙", "unit": "eV", "description": "DFT-calculated Band Gap"},
    "energy_above_hull": {"name": "凸包上方能量", "unit": "eV/atom", "description": "Energy Above Convex Hull"}
}

# 支持的模型类型
MODEL_TYPES = ["RF", "GBDT", "NN"]

# 输入模式
INPUT_MODES = ["comp_only", "cif_only"]


# =============================================================================
# 特征生成函数
# =============================================================================

def generate_cbfv_features(composition_str: str, elem_prop: str = "oliynyk") -> np.ndarray:
    """
    生成基于成分的特征向量 (CBFV)
    
    Args:
        composition_str: 分子式字符串
        elem_prop: 元素属性集 (oliynyk, magpie)
    
    Returns:
        特征向量
    """
    if not CBFV_AVAILABLE:
        raise ImportError("revised_CBFV 模块未安装，无法生成组分特征")
    
    # 清理输入
    composition_str = composition_str.replace("|", "")
    
    # 加载缩写对照表
    corr_path = SCRIPT_DIR / "revised_CBFV" / "Perovskite_a_ion_correspond_arr.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        for i in range(len(corr)):
            composition_str = composition_str.replace(
                corr["Abbreviation"].iloc[i], 
                corr["Chemical Formula"].iloc[i]
            )
    
    try:
        df_temp = pd.DataFrame([[composition_str, 0]], columns=["formula", "target"])
        X, y, formulae, skipped = composition.generate_features(df_temp, elem_prop=elem_prop)
        return X.fillna(0).values[0]
    except Exception as e:
        print(f"CBFV 特征生成失败 '{composition_str}': {e}")
        # 返回默认零向量
        if elem_prop == "oliynyk":
            return np.zeros(264)
        elif elem_prop == "magpie":
            return np.zeros(132)
        else:
            return np.zeros(264)


def generate_cif_features(cif_content: str) -> np.ndarray:
    """
    从 CIF 文件内容生成晶体结构特征
    
    Args:
        cif_content: CIF 文件内容字符串
    
    Returns:
        9维特征向量 [density, volume, a, b, c, alpha, beta, gamma, num_sites]
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgen 未安装，无法解析 CIF 文件")
    
    # 修复可能的转义字符
    if "\\n" in cif_content:
        cif_content = cif_content.replace("\\n", "\n")
    
    default_feats = np.zeros(9)
    
    try:
        parser = CifParser.from_string(cif_content)
        structure = parser.get_structures()[0]
    except:
        try:
            structure = Structure.from_str(cif_content, fmt="cif")
        except Exception as e:
            print(f"CIF 解析失败: {e}")
            return default_feats
    
    return np.array([
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


# =============================================================================
# 模型管理类
# =============================================================================

class PerovskiteModelManager:
    """钙钛矿预测模型管理器"""
    
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded_models: Dict[str, object] = {}
        self._scan_available_models()
    
    def _scan_available_models(self):
        """扫描可用的模型文件"""
        self.available_models = {}
        
        for input_mode in INPUT_MODES:
            self.available_models[input_mode] = {}
            for model_type in MODEL_TYPES:
                self.available_models[input_mode][model_type] = []
                for target in ALL_TARGETS:
                    model_path = self.model_dir / f"{input_mode}_{model_type}_{target}.pkl"
                    if model_path.exists():
                        self.available_models[input_mode][model_type].append(target)
    
    def print_available_models(self):
        """打印所有可用的模型"""
        print("\n" + "=" * 70)
        print("📊 可用的预测模型")
        print("=" * 70)
        
        for input_mode in INPUT_MODES:
            mode_name = "分子式输入 (Composition)" if input_mode == "comp_only" else "CIF结构输入 (CIF)"
            print(f"\n🔹 {mode_name}")
            print("-" * 60)
            
            for model_type in MODEL_TYPES:
                targets = self.available_models[input_mode][model_type]
                if targets:
                    targets_str = ", ".join(targets)
                    print(f"   {model_type:5s}: {targets_str}")
                else:
                    print(f"   {model_type:5s}: (无可用模型)")
        
        print("\n" + "=" * 70)
    
    def get_model_key(self, input_mode: str, model_type: str, target: str) -> str:
        """生成模型缓存键"""
        return f"{input_mode}_{model_type}_{target}"
    
    def load_model(self, input_mode: str, model_type: str, target: str) -> Optional[object]:
        """加载指定模型"""
        key = self.get_model_key(input_mode, model_type, target)
        
        # 检查缓存
        if key in self.loaded_models:
            return self.loaded_models[key]
        
        # 检查模型是否存在
        if target not in self.available_models.get(input_mode, {}).get(model_type, []):
            print(f"❌ 模型不存在: {input_mode}/{model_type}/{target}")
            return None
        
        # 加载模型
        model_path = self.model_dir / f"{input_mode}_{model_type}_{target}.pkl"
        try:
            model = joblib.load(model_path)
            self.loaded_models[key] = model
            return model
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return None
    
    def predict_single(
        self, 
        input_data: str,
        input_mode: str,
        model_type: str,
        target: str
    ) -> Optional[float]:
        """
        单个预测
        
        Args:
            input_data: 分子式或CIF内容
            input_mode: comp_only 或 cif_only
            model_type: RF, GBDT, NN
            target: 预测目标
        
        Returns:
            预测值或None
        """
        model = self.load_model(input_mode, model_type, target)
        if model is None:
            return None
        
        # 生成特征
        if input_mode == "comp_only":
            features = generate_cbfv_features(input_data)
        else:
            features = generate_cif_features(input_data)
        
        X = features.reshape(1, -1)
        
        try:
            pred = model.predict(X)[0]
            return float(pred)
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None
    
    def predict_all_targets(
        self,
        input_data: str,
        input_mode: str,
        model_type: str
    ) -> Dict[str, Optional[float]]:
        """预测所有可用目标"""
        results = {}
        available_targets = self.available_models.get(input_mode, {}).get(model_type, [])
        
        for target in ALL_TARGETS:
            if target in available_targets:
                results[target] = self.predict_single(input_data, input_mode, model_type, target)
            else:
                results[target] = None
        
        return results
    
    def compare_models(
        self,
        input_data: str,
        input_mode: str,
        target: str
    ) -> Dict[str, Optional[float]]:
        """使用不同模型类型预测同一目标，进行对比"""
        results = {}
        
        for model_type in MODEL_TYPES:
            pred = self.predict_single(input_data, input_mode, model_type, target)
            results[model_type] = pred
        
        return results


# =============================================================================
# 结果展示函数
# =============================================================================

def print_prediction_result(
    input_data: str,
    results: Dict[str, Optional[float]],
    title: str = "预测结果"
):
    """美观地打印预测结果"""
    print(f"\n{'=' * 70}")
    print(f"📋 {title}")
    print(f"{'=' * 70}")
    print(f"输入: {input_data[:80]}{'...' if len(input_data) > 80 else ''}")
    print("-" * 70)
    
    for target, value in results.items():
        info = TARGET_INFO.get(target, {"name": target, "unit": "", "description": ""})
        if value is not None:
            print(f"  {info['name']:25s}: {value:>10.4f} {info['unit']}")
        else:
            print(f"  {info['name']:25s}: {'N/A':>10s}")
    
    print("=" * 70)


def print_model_comparison(
    input_data: str,
    target: str,
    results: Dict[str, Optional[float]]
):
    """打印模型对比结果"""
    info = TARGET_INFO.get(target, {"name": target, "unit": ""})
    
    print(f"\n{'=' * 70}")
    print(f"🔬 模型对比 - {info['name']}")
    print(f"{'=' * 70}")
    print(f"输入: {input_data[:60]}{'...' if len(input_data) > 60 else ''}")
    print("-" * 70)
    
    valid_results = [(k, v) for k, v in results.items() if v is not None]
    
    if valid_results:
        values = [v for _, v in valid_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        for model_type, value in results.items():
            if value is not None:
                diff = value - mean_val
                print(f"  {model_type:6s}: {value:>10.4f} {info['unit']:8s} (差异: {diff:+.4f})")
            else:
                print(f"  {model_type:6s}: {'N/A':>10s}")
        
        print("-" * 70)
        print(f"  平均值: {mean_val:.4f} ± {std_val:.4f} {info['unit']}")
    else:
        print("  没有可用的预测结果")
    
    print("=" * 70)


# =============================================================================
# 交互模式
# =============================================================================

def interactive_mode(manager: PerovskiteModelManager):
    """交互式预测模式"""
    print("\n" + "=" * 70)
    print("🚀 钙钛矿太阳能电池性质预测系统 - 交互模式")
    print("=" * 70)
    print("输入 'help' 查看帮助, 'quit' 退出")
    print("=" * 70)
    
    while True:
        print("\n📌 请选择操作:")
        print("  1. 查看可用模型")
        print("  2. 分子式预测 (单个目标)")
        print("  3. 分子式预测 (所有目标)")
        print("  4. 模型对比预测")
        print("  5. 批量预测")
        print("  6. CIF 结构预测")
        print("  q. 退出")
        
        choice = input("\n请输入选项 (1-6 或 q): ").strip().lower()
        
        if choice in ['q', 'quit', 'exit']:
            print("👋 再见!")
            break
        
        elif choice == '1':
            manager.print_available_models()
        
        elif choice == '2':
            # 单个目标预测
            composition = input("请输入分子式 (如 FA0.25MA0.75PbI3): ").strip()
            if not composition:
                print("❌ 分子式不能为空")
                continue
            
            print(f"\n可用模型类型: {', '.join(MODEL_TYPES)}")
            model_type = input("请选择模型类型 [RF]: ").strip().upper() or "RF"
            if model_type not in MODEL_TYPES:
                print(f"❌ 无效的模型类型，请选择 {MODEL_TYPES}")
                continue
            
            print(f"\n可用预测目标: {', '.join(ALL_TARGETS)}")
            target = input("请选择预测目标 [pce]: ").strip().lower() or "pce"
            if target not in ALL_TARGETS:
                print(f"❌ 无效的目标，请选择 {ALL_TARGETS}")
                continue
            
            print(f"\n⏳ 正在预测...")
            result = manager.predict_single(composition, "comp_only", model_type, target)
            
            if result is not None:
                print_prediction_result(composition, {target: result}, 
                                       f"预测结果 ({model_type} 模型)")
            else:
                print("❌ 预测失败")
        
        elif choice == '3':
            # 所有目标预测
            composition = input("请输入分子式 (如 CsPbI3): ").strip()
            if not composition:
                print("❌ 分子式不能为空")
                continue
            
            model_type = input("请选择模型类型 [RF]: ").strip().upper() or "RF"
            if model_type not in MODEL_TYPES:
                print(f"❌ 无效的模型类型")
                continue
            
            print(f"\n⏳ 正在预测所有目标...")
            results = manager.predict_all_targets(composition, "comp_only", model_type)
            print_prediction_result(composition, results, f"全属性预测 ({model_type} 模型)")
        
        elif choice == '4':
            # 模型对比
            composition = input("请输入分子式: ").strip()
            if not composition:
                print("❌ 分子式不能为空")
                continue
            
            target = input(f"请选择预测目标 [{', '.join(ALL_TARGETS)}] [pce]: ").strip().lower() or "pce"
            if target not in ALL_TARGETS:
                print(f"❌ 无效的目标")
                continue
            
            print(f"\n⏳ 正在使用不同模型进行对比预测...")
            results = manager.compare_models(composition, "comp_only", target)
            print_model_comparison(composition, target, results)
        
        elif choice == '5':
            # 批量预测
            compositions_str = input("请输入多个分子式 (用逗号分隔): ").strip()
            if not compositions_str:
                print("❌ 输入不能为空")
                continue
            
            compositions = [c.strip() for c in compositions_str.split(",")]
            model_type = input("请选择模型类型 [RF]: ").strip().upper() or "RF"
            
            print(f"\n⏳ 正在批量预测 {len(compositions)} 个分子式...")
            
            all_results = []
            for comp in compositions:
                results = manager.predict_all_targets(comp, "comp_only", model_type)
                results["composition"] = comp
                all_results.append(results)
            
            # 转换为 DataFrame 显示
            df = pd.DataFrame(all_results)
            cols = ["composition"] + ALL_TARGETS
            df = df[cols]
            
            print(f"\n{'=' * 90}")
            print(f"📊 批量预测结果 ({model_type} 模型)")
            print("=" * 90)
            print(df.to_string(index=False))
            print("=" * 90)
            
            # 询问是否保存
            save = input("\n是否保存结果到CSV? [y/N]: ").strip().lower()
            if save == 'y':
                output_path = SCRIPT_DIR / "test_output" / f"batch_prediction_{model_type}.csv"
                output_path.parent.mkdir(exist_ok=True)
                df.to_csv(output_path, index=False)
                print(f"✅ 结果已保存到: {output_path}")
        
        elif choice == '6':
            # CIF 预测
            print("\n请选择 CIF 输入方式:")
            print("  1. 从文件读取")
            print("  2. 直接输入 CIF 内容")
            
            cif_choice = input("选择 [1]: ").strip() or "1"
            
            if cif_choice == '1':
                cif_path = input("请输入 CIF 文件路径: ").strip()
                if not os.path.exists(cif_path):
                    print(f"❌ 文件不存在: {cif_path}")
                    continue
                with open(cif_path, 'r') as f:
                    cif_content = f.read()
            else:
                print("请输入 CIF 内容 (输入 END 结束):")
                lines = []
                while True:
                    line = input()
                    if line.strip().upper() == "END":
                        break
                    lines.append(line)
                cif_content = "\n".join(lines)
            
            model_type = input("请选择模型类型 [RF]: ").strip().upper() or "RF"
            
            print(f"\n⏳ 正在预测...")
            results = manager.predict_all_targets(cif_content, "cif_only", model_type)
            print_prediction_result("CIF Structure", results, f"CIF 结构预测 ({model_type} 模型)")
        
        else:
            print("❌ 无效的选项，请重试")


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="钙钛矿太阳能电池性质预测测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互模式
  python model_predictor_test.py --interactive
  
  # 预测单个分子式的 PCE
  python model_predictor_test.py --composition "FA0.25MA0.75PbI3" --model RF --target pce
  
  # 预测所有属性
  python model_predictor_test.py --composition "CsPbI3" --model RF --all-targets
  
  # 批量预测
  python model_predictor_test.py --compositions "FA0.25MA0.75PbI3,CsPbI3,MAPbI3" --model RF
  
  # 对比不同模型
  python model_predictor_test.py --composition "CsPbI3" --compare-models --target pce
  
  # 列出所有可用模型
  python model_predictor_test.py --list-models
        """
    )
    
    # 模式选择
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='启动交互模式')
    parser.add_argument('--list-models', '-l', action='store_true',
                       help='列出所有可用模型')
    
    # 输入参数
    parser.add_argument('--composition', '-c', type=str,
                       help='单个分子式 (如 FA0.25MA0.75PbI3)')
    parser.add_argument('--compositions', type=str,
                       help='多个分子式，用逗号分隔')
    parser.add_argument('--cif-file', type=str,
                       help='CIF 文件路径')
    parser.add_argument('--cif-content', type=str,
                       help='CIF 内容字符串')
    
    # 模型参数
    parser.add_argument('--model', '-m', type=str, default='RF',
                       choices=MODEL_TYPES,
                       help='模型类型 (默认: RF)')
    parser.add_argument('--target', '-t', type=str, default='pce',
                       choices=ALL_TARGETS,
                       help='预测目标 (默认: pce)')
    parser.add_argument('--all-targets', '-a', action='store_true',
                       help='预测所有目标属性')
    parser.add_argument('--compare-models', action='store_true',
                       help='使用所有模型类型进行对比预测')
    
    # 输出参数
    parser.add_argument('--output', '-o', type=str,
                       help='输出文件路径 (CSV格式)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='安静模式，仅输出结果')
    
    args = parser.parse_args()
    
    # 初始化模型管理器
    manager = PerovskiteModelManager()
    
    # 处理不同模式
    if args.list_models:
        manager.print_available_models()
        return
    
    if args.interactive:
        interactive_mode(manager)
        return
    
    # 命令行模式
    if args.composition:
        if args.compare_models:
            # 模型对比
            results = manager.compare_models(args.composition, "comp_only", args.target)
            print_model_comparison(args.composition, args.target, results)
        elif args.all_targets:
            # 预测所有目标
            results = manager.predict_all_targets(args.composition, "comp_only", args.model)
            print_prediction_result(args.composition, results, f"全属性预测 ({args.model})")
        else:
            # 单目标预测
            result = manager.predict_single(args.composition, "comp_only", args.model, args.target)
            print_prediction_result(args.composition, {args.target: result}, 
                                   f"预测结果 ({args.model})")
    
    elif args.compositions:
        # 批量预测
        compositions = [c.strip() for c in args.compositions.split(",")]
        all_results = []
        
        if not args.quiet:
            print(f"⏳ 正在批量预测 {len(compositions)} 个分子式...")
        
        for comp in compositions:
            results = manager.predict_all_targets(comp, "comp_only", args.model)
            results["composition"] = comp
            all_results.append(results)
        
        df = pd.DataFrame(all_results)
        cols = ["composition"] + ALL_TARGETS
        df = df[cols]
        
        if not args.quiet:
            print(f"\n{'=' * 90}")
            print(f"📊 批量预测结果 ({args.model} 模型)")
            print("=" * 90)
            print(df.to_string(index=False))
            print("=" * 90)
        
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"✅ 结果已保存到: {args.output}")
    
    elif args.cif_file:
        # CIF 文件预测
        if not os.path.exists(args.cif_file):
            print(f"❌ CIF 文件不存在: {args.cif_file}")
            return
        
        with open(args.cif_file, 'r') as f:
            cif_content = f.read()
        
        if args.all_targets:
            results = manager.predict_all_targets(cif_content, "cif_only", args.model)
            print_prediction_result(args.cif_file, results, f"CIF 预测 ({args.model})")
        else:
            result = manager.predict_single(cif_content, "cif_only", args.model, args.target)
            print_prediction_result(args.cif_file, {args.target: result}, 
                                   f"CIF 预测 ({args.model})")
    
    elif args.cif_content:
        # CIF 内容预测
        if args.all_targets:
            results = manager.predict_all_targets(args.cif_content, "cif_only", args.model)
            print_prediction_result("CIF Content", results, f"CIF 预测 ({args.model})")
        else:
            result = manager.predict_single(args.cif_content, "cif_only", args.model, args.target)
            print_prediction_result("CIF Content", {args.target: result},
                                   f"CIF 预测 ({args.model})")
    
    else:
        # 没有提供输入，显示帮助
        parser.print_help()
        print("\n💡 提示: 使用 --interactive 启动交互模式")


# =============================================================================
# 快速测试函数
# =============================================================================

def quick_test():
    """快速测试函数"""
    print("=" * 70)
    print("🧪 快速测试 - 钙钛矿性质预测")
    print("=" * 70)
    
    manager = PerovskiteModelManager()
    manager.print_available_models()
    
    # 测试分子式
    test_compositions = [
        "CsPbI3",
        "MAPbI3",
        "FA0.25MA0.75PbI3",
        "Cs0.05FA0.79MA0.16Pb(I0.83Br0.17)3"
    ]
    
    print("\n" + "=" * 70)
    print("📊 测试组分预测 (RF 模型)")
    print("=" * 70)
    
    for comp in test_compositions:
        print(f"\n🔹 测试: {comp}")
        try:
            results = manager.predict_all_targets(comp, "comp_only", "RF")
            for target, value in results.items():
                if value is not None:
                    info = TARGET_INFO[target]
                    print(f"   {info['name']:20s}: {value:>8.4f} {info['unit']}")
        except Exception as e:
            print(f"   ❌ 错误: {e}")
    
    # 模型对比测试
    print("\n" + "=" * 70)
    print("📊 模型对比测试 (PCE 预测)")
    print("=" * 70)
    
    test_comp = "FA0.25MA0.75PbI3"
    results = manager.compare_models(test_comp, "comp_only", "pce")
    print_model_comparison(test_comp, "pce", results)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # 无参数时显示帮助并运行快速测试
        print("💡 使用 --help 查看完整帮助")
        print("💡 使用 --interactive 启动交互模式")
        print("\n正在运行快速测试...\n")
        quick_test()
    else:
        main()
