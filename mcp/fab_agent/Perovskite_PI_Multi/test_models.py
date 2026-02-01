#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型测试脚本 - 测试新训练的 Composition-only 和 CIF-only 模型

Usage:
    python test_models.py                    # 测试所有模型
    python test_models.py --mode comp_only   # 只测试 Composition-only 模型
    python test_models.py --mode cif_only    # 只测试 CIF-only 模型
    python test_models.py --model RF         # 只测试 RF 模型
"""

import os
import sys
import glob
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# 添加当前目录到路径
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from process import file2vector

# =============================================================================
# 配置
# =============================================================================

MODEL_DIR = SCRIPT_DIR / "data" / "model"
DATA_FILE = SCRIPT_DIR / "data" / "raw" / "full_dataset.csv"

# 目标属性
TARGET_PROPERTIES = ["pce", "dft_band_gap", "energy_above_hull", "voc", "jsc", "ff"]

# 特征配置
FEATURE_CONFIG = {
    "split_way": 1,
    "per_elem_prop": "oliynyk",
    "fill_way": "zero",
    "num_list": None,
}

# =============================================================================
# 辅助函数
# =============================================================================

def find_models(mode=None, model_type=None):
    """查找可用的模型文件"""
    models = {}
    
    # 查找模型文件模式
    patterns = []
    if mode:
        if model_type:
            patterns.append(f"model_{mode}_{model_type}_*.pkl")
        else:
            patterns.append(f"model_{mode}_*.pkl")
    else:
        if model_type:
            patterns.append(f"model_comp_only_{model_type}_*.pkl")
            patterns.append(f"model_cif_only_{model_type}_*.pkl")
        else:
            patterns.append("model_comp_only_*.pkl")
            patterns.append("model_cif_only_*.pkl")
    
    for pattern in patterns:
        for model_path in glob.glob(str(MODEL_DIR / pattern)):
            model_name = Path(model_path).stem
            # 解析模型信息: model_{mode}_{type}_...
            parts = model_name.split("_")
            if len(parts) >= 4:
                m_mode = f"{parts[1]}_{parts[2]}"  # comp_only or cif_only
                m_type = parts[3]  # RF, NN, GBDT
                key = f"{m_mode}_{m_type}"
                models[key] = {
                    "path": model_path,
                    "mode": m_mode,
                    "type": m_type,
                    "name": model_name
                }
    
    return models


def load_model(model_path):
    """加载模型"""
    print(f"  Loading: {Path(model_path).name}")
    return joblib.load(model_path)


def prepare_data(mode):
    """准备测试数据"""
    print(f"\n  Preparing data for mode: {mode}")
    
    # 检查缓存
    from scipy.sparse import load_npz
    csr_path = f"data/csr/{mode}_sp1_oliynyk_zero_csr.npz"
    col_path = f"data/csr/{mode}_sp1_oliynyk_zero_columns.npy"
    
    if os.path.exists(csr_path) and os.path.exists(col_path):
        print(f"  Loading cached features from {csr_path}")
        csr = load_npz(csr_path)
        columns = np.load(col_path, allow_pickle=True)
        X = pd.DataFrame(csr.toarray(), columns=columns)
    else:
        print(f"  Processing features from scratch...")
        X = file2vector(
            str(DATA_FILE),
            FEATURE_CONFIG["split_way"],
            FEATURE_CONFIG["per_elem_prop"],
            FEATURE_CONFIG["fill_way"],
            FEATURE_CONFIG["num_list"],
            mode
        )
    
    # 加载目标值
    df_raw = pd.read_csv(DATA_FILE)
    df_clean = df_raw.dropna(subset=TARGET_PROPERTIES)
    y = df_clean[TARGET_PROPERTIES]
    
    # 对齐数据
    X = X.iloc[df_clean.index].reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    print(f"  Data shape: X={X.shape}, y={y.shape}")
    return X, y


def evaluate_model(model, X_test, y_test, target_names):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    
    results = {
        "overall": {
            "R2": r2_score(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred)
        },
        "per_target": {}
    }
    
    # 每个目标的指标
    y_test_np = np.array(y_test)
    y_pred_np = np.array(y_pred)
    
    for i, name in enumerate(target_names):
        results["per_target"][name] = {
            "R2": r2_score(y_test_np[:, i], y_pred_np[:, i]),
            "RMSE": np.sqrt(mean_squared_error(y_test_np[:, i], y_pred_np[:, i])),
            "MAE": mean_absolute_error(y_test_np[:, i], y_pred_np[:, i])
        }
    
    return results


def print_results(model_info, results):
    """打印评估结果"""
    print(f"\n  {'='*60}")
    print(f"  Model: {model_info['name']}")
    print(f"  Mode: {model_info['mode']}, Type: {model_info['type']}")
    print(f"  {'='*60}")
    
    print(f"\n  Overall Test Metrics:")
    print(f"    R2:   {results['overall']['R2']:.4f}")
    print(f"    RMSE: {results['overall']['RMSE']:.4f}")
    print(f"    MAE:  {results['overall']['MAE']:.4f}")
    
    print(f"\n  Per-Target Test Metrics:")
    print(f"    {'Target':<20} {'R2':>10} {'RMSE':>10} {'MAE':>10}")
    print(f"    {'-'*50}")
    
    for target, metrics in results['per_target'].items():
        print(f"    {target:<20} {metrics['R2']:>10.4f} {metrics['RMSE']:>10.4f} {metrics['MAE']:>10.4f}")


def test_prediction_example(model, X, mode, model_type):
    """测试单个样本预测"""
    print(f"\n  Sample Prediction (first 3 samples):")
    
    X_sample = X.iloc[:3]
    y_pred = model.predict(X_sample)
    
    for i in range(3):
        print(f"\n    Sample {i+1}:")
        for j, target in enumerate(TARGET_PROPERTIES):
            print(f"      {target}: {y_pred[i][j]:.4f}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test trained perovskite prediction models")
    parser.add_argument("--mode", choices=["comp_only", "cif_only"], 
                        help="Feature mode to test")
    parser.add_argument("--model", choices=["RF", "NN", "GBDT"],
                        help="Model type to test")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Test set ratio (default: 0.2)")
    args = parser.parse_args()
    
    print("="*70)
    print(" 钙钛矿性质预测模型测试")
    print(f" 时间: {datetime.now()}")
    print("="*70)
    
    # 查找模型
    models = find_models(mode=args.mode, model_type=args.model)
    
    if not models:
        print("\n❌ 没有找到匹配的模型文件！")
        print(f"   模型目录: {MODEL_DIR}")
        print(f"   请确保模型已训练完成。")
        return
    
    print(f"\n找到 {len(models)} 个模型:")
    for key, info in models.items():
        print(f"  - {info['name']}")
    
    # 按 mode 分组准备数据 (避免重复加载)
    data_cache = {}
    
    # 测试每个模型
    all_results = {}
    
    for key, model_info in models.items():
        mode = model_info["mode"]
        
        print(f"\n{'='*70}")
        print(f" Testing: {model_info['name']}")
        print(f"{'='*70}")
        
        # 准备数据 (带缓存)
        if mode not in data_cache:
            X, y = prepare_data(mode)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_ratio, random_state=0
            )
            data_cache[mode] = (X_train, X_test, y_train, y_test)
        else:
            X_train, X_test, y_train, y_test = data_cache[mode]
        
        # 加载模型
        try:
            model = load_model(model_info["path"])
        except Exception as e:
            print(f"  ❌ 加载模型失败: {e}")
            continue
        
        # 评估模型
        try:
            results = evaluate_model(model, X_test, y_test, TARGET_PROPERTIES)
            all_results[key] = results
            print_results(model_info, results)
            
            # 测试单个预测
            test_prediction_example(model, X_test, mode, model_info["type"])
            
        except Exception as e:
            print(f"  ❌ 评估失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 汇总对比
    if len(all_results) > 1:
        print("\n" + "="*70)
        print(" 模型性能对比汇总")
        print("="*70)
        
        print(f"\n {'Model':<35} {'Overall R2':>12} {'Overall RMSE':>12}")
        print(f" {'-'*60}")
        
        for key, results in sorted(all_results.items()):
            print(f" {key:<35} {results['overall']['R2']:>12.4f} {results['overall']['RMSE']:>12.4f}")
        
        # 每个目标的最佳模型
        print(f"\n 每个目标属性的最佳模型 (按 R2):")
        print(f" {'-'*60}")
        
        for target in TARGET_PROPERTIES:
            best_model = None
            best_r2 = -float('inf')
            
            for key, results in all_results.items():
                r2 = results['per_target'][target]['R2']
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = key
            
            print(f"   {target:<20}: {best_model:<30} (R2={best_r2:.4f})")
    
    print("\n" + "="*70)
    print(" 测试完成!")
    print("="*70)


if __name__ == "__main__":
    main()
