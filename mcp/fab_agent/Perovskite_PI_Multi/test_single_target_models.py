#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_single_target_models.py
测试单属性训练的模型性能
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from process import file2vector, csr2vec


ALL_TARGETS = ["pce", "dft_band_gap", "energy_above_hull", "voc", "jsc", "ff"]
MODEL_DIR = "data/model/single_target"


def load_features(feature_mode, split_way=1, per_elem_prop="oliynyk", fill_way="zero"):
    """加载缓存的特征"""
    csr_path = f"data/csr/{feature_mode}_sp{split_way}_{per_elem_prop}_{fill_way}_csr.npz"
    col_path = f"data/csr/{feature_mode}_sp{split_way}_{per_elem_prop}_{fill_way}_columns.npy"
    
    if os.path.exists(csr_path):
        return csr2vec(csr_file_name=csr_path, columns_file_name=col_path)
    else:
        print(f"Cache not found: {csr_path}")
        return None


def test_all_models():
    """测试所有训练好的模型"""
    print("=" * 70)
    print("Testing Single-Target Models")
    print("=" * 70)
    
    # 加载数据
    raw_file_name = "data/raw/full_dataset.csv"
    df_raw = pd.read_csv(raw_file_name)
    print(f"Raw data: {df_raw.shape}")
    
    results = []
    
    for feature_mode in ["comp_only", "cif_only"]:
        print(f"\n{'#'*70}")
        print(f"# Feature Mode: {feature_mode}")
        print(f"{'#'*70}")
        
        X_full = load_features(feature_mode)
        if X_full is None:
            print(f"Skipping {feature_mode} - no cached features")
            continue
        
        print(f"Feature shape: {X_full.shape}")
        
        for model_type in ["RF", "GBDT", "NN"]:
            print(f"\n--- Model: {model_type} ---")
            
            for target in ALL_TARGETS:
                model_path = os.path.join(MODEL_DIR, f"{feature_mode}_{model_type}_{target}.pkl")
                
                if not os.path.exists(model_path):
                    print(f"  {target}: Model not found")
                    continue
                
                # 过滤有效数据
                valid_idx = df_raw[target].notna()
                X = X_full.iloc[valid_idx.values].reset_index(drop=True)
                y = df_raw.loc[valid_idx, target].reset_index(drop=True)
                
                # 划分数据集 (使用相同的随机种子)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # 加载模型
                model = joblib.load(model_path)
                
                # 预测
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # 计算指标
                r2_train = r2_score(y_train, y_train_pred)
                r2_test = r2_score(y_test, y_test_pred)
                rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
                rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
                mae_test = mean_absolute_error(y_test, y_test_pred)
                
                print(f"  {target}: R2={r2_test:.4f}, RMSE={rmse_test:.4f}, MAE={mae_test:.4f}")
                
                results.append({
                    'feature_mode': feature_mode,
                    'model': model_type,
                    'target': target,
                    'n_samples': len(y),
                    'R2_train': r2_train,
                    'RMSE_train': rmse_train,
                    'R2_test': r2_test,
                    'RMSE_test': rmse_test,
                    'MAE_test': mae_test
                })
    
    # 输出汇总
    if results:
        results_df = pd.DataFrame(results)
        print("\n" + "=" * 70)
        print("Summary by Feature Mode and Model:")
        print("=" * 70)
        
        for feature_mode in results_df['feature_mode'].unique():
            print(f"\n{feature_mode}:")
            for model_type in results_df['model'].unique():
                subset = results_df[
                    (results_df['feature_mode'] == feature_mode) & 
                    (results_df['model'] == model_type)
                ]
                if len(subset) > 0:
                    avg_r2 = subset['R2_test'].mean()
                    avg_rmse = subset['RMSE_test'].mean()
                    print(f"  {model_type}: Avg R2={avg_r2:.4f}, Avg RMSE={avg_rmse:.4f}")
        
        # 保存结果
        results_path = "data/results/test_single_target_results.csv"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")
        
        # 打印完整表格
        print("\nFull Results Table:")
        print(results_df.to_string())
    
    return results


if __name__ == "__main__":
    test_all_models()
