#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_single_target.py
单属性训练主程序 - 为每个属性单独训练一个模型

架构设计:
- 6 个属性 (pce, dft_band_gap, energy_above_hull, voc, jsc, ff)
- 2 种特征模式 (comp_only, cif_only)
- 3 种模型 (RF, GBDT, NN)
- = 36 个独立模型

使用方法:
    # 训练单个属性的模型
    python main_single_target.py --target pce --feature comp_only --model RF --hyperopt
    
    # 训练所有属性
    python main_single_target.py --target all --feature comp_only --model RF --hyperopt
"""

import argparse
import datetime
import os
import sys
import joblib
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split

from train_single_target import (
    metrics_single, 
    objective_single_target, 
    build_optimized_model,
    makemodel_single
)
from process import file2vector, csr2vec


# 所有可训练的目标属性
ALL_TARGETS = ["pce", "dft_band_gap", "energy_above_hull", "voc", "jsc", "ff"]


def ensure_directories():
    """确保输出目录存在"""
    dirs = [
        "data/csr",
        "data/model",
        "data/model/hyperopt",
        "data/model/single_target",
        "data/results"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_features(feature_mode, raw_file_name, split_way=1, per_elem_prop="oliynyk", fill_way="zero"):
    """加载或生成特征"""
    csr_path = f"data/csr/{feature_mode}_sp{split_way}_{per_elem_prop}_{fill_way}_csr.npz"
    col_path = f"data/csr/{feature_mode}_sp{split_way}_{per_elem_prop}_{fill_way}_columns.npy"
    
    if os.path.exists(csr_path):
        print(f"Loading cached features: {csr_path}")
        X = csr2vec(csr_file_name=csr_path, columns_file_name=col_path)
    else:
        print(f"Generating features for {feature_mode}...")
        X = file2vector(raw_file_name, split_way, per_elem_prop, fill_way, [], feature_mode)
    
    return X


def train_single_target_model(
    target_name,
    feature_mode,
    model_name,
    X_train, y_train,
    X_test, y_test,
    n_trials=100,
    random_state=42,
    do_hyperopt=True
):
    """训练单个属性的模型"""
    
    print(f"\n{'='*60}")
    print(f"Training: {target_name} | Feature: {feature_mode} | Model: {model_name}")
    print(f"{'='*60}")
    
    # 模型保存路径
    model_save_path = f"data/model/single_target/{feature_mode}_{model_name}_{target_name}.pkl"
    hyperopt_db = f"data/model/hyperopt/{feature_mode}_{model_name}_{target_name}_optuna.db"
    
    if do_hyperopt:
        # 超参数优化
        print(f"Starting hyperparameter optimization (n_trials={n_trials})...")
        
        study = optuna.create_study(
            study_name=f"{feature_mode}_{model_name}_{target_name}",
            storage=f"sqlite:///{hyperopt_db}",
            load_if_exists=True,
            direction="minimize"
        )
        
        # 设置日志级别
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study.optimize(
            objective_single_target(model_name, X_train, y_train, 0.25, random_state),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        print(f"\nBest trial:")
        print(f"  MSE: {study.best_trial.value:.4f}")
        print(f"  Params: {study.best_params}")
        
        # 使用最优参数构建模型
        model = build_optimized_model(model_name, study.best_params, random_state)
    else:
        # 使用默认参数
        default_params = {
            'n_estimators': 300,
            'max_depth': 20,
            'max_features': 'sqrt',
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': random_state,
            'lr': 0.1,
            'subsample': 0.8,
            'dim': 128,
            'n_mid': 3,
            'activation': 'relu',
            'solver': 'adam',
            'epoch': 1000,
            'alpha': 1e-4
        }
        model = makemodel_single(model_name, default_params)
    
    # 训练最终模型
    print("Training final model on full training set...")
    model.fit(X_train, y_train)
    
    # 评估
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    results = metrics_single(y_train, y_train_pred, y_test, y_test_pred, target_name)
    
    # 保存模型
    joblib.dump(model, model_save_path)
    print(f"Model saved: {model_save_path}")
    
    return results, model


def main():
    parser = argparse.ArgumentParser(description='单属性钙钛矿性质预测模型训练')
    parser.add_argument('--target', type=str, default='all',
                        help='目标属性: pce, dft_band_gap, energy_above_hull, voc, jsc, ff, 或 all')
    parser.add_argument('--feature', type=str, choices=['comp_only', 'cif_only', 'all'],
                        default='comp_only', help='特征模式')
    parser.add_argument('--model', type=str, choices=['RF', 'GBDT', 'NN', 'all'],
                        default='RF', help='模型类型')
    parser.add_argument('--hyperopt', action='store_true', help='是否进行超参数优化')
    parser.add_argument('--n_trials', type=int, default=100, help='超参数优化试验次数')
    parser.add_argument('--clear_cache', action='store_true', help='清除特征缓存')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    print("=" * 70)
    print("钙钛矿性质预测 - 单属性模型训练")
    print("=" * 70)
    print(f"开始时间: {datetime.datetime.now()}")
    print(f"目标属性: {args.target}")
    print(f"特征模式: {args.feature}")
    print(f"模型类型: {args.model}")
    print(f"超参数优化: {args.hyperopt}")
    print("=" * 70)
    
    ensure_directories()
    
    # 清除缓存
    if args.clear_cache:
        import glob
        for f in glob.glob("data/csr/*.npz") + glob.glob("data/csr/*.npy"):
            os.remove(f)
            print(f"Deleted cache: {f}")
    
    # 确定训练目标
    if args.target == 'all':
        targets = ALL_TARGETS
    else:
        targets = [args.target]
    
    # 确定特征模式
    if args.feature == 'all':
        feature_modes = ['comp_only', 'cif_only']
    else:
        feature_modes = [args.feature]
    
    # 确定模型类型
    if args.model == 'all':
        models = ['RF', 'GBDT', 'NN']
    else:
        models = [args.model]
    
    # 数据文件
    raw_file_name = "data/raw/full_dataset.csv"
    
    # 加载原始数据
    print("\nLoading raw data...")
    df_raw = pd.read_csv(raw_file_name)
    print(f"Raw data shape: {df_raw.shape}")
    
    # 汇总结果
    all_results = []
    
    for feature_mode in feature_modes:
        print(f"\n{'#'*70}")
        print(f"# Feature Mode: {feature_mode}")
        print(f"{'#'*70}")
        
        # 加载特征
        X_full = load_features(feature_mode, raw_file_name)
        print(f"Feature shape: {X_full.shape}")
        
        for target_name in targets:
            print(f"\n--- Processing target: {target_name} ---")
            
            # 过滤有效数据 (该属性非空)
            valid_idx = df_raw[target_name].notna()
            X = X_full.iloc[valid_idx.values].reset_index(drop=True)
            y = df_raw.loc[valid_idx, target_name].reset_index(drop=True)
            
            print(f"Valid samples for {target_name}: {len(y)}")
            
            if len(y) < 50:
                print(f"WARNING: Too few samples ({len(y)}), skipping...")
                continue
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=args.random_state
            )
            print(f"Train: {len(y_train)}, Test: {len(y_test)}")
            
            for model_name in models:
                try:
                    results, _ = train_single_target_model(
                        target_name=target_name,
                        feature_mode=feature_mode,
                        model_name=model_name,
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test,
                        n_trials=args.n_trials,
                        random_state=args.random_state,
                        do_hyperopt=args.hyperopt
                    )
                    results['feature_mode'] = feature_mode
                    results['model'] = model_name
                    all_results.append(results)
                except Exception as e:
                    print(f"ERROR training {target_name}/{feature_mode}/{model_name}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # 保存汇总结果
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = f"data/results/training_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n{'='*70}")
        print("Training Summary:")
        print(results_df.to_string())
        print(f"\nResults saved to: {results_path}")
    
    print(f"\n{'='*70}")
    print(f"All training completed!")
    print(f"End time: {datetime.datetime.now()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
