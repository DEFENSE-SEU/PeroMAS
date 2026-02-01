#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_models.py
统一训练入口：支持训练 Composition-only 和 CIF-only 两种模型

使用方法:
    # 训练分子式模型 (RF)
    python train_models.py --mode comp_only --model RF
    
    # 训练 CIF 结构模型 (RF)
    python train_models.py --mode cif_only --model RF
    
    # 训练所有模型
    python train_models.py --mode all --model RF
    
    # 使用超参数优化
    python train_models.py --mode comp_only --model RF --hyperopt
"""

import argparse
import datetime
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='训练钙钛矿性质预测模型')
    parser.add_argument('--mode', type=str, choices=['comp_only', 'cif_only', 'all'], 
                        default='all', help='训练模式: comp_only(分子式), cif_only(CIF结构), all(两者)')
    parser.add_argument('--model', type=str, choices=['RF', 'GBDT', 'NN'], 
                        default='RF', help='模型类型: RF, GBDT, NN')
    parser.add_argument('--hyperopt', action='store_true', 
                        help='是否进行超参数优化')
    parser.add_argument('--n_trials', type=int, default=100, 
                        help='超参数优化的试验次数')
    parser.add_argument('--clear_cache', action='store_true', 
                        help='清除缓存的特征文件')
    args = parser.parse_args()
    
    print("=" * 60)
    print("钙钛矿性质预测模型训练")
    print("=" * 60)
    print(f"开始时间: {datetime.datetime.now()}")
    print(f"训练模式: {args.mode}")
    print(f"模型类型: {args.model}")
    print(f"超参数优化: {args.hyperopt}")
    print("=" * 60)
    
    # 确保目录存在
    os.makedirs("data/csr", exist_ok=True)
    os.makedirs("data/model", exist_ok=True)
    os.makedirs("data/model/hyperopt", exist_ok=True)
    
    # 清除缓存
    if args.clear_cache:
        import glob
        for f in glob.glob("data/csr/*.npz") + glob.glob("data/csr/*.npy"):
            os.remove(f)
            print(f"已删除缓存: {f}")
    
    modes_to_train = []
    if args.mode == 'all':
        modes_to_train = ['comp_only', 'cif_only']
    else:
        modes_to_train = [args.mode]
    
    for mode in modes_to_train:
        print(f"\n{'='*60}")
        print(f"训练模式: {mode}")
        print(f"{'='*60}")
        
        # 动态导入对应的配置
        if mode == 'comp_only':
            import settings_comp_only as settings
        elif mode == 'cif_only':
            import settings_cif_only as settings
        else:
            raise ValueError(f"未知模式: {mode}")
        
        # 重新加载模块以获取最新配置
        import importlib
        if mode == 'comp_only':
            import settings_comp_only
            importlib.reload(settings_comp_only)
            settings = settings_comp_only
        else:
            import settings_cif_only
            importlib.reload(settings_cif_only)
            settings = settings_cif_only
        
        # 导入训练函数
        from main2_train import main as train_main
        
        # 覆盖配置
        run_mode = "Hyperopt" if args.hyperopt else "Train"
        model_name = args.model
        
        # 更新保存名称
        save_name = f"{mode}_{run_mode}_{model_name}_sp{settings.split_way}_{settings.per_elem_prop}_{settings.fill_way}_r{settings.random_state}"
        model_save_name = f"model_{mode}_{model_name}_sp{settings.split_way}_{settings.per_elem_prop}_{settings.fill_way}_r{settings.random_state}"
        
        params = {
            "run_mode": run_mode,
            "model_name": model_name,
            "random_state": settings.random_state,
            "raw_file_name": settings.raw_file_name,
            "split_way": settings.split_way,
            "per_elem_prop": settings.per_elem_prop,
            "fill_way": settings.fill_way,
            "save_name": save_name,
            "model_save_name": model_save_name,
            "num_list": settings.num_list,
            "target": settings.target,
            "use_X": mode,  # 使用当前模式
            "test_ratio": settings.test_ratio,
            "valid_ratio_in_train": settings.valid_ratio_in_train,
            "n_estimators": settings.n_estimators,
            "max_depth": settings.max_depth,
            "max_leaf_nodes": settings.max_leaf_nodes,
            "min_samples_split": settings.min_samples_split,
            "min_samples_leaf": settings.min_samples_leaf,
            "n_trials": args.n_trials if args.hyperopt else settings.n_trials,
            "dim": settings.dim,
            "n_mid": settings.n_mid,
            "lr": settings.lr,
            "epoch": settings.epoch,
            "solver": settings.solver,
            "activation": settings.activation,
            "storage_name": settings.storage_name,
            "calc_shap": settings.calc_shap,
        }
        
        print(f"\n配置参数:")
        print(f"  数据文件: {params['raw_file_name']}")
        print(f"  特征模式: {params['use_X']}")
        print(f"  模型: {params['model_name']}")
        print(f"  目标: {params['target']}")
        print(f"  测试集比例: {params['test_ratio']}")
        print()
        
        train_main(**params)
        
        print(f"\n模式 {mode} 训练完成!")
    
    print("\n" + "=" * 60)
    print(f"所有训练完成! 结束时间: {datetime.datetime.now()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
