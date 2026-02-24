#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_models.py
Unified training entry point for Composition-only and CIF-only models.

Usage:
    # Train formula model (RF)
    python train_models.py --mode comp_only --model RF
    
    # Train CIF structure model (RF)
    python train_models.py --mode cif_only --model RF
    
    # Train all models
    python train_models.py --mode all --model RF
    
    # Use hyperparameter optimization
    python train_models.py --mode comp_only --model RF --hyperopt
"""

import argparse
import datetime
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Train perovskite property prediction models')
    parser.add_argument('--mode', type=str, choices=['comp_only', 'cif_only', 'all'], 
                        default='all', help='Training mode: comp_only(formula), cif_only(CIF structure), all(both)')
    parser.add_argument('--model', type=str, choices=['RF', 'GBDT', 'NN'], 
                        default='RF', help='Model type: RF, GBDT, NN')
    parser.add_argument('--hyperopt', action='store_true', 
                        help='Enable hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=100, 
                        help='Number of hyperopt trials')
    parser.add_argument('--clear_cache', action='store_true', 
                        help='Clear cached feature files')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Perovskite property prediction model training")
    print("=" * 60)
    print(f"Start time: {datetime.datetime.now()}")
    print(f"Training mode: {args.mode}")
    print(f"Model type: {args.model}")
    print(f"Hyperopt: {args.hyperopt}")
    print("=" * 60)
    
    # Ensure directories exist.
    os.makedirs("data/csr", exist_ok=True)
    os.makedirs("data/model", exist_ok=True)
    os.makedirs("data/model/hyperopt", exist_ok=True)
    
    # Clear cache.
    if args.clear_cache:
        import glob
        for f in glob.glob("data/csr/*.npz") + glob.glob("data/csr/*.npy"):
            os.remove(f)
            print(f"Deleted cache file: {f}")
    
    modes_to_train = []
    if args.mode == 'all':
        modes_to_train = ['comp_only', 'cif_only']
    else:
        modes_to_train = [args.mode]
    
    for mode in modes_to_train:
        print(f"\n{'='*60}")
        print(f"Training mode: {mode}")
        print(f"{'='*60}")
        
        # Dynamically import the configuration for this mode.
        if mode == 'comp_only':
            import settings_comp_only as settings
        elif mode == 'cif_only':
            import settings_cif_only as settings
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Reload to pick up the latest config values.
        import importlib
        if mode == 'comp_only':
            import settings_comp_only
            importlib.reload(settings_comp_only)
            settings = settings_comp_only
        else:
            import settings_cif_only
            importlib.reload(settings_cif_only)
            settings = settings_cif_only
        
        # Import training entry.
        from main2_train import main as train_main
        
        # Override config values.
        run_mode = "Hyperopt" if args.hyperopt else "Train"
        model_name = args.model
        
        # Update save names.
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
            "use_X": mode,  # Use current mode.
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
        
        print(f"\nConfiguration:")
        print(f"  Data file: {params['raw_file_name']}")
        print(f"  Feature mode: {params['use_X']}")
        print(f"  Model: {params['model_name']}")
        print(f"  Target: {params['target']}")
        print(f"  Test split: {params['test_ratio']}")
        print()
        
        train_main(**params)
        
        print(f"\nMode {mode} training completed!")
    
    print("\n" + "=" * 60)
    print(f"All training completed! End time: {datetime.datetime.now()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
