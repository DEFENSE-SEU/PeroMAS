
from train import *
from process import *
import settings
import datetime
import joblib
import numpy as np
import shap
import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance

def ensure_directories():
    """Ensure all necessary output directories exist."""
    dirs = [
        "data/csr",
        "data/model",
        "data/model/hyperopt",
        "data/model/interpret",
        "data/model/regression"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def save_prediction_results(y_real, y_pred, target_names, filepath):
    y_real = y_real.reset_index(drop=True)
    y_pred_df = pd.DataFrame(y_pred, columns=[f"{t}_pred" for t in target_names])
    y_real_renamed = y_real.copy()
    y_real_renamed.columns = [f"{t}_real" for t in target_names]
    result_df = pd.concat([y_real_renamed, y_pred_df], axis=1)
    
    ordered_cols = []
    for t in target_names:
        ordered_cols.append(f"{t}_real")
        ordered_cols.append(f"{t}_pred")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    result_df = result_df[ordered_cols]
    result_df.to_csv(filepath, index=False)
    print(f"Saved prediction results to: {filepath}")

def save_importance_csv(importances, columns, save_name, suffix=""):
    """辅助函数：保存重要性CSV"""
    fti = pd.Series(importances, index=list(columns))
    fti_sort = fti.sort_values(ascending=False)
    
    fname_fti = f"data/model/interpret/fti_{suffix}_{save_name}.csv"
    os.makedirs(os.path.dirname(fname_fti), exist_ok=True)
    # [修复] columns 参数必须是列表
    pd.DataFrame(fti_sort, columns=["feature_importance"]).to_csv(fname_fti, index=True)
    
    fti_summary_dict = {}
    for col_name, score in fti.items():
        if "_" in col_name:
            prefix = col_name.split("_")[0]
        else:
            prefix = "Other"
        fti_summary_dict[prefix] = fti_summary_dict.get(prefix, 0) + score
    
    fname_sum = f"data/model/interpret/fti_sum_{suffix}_{save_name}.csv"
    pd.DataFrame.from_dict(fti_summary_dict, orient='index', columns=['importance']).sort_values('importance', ascending=False).to_csv(fname_sum)
    print(f"Saved importance for: {suffix}")

def vec2csr(vec, csr_file_name, columns_file_name):
    if os.path.dirname(csr_file_name):
        os.makedirs(os.path.dirname(csr_file_name), exist_ok=True)
    csr = csr_matrix(vec)
    save_npz(csr_file_name, csr)
    if columns_file_name is not None:
        columns_arr = np.array(vec.columns)
        np.save(columns_file_name, columns_arr)

def main(
    run_mode,
    model_name = None,
    random_state = None,
    raw_file_name = None,
    split_way = None,
    per_elem_prop = None,
    fill_way = None,
    save_name = None,
    model_save_name = None,
    num_list = None,
    target = None,
    use_X = None,
    test_ratio = None,
    valid_ratio_in_train = None,
    n_estimators = None,
    max_depth = None,
    max_leaf_nodes= None,
    min_samples_split = None,
    min_samples_leaf = None,
    n_trials = None,
    dim = None,
    n_mid = None,
    lr = None,
    epoch = None,
    solver = None,
    activation = None,
    storage_name = None,
    calc_shap = None,
):
    print(datetime.datetime.now(), "Start")
    ensure_directories()
    
    print(f"Mode: {run_mode}, Targets ({len(target)}): {target}")

    # --- 特征加载 ---
    csr_path = f"data/csr/{use_X}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_csr.npz"
    col_path = f"data/csr/{use_X}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_columns.npy"

    if os.path.exists(csr_path):
        print(f"Loading cached features from {csr_path}...")
        X = csr2vec(csr_file_name = csr_path, columns_file_name = col_path)
    else:
        print("Processing features from scratch...")
        X = file2vector(raw_file_name, split_way, per_elem_prop, fill_way, num_list, use_X)

    # --- 数据对齐 ---
    df_raw = pd.read_csv(raw_file_name)
    df_clean = df_raw.dropna(subset=target)
    y = df_clean[target]
    
    X = X.iloc[df_clean.index].reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio, random_state = random_state)

    # ---------------- TRAIN ----------------
    if run_mode == "Train":
        if os.path.exists("data/model/hyperopt/" + model_save_name + ".pkl"):
            print("Loading optimized model...")
            model = joblib.load("data/model/hyperopt/" + model_save_name + ".pkl")
        else:
            print("Using manual parameters...")
            model = makemodel(model_name, max_depth, max_leaf_nodes, n_estimators, min_samples_split, min_samples_leaf, dim, n_mid, activation, solver, lr, epoch)

        print(datetime.datetime.now(), "Regression Start")
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        metrics(y_train, y_train_pred, y_test, y_test_pred, target_names=target)
        joblib.dump(model, "data/model/" + model_save_name + ".pkl")
        save_prediction_results(y_train, y_train_pred, target, "data/model/" + save_name + ".csv")

    # ---------------- PREDICT ----------------
    elif run_mode == "Predict":
        print(f"Loading model data/model/{model_save_name}.pkl ...")
        model = joblib.load("data/model/" + model_save_name + ".pkl")
        y_test_pred = model.predict(X_test)
        
        metrics(y_train, model.predict(X_train), y_test, y_test_pred, target_names=target)
        save_prediction_results(y_test, y_test_pred, target, "data/model/regression" + save_name + ".csv")

    # ---------------- HYPEROPT ----------------
    elif run_mode == "Hyperopt":
        os.makedirs(os.path.dirname(storage_name), exist_ok=True)
        study = optuna.create_study(study_name = f"{use_X}_{model_name}_sp{str(split_way)}_{per_elem_prop}_{fill_way}",
                                    storage = "sqlite:///" + storage_name + ".db",
                                        load_if_exists = True,
                                        direction = "minimize")
        study.optimize(objective_variable(model_name, X_train, y_train, valid_ratio_in_train), n_trials = n_trials)

        print("Best trial:", study.best_trial)
        
        if model_name == "RF":
            optimised_model = RandomForestRegressor(
                n_estimators = study.best_params["n_estimators"],
                min_samples_split = study.best_params["min_samples_split"],
                min_samples_leaf = study.best_params["min_samples_leaf"],
                n_jobs=-1,
                random_state=0
            )
        elif model_name == "GBDT":
            gbdt_inner = GradientBoostingRegressor(
                n_estimators = study.best_params["n_estimators"],
                min_samples_split = study.best_params["min_samples_split"],
                min_samples_leaf = study.best_params["min_samples_leaf"],
                learning_rate = study.best_params["lr"],
                random_state=0)
            optimised_model = MultiOutputRegressor(gbdt_inner)
        elif model_name == "NN":
            optimised_model = MLPRegressor(
                hidden_layer_sizes = tuple([study.best_params["dim"] for dim_count in range(study.best_params["n_mid"])]),
                activation = study.best_params["activation"],
                solver = study.best_params["solver"],
                learning_rate_init = study.best_params["lr"],
                max_iter = study.best_params["epoch"],
                random_state=0)

        optimised_model.fit(X_train, y_train)
        joblib.dump(optimised_model, "data/model/hyperopt/" + model_save_name + ".pkl")

    # ---------------- INTERPRET ----------------
    elif run_mode == 'Interpret':
        model = joblib.load("data/model/"+ model_save_name + ".pkl")
        columns = np.load(f"data/csr/{use_X}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_columns.npy", allow_pickle = True)
        
        # 1. Feature Importance
        if isinstance(model, MultiOutputRegressor):
            estimators = model.estimators_
            print(f"Extracting importance for {len(estimators)} targets (MultiOutput)...")
            for i, est in enumerate(estimators):
                current_target = target[i]
                if hasattr(est, "feature_importances_"):
                    save_importance_csv(est.feature_importances_, columns, save_name, suffix=current_target)

        elif isinstance(model, RandomForestRegressor):
            print("RandomForest detected: Feature importances are aggregated globally.")
            if hasattr(model, "feature_importances_"):
                save_importance_csv(model.feature_importances_, columns, save_name, suffix="Global_AllTargets")

        elif isinstance(model, MLPRegressor):
            print("Neural Network detected: Using Permutation Importance (Global).")
            r = permutation_importance(model, X_test, y_test,
                                       n_repeats=5, 
                                       random_state=0, n_jobs=-1)
            save_importance_csv(r.importances_mean, columns, save_name, suffix="Global_Permutation_NN")

        else:
            print(f"Model {model_name} structure not supported for importance extraction.")

        # 2. SHAP Calculation
        if calc_shap:
            print("Calculating SHAP values...")
            X_arr = load_npz(f"data/csr/{use_X}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_csr.npz").toarray()
            X_df = pd.DataFrame(X_arr, columns=columns)
            
            # 背景数据采样
            X_sample = X_df.sample(min(100, len(X_df)), random_state=0)

            if isinstance(model, RandomForestRegressor):
                explainer = shap.TreeExplainer(model)
                shap_values_raw = explainer.shap_values(X_sample)
                
                # --- 关键修复：兼容 SHAP 3D 数组输出 ---
                shap_values_list = []
                # 检查是否为 3D 数组 (samples, features, targets)
                if isinstance(shap_values_raw, np.ndarray) and len(shap_values_raw.shape) == 3:
                    print(f"Detected 3D SHAP array {shap_values_raw.shape}, converting to list...")
                    # 按最后一个维度切片，转换为 [ (samples, features) for target_0, ... ]
                    num_targets = shap_values_raw.shape[2]
                    for i in range(num_targets):
                        shap_values_list.append(shap_values_raw[:, :, i])
                elif isinstance(shap_values_raw, list):
                    shap_values_list = shap_values_raw
                else:
                    # 单目标情况
                    shap_values_list = [shap_values_raw]
                # -------------------------------------

                print(f"SHAP (RF) calculated for {len(shap_values_list)} targets.")
                for i, sv in enumerate(shap_values_list):
                    if i < len(target):
                        t_name = target[i]
                        fname_shap = f"data/model/interpret/shap_{t_name}_{model_save_name}_csr.npz"
                        vec2csr(pd.DataFrame(sv, columns=columns), fname_shap, None)
                        print(f"  Saved SHAP for {t_name}")

            elif isinstance(model, MultiOutputRegressor):
                for i, est in enumerate(model.estimators_):
                    t_name = target[i]
                    print(f"  Calculating SHAP (GBDT) for {t_name}...")
                    explainer = shap.TreeExplainer(est)
                    shap_val = explainer.shap_values(X_sample)
                    
                    fname_shap = f"data/model/interpret/shap_{t_name}_{model_save_name}_csr.npz"
                    vec2csr(pd.DataFrame(shap_val, columns=columns), fname_shap, None)

            elif isinstance(model, MLPRegressor):
                print("  Calculating SHAP (NN) using KernelExplainer (Slow!)...")
                X_summary = shap.kmeans(X_train, 10) 
                explainer = shap.KernelExplainer(model.predict, X_summary)
                shap_values_raw = explainer.shap_values(X_sample)
                
                # 同样处理 NN 的多输出格式
                shap_values_list = []
                if isinstance(shap_values_raw, np.ndarray) and len(shap_values_raw.shape) == 3:
                     for i in range(shap_values_raw.shape[2]):
                        shap_values_list.append(shap_values_raw[:, :, i])
                elif isinstance(shap_values_raw, list):
                    shap_values_list = shap_values_raw
                
                print(f"SHAP (NN) calculated for {len(shap_values_list)} targets.")
                for i, sv in enumerate(shap_values_list):
                    t_name = target[i]
                    fname_shap = f"data/model/interpret/shap_{t_name}_{model_save_name}_csr.npz"
                    vec2csr(pd.DataFrame(sv, columns=columns), fname_shap, None)
                    print(f"  Saved SHAP for {t_name}")

    save_name = save_name + str(random_state) + ".csv"
    print(datetime.datetime.now(), "Done")

if __name__ == "__main__":
    params = {
        "run_mode": settings.run_mode,
        "model_name": settings.model_name,
        "random_state": settings.random_state,
        "raw_file_name": settings.raw_file_name,
        "split_way": settings.split_way,
        "per_elem_prop": settings.per_elem_prop,
        "fill_way": settings.fill_way,
        "save_name": settings.save_name,
        "model_save_name": settings.model_save_name,
        "split_way": settings.split_way,
        "num_list": settings.num_list,
        "target": settings.target,
        "use_X": settings.use_X,
        "test_ratio": settings.test_ratio,
        "valid_ratio_in_train": settings.valid_ratio_in_train,
        "n_estimators": settings.n_estimators,
        "max_depth": settings.max_depth,
        "max_leaf_nodes": settings.max_leaf_nodes,
        "min_samples_split": settings.min_samples_split,
        "min_samples_leaf": settings.min_samples_leaf,
        "n_trials": settings.n_trials,
        "dim": settings.dim,
        "n_mid": settings.n_mid,
        "lr": settings.lr,
        "epoch": settings.epoch,
        "solver": settings.solver,
        "activation": settings.activation,
        "storage_name": settings.storage_name,
        "calc_shap": settings.calc_shap,
    }
    main(**params)
