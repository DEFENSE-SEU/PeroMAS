# train.py
# Optimized for Multi-Output & Sklearn 1.5+

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna

def metrics(Y_train_real, Y_train_pred, Y_test_real, Y_test_pred, target_names=None):
    # Use np.sqrt with MSE to compute RMSE (version-safe across sklearn).
    R2_train = r2_score(Y_train_real, Y_train_pred)
    RMSE_train = np.sqrt(mean_squared_error(Y_train_real, Y_train_pred))
    MAE_train = mean_absolute_error(Y_train_real, Y_train_pred)
    
    R2_test = r2_score(Y_test_real, Y_test_pred)
    RMSE_test = np.sqrt(mean_squared_error(Y_test_real, Y_test_pred))
    MAE_test = mean_absolute_error(Y_test_real, Y_test_pred)
    
    print("-" * 30)
    print(f"Overall Train - R2: {R2_train:.4f}, RMSE: {RMSE_train:.4f}, MAE: {MAE_train:.4f}")
    print(f"Overall Test  - R2: {R2_test:.4f}, RMSE: {RMSE_test:.4f}, MAE: {MAE_test:.4f}")
    
    if target_names is not None:
        print("-" * 30)
        print("Metrics per target:")
        Y_te_real_np = np.array(Y_test_real)
        Y_te_pred_np = np.array(Y_test_pred)
        
        for i, name in enumerate(target_names):
            r2 = r2_score(Y_te_real_np[:, i], Y_te_pred_np[:, i])
            rmse = np.sqrt(mean_squared_error(Y_te_real_np[:, i], Y_te_pred_np[:, i]))
            print(f"  {name}: Test R2 = {r2:.4f}, Test RMSE = {rmse:.4f}")
    print("-" * 30)
    
    return R2_train, RMSE_train, MAE_train, R2_test, RMSE_test, MAE_test

def makemodel(model_name, max_depth, max_leaf_nodes, n_estimators, min_samples_split, min_samples_leaf, dim, n_mid, activation, solver, lr, epoch):
    
    if model_name == "RF":
        model = RandomForestRegressor(max_depth = max_depth,
                                     max_leaf_nodes = max_leaf_nodes,
                                     n_estimators = n_estimators,
                                     min_samples_split = min_samples_split,
                                     min_samples_leaf = min_samples_leaf,
                                     n_jobs=-1,
                                     random_state=0)
        
    elif model_name == "GBDT":
        gbdt_inner = GradientBoostingRegressor(max_depth = max_depth,
                                     max_leaf_nodes = max_leaf_nodes,
                                     n_estimators = n_estimators,
                                     min_samples_split = min_samples_split,
                                     min_samples_leaf = min_samples_leaf,
                                     random_state=0)
        # GBDT must be wrapped for multi-target prediction.
        model = MultiOutputRegressor(gbdt_inner) 

    elif model_name == "NN":
        mlp = MLPRegressor(hidden_layer_sizes = \
                             tuple([dim for dim_count in range(n_mid)]),
                             activation = activation,
                             solver = solver,
                             learning_rate_init = lr,
                             max_iter = epoch,
                             random_state=0)
        # Pipeline wrapper to stabilize training.
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', mlp)
        ])
        
    return model

# optuna
def objective_variable(model_name, X_train, y_train, valid_ratio_in_train):
    
    def objective(trial):
        if model_name == "RF":
            # Expand hyperparameter search space.
            n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
            max_depth = trial.suggest_int("max_depth", 5, 50)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0])
            
            regr = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                n_jobs=-1,
                random_state=0
            )
            
        elif model_name == "GBDT":
            # Expand GBDT hyperparameters.
            n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
            max_depth = trial.suggest_int("max_depth", 3, 15)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
            lr = trial.suggest_float("lr", 0.01, 0.3, log=True)
            subsample = trial.suggest_float("subsample", 0.6, 1.0)
            
            gbdt_inner = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                learning_rate=lr,
                subsample=subsample,
                random_state=0
            )
            regr = MultiOutputRegressor(gbdt_inner)
            
        elif model_name == "NN":
            # Tune NN hyperparameter ranges.
            dim = trial.suggest_categorical("dim", [32, 64, 128, 256, 512])
            n_mid = trial.suggest_int("n_mid", 2, 6)
            activation = trial.suggest_categorical("activation", ["tanh", "relu"])
            solver = trial.suggest_categorical("solver", ["adam"])  # Adam is more stable.
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            epoch = trial.suggest_int("epoch", 200, 2000, step=100)
            alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)  # L2 regularization.
            
            # Pipeline wrapper to stabilize training.
            mlp = MLPRegressor(
                hidden_layer_sizes=tuple([dim for _ in range(n_mid)]),
                activation=activation,
                solver=solver,
                learning_rate_init=lr,
                max_iter=epoch,
                alpha=alpha,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=0
            )
            regr = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', mlp)
            ])
            
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=valid_ratio_in_train, random_state=0)
        
        # Catch training errors and return a large value to skip unstable configs.
        try:
            regr.fit(X_tr, y_tr)
            y_val_pred = regr.predict(X_val)
            mse = mean_squared_error(y_val, y_val_pred)
            return mse
        except (ValueError, RuntimeWarning) as e:
            # Return a large value to indicate an unstable configuration.
            print(f"  Trial failed with error: {e}, returning large value")
            return 1e10

    return objective