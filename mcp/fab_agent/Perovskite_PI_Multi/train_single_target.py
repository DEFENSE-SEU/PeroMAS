# train_single_target.py
# 单属性训练版本 - 每个属性单独训练一个模型

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna


def metrics_single(y_train_real, y_train_pred, y_test_real, y_test_pred, target_name=None):
    """单属性模型的评估指标"""
    R2_train = r2_score(y_train_real, y_train_pred)
    RMSE_train = np.sqrt(mean_squared_error(y_train_real, y_train_pred))
    MAE_train = mean_absolute_error(y_train_real, y_train_pred)
    
    R2_test = r2_score(y_test_real, y_test_pred)
    RMSE_test = np.sqrt(mean_squared_error(y_test_real, y_test_pred))
    MAE_test = mean_absolute_error(y_test_real, y_test_pred)
    
    print("-" * 50)
    if target_name:
        print(f"Target: {target_name}")
    print(f"Train - R2: {R2_train:.4f}, RMSE: {RMSE_train:.4f}, MAE: {MAE_train:.4f}")
    print(f"Test  - R2: {R2_test:.4f}, RMSE: {RMSE_test:.4f}, MAE: {MAE_test:.4f}")
    print("-" * 50)
    
    return {
        'target': target_name,
        'R2_train': R2_train, 'RMSE_train': RMSE_train, 'MAE_train': MAE_train,
        'R2_test': R2_test, 'RMSE_test': RMSE_test, 'MAE_test': MAE_test
    }


def makemodel_single(model_name, params):
    """创建单输出模型"""
    if model_name == "RF":
        model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 300),
            max_depth=params.get('max_depth', None),
            max_features=params.get('max_features', 'sqrt'),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            n_jobs=-1,
            random_state=params.get('random_state', 42)
        )
        
    elif model_name == "GBDT":
        model = GradientBoostingRegressor(
            n_estimators=params.get('n_estimators', 300),
            max_depth=params.get('max_depth', 5),
            learning_rate=params.get('lr', 0.1),
            subsample=params.get('subsample', 0.8),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            random_state=params.get('random_state', 42)
        )

    elif model_name == "NN":
        dim = params.get('dim', 128)
        n_mid = params.get('n_mid', 3)
        mlp = MLPRegressor(
            hidden_layer_sizes=tuple([dim for _ in range(n_mid)]),
            activation=params.get('activation', 'relu'),
            solver=params.get('solver', 'adam'),
            learning_rate_init=params.get('lr', 1e-3),
            max_iter=params.get('epoch', 1000),
            alpha=params.get('alpha', 1e-4),
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=params.get('random_state', 42)
        )
        # Pipeline 包装，避免梯度爆炸
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', mlp)
        ])
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    return model


def objective_single_target(model_name, X_train, y_train, valid_ratio, random_state=42):
    """单属性超参数优化目标函数 - 优化版，防止过拟合"""
    
    def objective(trial):
        if model_name == "RF":
            # RF 优化：增加正则化，防止过拟合
            params = {
                'n_estimators': trial.suggest_int("n_estimators", 200, 1000, step=100),
                'max_depth': trial.suggest_int("max_depth", 8, 25),  # 限制深度防止过拟合
                'max_features': trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
                'min_samples_split': trial.suggest_int("min_samples_split", 5, 30),  # 增大防止过拟合
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 3, 20),    # 增大防止过拟合
                'max_samples': trial.suggest_float("max_samples", 0.6, 0.95),        # Bootstrap 采样比例
                'random_state': random_state
            }
            regr = RandomForestRegressor(**params, n_jobs=-1, oob_score=True)
            
        elif model_name == "GBDT":
            # GBDT 优化：更小学习率 + 更多树 + 正则化
            params = {
                'n_estimators': trial.suggest_int("n_estimators", 200, 1500, step=100),
                'max_depth': trial.suggest_int("max_depth", 3, 10),      # 浅树防止过拟合
                'learning_rate': trial.suggest_float("lr", 0.005, 0.1, log=True),  # 更小学习率
                'subsample': trial.suggest_float("subsample", 0.5, 0.9),
                'min_samples_split': trial.suggest_int("min_samples_split", 5, 30),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 3, 20),
                'max_features': trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
                'random_state': random_state
            }
            regr = GradientBoostingRegressor(**params)
            
        elif model_name == "NN":
            # NN 优化：更多 epoch + 更强正则化
            dim = trial.suggest_categorical("dim", [64, 128, 256, 512])
            n_mid = trial.suggest_int("n_mid", 2, 5)
            activation = trial.suggest_categorical("activation", ["tanh", "relu"])
            lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)  # 更小学习率范围
            epoch = trial.suggest_int("epoch", 1000, 5000, step=500)  # 更多 epoch
            alpha = trial.suggest_float("alpha", 1e-4, 1e-1, log=True)  # 更强 L2 正则化
            
            mlp = MLPRegressor(
                hidden_layer_sizes=tuple([dim for _ in range(n_mid)]),
                activation=activation,
                solver='adam',
                learning_rate_init=lr,
                max_iter=epoch,
                alpha=alpha,
                early_stopping=True,
                validation_fraction=0.15,  # 增加验证集比例
                n_iter_no_change=50,       # 增加耐心值
                random_state=random_state
            )
            regr = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', mlp)
            ])
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=valid_ratio, random_state=random_state
        )
        
        try:
            regr.fit(X_tr, y_tr)
            y_val_pred = regr.predict(X_val)
            mse = mean_squared_error(y_val, y_val_pred)
            return mse
        except Exception as e:
            print(f"  Trial failed: {e}")
            return 1e10

    return objective


def build_optimized_model(model_name, best_params, random_state=42):
    """根据最优超参数构建模型"""
    if model_name == "RF":
        model = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            max_features=best_params["max_features"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            max_samples=best_params.get("max_samples", None),
            n_jobs=-1,
            random_state=random_state
        )
    elif model_name == "GBDT":
        model = GradientBoostingRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["lr"],
            subsample=best_params["subsample"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            max_features=best_params.get("max_features", None),
            random_state=random_state
        )
    elif model_name == "NN":
        mlp = MLPRegressor(
            hidden_layer_sizes=tuple([best_params["dim"] for _ in range(best_params["n_mid"])]),
            activation=best_params["activation"],
            solver='adam',
            learning_rate_init=best_params["lr"],
            max_iter=best_params["epoch"],
            alpha=best_params["alpha"],
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
            random_state=random_state
        )
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', mlp)
        ])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model
