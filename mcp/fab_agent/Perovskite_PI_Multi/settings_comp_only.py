# settings_comp_only.py
# 训练配置：仅使用分子式 (Composition) 作为输入特征
# 创建时间: 2026/01/23

# ============================================================
# 数据输入
# ============================================================
raw_file_name = "data/raw/full_dataset.csv"
run_mode = "Train"  # "Hyperopt", "Train", "Predict", "Interpret"
calc_shap = False

# 预测目标 (多输出回归)
# 注意: stability_t80 数据缺失严重(仅39/711有效)，建议移除
target = [
    "pce",                  # 光电转换效率
    "dft_band_gap",         # DFT计算的带隙
    "energy_above_hull",    # 热力学稳定性
    "voc",                  # 开路电压
    "jsc",                  # 短路电流密度
    "ff"                    # 填充因子
    # "stability_t80",      # T80稳定性(数据太少，暂不训练)
]

# ============================================================
# 特征模式
# ============================================================
use_X = "comp_only"  # 仅使用分子式特征 (CBFV)

num_list = []
fill_way = "zero"                    # "dummy", "zero", "median"
per_elem_prop = "oliynyk"            # "oliynyk", "magpie", "mat2vec"
split_way = 1                        # 0:onehot, 1:multihot_1

# ============================================================
# 训练参数
# ============================================================
random_state = 42
test_ratio = 0.2
valid_ratio_in_train = 0.25          # 0.8 * 0.25 = 0.2

# Hyperopt
n_trials = 100
storage_name = f"data/model/hyperopt/{use_X}_optuna_study"

# ============================================================
# 模型参数
# ============================================================
model_name = "RF"  # "RF", "GBDT", "NN"

# Random Forest / GBDT 参数
n_estimators = 300
max_depth = None
max_leaf_nodes = None
min_samples_split = 2
min_samples_leaf = 1

# Neural Network 参数
dim = 128
n_mid = 3
activation = "relu"
solver = "adam"
lr = 1e-3
epoch = 500

# ============================================================
# 保存名称
# ============================================================
save_name = f"{use_X}_{run_mode}_{model_name}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_r{str(random_state)}"
model_save_name = f"model_{use_X}_{model_name}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_r{str(random_state)}"
