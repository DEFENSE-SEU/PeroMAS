# settings_comp_only.py
# Training config: use composition-only features as inputs.
# Created: 2026/01/23

# ============================================================
# Data input
# ============================================================
raw_file_name = "data/raw/full_dataset.csv"
run_mode = "Train"  # "Hyperopt", "Train", "Predict", "Interpret"
calc_shap = False

# Prediction targets (multi-output regression)
# Note: stability_t80 has severe missing data (only 39/711 valid); consider removing.
target = [
    "pce",                  # Power conversion efficiency
    "dft_band_gap",         # DFT band gap
    "energy_above_hull",    # Thermodynamic stability
    "voc",                  # Open-circuit voltage
    "jsc",                  # Short-circuit current density
    "ff"                    # Fill factor
    # "stability_t80",      # T80 stability (too little data to train)
]

# ============================================================
# Feature mode
# ============================================================
use_X = "comp_only"  # Use composition-only features (CBFV)

num_list = []
fill_way = "zero"                    # "dummy", "zero", "median"
per_elem_prop = "oliynyk"            # "oliynyk", "magpie", "mat2vec"
split_way = 1                        # 0:onehot, 1:multihot_1

# ============================================================
# Training parameters
# ============================================================
random_state = 42
test_ratio = 0.2
valid_ratio_in_train = 0.25          # 0.8 * 0.25 = 0.2

# Hyperopt
n_trials = 100
storage_name = f"data/model/hyperopt/{use_X}_optuna_study"

# ============================================================
# Model parameters
# ============================================================
model_name = "RF"  # "RF", "GBDT", "NN"

# Random Forest / GBDT parameters
n_estimators = 300
max_depth = None
max_leaf_nodes = None
min_samples_split = 2
min_samples_leaf = 1

# Neural Network parameters
dim = 128
n_mid = 3
activation = "relu"
solver = "adam"
lr = 1e-3
epoch = 500

# ============================================================
# Save names
# ============================================================
save_name = f"{use_X}_{run_mode}_{model_name}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_r{str(random_state)}"
model_save_name = f"model_{use_X}_{model_name}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_r{str(random_state)}"
