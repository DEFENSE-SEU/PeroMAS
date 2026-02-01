# Perovskite_PI
![TOC_3_3](https://github.com/Fukapy/Perovskite_PI/assets/79046839/05d845e9-41f6-4966-b3f6-fb665af5182f)


## Dependencies
We implemented the codes on Python 3.9 in a computer of Windows 10.

Minimum packages we used are following.
- cbfv == 1.1.0
- joblib == 1.1.0
- matplotlib == 3.5.3
- numpy == 1.22.3
- optuna == 3.1.0
- pandas == 1.4.4
- scikit-learn == 1.1.2
- scipy == 1.9.1
- seaborn == 0.12.2
- shap == 0.41.0


## Dataset
"Perovskite_database_content_all_data.csv" is the raw data downloaded from "The Perovskite Database Project" as at 31 March 2022. (#=42459)

The "Perovskite_36937data.csv" is the csv file that was saved in the "Data Curation.ipynb" notebook with unnecessary rows and columns deleted.

The formatted "Perovskite_36937data.csv" was used in the regression analysis.(#=36937)

## Files
Each executable file has the following roles.

`main.py`: Work mainly

`train.py`: Train the machine learning models

`process.py`: Calculate molecular descriptors

`settings.py`: Input the setting

revised_CBFV folder is imported in `process.py`.
It is required to correct abbreviations in the composition and to convert the composition into a chemical feature vector.
It is a modified version of the Python open library CBFV for ease of use in this project.

## Setting arguments
The types and meanings of the arguments of `settings.py` correspond to the following, respectively.

- raw_file_name (str): "data/raw/Perovskite_36937data.csv"
- run_mode (str): "Hyperopt", "Train", "Predict", "Interpret"
- calc_shap (bool): Whether SHAP value is calculated on the interpretability of the model or not.
- target (str): "JV_default_PCE"
- use_X (str): "all"(Materials and processes for all layers), "per"(Only perovskite compositions), "mat"(Materials for all layers)
- num_list (list): List of column names that can be treated as float type numbers. ["Cell_area_measured","Substrate_thickness","ETL_thickness", "Perovskite_thickness","HTL_thickness_list","Backcontact_thickness_list"]
- fill_way (str) : "zero" # "dummy", "zero", "median"
- per_elem_prop (str) :  "oliynyk" # "dummy", "oliynyk", "magpie", "mat2vec" -> for PCE, only oliynyk.
- split_way (int): 0 = onehot, 1 = multihot_1, 2 = multihot_2, 3 = multihot_3
- random_state (int): 0
- test_ratio (float): 0.2
- valid_ratio_in_train (float): 0.25 # 0.8*0.25=0.2
- n_trials (int): 25
- storage_name (str) : f"data/model/hyperopt/{use_X}_optuna_study_attempt"
- model_name (str) : "RF", "GBDT", "NN"
- n_estimators (int) : 270
- max_depth (int or None) : None
- max_leaf_nodes (int or None) : None
- min_samples_split (int) : 3
- min_samples_leaf (int) : 1
- dim (int) : 100
- n_mid (int) : 2
- activation (str) : "relu"
- solver (str) : "adam"
- lr (float) : 1e-3
- epoch (int) : 200
- save_name (str) : f"{use_X} _ {run_mode} _ {model_name} _ sp{str(split_way)} _ {per_elem_prop} _ {fill_way} _ r{str(random_state)}"
- model_save_name (str) : f"model _ {use_X} _ {model_name} _ sp{str(split_way)} _ {per_elem_prop} _ {fill_way} _ r{str(random_state)}"

## Default Output Folder

`./data/csr/` : Folder where created vectors are stored in "Hyperopt" or "Train" mode

`./data/model` : Folder where created regression models are stored in "Train" mode

`./data/model/hyperopt` : Folder where hyperparameters tuned models are stored in "Hyperopt" mode

`./data/model/regression` : Folder where csv file of regression results are stored in "Train" or "Predict" mode

`./data/model/interpret` : Folder where feature importance or shap value are stored in "Interpret" mode


## Examples
To run:
`python main.py`

By this code, `main.py` reads the argument of `setting.py` and start the vectorization and training models.

