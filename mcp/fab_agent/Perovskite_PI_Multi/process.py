# process.py
# Optimized for Python 3.10+, Pandas 2.x, and CIF Parsing

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')
from revised_CBFV import composition
import statistics
from scipy.sparse import csr_matrix, save_npz, load_npz

# Import pymatgen for CIF parsing.
from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure

# -----------------------------------------------------------
# 1. Text/categorical feature processing
# -----------------------------------------------------------
def onehot_vector_table(x_name, df):
    vector_table_data = pd.get_dummies(df[x_name]).add_prefix(x_name+"_")
    return vector_table_data

def multihot_vector_table_1(x_name, df):
    first_split = df[x_name].astype(str).str.split("|")
    single_split = pd.DataFrame(list(first_split))
    
    start = 0
    small_box = []
    # Note: reshape(-1) remains valid in newer pandas/numpy.
    single_split_reshape = np.array(single_split).transpose().reshape(-1,) 
    for j in range(np.shape(single_split)[1]):
        stop = start + len(df)
        dummy = pd.get_dummies(single_split_reshape).iloc[start:stop].reset_index(drop = True)
        small_box.append(dummy)
        start = start+len(df)

    sum_box = sum(small_box)
    if "nan" in sum_box.columns:
        sum_box = sum_box.drop("nan", axis = 1)
    sum_box.where(sum_box <= 0, 1, inplace = True)
    sum_box = sum_box.add_prefix(x_name + "_")
    return sum_box

# ... (multihot_vector_table_2 and 3 follow the same logic, using concat instead of append) ...

# -----------------------------------------------------------
# 2. Numerical feature processing
# -----------------------------------------------------------
def numerical_sum_table(num_name, df, fill_way = "zero"):
    delimit = df[num_name].astype(str).str.split("|")
    delimit = pd.DataFrame(list(delimit))
    num_sum=[]
    for i in range(len(df)):
        box = []
        for j in range(delimit.shape[1]):
            try:
                # Use native float for better compatibility.
                box.append(float(delimit[j][i]))
            except:
                box.append(0)
        num_sum.append(sum(np.array(pd.Series(box).fillna(0))))
        
    numerical_table_data = pd.DataFrame(num_sum, columns = {num_name})
    
    if fill_way == "median":
        except_zero = numerical_table_data[num_name].iloc[np.nonzero(np.array(numerical_table_data[num_name]))]
        try:
            median = statistics.median(except_zero)
        except:
            median = 0
        for i in range(len(numerical_table_data)):
            if numerical_table_data[num_name].iloc[i] == 0:
                numerical_table_data[num_name].iloc[i] = median
    return numerical_table_data

# -----------------------------------------------------------
# 3. Composition features (CBFV)
# -----------------------------------------------------------
def cbfv_table(x_name, df, elem_prop="oliynyk"):
    corr = pd.read_csv("revised_CBFV/Perovskite_a_ion_correspond_arr.csv") 
    df[x_name] = df[x_name].astype(str).str.replace("|", "", regex=True)
    for corr_i in range(len(corr)): 
        df[x_name] = df[x_name].str.replace(corr["Abbreviation"][corr_i], corr["Chemical Formula"][corr_i], regex=True)

    try:
        data = []
        for i in range(len(df)):
            data.append([df[x_name][i], 0])
        df_temp = pd.DataFrame(data, columns = ["formula", "target"])
        X, y, formulae, skipped = composition.generate_features(df_temp, elem_prop = elem_prop)
        table = pd.DataFrame(X)
    except:    
        cbfv = []
        for i in range(len(df)):
            df_temp = pd.DataFrame([[df[x_name][i], 0]], columns = ["formula","target"])
            try:
                X, y, formulae, skipped = composition.generate_features(df_temp, elem_prop = elem_prop)
                cbfv.append(np.array(X.fillna(0))[0])
            except:
                # Pad zeros based on elem_prop dimension.
                if elem_prop == "oliynyk":
                    cbfv.append(np.zeros(264))
                elif elem_prop == "magpie":
                    cbfv.append(np.zeros(132))
                elif elem_prop == "mat2vec":
                    cbfv.append(np.zeros(1200))
                else:
                    print("elem_prop name error!")
        try:
            table = pd.DataFrame(cbfv, columns = X.columns)
        except:
            table = pd.DataFrame(cbfv)
    return table

# -----------------------------------------------------------
# 4. CIF structure features (pymatgen)
# -----------------------------------------------------------
def cif_vector_table(x_name, df):
    print(f"Processing CIF data column '{x_name}'...")
    cif_features = []
    
    default_feats = [0.0] * 9
    feat_names = ["density", "volume", "a", "b", "c", "alpha", "beta", "gamma", "nsites"]

    for i in range(len(df)):
        cif_str = str(df[x_name].iloc[i])
        # Fix escaped newlines introduced by CSV.
        if "\\n" in cif_str:
            cif_str = cif_str.replace("\\n", "\n")
            
        try:
            # Try to parse.
            parser = CifParser.from_string(cif_str)
            structure = parser.get_structures()[0]
            
            feats = {
                "density": structure.density,
                "volume": structure.volume,
                "a": structure.lattice.a,
                "b": structure.lattice.b,
                "c": structure.lattice.c,
                "alpha": structure.lattice.alpha,
                "beta": structure.lattice.beta,
                "gamma": structure.lattice.gamma,
                "nsites": structure.num_sites
            }
            cif_features.append(list(feats.values()))
            
        except Exception:
            # Fallback: Structure.from_str can be more robust if CifParser fails.
            try:
                structure = Structure.from_str(cif_str, fmt="cif")
                feats = {
                    "density": structure.density,
                    "volume": structure.volume,
                    "a": structure.lattice.a,
                    "b": structure.lattice.b,
                    "c": structure.lattice.c,
                    "alpha": structure.lattice.alpha,
                    "beta": structure.lattice.beta,
                    "gamma": structure.lattice.gamma,
                    "nsites": structure.num_sites
                }
                cif_features.append(list(feats.values()))
            except Exception:
                cif_features.append(default_feats)

    return pd.DataFrame(cif_features, columns=[f"CIF_{n}" for n in feat_names])


# -----------------------------------------------------------
# 5. Save/load helpers
# -----------------------------------------------------------
def vec2csr(vec, csr_file_name, columns_file_name):
    # Auto-create directories.
    if os.path.dirname(csr_file_name):
        os.makedirs(os.path.dirname(csr_file_name), exist_ok=True)
    
    csr = csr_matrix(vec)
    save_npz(csr_file_name, csr)
    if columns_file_name is not None:
        columns_arr = np.array(vec.columns)
        np.save(columns_file_name, columns_arr)
        
def csr2vec(csr_file_name, columns_file_name):
    if columns_file_name is None:
        vec = load_npz(csr_file_name).toarray()
    else:
        vec = pd.DataFrame(load_npz(csr_file_name).toarray(),
             columns=np.load(columns_file_name, allow_pickle=True))
    return vec

# -----------------------------------------------------------
# 6. Main feature generation entry
# -----------------------------------------------------------
def file2vector(raw_file_name, split_way, per_elem_prop, fill_way, num_list, use_X):
    df = pd.read_csv(raw_file_name)
    x_list = []
    
    # Mode: "comp_only" - composition only
    if use_X == "comp_only":
        print("Feature Mode: Composition Only (CBFV)")
        if "composition" in df.columns:
            x_comp = cbfv_table("composition", df.copy(), per_elem_prop)
            x_list.append(x_comp)
            print(f"  -> Composition features shape: {x_comp.shape}")
        else:
            print("Error: 'composition' column not found!")
            
    # Mode: "cif_only" - CIF features only
    elif use_X == "cif_only":
        print("Feature Mode: CIF Structure Only")
        if "cif" in df.columns:
            x_cif = cif_vector_table("cif", df)
            x_list.append(x_cif)
            print(f"  -> CIF features shape: {x_cif.shape}")
        else:
            print("Error: 'cif' column not found!")
    
    # Mode: "cif_comp" - CIF + composition
    elif use_X == "cif_comp":
        print("Feature Mode: CIF + Composition")
        if "composition" in df.columns:
            x_comp = cbfv_table("composition", df.copy(), per_elem_prop)
            x_list.append(x_comp)
            print(f"  -> Composition features shape: {x_comp.shape}")
        if "cif" in df.columns:
            x_cif = cif_vector_table("cif", df)
            x_list.append(x_cif)
            print(f"  -> CIF features shape: {x_cif.shape}")
            
    # Mode: legacy (all, per, mat)
    else:
        if use_X == "all":
            df = df.iloc[:,0:248] 
        elif use_X == "per":
            df = pd.DataFrame(df["Perovskite_composition_long_form"])
        elif use_X == "mat":
            df = df[["Substrate_stack_sequence", "ETL_stack_sequence","Perovskite_composition_long_form","HTL_stack_sequence","Backcontact_stack_sequence"]]
        
        for x_name in list(df.columns):
            if x_name in num_list and fill_way != "dummy":
                x = numerical_sum_table(x_name, df, fill_way)
                x_list.append(x)
            elif x_name == "Perovskite_composition_long_form" and per_elem_prop != "dummy":
                x = cbfv_table(x_name, df, per_elem_prop)
                x_list.append(x)
            else:
                if split_way == 0:
                    x = onehot_vector_table(x_name, df)
                elif split_way == 1:
                    x = multihot_vector_table_1(x_name, df)
                # ... (split_way 2, 3)
                x_list.append(x)

    if len(x_list) > 0:
        X = pd.concat(x_list, axis=1)
    else:
        print("Error: No features generated.")
        return pd.DataFrame()

    X = X.fillna(0)
    print(f"Total features shape: {X.shape}")

    vec2csr(vec = X,
            csr_file_name = f"data/csr/{use_X}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_csr.npz",
            columns_file_name = f"data/csr/{use_X}_sp{str(split_way)}_{per_elem_prop}_{fill_way}_columns.npy")
    
    return X