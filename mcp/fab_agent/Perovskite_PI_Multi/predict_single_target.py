#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_single_target.py
Single-target model prediction API.

Usage:
    from predict_single_target import PerovskitePredictor
    
    predictor = PerovskitePredictor()
    
    # Predict a single material
    results = predictor.predict_composition("FA0.25MA0.75PbI3")
    
    # Predict CIF structure
    results = predictor.predict_cif(cif_string)
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Import feature generation modules.
from revised_CBFV import composition
from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure


# All target properties
ALL_TARGETS = ["pce", "dft_band_gap", "energy_above_hull", "voc", "jsc", "ff"]

# Model directory
MODEL_DIR = "data/model/single_target"


class PerovskitePredictor:
    """Perovskite property predictor (single-target models)."""
    
    def __init__(self, model_type: str = "RF", model_dir: str = MODEL_DIR):
        """
        Initialize the predictor.
        
        Args:
            model_type: Model type ("RF", "GBDT", "NN")
            model_dir: Model directory
        """
        self.model_type = model_type
        self.model_dir = model_dir
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load all models."""
        print(f"Loading {self.model_type} models...")
        
        for feature_mode in ["comp_only", "cif_only"]:
            self.models[feature_mode] = {}
            for target in ALL_TARGETS:
                model_path = os.path.join(
                    self.model_dir, 
                    f"{feature_mode}_{self.model_type}_{target}.pkl"
                )
                if os.path.exists(model_path):
                    self.models[feature_mode][target] = joblib.load(model_path)
                    print(f"  Loaded: {feature_mode}/{target}")
                else:
                    print(f"  Missing: {model_path}")
    
    def _generate_cbfv_features(self, composition_str: str, elem_prop: str = "oliynyk") -> np.ndarray:
        """Generate CBFV features."""
        # Normalize input.
        composition_str = composition_str.replace("|", "")
        
        # Load abbreviation mapping.
        corr_path = "revised_CBFV/Perovskite_a_ion_correspond_arr.csv"
        if os.path.exists(corr_path):
            corr = pd.read_csv(corr_path)
            for i in range(len(corr)):
                composition_str = composition_str.replace(
                    corr["Abbreviation"].iloc[i], 
                    corr["Chemical Formula"].iloc[i]
                )
        
        try:
            df_temp = pd.DataFrame([[composition_str, 0]], columns=["formula", "target"])
            X, y, formulae, skipped = composition.generate_features(df_temp, elem_prop=elem_prop)
            return X.fillna(0).values[0]
        except Exception as e:
            print(f"CBFV generation failed for '{composition_str}': {e}")
            if elem_prop == "oliynyk":
                return np.zeros(264)
            elif elem_prop == "magpie":
                return np.zeros(132)
            else:
                return np.zeros(264)
    
    def _generate_cif_features(self, cif_string: str) -> np.ndarray:
        """Generate CIF structure features."""
        # Fix escaped newlines.
        if "\\n" in cif_string:
            cif_string = cif_string.replace("\\n", "\n")
        
        default_feats = np.zeros(9)
        
        try:
            parser = CifParser.from_string(cif_string)
            structure = parser.get_structures()[0]
        except:
            try:
                structure = Structure.from_str(cif_string, fmt="cif")
            except Exception as e:
                print(f"CIF parsing failed: {e}")
                return default_feats
        
        return np.array([
            structure.density,
            structure.volume,
            structure.lattice.a,
            structure.lattice.b,
            structure.lattice.c,
            structure.lattice.alpha,
            structure.lattice.beta,
            structure.lattice.gamma,
            structure.num_sites
        ])
    
    def predict_composition(
        self, 
        composition_str: str, 
        targets: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Predict properties from a formula.
        
        Args:
            composition_str: Formula string, e.g., "FA0.25MA0.75PbI3"
            targets: Target list to predict (defaults to all)
            
        Returns:
            Dict[str, float]: Predicted values
        """
        if targets is None:
            targets = ALL_TARGETS
        
        # Generate features.
        features = self._generate_cbfv_features(composition_str)
        X = features.reshape(1, -1)
        
        results = {}
        for target in targets:
            if target in self.models.get("comp_only", {}):
                model = self.models["comp_only"][target]
                pred = model.predict(X)[0]
                results[target] = float(pred)
            else:
                results[target] = None
        
        return results
    
    def predict_cif(
        self, 
        cif_string: str, 
        targets: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Predict properties from CIF structure.
        
        Args:
            cif_string: CIF structure string
            targets: Target list to predict (defaults to all)
            
        Returns:
            Dict[str, float]: Predicted values
        """
        if targets is None:
            targets = ALL_TARGETS
        
        # Generate features.
        features = self._generate_cif_features(cif_string)
        X = features.reshape(1, -1)
        
        results = {}
        for target in targets:
            if target in self.models.get("cif_only", {}):
                model = self.models["cif_only"][target]
                pred = model.predict(X)[0]
                results[target] = float(pred)
            else:
                results[target] = None
        
        return results
    
    def predict_batch(
        self, 
        compositions: Optional[List[str]] = None,
        cif_strings: Optional[List[str]] = None,
        feature_mode: str = "comp_only",
        targets: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Batch prediction.
        
        Args:
            compositions: Formula list (when feature_mode="comp_only")
            cif_strings: CIF string list (when feature_mode="cif_only")
            feature_mode: Feature mode
            targets: Target list to predict
            
        Returns:
            pd.DataFrame: Prediction table
        """
        if targets is None:
            targets = ALL_TARGETS
        
        if feature_mode == "comp_only":
            if compositions is None:
                raise ValueError("compositions required for comp_only mode")
            
            features_list = [self._generate_cbfv_features(c) for c in compositions]
            X = np.vstack(features_list)
            input_col = compositions
            
        elif feature_mode == "cif_only":
            if cif_strings is None:
                raise ValueError("cif_strings required for cif_only mode")
            
            features_list = [self._generate_cif_features(c) for c in cif_strings]
            X = np.vstack(features_list)
            input_col = [f"cif_{i}" for i in range(len(cif_strings))]
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")
        
        results = {"input": input_col}
        for target in targets:
            if target in self.models.get(feature_mode, {}):
                model = self.models[feature_mode][target]
                preds = model.predict(X)
                results[target] = preds
            else:
                results[target] = [None] * len(input_col)
        
        return pd.DataFrame(results)
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models."""
        available = {}
        for feature_mode, targets_dict in self.models.items():
            available[feature_mode] = list(targets_dict.keys())
        return available


def test_predictor():
    """Test the predictor."""
    print("=" * 60)
    print("Testing PerovskitePredictor")
    print("=" * 60)
    
    # Check model directory.
    if not os.path.exists(MODEL_DIR):
        print(f"Model directory not found: {MODEL_DIR}")
        print("Please train models first using train_single_target.sh")
        return
    
    predictor = PerovskitePredictor(model_type="RF")
    
    print("\nAvailable models:")
    print(predictor.get_available_models())
    
    # Test formula prediction.
    test_compositions = [
        "Cs0.05FA0.79MA0.16Pb(I0.83Br0.17)3",
        "FA0.25MA0.75PbI3",
        "CsPbI3"
    ]
    
    print("\n" + "=" * 60)
    print("Testing composition prediction:")
    print("=" * 60)
    
    for comp in test_compositions:
        print(f"\nComposition: {comp}")
        try:
            results = predictor.predict_composition(comp)
            for k, v in results.items():
                if v is not None:
                    print(f"  {k}: {v:.4f}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    test_predictor()
