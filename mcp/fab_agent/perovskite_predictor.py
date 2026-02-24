#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
perovskite_predictor.py
Perovskite solar cell property predictor for FabAgent integration.

Provides RF model-based property prediction:
- PCE (power conversion efficiency)
- Voc (open-circuit voltage)
- Jsc (short-circuit current density)
- FF (fill factor)
- dft_band_gap (DFT band gap)
- energy_above_hull (energy above hull)

Author: PSC_Agents Team
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Path setup
PREDICTOR_DIR = Path(__file__).parent.absolute()
MODEL_PROJECT_DIR = PREDICTOR_DIR / "Perovskite_PI_Multi"

# Add model project to path
sys.path.insert(0, str(MODEL_PROJECT_DIR))

# Lazy imports to avoid heavy dependencies at startup
_numpy = None
_pandas = None
_joblib = None
_CBFV_AVAILABLE = False
_PYMATGEN_AVAILABLE = False


def _lazy_import():
    """Lazy import dependencies."""
    global _numpy, _pandas, _joblib, _CBFV_AVAILABLE, _PYMATGEN_AVAILABLE
    
    if _numpy is None:
        import numpy as np
        import pandas as pd
        import joblib
        _numpy = np
        _pandas = pd
        _joblib = joblib
        
        # Check CBFV
        try:
            from revised_CBFV import composition
            _CBFV_AVAILABLE = True
        except ImportError:
            _CBFV_AVAILABLE = False
        
        # Check pymatgen
        try:
            from pymatgen.io.cif import CifParser
            from pymatgen.core.structure import Structure
            _PYMATGEN_AVAILABLE = True
        except ImportError:
            _PYMATGEN_AVAILABLE = False


# =============================================================================
# Constants
# =============================================================================

MODEL_DIR = MODEL_PROJECT_DIR / "data" / "model" / "single_target"

# All supported target properties
ALL_TARGETS = ["pce", "voc", "jsc", "ff", "dft_band_gap", "energy_above_hull"]

# Target metadata
TARGET_INFO = {
    "pce": {"name": "PCE", "unit": "%", "full_name": "Power Conversion Efficiency"},
    "voc": {"name": "Voc", "unit": "V", "full_name": "Open-Circuit Voltage"},
    "jsc": {"name": "Jsc", "unit": "mA/cm²", "full_name": "Short-Circuit Current Density"},
    "ff": {"name": "FF", "unit": "%", "full_name": "Fill Factor"},
    "dft_band_gap": {"name": "Band Gap", "unit": "eV", "full_name": "DFT Band Gap"},
    "energy_above_hull": {"name": "E_hull", "unit": "eV/atom", "full_name": "Energy Above Hull"}
}

# Default model type
DEFAULT_MODEL_TYPE = "RF"


# =============================================================================
# Feature generation functions
# =============================================================================

def generate_cbfv_features(composition_str: str, elem_prop: str = "oliynyk"):
    """Generate composition-based feature vector (CBFV)."""
    _lazy_import()
    
    if not _CBFV_AVAILABLE:
        print("Warning: CBFV not available, using zero features")
        return _numpy.zeros(264)
    
    # Normalize input.
    composition_str = composition_str.replace("|", "")
    
    # Load abbreviation mapping.
    corr_path = MODEL_PROJECT_DIR / "revised_CBFV" / "Perovskite_a_ion_correspond_arr.csv"
    if corr_path.exists():
        try:
            corr = _pandas.read_csv(corr_path)
            for i in range(len(corr)):
                composition_str = composition_str.replace(
                    str(corr["Abbreviation"].iloc[i]), 
                    str(corr["Chemical Formula"].iloc[i])
                )
        except Exception:
            pass
    
    try:
        from revised_CBFV import composition
        df_temp = _pandas.DataFrame([[composition_str, 0]], columns=["formula", "target"])
        X, y, formulae, skipped = composition.generate_features(df_temp, elem_prop=elem_prop)
        return X.fillna(0).values[0]
    except Exception as e:
        print(f"CBFV feature generation failed for '{composition_str}': {e}")
        return _numpy.zeros(264)


def generate_cif_features(cif_content: str):
    """Generate structure features from CIF content."""
    _lazy_import()
    
    if not _PYMATGEN_AVAILABLE:
        print("Warning: pymatgen not available, using zero features")
        return _numpy.zeros(9)
    
    # Fix escaped newlines.
    if "\\n" in cif_content:
        cif_content = cif_content.replace("\\n", "\n")
    
    default_feats = _numpy.zeros(9)
    
    try:
        from pymatgen.io.cif import CifParser
        from pymatgen.core.structure import Structure
        
        try:
            parser = CifParser.from_string(cif_content)
            structure = parser.get_structures()[0]
        except:
            structure = Structure.from_str(cif_content, fmt="cif")
        
        return _numpy.array([
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
    except Exception as e:
        print(f"CIF parsing failed: {e}")
        return default_feats


# =============================================================================
# Predictor class
# =============================================================================

class PerovskitePredictor:
    """Perovskite solar cell property predictor."""
    
    _instance = None  # Singleton instance.
    
    def __new__(cls, *args, **kwargs):
        """Singleton to avoid reloading models."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_type: str = DEFAULT_MODEL_TYPE):
        """
        Initialize the predictor.
        
        Args:
            model_type: Model type (RF, GBDT, NN). Default: RF.
        """
        if self._initialized and self.model_type == model_type:
            return
        
        _lazy_import()
        
        self.model_type = model_type
        self.model_dir = MODEL_DIR
        self.models: Dict[str, Dict[str, Any]] = {}
        self._initialized = True
        self._scan_and_load_models()
    
    def _scan_and_load_models(self):
        """Scan and load models."""
        print(f"[PerovskitePredictor] Loading {self.model_type} models...")
        
        for input_mode in ["comp_only", "cif_only"]:
            self.models[input_mode] = {}
            for target in ALL_TARGETS:
                model_path = self.model_dir / f"{input_mode}_{self.model_type}_{target}.pkl"
                if model_path.exists():
                    try:
                        self.models[input_mode][target] = _joblib.load(model_path)
                        print(f"  ✓ Loaded: {input_mode}/{target}")
                    except Exception as e:
                        print(f"  ✗ Failed to load {input_mode}/{target}: {e}")
                else:
                    print(f"  - Not found: {model_path.name}")
    
    def get_available_targets(self, input_mode: str = "comp_only") -> List[str]:
        """Get available prediction targets."""
        return list(self.models.get(input_mode, {}).keys())
    
    def predict_from_composition(
        self, 
        composition: str,
        targets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Predict properties from a formula.
        
        Args:
            composition: Formula string, e.g., "CsPbI3", "FA0.25MA0.75PbI3"
            targets: Target list to predict (defaults to all).
            
        Returns:
            Prediction results dict.
        """
        if targets is None:
            targets = ALL_TARGETS
        
        # Generate features.
        features = generate_cbfv_features(composition)
        X = features.reshape(1, -1)
        
        results = {
            "input": composition,
            "input_mode": "composition",
            "model_type": self.model_type,
            "predictions": {}
        }
        
        for target in targets:
            if target in self.models.get("comp_only", {}):
                try:
                    model = self.models["comp_only"][target]
                    pred = model.predict(X)[0]
                    info = TARGET_INFO.get(target, {})
                    results["predictions"][target] = {
                        "value": float(pred),
                        "unit": info.get("unit", ""),
                        "name": info.get("name", target)
                    }
                except Exception as e:
                    results["predictions"][target] = {"error": str(e)}
            else:
                results["predictions"][target] = {"error": "Model not available"}
        
        return results
    
    def predict_from_cif(
        self,
        cif_content: str,
        targets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Predict properties from CIF structure.
        
        Args:
            cif_content: CIF content string.
            targets: Target list to predict (defaults to all).
            
        Returns:
            Prediction results dict.
        """
        if targets is None:
            targets = ALL_TARGETS
        
        # Generate features.
        features = generate_cif_features(cif_content)
        X = features.reshape(1, -1)
        
        results = {
            "input": "CIF structure",
            "input_mode": "cif",
            "model_type": self.model_type,
            "predictions": {}
        }
        
        for target in targets:
            if target in self.models.get("cif_only", {}):
                try:
                    model = self.models["cif_only"][target]
                    pred = model.predict(X)[0]
                    info = TARGET_INFO.get(target, {})
                    results["predictions"][target] = {
                        "value": float(pred),
                        "unit": info.get("unit", ""),
                        "name": info.get("name", target)
                    }
                except Exception as e:
                    results["predictions"][target] = {"error": str(e)}
            else:
                results["predictions"][target] = {"error": "Model not available"}
        
        return results
    
    def predict(
        self,
        composition: Optional[str] = None,
        cif_content: Optional[str] = None,
        cif_file: Optional[str] = None,
        targets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Unified prediction interface.
        
        Args:
            composition: Formula string.
            cif_content: CIF content string.
            cif_file: CIF file path.
            targets: Target list.
            
        Returns:
            Prediction results.
        """
        # Prefer formula input.
        if composition:
            return self.predict_from_composition(composition, targets)
        
        # Otherwise use CIF content.
        if cif_content:
            return self.predict_from_cif(cif_content, targets)
        
        # Read CIF from file.
        if cif_file and os.path.exists(cif_file):
            with open(cif_file, 'r') as f:
                cif_content = f.read()
            return self.predict_from_cif(cif_content, targets)
        
        return {"error": "No valid input provided. Use composition, cif_content, or cif_file."}
    
    def format_results_for_visualization(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Format predictions for visualization tools.
        
        Returns:
            {PCE_percent, Voc_V, Jsc_mA_cm2, FF_percent, ...}
        """
        predictions = results.get("predictions", {})
        formatted = {}
        
        # Map to visualization format.
        mapping = {
            "pce": "PCE_percent",
            "voc": "Voc_V",
            "jsc": "Jsc_mA_cm2",
            "ff": "FF_percent",
            "dft_band_gap": "BandGap_eV",
            "energy_above_hull": "E_hull_eV"
        }
        
        for target, viz_name in mapping.items():
            if target in predictions:
                pred = predictions[target]
                if isinstance(pred, dict) and "value" in pred:
                    formatted[viz_name] = pred["value"]
        
        return formatted


# =============================================================================
# Convenience functions
# =============================================================================

_predictor_instance: Optional[PerovskitePredictor] = None


def get_predictor(model_type: str = DEFAULT_MODEL_TYPE) -> PerovskitePredictor:
    """Get predictor instance (singleton)."""
    global _predictor_instance
    if _predictor_instance is None or _predictor_instance.model_type != model_type:
        _predictor_instance = PerovskitePredictor(model_type)
    return _predictor_instance


def predict_perovskite_properties(
    composition: Optional[str] = None,
    cif_content: Optional[str] = None,
    cif_file: Optional[str] = None,
    targets: Optional[List[str]] = None,
    model_type: str = DEFAULT_MODEL_TYPE
) -> Dict[str, Any]:
    """
    Predict perovskite solar cell properties (main entry point).
    
    Args:
        composition: Formula string, e.g., "CsPbI3", "FA0.25MA0.75PbI3", "MAPbI3"
        cif_content: CIF content string.
        cif_file: CIF file path.
        targets: Target list, e.g., pce, voc, jsc, ff, dft_band_gap, energy_above_hull.
        model_type: Model type (default: "RF").
        
    Returns:
        Prediction results dict with predictions and formatted_for_viz.
        
    Example:
        >>> result = predict_perovskite_properties(composition="CsPbI3")
        >>> print(result["predictions"]["pce"]["value"])  # PCE prediction
    """
    predictor = get_predictor(model_type)
    results = predictor.predict(
        composition=composition,
        cif_content=cif_content,
        cif_file=cif_file,
        targets=targets
    )
    
    # Add formatted results and status.
    if "predictions" in results:
        results["formatted_for_viz"] = predictor.format_results_for_visualization(results)
        # Check for valid predictions.
        has_valid_prediction = any(
            isinstance(p, dict) and "value" in p 
            for p in results["predictions"].values()
        )
        results["status"] = "success" if has_valid_prediction else "partial"
    else:
        results["status"] = "error"
    
    return results


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing PerovskitePredictor")
    print("=" * 60)
    
    # Test formula prediction.
    test_compositions = ["CsPbI3", "MAPbI3", "FA0.25MA0.75PbI3"]
    
    for comp in test_compositions:
        print(f"\n🔹 Testing: {comp}")
        result = predict_perovskite_properties(composition=comp)
        
        if "predictions" in result:
            for target, pred in result["predictions"].items():
                if isinstance(pred, dict) and "value" in pred:
                    print(f"   {pred.get('name', target)}: {pred['value']:.4f} {pred.get('unit', '')}")
                elif isinstance(pred, dict) and "error" in pred:
                    print(f"   {target}: ERROR - {pred['error']}")
        
        print(f"   Viz format: {result.get('formatted_for_viz', {})}")
