"""
SHAP Analysis Tool for Perovskite Solar Cell Materials

Provides SHAP (SHapley Additive exPlanations) analysis for understanding
feature importance and model interpretability in material property prediction.

Features:
- Feature importance analysis for PCE prediction
- SHAP summary plots
- SHAP dependence plots for specific features
- Waterfall plots for individual predictions
- Force plots for local explanations

Author: PSC_Agents Team
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

# Try to import SHAP and ML libraries
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================
# Default Feature Names for Perovskite Analysis
# ============================================================

PEROVSKITE_FEATURES = [
    "A_site_ionic_radius",      # A位离子半径 (Å)
    "B_site_ionic_radius",      # B位离子半径 (Å)  
    "X_site_ionic_radius",      # X位离子半径 (Å)
    "tolerance_factor",         # 容忍因子 t
    "octahedral_factor",        # 八面体因子 μ
    "bandgap_eV",               # 带隙 (eV)
    "formation_energy",         # 形成能 (eV/atom)
    "Cs_fraction",              # Cs含量比例
    "FA_fraction",              # FA含量比例
    "MA_fraction",              # MA含量比例
    "Pb_fraction",              # Pb含量比例
    "Sn_fraction",              # Sn含量比例
    "I_fraction",               # I含量比例
    "Br_fraction",              # Br含量比例
    "Cl_fraction",              # Cl含量比例
    "film_thickness_nm",        # 薄膜厚度 (nm)
    "annealing_temp_C",         # 退火温度 (°C)
    "annealing_time_min",       # 退火时间 (min)
]


# ============================================================
# SHAP Analysis Tool Class
# ============================================================

class SHAPAnalyzer:
    """
    SHAP Analysis tool for perovskite material property prediction.
    
    Provides interpretable machine learning analysis for understanding
    which material features contribute most to predicted properties.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize SHAP Analyzer.
        
        Args:
            output_dir: Directory for saving analysis outputs
        """
        self.output_dir = Path(output_dir) if output_dir else Path("analysis_output/shap")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check dependencies
        self.has_shap = HAS_SHAP
        self.has_matplotlib = HAS_MATPLOTLIB
        self.has_pandas = HAS_PANDAS
        self.has_numpy = HAS_NUMPY
        
        # Cache for loaded models and data
        self._model = None
        self._explainer = None
        self._shap_values = None
    
    def get_feature_importance(
        self,
        feature_importance: Dict[str, float],
        material_composition: Optional[str] = None,
        target_property: str = "PCE",
        n_top_features: int = 10,
    ) -> Dict[str, Any]:
        """
        Get SHAP-based feature importance for material property prediction.
        
        Args:
            feature_importance: Dict of feature name -> importance value (required)
            material_composition: Material formula (e.g., "Cs0.05FA0.95PbI3")
            target_property: Property to analyze (PCE, Voc, Jsc, FF, T80)
            n_top_features: Number of top features to return
            
        Returns:
            Dict with feature importance ranking and analysis
        """
        result = {
            "success": False,
            "target_property": target_property,
            "material": material_composition,
            "feature_importance": {},
            "top_features": [],
            "analysis_summary": "",
            "error": None,
        }
        
        try:
            if not feature_importance:
                result["error"] = "feature_importance dict is required"
                return result
            
            # Sort by importance
            sorted_features = sorted(
                feature_importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:n_top_features]
            
            result["feature_importance"] = feature_importance
            result["top_features"] = [
                {"rank": i+1, "feature": f, "importance": round(v, 4)}
                for i, (f, v) in enumerate(sorted_features)
            ]
            
            # Generate analysis summary
            top_3 = [f[0] for f in sorted_features[:3]]
            result["analysis_summary"] = (
                f"[SHAP Analysis] For {target_property} prediction, "
                f"the top 3 most important features are: {', '.join(top_3)}."
            )
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def generate_summary_plot(
        self,
        feature_importance: Dict[str, float],
        shap_values: Optional[List[List[float]]] = None,
        feature_values: Optional[List[List[float]]] = None,
        feature_names: Optional[List[str]] = None,
        target_property: str = "PCE",
        plot_type: str = "bar",
        max_features: int = 15,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate SHAP summary plot.
        
        Args:
            feature_importance: Dict of feature name -> importance value (required for bar plot)
            shap_values: 2D list of SHAP values [n_samples, n_features] (required for beeswarm)
            feature_values: 2D list of feature values [n_samples, n_features] (required for beeswarm)
            feature_names: List of feature names
            target_property: Property to analyze
            plot_type: "bar" for bar plot, "beeswarm" for beeswarm plot
            max_features: Maximum features to display
            save: Whether to save the plot
            
        Returns:
            Dict with plot filepath and metadata
        """
        result = {
            "success": False,
            "filepath": None,
            "plot_type": plot_type,
            "target_property": target_property,
            "error": None,
        }
        
        if not self.has_matplotlib:
            result["error"] = "matplotlib not installed. Install with: pip install matplotlib"
            return result
        
        if not self.has_numpy:
            result["error"] = "numpy not installed. Install with: pip install numpy"
            return result
        
        import numpy as np
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if plot_type == "bar":
                if not feature_importance:
                    result["error"] = "feature_importance dict is required for bar plot"
                    return result
                
                # Bar plot of mean |SHAP|
                importance = list(feature_importance.items())[:max_features]
                importance.sort(key=lambda x: abs(x[1]), reverse=True)
                
                features = [f[0] for f in importance]
                values = [abs(f[1]) for f in importance]
                
                colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(features)))
                
                bars = ax.barh(range(len(features)), values, color=colors)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features)
                ax.invert_yaxis()
                ax.set_xlabel("Mean |SHAP Value| (Impact on Prediction)")
                ax.set_title(f"SHAP Feature Importance for {target_property}")
                
            else:  # beeswarm plot
                if shap_values is None or feature_values is None:
                    result["error"] = "shap_values and feature_values are required for beeswarm plot"
                    return result
                
                shap_arr = np.array(shap_values)
                feat_arr = np.array(feature_values)
                
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(shap_arr.shape[1])]
                
                n_features_plot = min(max_features, shap_arr.shape[1])
                
                for i in range(n_features_plot):
                    y_pos = np.ones(shap_arr.shape[0]) * i
                    colors = feat_arr[:, i]
                    ax.scatter(shap_arr[:, i], y_pos, c=colors, cmap='RdBu_r', alpha=0.6, s=10)
                
                ax.set_yticks(range(n_features_plot))
                ax.set_yticklabels(feature_names[:n_features_plot])
                ax.invert_yaxis()
                ax.set_xlabel("SHAP Value (Impact on Prediction)")
                ax.set_title(f"SHAP Summary Plot for {target_property}")
            
            plt.tight_layout()
            
            if save:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = self.output_dir / f"shap_summary_{target_property}_{plot_type}_{ts}.png"
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                result["filepath"] = str(filepath)
            
            plt.close(fig)
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def analyze_single_prediction(
        self,
        contributions: List[Dict[str, Any]],
        base_value: float,
        predicted_value: float,
        target_property: str = "PCE",
    ) -> Dict[str, Any]:
        """
        Analyze SHAP contributions for a single material prediction.
        
        Args:
            contributions: List of dicts with keys: feature, value, contribution
                Example: [{"feature": "bandgap_eV", "value": 1.55, "contribution": 0.8}, ...]
            base_value: The base/expected value (average prediction)
            predicted_value: The predicted property value
            target_property: Property being predicted
            
        Returns:
            Dict with feature contributions breakdown
        """
        result = {
            "success": False,
            "target_property": target_property,
            "predicted_value": predicted_value,
            "base_value": base_value,
            "contributions": [],
            "top_positive": [],
            "top_negative": [],
            "interpretation": "",
            "error": None,
        }
        
        try:
            if not contributions:
                result["error"] = "contributions list is required"
                return result
            
            # Ensure contributions have required fields
            processed_contributions = []
            for c in contributions:
                if "feature" not in c or "contribution" not in c:
                    continue
                processed_contributions.append({
                    "feature": c["feature"],
                    "value": c.get("value", None),
                    "contribution": round(c["contribution"], 3),
                    "direction": "positive" if c["contribution"] > 0 else "negative"
                })
            
            # Sort by absolute contribution
            processed_contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            result["contributions"] = processed_contributions
            
            # Top positive and negative contributors
            result["top_positive"] = [c for c in processed_contributions if c["contribution"] > 0][:3]
            result["top_negative"] = [c for c in processed_contributions if c["contribution"] < 0][:3]
            
            # Generate interpretation
            if result["top_positive"]:
                pos_features = [c["feature"] for c in result["top_positive"]]
                pos_text = f"Positive contributors: {', '.join(pos_features)}"
            else:
                pos_text = "No significant positive contributors"
                
            if result["top_negative"]:
                neg_features = [c["feature"] for c in result["top_negative"]]
                neg_text = f"Negative contributors: {', '.join(neg_features)}"
            else:
                neg_text = "No significant negative contributors"
            
            result["interpretation"] = (
                f"[SHAP Waterfall] Starting from base {target_property}={base_value:.1f}%, "
                f"the model predicts {predicted_value:.1f}%. "
                f"{pos_text}. {neg_text}."
            )
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def generate_dependence_plot(
        self,
        feature_name: str,
        feature_values: List[float],
        shap_values: List[float],
        interaction_values: Optional[List[float]] = None,
        interaction_feature: Optional[str] = None,
        target_property: str = "PCE",
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate SHAP dependence plot for a specific feature.
        
        Shows how a feature's value affects predictions, optionally
        colored by an interaction feature.
        
        Args:
            feature_name: Feature to analyze
            feature_values: List of feature values (required)
            shap_values: List of SHAP values for this feature (required)
            interaction_values: Values for coloring (optional)
            interaction_feature: Feature name for coloring (optional)
            target_property: Property being predicted
            save: Whether to save the plot
            
        Returns:
            Dict with plot filepath and metadata
        """
        result = {
            "success": False,
            "filepath": None,
            "feature": feature_name,
            "interaction_feature": interaction_feature,
            "target_property": target_property,
            "correlation": None,
            "trend": None,
            "error": None,
        }
        
        if not self.has_matplotlib:
            result["error"] = "matplotlib not installed"
            return result
        
        if not self.has_numpy:
            result["error"] = "numpy not installed"
            return result
        
        import numpy as np
        
        try:
            if not feature_values or not shap_values:
                result["error"] = "feature_values and shap_values are required"
                return result
            
            if len(feature_values) != len(shap_values):
                result["error"] = "feature_values and shap_values must have same length"
                return result
            
            feat_arr = np.array(feature_values)
            shap_arr = np.array(shap_values)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(8, 6))
            
            if interaction_values is not None:
                inter_arr = np.array(interaction_values)
                scatter = ax.scatter(
                    feat_arr, shap_arr,
                    c=inter_arr, cmap='RdBu_r',
                    alpha=0.7, s=30
                )
                plt.colorbar(scatter, ax=ax, label=interaction_feature or "Interaction")
            else:
                ax.scatter(feat_arr, shap_arr, alpha=0.7, s=30, c='steelblue')
            
            # Add trend line
            z = np.polyfit(feat_arr, shap_arr, 1)
            p = np.poly1d(z)
            x_line = np.linspace(feat_arr.min(), feat_arr.max(), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.8, label=f'Trend (slope={z[0]:.2f})')
            
            ax.set_xlabel(f"{feature_name} Value")
            ax.set_ylabel(f"SHAP Value for {target_property}")
            ax.set_title(f"SHAP Dependence Plot: {feature_name}")
            ax.legend()
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            
            if save:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = feature_name.replace("/", "_").replace(" ", "_")
                filepath = self.output_dir / f"shap_dependence_{safe_name}_{ts}.png"
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                result["filepath"] = str(filepath)
            
            plt.close(fig)
            
            # Calculate correlation
            correlation = np.corrcoef(feat_arr, shap_arr)[0, 1]
            result["correlation"] = round(correlation, 3)
            result["trend"] = "positive" if z[0] > 0 else "negative"
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def compare_materials(
        self,
        materials: List[Dict[str, Any]],
        target_property: str = "PCE",
    ) -> Dict[str, Any]:
        """
        Compare SHAP analysis across multiple materials.
        
        Args:
            materials: List of material dicts with keys:
                - name: Material name (required)
                - predicted_value: Predicted property value (required)
                - contributions: List of {feature, contribution} dicts (required)
            target_property: Property to compare
            
        Returns:
            Dict with comparative analysis
        """
        result = {
            "success": False,
            "target_property": target_property,
            "materials": [],
            "ranking": [],
            "key_differences": [],
            "recommendation": "",
            "error": None,
        }
        
        try:
            if not materials:
                result["error"] = "materials list is required"
                return result
            
            comparisons = []
            
            for mat in materials:
                name = mat.get("name", "Unknown")
                predicted = mat.get("predicted_value")
                contributions = mat.get("contributions", [])
                
                if predicted is None:
                    continue
                
                # Process contributions
                top_positive = [c for c in contributions if c.get("contribution", 0) > 0][:3]
                top_negative = [c for c in contributions if c.get("contribution", 0) < 0][:3]
                
                comparisons.append({
                    "name": name,
                    "predicted_value": predicted,
                    "top_positive": top_positive,
                    "top_negative": top_negative,
                })
            
            result["materials"] = comparisons
            
            # Rank by predicted value
            ranking = sorted(comparisons, key=lambda x: x["predicted_value"], reverse=True)
            result["ranking"] = [
                {"rank": i+1, "name": m["name"], "predicted": m["predicted_value"]}
                for i, m in enumerate(ranking)
            ]
            
            # Identify key differences
            if len(comparisons) >= 2:
                best = ranking[0]
                worst = ranking[-1]
                result["key_differences"] = [
                    f"Best material ({best['name']}) has higher {target_property} "
                    f"({best['predicted_value']:.1f}%) vs worst ({worst['name']}, {worst['predicted_value']:.1f}%)",
                ]
                if best['top_positive']:
                    pos_features = [c.get('feature', 'unknown') for c in best['top_positive'][:2]]
                    result["key_differences"].append(
                        f"Key positive factors for {best['name']}: {pos_features}"
                    )
            
            if ranking:
                result["recommendation"] = (
                    f"Based on SHAP analysis, {ranking[0]['name']} shows the highest predicted {target_property}."
                )
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
        
        return result


# ============================================================
# Convenience Functions
# ============================================================

def get_shap_feature_importance(
    feature_importance: Dict[str, float],
    material_composition: str = None,
    target_property: str = "PCE",
    n_top_features: int = 10,
    output_dir: str = None,
) -> Dict[str, Any]:
    """Convenience function for feature importance analysis."""
    analyzer = SHAPAnalyzer(output_dir=output_dir)
    return analyzer.get_feature_importance(
        feature_importance=feature_importance,
        material_composition=material_composition,
        target_property=target_property,
        n_top_features=n_top_features,
    )


def generate_shap_summary_plot(
    feature_importance: Dict[str, float],
    shap_values: Optional[List[List[float]]] = None,
    feature_values: Optional[List[List[float]]] = None,
    feature_names: Optional[List[str]] = None,
    target_property: str = "PCE",
    plot_type: str = "bar",
    max_features: int = 15,
    output_dir: str = None,
) -> Dict[str, Any]:
    """Convenience function for summary plot generation."""
    analyzer = SHAPAnalyzer(output_dir=output_dir)
    return analyzer.generate_summary_plot(
        feature_importance=feature_importance,
        shap_values=shap_values,
        feature_values=feature_values,
        feature_names=feature_names,
        target_property=target_property,
        plot_type=plot_type,
        max_features=max_features,
    )


def analyze_material_shap(
    contributions: List[Dict[str, Any]],
    base_value: float,
    predicted_value: float,
    target_property: str = "PCE",
    output_dir: str = None,
) -> Dict[str, Any]:
    """Convenience function for single material SHAP analysis."""
    analyzer = SHAPAnalyzer(output_dir=output_dir)
    return analyzer.analyze_single_prediction(
        contributions=contributions,
        base_value=base_value,
        predicted_value=predicted_value,
        target_property=target_property,
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("Testing SHAP Analysis Tool...")
    
    analyzer = SHAPAnalyzer()
    
    # Test feature importance
    print("\n1. Feature Importance:")
    test_importance = {
        "tolerance_factor": 0.25,
        "bandgap_eV": 0.20,
        "FA_fraction": 0.15,
        "Pb_fraction": 0.12,
        "defect_density": 0.10,
        "film_thickness_nm": 0.08,
        "annealing_temp_C": 0.05,
        "I_fraction": 0.03,
        "Cs_fraction": 0.02,
    }
    result = analyzer.get_feature_importance(
        feature_importance=test_importance,
        target_property="PCE"
    )
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Top 5 features: {result['top_features'][:5]}")
    else:
        print(f"   Error: {result['error']}")
    
    # Test summary plot
    print("\n2. Summary Plot (bar):")
    result = analyzer.generate_summary_plot(
        feature_importance=test_importance,
        plot_type="bar"
    )
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Filepath: {result['filepath']}")
    else:
        print(f"   Error: {result['error']}")
    
    # Test single prediction analysis
    print("\n3. Single Prediction Analysis:")
    test_contributions = [
        {"feature": "tolerance_factor", "value": 0.95, "contribution": 1.2},
        {"feature": "bandgap_eV", "value": 1.55, "contribution": 0.8},
        {"feature": "FA_fraction", "value": 0.8, "contribution": 0.5},
        {"feature": "defect_density", "value": 1e15, "contribution": -1.0},
        {"feature": "humidity_percent", "value": 60, "contribution": -0.5},
    ]
    result = analyzer.analyze_single_prediction(
        contributions=test_contributions,
        base_value=15.0,
        predicted_value=18.5,
    )
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Interpretation: {result['interpretation']}")
    else:
        print(f"   Error: {result['error']}")
    
    # Test dependence plot
    print("\n4. Dependence Plot:")
    test_feature_values = [0.85, 0.88, 0.90, 0.92, 0.95, 0.97, 1.0, 1.02, 1.05]
    test_shap_values = [-0.5, -0.3, -0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5]
    result = analyzer.generate_dependence_plot(
        feature_name="tolerance_factor",
        feature_values=test_feature_values,
        shap_values=test_shap_values,
    )
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Correlation: {result['correlation']}")
        print(f"   Filepath: {result['filepath']}")
    else:
        print(f"   Error: {result['error']}")
    
    # Test compare materials
    print("\n5. Compare Materials:")
    test_materials = [
        {
            "name": "CsPbI3",
            "predicted_value": 19.5,
            "contributions": [
                {"feature": "tolerance_factor", "contribution": 1.0},
                {"feature": "Cs_fraction", "contribution": 0.5},
            ]
        },
        {
            "name": "MAPbI3",
            "predicted_value": 17.2,
            "contributions": [
                {"feature": "MA_fraction", "contribution": 0.3},
                {"feature": "humidity_stability", "contribution": -0.8},
            ]
        },
    ]
    result = analyzer.compare_materials(materials=test_materials)
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Ranking: {result['ranking']}")
        print(f"   Recommendation: {result['recommendation']}")
    else:
        print(f"   Error: {result['error']}")
    
    print("\n✅ All tests completed!")
