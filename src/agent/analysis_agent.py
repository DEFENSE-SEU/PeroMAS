"""
AnalysisAgent - Lead Analyst for PSC_Agents.

Responsible for the "Check" phase of the PDCA loop.
It performs Gap Analysis (Target vs. Actual) and Root Cause Diagnosis 
to guide the next research iteration.

Tools:
- visualize_structure: Visualize crystal structure from CIF
- analyze_mechanism: Analyze degradation/performance mechanism (Mock)

Author: PSC_Agents Team
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, ClassVar
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.base_agent import BaseAgent
from core.config import Settings


# === Type-safe Helper Functions ===
def safe_str(value: Any, default: str = "") -> str:
    """Safely convert any value to string, handling None, list, dict."""
    if value is None:
        return default
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else default
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False) if value else default
    return str(value)


def safe_truncate(value: Any, max_len: int, suffix: str = "...", default: str = "N/A") -> str:
    """Safely truncate any value to max_len characters."""
    str_value = safe_str(value, default)
    if len(str_value) > max_len:
        return str_value[:max_len] + suffix
    return str_value


# Import visualization tools
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mcp" / "analysis_agent"))
    from visualization_plotly import PlotlyCrystalVisualizer
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

# Import SHAP analysis tools
try:
    from shap_analysis import SHAPAnalyzer
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Import chemistry analysis tools from mcp/analysis_agent
try:
    from chemistry_tools import (
        analyze_stoichiometry,
        analyze_organic_cation,
        get_cation_info,
        compare_cations,
        calculate_correlation,
        calculate_feature_statistics,
        COMMON_CATIONS,
        HAS_PYMATGEN,
        HAS_RDKIT,
        HAS_PANDAS,
    )
    HAS_CHEMISTRY_TOOLS = True
except ImportError:
    HAS_CHEMISTRY_TOOLS = False
    HAS_PYMATGEN = False
    HAS_RDKIT = False
    HAS_PANDAS = False


# ============================================================
# Analysis Agent Specific Tools (dict format)
# ============================================================

ANALYSIS_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "visualize_structure",
            "description": """Visualize a crystal structure from CIF file content.
Creates an interactive 3D HTML visualization.

Args:
  cif_content: CIF file content as string
  name: Name for the output file (optional)
  supercell: Supercell dimensions like "2,2,2" (optional)
  theme: "light" or "dark" (optional)

Returns: JSON with filepath to the generated HTML file.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "cif_content": {
                        "type": "string",
                        "description": "CIF file content as string"
                    },
                    "name": {
                        "type": "string",
                        "description": "Name for the output file"
                    },
                    "supercell": {
                        "type": "string",
                        "description": "Supercell dimensions like '2,2,2'"
                    },
                    "theme": {
                        "type": "string",
                        "description": "'light' or 'dark'"
                    }
                },
                "required": ["cif_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_mechanism",
            "description": """Analyze the mechanism behind perovskite solar cell performance or degradation.

This is a scientific analysis tool that provides insights into:
- Degradation pathways (moisture, heat, light, ion migration)
- Performance limiting factors (recombination, bandgap, defects)
- Structure-property relationships

Args:
  analysis_type: "degradation", "performance", or "structure_property"
  material_info: Material composition (e.g., "CsPbI3", "MAPbI3")
  conditions: Experimental conditions or observations (optional)
  metrics: Performance metrics as JSON string (optional)

Returns: JSON with mechanism analysis results and recommendations.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string",
                        "description": "'degradation', 'performance', or 'structure_property'"
                    },
                    "material_info": {
                        "type": "string",
                        "description": "Material composition (e.g., 'CsPbI3')"
                    },
                    "conditions": {
                        "type": "string",
                        "description": "Experimental conditions"
                    },
                    "metrics": {
                        "type": "string",
                        "description": "Performance metrics as JSON string"
                    }
                },
                "required": ["analysis_type", "material_info"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "shap_feature_importance",
            "description": """Analyze SHAP-based feature importance for material property prediction.

Provides interpretable ML analysis showing which material features
contribute most to predicted properties (PCE, Voc, Jsc, FF, T80).

Args:
  feature_importance: Dict of feature name -> importance value (required)
    Example: {"tolerance_factor": 0.25, "bandgap_eV": 0.20, "FA_fraction": 0.15}
  material_composition: Material formula (e.g., "Cs0.05FA0.95PbI3") (optional)
  target_property: Property to analyze - "PCE", "Voc", "Jsc", "FF", "T80" (default: "PCE")
  n_top_features: Number of top features to return (default: 10)

Returns: JSON with feature importance ranking, top features, and analysis summary.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "feature_importance": {
                        "type": "string",
                        "description": "JSON string of feature importance dict, e.g., '{\"tolerance_factor\": 0.25, \"bandgap_eV\": 0.20}'"
                    },
                    "material_composition": {
                        "type": "string",
                        "description": "Material formula (e.g., 'Cs0.05FA0.95PbI3')"
                    },
                    "target_property": {
                        "type": "string",
                        "description": "Property to analyze: PCE, Voc, Jsc, FF, T80"
                    },
                    "n_top_features": {
                        "type": "integer",
                        "description": "Number of top features to return"
                    }
                },
                "required": ["feature_importance"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "shap_summary_plot",
            "description": """Generate SHAP summary plot visualization.

Creates either a bar plot (feature importance) or beeswarm plot (value distribution)
showing how features impact predictions.

Args:
  feature_importance: Dict of feature name -> importance value (required for bar plot)
    Example: {"tolerance_factor": 0.25, "bandgap_eV": 0.20}
  target_property: Property to analyze (default: "PCE")
  plot_type: "bar" for importance bar chart, "beeswarm" for beeswarm plot (default: "bar")
  max_features: Maximum features to display (default: 15)

Returns: JSON with filepath to saved PNG plot and metadata.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "feature_importance": {
                        "type": "string",
                        "description": "JSON string of feature importance dict"
                    },
                    "target_property": {
                        "type": "string",
                        "description": "Property to analyze: PCE, Voc, Jsc, FF, T80"
                    },
                    "plot_type": {
                        "type": "string",
                        "description": "'bar' or 'beeswarm'"
                    },
                    "max_features": {
                        "type": "integer",
                        "description": "Maximum features to display"
                    }
                },
                "required": ["feature_importance"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "shap_analyze_prediction",
            "description": """Analyze SHAP contributions for a single material prediction.

Provides a waterfall-style breakdown showing how each feature contributes
to the final prediction, starting from the base value.

Args:
  contributions: List of feature contribution dicts (required)
    Example: [{"feature": "bandgap_eV", "value": 1.55, "contribution": 0.8}, ...]
  base_value: The base/expected value (average prediction) (required)
  predicted_value: The predicted property value (required)
  target_property: Property being predicted (default: "PCE")

Returns: JSON with base value, contributions breakdown, top positive/negative factors, and interpretation.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "contributions": {
                        "type": "string",
                        "description": "JSON array of contribution dicts, e.g., '[{\"feature\": \"bandgap_eV\", \"value\": 1.55, \"contribution\": 0.8}]'"
                    },
                    "base_value": {
                        "type": "number",
                        "description": "The base/expected value (average prediction)"
                    },
                    "predicted_value": {
                        "type": "number",
                        "description": "The predicted property value"
                    },
                    "target_property": {
                        "type": "string",
                        "description": "Property being predicted: PCE, Voc, Jsc, FF, T80"
                    }
                },
                "required": ["contributions", "base_value", "predicted_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "shap_dependence_plot",
            "description": """Generate SHAP dependence plot for a specific feature.

Shows how a feature's value affects predictions, with optional
coloring by an interaction feature to reveal feature interactions.

Args:
  feature_name: Feature to analyze (e.g., "tolerance_factor", "bandgap_eV") (required)
  feature_values: List of feature values (required)
  shap_values: List of SHAP values for this feature (required)
  interaction_values: List of values for coloring (optional)
  interaction_feature: Feature name for coloring (optional)
  target_property: Property being predicted (default: "PCE")

Returns: JSON with filepath to saved PNG plot, correlation coefficient, and trend direction.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "feature_name": {
                        "type": "string",
                        "description": "Feature to analyze (e.g., 'tolerance_factor')"
                    },
                    "feature_values": {
                        "type": "string",
                        "description": "JSON array of feature values, e.g., '[0.85, 0.90, 0.95, 1.0]'"
                    },
                    "shap_values": {
                        "type": "string",
                        "description": "JSON array of SHAP values, e.g., '[-0.5, 0.0, 0.5, 1.0]'"
                    },
                    "interaction_values": {
                        "type": "string",
                        "description": "JSON array of interaction values for coloring (optional)"
                    },
                    "interaction_feature": {
                        "type": "string",
                        "description": "Feature name for coloring (optional)"
                    },
                    "target_property": {
                        "type": "string",
                        "description": "Property being predicted: PCE, Voc, Jsc, FF, T80"
                    }
                },
                "required": ["feature_name", "feature_values", "shap_values"]
            }
        }
    },
    # ============================================================
    # New: physics/chemistry/data analysis tools
    # ============================================================
    {
        "type": "function",
        "function": {
            "name": "analyze_stoichiometry",
            "description": """Analyze chemical formula stoichiometry using pymatgen.

Performs comprehensive chemical analysis including:
- Molecular weight calculation
- Atomic fractions of each element
- Oxidation state guessing
- Charge balance validation

This is essential for validating perovskite compositions before simulation.

Args:
  formula: Chemical formula string (e.g., "CsPbI3", "FA0.8Cs0.2PbI2.4Br0.6")

Returns: JSON with molecular weight, atomic fractions, oxidation states, and charge balance status.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "formula": {
                        "type": "string",
                        "description": "Chemical formula (e.g., 'CsPbI3', 'MAPbI3')"
                    }
                },
                "required": ["formula"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_organic_cation",
            "description": """Analyze organic cation properties using RDKit.

Calculates molecular descriptors critical for humidity stability assessment:
- LogP (hydrophobicity): Higher = more moisture resistant
- TPSA (Topological Polar Surface Area): Lower = less polar
- Molecular weight

Common perovskite organic cations:
- MA (methylammonium): SMILES = "C[NH3+]" or "CN"
- FA (formamidinium): SMILES = "[NH2+]=CN" or "C(=N)N"
- PEA (phenethylammonium): SMILES = "NCCc1ccccc1"
- BA (butylammonium): SMILES = "CCCCN"

Args:
  smiles: SMILES string of the organic cation
  name: Optional name for the cation

Returns: JSON with LogP, TPSA, molecular weight, and hydrophobicity assessment.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "SMILES string (e.g., 'CN' for MA, 'NCCc1ccccc1' for PEA)"
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional name of the cation (e.g., 'MA', 'FA', 'PEA')"
                    }
                },
                "required": ["smiles"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_correlation",
            "description": """Calculate Pearson correlation coefficients between features.

Performs statistical analysis on experimental data to find:
- Correlation matrix between all numeric features
- Top factors most correlated with target property (e.g., PCE)

Essential for identifying which process parameters most strongly affect device performance.

Args:
  data_json: JSON string of experimental data (list of dicts with numeric values)
  target_column: Column to analyze correlations against (default: "PCE")

Returns: JSON with correlation matrix and top correlated features.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_json": {
                        "type": "string",
                        "description": "JSON string of data, e.g., '[{\"PCE\": 18.5, \"Voc\": 1.1}, ...]'"
                    },
                    "target_column": {
                        "type": "string",
                        "description": "Target column for correlation analysis (default: 'PCE')"
                    }
                },
                "required": ["data_json"]
            }
        }
    }
]


# ============================================================
# Tool Implementation Functions
# ============================================================

def _execute_visualize_structure(**kwargs) -> str:
    """Execute structure visualization."""
    if not HAS_VIZ:
        return json.dumps({"error": "Visualization not available. Install plotly."})
    
    cif_content = kwargs.get("cif_content", "")
    name = kwargs.get("name", "structure")
    supercell_str = kwargs.get("supercell", None)
    theme = kwargs.get("theme", "light")
    
    # Parse supercell
    supercell = None
    if supercell_str:
        try:
            supercell = tuple(map(int, supercell_str.split(",")))
        except:
            pass
    
    try:
        output_dir = Path("analysis_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        viz = PlotlyCrystalVisualizer(theme=theme)
        fig = viz.visualize(cif_content, supercell=supercell, title=name)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"{name}_{ts}.html"
        viz.save_html(fig, filepath)
        
        return json.dumps({
            "success": True,
            "filepath": str(filepath),
            "message": f"Visualization saved to {filepath}"
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def _execute_analyze_mechanism(**kwargs) -> str:
    """Execute mechanism analysis (Mock)."""
    analysis_type = kwargs.get("analysis_type", "performance")
    material_info = kwargs.get("material_info", "")
    conditions = kwargs.get("conditions", "")
    metrics_str = kwargs.get("metrics", "{}")
    
    try:
        metrics = json.loads(metrics_str) if metrics_str else {}
    except:
        metrics = {}
    
    # Mock analysis based on type
    if analysis_type == "degradation":
        result = _mock_degradation_analysis(material_info, conditions, metrics)
    elif analysis_type == "structure_property":
        result = _mock_structure_property_analysis(material_info, conditions)
    else:
        result = _mock_performance_analysis(material_info, metrics)
    
    return json.dumps(result, indent=2)


def _mock_degradation_analysis(material: str, conditions: str, metrics: dict) -> dict:
    """Mock degradation mechanism analysis."""
    pathways = []
    recommendations = []
    
    if "MA" in material or "FA" in material:
        pathways.append({
            "pathway": "Organic cation volatilization",
            "trigger": "Thermal stress (>85°C)",
            "mechanism": "MA+/FA+ ions escape from the lattice, leaving behind PbI2",
            "severity": "High"
        })
        recommendations.append("Consider mixed-cation approach (Cs/FA) to improve thermal stability")
    
    if "Pb" in material:
        pathways.append({
            "pathway": "Ion migration",
            "trigger": "Electric field / Light soaking",
            "mechanism": "I- and Pb2+ ions migrate under bias, causing hysteresis",
            "severity": "Medium"
        })
        recommendations.append("Interface passivation with 2D perovskite layer")
    
    if "I" in material:
        pathways.append({
            "pathway": "Moisture-induced degradation",
            "trigger": "Humidity > 50% RH",
            "mechanism": "Water molecules penetrate grain boundaries",
            "severity": "High"
        })
        recommendations.append("Encapsulation required; consider hydrophobic HTL")
    
    if not pathways:
        pathways.append({"pathway": "General oxidation", "trigger": "Oxygen + Light", "severity": "Medium"})
    
    return {
        "analysis_type": "degradation",
        "material": material,
        "identified_pathways": pathways,
        "dominant_mechanism": pathways[0]["pathway"],
        "recommendations": recommendations,
        "note": "[Mock Analysis] Real analysis requires experimental data"
    }


def _mock_performance_analysis(material: str, metrics: dict) -> dict:
    """Mock performance mechanism analysis."""
    pce = metrics.get("pce", metrics.get("PCE", 0))
    voc = metrics.get("voc", metrics.get("Voc", 0))
    jsc = metrics.get("jsc", metrics.get("Jsc", 0))
    ff = metrics.get("ff", metrics.get("FF", 0))
    
    limiting_factors = []
    
    if voc and voc < 1.0:
        limiting_factors.append({
            "parameter": "Voc",
            "issue": "Low open-circuit voltage",
            "cause": "Non-radiative recombination",
            "solution": "Interface passivation"
        })
    
    if jsc and jsc < 20:
        limiting_factors.append({
            "parameter": "Jsc",
            "issue": "Low short-circuit current",
            "cause": "Poor light absorption",
            "solution": "Optimize film thickness"
        })
    
    if ff and ff < 0.75:
        limiting_factors.append({
            "parameter": "FF",
            "issue": "Low fill factor",
            "cause": "Series resistance",
            "solution": "Improve contact quality"
        })
    
    return {
        "analysis_type": "performance",
        "material": material,
        "metrics_analyzed": metrics,
        "limiting_factors": limiting_factors,
        "rate_limiting_step": limiting_factors[0]["parameter"] if limiting_factors else "Unknown",
        "note": "[Mock Analysis]"
    }


def _mock_structure_property_analysis(material: str, conditions: str) -> dict:
    """Mock structure-property relationship analysis."""
    relationships = []
    
    if "Cs" in material:
        relationships.append({
            "feature": "Cs+ incorporation",
            "effect": "Improved phase stability, wider bandgap",
            "mechanism": "Cs+ has ideal tolerance factor"
        })
    
    if "Br" in material:
        relationships.append({
            "feature": "Br- substitution",
            "effect": "Wider bandgap, blue-shifted absorption",
            "mechanism": "Br higher electronegativity than I"
        })
    
    if "Sn" in material:
        relationships.append({
            "feature": "Sn2+ B-site",
            "effect": "Lower bandgap (~1.3 eV), oxidation-prone",
            "mechanism": "Sn 5s2 lone pair raises VBM"
        })
    
    if not relationships:
        relationships.append({
            "feature": "ABX3 structure",
            "effect": "Direct bandgap, high absorption",
            "mechanism": "Corner-sharing octahedra"
        })
    
    return {
        "analysis_type": "structure_property",
        "material": material,
        "relationships": relationships,
        "note": "[Mock Analysis]"
    }


# ============================================================
# SHAP Tool Implementation Functions
# ============================================================

def _execute_shap_feature_importance(**kwargs) -> str:
    """Execute SHAP feature importance analysis."""
    if not HAS_SHAP:
        return json.dumps({"error": "SHAP analysis not available. Install with: pip install shap matplotlib"})
    
    try:
        # Parse feature_importance from JSON string
        importance_str = kwargs.get("feature_importance", "{}")
        if isinstance(importance_str, str):
            feature_importance = json.loads(importance_str)
        else:
            feature_importance = importance_str
        
        if not feature_importance:
            return json.dumps({"error": "feature_importance parameter is required"})
        
        analyzer = SHAPAnalyzer(output_dir="analysis_output/shap")
        result = analyzer.get_feature_importance(
            feature_importance=feature_importance,
            material_composition=kwargs.get("material_composition"),
            target_property=kwargs.get("target_property", "PCE"),
            n_top_features=kwargs.get("n_top_features", 10),
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def _execute_shap_summary_plot(**kwargs) -> str:
    """Execute SHAP summary plot generation."""
    if not HAS_SHAP:
        return json.dumps({"error": "SHAP analysis not available. Install with: pip install shap matplotlib"})
    
    try:
        # Parse feature_importance from JSON string
        importance_str = kwargs.get("feature_importance", "{}")
        if isinstance(importance_str, str):
            feature_importance = json.loads(importance_str)
        else:
            feature_importance = importance_str
        
        if not feature_importance:
            return json.dumps({"error": "feature_importance parameter is required"})
        
        analyzer = SHAPAnalyzer(output_dir="analysis_output/shap")
        result = analyzer.generate_summary_plot(
            feature_importance=feature_importance,
            target_property=kwargs.get("target_property", "PCE"),
            plot_type=kwargs.get("plot_type", "bar"),
            max_features=kwargs.get("max_features", 15),
            save=True,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def _execute_shap_analyze_prediction(**kwargs) -> str:
    """Execute SHAP analysis for single prediction."""
    if not HAS_SHAP:
        return json.dumps({"error": "SHAP analysis not available. Install with: pip install shap matplotlib"})
    
    try:
        # Parse contributions from JSON string
        contributions_str = kwargs.get("contributions", "[]")
        if isinstance(contributions_str, str):
            contributions = json.loads(contributions_str)
        else:
            contributions = contributions_str
        
        if not contributions:
            return json.dumps({"error": "contributions parameter is required"})
        
        base_value = kwargs.get("base_value")
        predicted_value = kwargs.get("predicted_value")
        
        if base_value is None or predicted_value is None:
            return json.dumps({"error": "base_value and predicted_value are required"})
        
        analyzer = SHAPAnalyzer(output_dir="analysis_output/shap")
        result = analyzer.analyze_single_prediction(
            contributions=contributions,
            base_value=base_value,
            predicted_value=predicted_value,
            target_property=kwargs.get("target_property", "PCE"),
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def _execute_shap_dependence_plot(**kwargs) -> str:
    """Execute SHAP dependence plot generation."""
    if not HAS_SHAP:
        return json.dumps({"error": "SHAP analysis not available. Install with: pip install shap matplotlib"})
    
    try:
        feature_name = kwargs.get("feature_name")
        if not feature_name:
            return json.dumps({"error": "feature_name parameter is required"})
        
        # Parse feature_values from JSON string
        fv_str = kwargs.get("feature_values", "[]")
        if isinstance(fv_str, str):
            feature_values = json.loads(fv_str)
        else:
            feature_values = fv_str
        
        # Parse shap_values from JSON string
        sv_str = kwargs.get("shap_values", "[]")
        if isinstance(sv_str, str):
            shap_values = json.loads(sv_str)
        else:
            shap_values = sv_str
        
        if not feature_values or not shap_values:
            return json.dumps({"error": "feature_values and shap_values are required"})
        
        # Parse optional interaction_values
        interaction_values = None
        iv_str = kwargs.get("interaction_values")
        if iv_str:
            if isinstance(iv_str, str):
                interaction_values = json.loads(iv_str)
            else:
                interaction_values = iv_str
        
        analyzer = SHAPAnalyzer(output_dir="analysis_output/shap")
        result = analyzer.generate_dependence_plot(
            feature_name=feature_name,
            feature_values=feature_values,
            shap_values=shap_values,
            interaction_values=interaction_values,
            interaction_feature=kwargs.get("interaction_feature"),
            target_property=kwargs.get("target_property", "PCE"),
            save=True,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# ============================================================
# New Chemistry Tools Execution Functions
# ============================================================

def _execute_analyze_stoichiometry(**kwargs) -> str:
    """Execute stoichiometry analysis using pymatgen."""
    if not HAS_CHEMISTRY_TOOLS or not HAS_PYMATGEN:
        return json.dumps({"error": "pymatgen not available. Install with: pip install pymatgen"})
    
    try:
        formula = kwargs.get("formula", "")
        if not formula:
            return json.dumps({"error": "formula parameter is required"})
        
        result = analyze_stoichiometry(formula)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def _execute_analyze_organic_cation(**kwargs) -> str:
    """Execute organic cation analysis using RDKit."""
    if not HAS_CHEMISTRY_TOOLS or not HAS_RDKIT:
        return json.dumps({"error": "rdkit not available. Install with: pip install rdkit"})
    
    try:
        smiles = kwargs.get("smiles", "")
        name = kwargs.get("name", None)
        
        if not smiles:
            # Try to resolve from name.
            if name and name.upper() in COMMON_CATIONS:
                result = get_cation_info(name)
            else:
                return json.dumps({
                    "error": "smiles parameter is required, or provide a known cation name",
                    "available_cations": list(COMMON_CATIONS.keys()) if HAS_CHEMISTRY_TOOLS else []
                })
        else:
            result = analyze_organic_cation(smiles, name)
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def _execute_calculate_correlation(**kwargs) -> str:
    """Execute correlation analysis using pandas."""
    if not HAS_CHEMISTRY_TOOLS or not HAS_PANDAS:
        return json.dumps({"error": "pandas not available. Install with: pip install pandas numpy"})
    
    try:
        data_json = kwargs.get("data_json", "")
        target_column = kwargs.get("target_column", "PCE")
        
        if not data_json:
            return json.dumps({"error": "data_json parameter is required"})
        
        result = calculate_correlation(data_json, target_column)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# === System Prompt ===
SYSTEM_PROMPT = """You are AnalysisAgent - Lead Strategic Analyst of PSC_Agents.

## Core Mission
Analyze perovskite materials and experimental results. Determine not just WHAT the results are, but WHY.

## Your Specialized Toolbox

### Chemistry Analysis
- `analyze_stoichiometry(formula)`: Validate formula, calculate molecular weight, check charge balance, list atomic fractions.
  - Use when: analyzing chemical formulas (e.g., CsPbI3, Cs2AgBiBr6)
  - Input: formula string
  
- `analyze_organic_cation(smiles, name?)`: Analyze organic cation properties - LogP, TPSA, molecular weight.
  - Use when: evaluating hydrophobicity, moisture resistance of organic molecules
  - Common SMILES: MA="CN", FA="[NH2+]=CN", PEA="NCCc1ccccc1", BA="CCCCN", OA="CCCCCCCCN"
  - Note: Higher LogP = more hydrophobic = better moisture resistance

### Mechanism Diagnosis
- `analyze_mechanism(analysis_type, material_info, conditions?, metrics?)`: Diagnose degradation or performance mechanisms.
  - analysis_type: "degradation", "performance", or "structure_property"
  - Use when: explaining why material behaves certain way (e.g., thermal degradation, voltage loss)

### Statistical Analysis
- `calculate_correlation(data_json, target_column?)`: Calculate Pearson correlation coefficients.
  - Use when: user provides experimental data and asks about correlations
  - Input: JSON array of data points, e.g., [{"Temp": 100, "PCE": 18.5}, ...]

### SHAP Interpretation
- `shap_feature_importance(feature_importance, material_composition?, target_property?)`: Rank feature importance.
  - Use when: user provides feature importance dict from ML model
  
- `shap_summary_plot(feature_importance)`: Generate SHAP bar chart visualization.
  - Use when: user wants a visual summary of feature importance
  
- `shap_analyze_prediction(contributions, base_value, predicted_value)`: Explain single prediction.
  - Use when: user provides contribution data for specific samples

### Visualization
- `visualize_structure(cif_content, name?, supercell?, theme?)`: Render crystal structure as 3D HTML.
  - CONSTRAINT: Requires actual CIF file content - skip if not available

## Tool Selection Principles
1. **Match tool to task**: Choose tools based on what the user is asking
   - Formula questions → analyze_stoichiometry
   - Organic molecule properties → analyze_organic_cation  
   - Why/mechanism questions → analyze_mechanism
   - Data correlation → calculate_correlation
   - Feature importance → shap_feature_importance
2. **Tool-first**: Call relevant tools before writing conclusions
3. **Data dependency**: Only use tools if you have the required input data
4. **Multi-tool OK**: Complex questions may need multiple tools in sequence

## Output Principles
- Base conclusions on actual tool outputs
- Include relevant numbers and units from tool results
- Give clear, actionable insights when applicable
- Adapt output format to the task (no fixed JSON structure required)
"""


# Tool execution dispatcher
ANALYSIS_TOOL_EXECUTORS = {
    # Structure & Mechanism
    "visualize_structure": _execute_visualize_structure,
    "analyze_mechanism": _execute_analyze_mechanism,
    # SHAP Analysis
    "shap_feature_importance": _execute_shap_feature_importance,
    "shap_summary_plot": _execute_shap_summary_plot,
    "shap_analyze_prediction": _execute_shap_analyze_prediction,
    "shap_dependence_plot": _execute_shap_dependence_plot,
    # Chemistry Tools (NEW)
    "analyze_stoichiometry": _execute_analyze_stoichiometry,
    "analyze_organic_cation": _execute_analyze_organic_cation,
    "calculate_correlation": _execute_calculate_correlation,
}


class AnalysisAgent(BaseAgent):
    """
    Data Analyst agent.
    
    It closes the loop by turning raw data into actionable insights for the MetaAgent.
    
    Exclusive Tools:
    - visualize_structure: Crystal structure visualization
    - analyze_mechanism: Mechanism analysis (degradation/performance)
    - shap_feature_importance: SHAP-based feature importance
    - shap_summary_plot: SHAP summary visualization
    - shap_analyze_prediction: Single prediction SHAP analysis
    - shap_dependence_plot: Feature dependence plot
    - analyze_stoichiometry: Chemical formula validation (pymatgen)
    - analyze_organic_cation: Organic cation analysis (rdkit)
    - calculate_correlation: Feature correlation analysis (pandas)
    """

    # Analysis agent specific tools
    REQUIRED_TOOL_TYPES: ClassVar[list[str]] = []

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(name="AnalysisAgent", settings=settings)
        # Register local tools
        self._local_tools = ANALYSIS_AGENT_TOOLS
        self._tool_executors = ANALYSIS_TOOL_EXECUTORS
    
    def get_tools_schema(self) -> list[dict]:
        """Return tool schemas including analysis-specific tools."""
        # Get base tools from parent
        base_tools = super().get_tools_schema() if hasattr(super(), 'get_tools_schema') else []
        return base_tools + self._local_tools
    
    async def execute_local_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute analysis-specific local tools."""
        if tool_name in self._tool_executors:
            return self._tool_executors[tool_name](**arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    async def autonomous_thinking(
        self,
        prompt: str,
        state: dict[str, Any],
        system_message: str | None = None,
        max_iterations: int = 10,
    ) -> dict[str, Any]:
        """
        Override autonomous_thinking to handle local analysis tools.
        """
        if not self.llm:
            self.logger.error("LLM client not available")
            return {
                "response": "[ERROR] LLM not configured",
                "tool_calls": [],
                "tool_results": [],
                "iterations": 0,
            }

        # Get tools (local + MCP)
        local_tool_names = {t.get('function', {}).get('name', '') for t in self._local_tools}
        mcp_tools = []
        if self.registry.is_initialized():
            mcp_tools = await self.registry.get_tools_schema()
        
        tools = self._local_tools + mcp_tools
        
        self.logger.info(f"Available tools: {len(tools)}")
        tool_names = [t.get('function', {}).get('name', 'unknown') for t in tools]
        self.logger.info(f"Tool names: {tool_names}")

        # Build messages
        messages: list[dict[str, Any]] = []
        
        final_system_prompt = self._get_system_prompt(state, system_message)
        if final_system_prompt:
            messages.append({"role": "system", "content": final_system_prompt})

        context_str = ""
        if state:
            context_str = f"\n\nCurrent context:\n{state}"

        messages.append({"role": "user", "content": prompt + context_str})

        all_tool_calls: list[dict[str, Any]] = []
        all_tool_results: list[dict[str, Any]] = []
        iterations = 0
        response = None
        
        # Track consecutive tool calls for deduplication
        _last_tool_name: str | None = None
        _consecutive_count: int = 0

        # ReAct loop
        while iterations < max_iterations:
            iterations += 1
            self.logger.debug(f"Thinking iteration {iterations}")

            response = await self.llm.ainvoke(messages, tools=tools if tools else None)

            if not self.llm.has_tool_calls(response):
                self.logger.debug("No tool calls, finishing")
                break

            tool_calls = self.llm.get_tool_calls(response)
            messages.append(response)

            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                self.logger.info(f"Executing tool: {tool_name}")
                
                # Tool call visualization with de-duplication.
                tool_type = "📍 Local" if tool_name in local_tool_names else "🌐 MCP"
                if tool_name == _last_tool_name:
                    _consecutive_count += 1
                    print(f"\r   🔄 [AnalysisAgent] {tool_name} called {_consecutive_count}x (consecutive)", end="", flush=True)
                else:
                    if _last_tool_name is not None and _consecutive_count > 1:
                        print()  # End the previous tool's counter line.
                    _consecutive_count = 1
                    _last_tool_name = tool_name
                    print(f"\n🔧 [AnalysisAgent] Calling {tool_type} Tool: {tool_name}")
                    print(f"   📥 Arguments: {str(tool_args)[:200]}{'...' if len(str(tool_args)) > 200 else ''}")
                
                all_tool_calls.append(tc)

                try:
                    # Check if it's a local tool
                    if tool_name in local_tool_names:
                        result_str = await self.execute_local_tool(tool_name, tool_args)
                    else:
                        # MCP tool
                        result = await self.registry.call_tool(tool_name, tool_args)
                        result_str = str(result) if result else "No result"
                    
                    result_str = self._truncate_tool_output(result_str, tool_name)
                except Exception as e:
                    self.logger.error(f"Tool execution failed: {e}")
                    result_str = f"[ERROR] {e}"

                all_tool_results.append({
                    "tool": tool_name,
                    "result": result_str,
                })
                
                # Tool result visualization (details only for first call).
                if _consecutive_count == 1:
                    result_preview = result_str[:150] if len(result_str) > 150 else result_str
                    print(f"   📤 Result: {result_preview}{'...' if len(result_str) > 150 else ''}")

                tool_message = self.llm.create_tool_message(tool_id, result_str)
                messages.append(tool_message)

        final_response = ""
        if response and hasattr(response, "content"):
            final_response = response.content or ""

        return {
            "response": final_response,
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "iterations": iterations,
        }

    def _get_system_prompt(
        self,
        state: dict[str, Any],
        default_prompt: str | None = None,
    ) -> str:
        return SYSTEM_PROMPT

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute analysis based on MetaAgent's task.
        """
        print(f"\n{'='*60}")
        print(f"📊 [AnalysisAgent] Gap Analysis")
        print(f"{'='*60}")
        
        # Show available tools with categorization
        local_tool_names = [t.get('function', {}).get('name', 'unknown') for t in self._local_tools]
        mcp_tool_names = []
        if self.registry.is_initialized():
            mcp_tools = await self.registry.get_tools_schema()
            mcp_tool_names = [t.get('function', {}).get('name', 'unknown') for t in mcp_tools]
        
        print(f"\n🛠️  Available Tools Summary:")
        print(f"   📍 Local Tools ({len(local_tool_names)}): {local_tool_names}")
        print(f"   🌐 MCP Tools ({len(mcp_tool_names)}): {mcp_tool_names if mcp_tool_names else 'None'}")
        print(f"   📊 Total: {len(local_tool_names) + len(mcp_tool_names)} tools")
        
        # Get context from state - include ALL upstream outputs
        goal = state.get("goal", "")
        plan = state.get("plan", "")
        data_context = state.get("data_context", "")  # From DataAgent
        experimental_params = state.get("experimental_params", {})  # From DesignAgent
        fab_results = state.get("fab_results", {})  # From FabAgent
        
        # Extract AnalysisAgent-specific task from MetaAgent's plan
        my_task = self._extract_my_task(plan, "AnalysisAgent")
        
        # === Display upstream context clearly ===
        print(f"\n📊 Upstream Context (Full Pipeline):")
        print(f"   ├─ 🎯 Goal: {(goal or '')[:60]}{'...' if len(goal or '') > 60 else ''}")
        print(f"   ├─ 📝 Task: {my_task}")
        print(f"   ├─ 📚 Data (DataAgent): {len(data_context)} chars")
        
        # DesignAgent output summary
        if experimental_params:
            formula = experimental_params.get("composition", {}).get("formula", "N/A")
            process = experimental_params.get("process", {})
            method = process.get("method", "N/A")
            protocol = process.get("synthesis_protocol", "")
            print(f"   ├─ 🧪 Design (DesignAgent):")
            print(f"   │   ├─ Formula: {formula}")
            print(f"   │   ├─ Method: {method}")
            if protocol:
                print(f"   │   └─ Protocol: {protocol[:150]}{'...' if len(protocol) > 150 else ''}")
            else:
                print(f"   │   └─ Protocol: N/A")
        else:
            print(f"   ├─ 🧪 Design: None")
        
        # FabAgent output summary
        if fab_results and isinstance(fab_results, dict):
            metrics = fab_results.get("predicted_metrics", {})
            if metrics:
                pce = metrics.get("PCE_percent", "N/A")
                voc = metrics.get("Voc_V", "N/A")
                print(f"   └─ 🏭 Fab (FabAgent):")
                print(f"       ├─ PCE: {pce}%")
                print(f"       └─ Voc: {voc}V")
            else:
                print(f"   └─ 🏭 Fab: No metrics")
        else:
            print(f"   └─ 🏭 Fab: None")

        # Format inputs
        params_str = json.dumps(experimental_params, indent=2) if experimental_params else "N/A"
        if isinstance(fab_results, dict):
            # Try to get predicted_metrics first, then metrics, then whole dict
            metrics = fab_results.get("predicted_metrics") or fab_results.get("metrics") or fab_results
            results_str = json.dumps(metrics, indent=2, ensure_ascii=False)
        else:
            results_str = str(fab_results) if fab_results else "No prediction results"

        prompt = f"""# 📊 Analysis Task
**Task**: {my_task}
**Research Goal**: {goal}

# 📥 Available Context

### Literature (DataAgent):
{safe_truncate(data_context, 2000, default='No literature data.')}

### Design Recipe (DesignAgent):
```json
{params_str}
```

### Prediction Results (FabAgent):
```json
{results_str}
```

# 🎯 Instructions
Analyze the above context and complete the task. Select and use appropriate tools based on the task requirements.

Provide your analysis and conclusions based on tool outputs.
"""

        result = await self.autonomous_thinking(
            prompt=prompt,
            state=state,
            system_message=SYSTEM_PROMPT,
            max_iterations=3,
        )

        response_text = result.get("response", "")
        analysis_summary = self._extract_json_block(response_text)
        
        # Display summary
        print(f"\n{'─'*60}")
        print(f"✅ [AnalysisAgent] Analysis Complete")
        if analysis_summary:
            is_met = analysis_summary.get("is_goal_met", "Unknown")
            print(f"   ├─ Goal Met: {is_met}")
            suggestion = analysis_summary.get("iteration_feedback", {}).get("suggested_adjustment") or \
                        analysis_summary.get("suggested_adjustment", "N/A")
            print(f"   └─ Suggestion: {suggestion[:60]}..." if len(str(suggestion)) > 60 else f"   └─ Suggestion: {suggestion}")
        else:
            print(f"   └─ Status: Analysis complete (no structured summary)")

        return {"analysis_report": response_text}
    
    def _extract_my_task(self, plan: str | dict, agent_name: str) -> str:
        """Extract specific task for this agent from MetaAgent's plan."""
        if not plan:
            return "Analyze results and identify improvements"
        
        # If plan is already a dict, use it directly
        if isinstance(plan, dict):
            plan_data = plan
        else:
            # Try to parse JSON from string plan
            try:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', plan)
                if match:
                    plan_data = json.loads(match.group(1))
                elif '{' in plan:
                    start = plan.find('{')
                    end = plan.rfind('}') + 1
                    plan_data = json.loads(plan[start:end])
                else:
                    return str(plan)[:500]
            except (json.JSONDecodeError, KeyError):
                return str(plan)[:500]
        
        # Extract agent-specific task
        agent_tasks = plan_data.get("agent_tasks", {})
        task = agent_tasks.get(agent_name, "")
        
        if task and str(task).upper() != "SKIP":
            return task
        else:
            return plan_data.get("iteration_focus", "Analyze gaps")

    def _extract_json_block(self, text: str) -> Any | None:
        """Robustly extract JSON from Markdown."""
        try:
            # Regex to match content wrapped in ```json and ```
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                return json.loads(match.group(1))
            
            # Fallback: find first and last curly braces
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start : end + 1])
        except Exception:
            pass
        return None