#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_fab_predictor.py
Test FabAgent perovskite prediction features.

Usage:
    cd f:\PSC_Agents\mcp\fab_agent
    conda activate psc_agent
    python test_fab_predictor.py
"""

import sys
from pathlib import Path

current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

from perovskite_predictor import (
    predict_perovskite_properties,
    PerovskitePredictor,
    ALL_TARGETS,
    TARGET_INFO
)


def test_composition_prediction():
    """Test composition-based prediction."""
    print("=" * 70)
    print("🧪 Testing FabAgent perovskite predictor")
    print("=" * 70)
    
    test_cases = [
        "CsPbI3",
        "MAPbI3",
        "FAPbI3",
        "FA0.25MA0.75PbI3",
        "Cs0.05FA0.79MA0.16PbI3",
    ]
    
    for comp in test_cases:
        print(f"\n📌 Test composition: {comp}")
        print("-" * 50)
        
        result = predict_perovskite_properties(composition=comp)
        
        if result.get("status") == "success":
            print(f"   Input mode: {result.get('input_mode')}")
            print(f"   Model type: {result.get('model_type')}")
            print("\n   Predictions:")
            
            for target, pred in result.get("predictions", {}).items():
                if isinstance(pred, dict) and "value" in pred:
                    name = pred.get("name", target)
                    value = pred["value"]
                    unit = pred.get("unit", "")
                    print(f"      {name:12s}: {value:>8.4f} {unit}")
            
            viz_data = result.get("formatted_for_viz", {})
            if viz_data:
                print(f"\n   Visualization payload: {viz_data}")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")


def test_predictor_class():
    """Test predictor class."""
    print("\n" + "=" * 70)
    print("🔬 Testing PerovskitePredictor class")
    print("=" * 70)
    
    predictor = PerovskitePredictor(model_type="RF")
    
    print(f"\nAvailable targets (comp_only): {predictor.get_available_targets('comp_only')}")
    print(f"Available targets (cif_only): {predictor.get_available_targets('cif_only')}")
    
    print("\nSingle PCE prediction:")
    result = predictor.predict(composition="MAPbI3", targets=["pce"])
    if "predictions" in result:
        pce = result["predictions"].get("pce", {})
        if "value" in pce:
            print(f"   MAPbI3 PCE: {pce['value']:.4f} {pce.get('unit', '')}")


def test_fab_agent_integration():
    """Test integration with FabAgent."""
    print("\n" + "=" * 70)
    print("🏭 Testing FabAgent integration")
    print("=" * 70)
    
    recipe = {
        "perovskite_composition": "Cs0.05FA0.79MA0.16Pb(I0.83Br0.17)3",
        "etl": "SnO2",
        "htl": "Spiro-OMeTAD",
        "thickness": "500nm"
    }
    
    composition = recipe.get("perovskite_composition")
    print(f"\nComposition extracted from recipe: {composition}")
    
    result = predict_perovskite_properties(composition=composition)
    
    if result.get("status") == "success":
        print("\n📊 Predictions:")
        for target, pred in result.get("predictions", {}).items():
            if isinstance(pred, dict) and "value" in pred:
                print(f"   {pred.get('name', target):12s}: {pred['value']:>8.4f} {pred.get('unit', '')}")
        
        # Build visualization payload.
        viz_metrics = result.get("formatted_for_viz", {})
        print(f"\n📈 Payload for visualize_predictions:")
        print(f"   {viz_metrics}")
    else:
        print(f"   ❌ Prediction failed: {result.get('message')}")


if __name__ == "__main__":
    test_composition_prediction()
    test_predictor_class()
    test_fab_agent_integration()
    
    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)
