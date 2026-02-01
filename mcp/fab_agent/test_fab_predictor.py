#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_fab_predictor.py
测试 FabAgent 的钙钛矿预测功能

Usage:
    cd f:\PSC_Agents\mcp\fab_agent
    conda activate psc_agent
    python test_fab_predictor.py
"""

import sys
from pathlib import Path

# 添加当前目录到路径
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

from perovskite_predictor import (
    predict_perovskite_properties,
    PerovskitePredictor,
    ALL_TARGETS,
    TARGET_INFO
)


def test_composition_prediction():
    """测试分子式预测"""
    print("=" * 70)
    print("🧪 测试 FabAgent 钙钛矿预测工具")
    print("=" * 70)
    
    test_cases = [
        "CsPbI3",
        "MAPbI3",
        "FAPbI3",
        "FA0.25MA0.75PbI3",
        "Cs0.05FA0.79MA0.16PbI3",
    ]
    
    for comp in test_cases:
        print(f"\n📌 测试组分: {comp}")
        print("-" * 50)
        
        result = predict_perovskite_properties(composition=comp)
        
        if result.get("status") == "success":
            print(f"   输入模式: {result.get('input_mode')}")
            print(f"   模型类型: {result.get('model_type')}")
            print("\n   预测结果:")
            
            for target, pred in result.get("predictions", {}).items():
                if isinstance(pred, dict) and "value" in pred:
                    name = pred.get("name", target)
                    value = pred["value"]
                    unit = pred.get("unit", "")
                    print(f"      {name:12s}: {value:>8.4f} {unit}")
            
            # 显示格式化后的可视化数据
            viz_data = result.get("formatted_for_viz", {})
            if viz_data:
                print(f"\n   可视化格式: {viz_data}")
        else:
            print(f"   ❌ 错误: {result.get('message', 'Unknown error')}")


def test_predictor_class():
    """测试预测器类"""
    print("\n" + "=" * 70)
    print("🔬 测试 PerovskitePredictor 类")
    print("=" * 70)
    
    predictor = PerovskitePredictor(model_type="RF")
    
    print(f"\n可用目标 (comp_only): {predictor.get_available_targets('comp_only')}")
    print(f"可用目标 (cif_only): {predictor.get_available_targets('cif_only')}")
    
    # 测试单独预测
    print("\n单独预测 PCE:")
    result = predictor.predict(composition="MAPbI3", targets=["pce"])
    if "predictions" in result:
        pce = result["predictions"].get("pce", {})
        if "value" in pce:
            print(f"   MAPbI3 PCE: {pce['value']:.4f} {pce.get('unit', '')}")


def test_fab_agent_integration():
    """测试与 FabAgent 的集成"""
    print("\n" + "=" * 70)
    print("🏭 测试 FabAgent 集成")
    print("=" * 70)
    
    # 模拟 FabAgent 调用
    # 这里直接调用预测函数，实际在 FabAgent 中会通过 tool 调用
    
    # 模拟从 DesignAgent 获取的配方
    recipe = {
        "perovskite_composition": "Cs0.05FA0.79MA0.16Pb(I0.83Br0.17)3",
        "etl": "SnO2",
        "htl": "Spiro-OMeTAD",
        "thickness": "500nm"
    }
    
    composition = recipe.get("perovskite_composition")
    print(f"\n从配方提取的组分: {composition}")
    
    result = predict_perovskite_properties(composition=composition)
    
    if result.get("status") == "success":
        print("\n📊 预测结果:")
        for target, pred in result.get("predictions", {}).items():
            if isinstance(pred, dict) and "value" in pred:
                print(f"   {pred.get('name', target):12s}: {pred['value']:>8.4f} {pred.get('unit', '')}")
        
        # 生成可视化数据
        viz_metrics = result.get("formatted_for_viz", {})
        print(f"\n📈 可传递给 visualize_predictions 的数据:")
        print(f"   {viz_metrics}")
    else:
        print(f"   ❌ 预测失败: {result.get('message')}")


if __name__ == "__main__":
    test_composition_prediction()
    test_predictor_class()
    test_fab_agent_integration()
    
    print("\n" + "=" * 70)
    print("✅ 所有测试完成!")
    print("=" * 70)
