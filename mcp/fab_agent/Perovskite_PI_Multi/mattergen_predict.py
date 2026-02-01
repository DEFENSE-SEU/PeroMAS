#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MatterGen + 性质预测 端到端流程

从 MatterGen 生成的 CIF 文件，直接预测 8 个钙钛矿性质。

使用方法:
    # 预测 MatterGen 输出目录中的所有结构
    python mattergen_predict.py --mattergen-dir /path/to/generation_results/xxx
    
    # 指定模型
    python mattergen_predict.py --mattergen-dir /path/to/output --model RF
    
    # 筛选高性能材料
    python mattergen_predict.py --mattergen-dir /path/to/output --filter "pce>18,dft_band_gap>1.2"

Author: PSC_Agents
Date: 2026-01-21
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# 导入预测 API
from predict_api import PerovskitePredictor, TARGET_PROPERTIES


def filter_results(
    results: List[Dict],
    filter_expr: str,
    model: str = "ensemble"
) -> List[Dict]:
    """筛选符合条件的结果
    
    Args:
        results: 预测结果列表
        filter_expr: 筛选表达式，如 "pce>18,dft_band_gap>1.2"
        model: 模型名称
        
    Returns:
        筛选后的结果
    """
    if not filter_expr:
        return results
    
    # 解析筛选条件
    conditions = []
    for cond in filter_expr.split(","):
        cond = cond.strip()
        for op in [">=", "<=", ">", "<", "=="]:
            if op in cond:
                prop, val = cond.split(op)
                conditions.append((prop.strip(), op, float(val.strip())))
                break
    
    # 筛选
    filtered = []
    for r in results:
        if "error" in r:
            continue
        
        if model == "ensemble":
            preds = r.get("ensemble_predictions", {})
        else:
            preds = r.get("predictions", {})
        
        match = True
        for prop, op, val in conditions:
            pred_val = preds.get(prop)
            if pred_val is None:
                match = False
                break
            if op == ">" and not pred_val > val:
                match = False
            elif op == ">=" and not pred_val >= val:
                match = False
            elif op == "<" and not pred_val < val:
                match = False
            elif op == "<=" and not pred_val <= val:
                match = False
            elif op == "==" and not pred_val == val:
                match = False
        
        if match:
            filtered.append(r)
    
    return filtered


def print_results_table(
    results: List[Dict],
    model: str = "ensemble",
    top_n: int = 20
):
    """打印结果表格
    
    Args:
        results: 预测结果列表
        model: 模型名称
        top_n: 显示前 N 个
    """
    print(f"\n{'='*100}")
    print(f"{'材料化学式':^15} | {'PCE (%)':^10} | {'带隙 (eV)':^10} | {'稳定性':^10} | {'Voc (V)':^10} | {'Jsc':^10} | {'FF':^10}")
    print(f"{'='*100}")
    
    for r in results[:top_n]:
        formula = r.get("formula", "N/A")
        if model == "ensemble":
            preds = r.get("ensemble_predictions", {})
        else:
            preds = r.get("predictions", {})
        
        pce = preds.get("pce", 0)
        bg = preds.get("dft_band_gap", 0)
        ehull = preds.get("energy_above_hull", 0)
        voc = preds.get("voc", 0)
        jsc = preds.get("jsc", 0)
        ff = preds.get("ff", 0)
        
        print(f"{formula:^15} | {pce:^10.2f} | {bg:^10.2f} | {ehull:^10.4f} | {voc:^10.2f} | {jsc:^10.2f} | {ff:^10.2f}")
    
    if len(results) > top_n:
        print(f"... 还有 {len(results) - top_n} 个结果")


def save_results_csv(
    results: List[Dict],
    output_path: str,
    model: str = "ensemble"
):
    """保存结果为 CSV
    
    Args:
        results: 预测结果列表
        output_path: 输出文件路径
        model: 模型名称
    """
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        header = ["formula", "filename", "has_structure"] + TARGET_PROPERTIES
        writer.writerow(header)
        
        # 写入数据
        for r in results:
            if "error" in r:
                continue
            
            row = [
                r.get("formula", ""),
                r.get("filename", ""),
                r.get("has_structure", False),
            ]
            
            if model == "ensemble":
                preds = r.get("ensemble_predictions", {})
            else:
                preds = r.get("predictions", {})
            
            for prop in TARGET_PROPERTIES:
                row.append(preds.get(prop, ""))
            
            writer.writerow(row)
    
    print(f"💾 结果已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MatterGen + 性质预测 端到端流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 预测 MatterGen 输出
    python mattergen_predict.py --mattergen-dir generation_results/multi_gpu_xxx/
    
    # 筛选高 PCE 材料
    python mattergen_predict.py --mattergen-dir output/ --filter "pce>18"
    
    # 按 PCE 排序
    python mattergen_predict.py --mattergen-dir output/ --sort pce
    
    # 保存结果
    python mattergen_predict.py --mattergen-dir output/ --output predictions.csv
        """
    )
    
    parser.add_argument("--mattergen-dir", type=str, required=True,
                        help="MatterGen 输出目录")
    parser.add_argument("--model", type=str, default="ensemble",
                        choices=["RF", "GBDT", "NN", "ensemble"],
                        help="预测模型 (默认: ensemble)")
    parser.add_argument("--filter", type=str, default=None,
                        help="筛选条件，如 'pce>18,dft_band_gap>1.2'")
    parser.add_argument("--sort", type=str, default="pce",
                        help="排序属性 (默认: pce)")
    parser.add_argument("--top", type=int, default=20,
                        help="显示前 N 个结果 (默认: 20)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 CSV 文件路径")
    parser.add_argument("--json", type=str, default=None,
                        help="输出 JSON 文件路径")
    
    args = parser.parse_args()
    
    # 初始化预测器
    print("正在加载预测模型...")
    predictor = PerovskitePredictor()
    
    # 预测
    print(f"\n正在预测 {args.mattergen_dir} 中的结构...")
    prediction_results = predictor.predict_from_mattergen_output(
        args.mattergen_dir, args.model
    )
    
    if "error" in prediction_results:
        print(f"❌ 错误: {prediction_results['error']}")
        return
    
    results = prediction_results["results"]
    print(f"✅ 预测完成: {len(results)} 个结构")
    
    # 筛选
    if args.filter:
        results = filter_results(results, args.filter, args.model)
        print(f"📊 筛选后: {len(results)} 个结构符合条件 ({args.filter})")
    
    # 排序
    if args.sort:
        def get_sort_key(r):
            if args.model == "ensemble":
                return r.get("ensemble_predictions", {}).get(args.sort, 0)
            return r.get("predictions", {}).get(args.sort, 0)
        results = sorted(results, key=get_sort_key, reverse=True)
    
    # 打印结果
    print_results_table(results, args.model, args.top)
    
    # 打印汇总
    print(f"\n📊 汇总统计:")
    print("-"*50)
    summary = prediction_results.get("summary", {})
    for prop in ["pce", "dft_band_gap", "energy_above_hull", "voc", "jsc", "ff"]:
        if prop in summary:
            stats = summary[prop]
            print(f"  {prop:20}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # 打印最佳材料
    if results:
        print(f"\n🏆 最佳材料 (按 {args.sort} 排序):")
        print("-"*50)
        best = results[0]
        print(f"  化学式: {best.get('formula')}")
        if args.model == "ensemble":
            preds = best.get("ensemble_predictions", {})
        else:
            preds = best.get("predictions", {})
        for prop, val in preds.items():
            print(f"  {prop}: {val:.4f}")
    
    # 输出钙钛矿化学式列表 (用于 CSLLM)
    print(f"\n📋 生成的材料化学式 (用于 CSLLM):")
    print("-"*50)
    formulas = [r.get("formula") for r in results if r.get("formula")]
    unique_formulas = list(set(formulas))
    print(f"formulas = {unique_formulas}")
    
    # 保存结果
    if args.output:
        save_results_csv(results, args.output, args.model)
    
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(prediction_results, f, indent=2, ensure_ascii=False)
        print(f"💾 JSON 已保存: {args.json}")


if __name__ == "__main__":
    main()
