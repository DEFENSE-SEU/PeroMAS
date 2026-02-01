"""分析当前数据集结构"""
import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv('data/raw/full_dataset.csv')

print("=" * 60)
print("数据集基本信息")
print("=" * 60)
print(f"总行数: {len(df)}")
print(f"总列数: {len(df.columns)}")
print(f"\n列名: {df.columns.tolist()}")

print("\n" + "=" * 60)
print("目标变量统计")
print("=" * 60)
targets = ['pce', 'dft_band_gap', 'energy_above_hull', 'voc', 'jsc', 'ff', 'stability_t80']
for col in targets:
    if col in df.columns:
        valid = df[col].notna().sum()
        missing = df[col].isna().sum()
        print(f"{col:20s}: {valid:4d} 有效 / {missing:4d} 缺失 ({valid/len(df)*100:.1f}%)")

print("\n" + "=" * 60)
print("Composition 列示例 (Long Form)")
print("=" * 60)
print(df['composition'].head(15).tolist())

print("\n" + "=" * 60)
print("CIF 列示例")
print("=" * 60)
print(f"CIF 非空数量: {df['cif'].notna().sum()}")
print(f"CIF 示例 (前200字符):\n{str(df['cif'].iloc[0])[:200]}...")

print("\n" + "=" * 60)
print("Composition Short 列示例")
print("=" * 60)
if 'composition_short' in df.columns:
    print(df['composition_short'].head(15).tolist())

print("\n" + "=" * 60)
print("数据类型")
print("=" * 60)
print(df.dtypes)

# 检查哪些目标变量同时有效
print("\n" + "=" * 60)
print("同时拥有所有目标变量的样本数")
print("=" * 60)
mask_all = df[targets].notna().all(axis=1)
print(f"所有目标都有效的样本: {mask_all.sum()}")

# 检查常用目标组合
main_targets = ['pce', 'dft_band_gap', 'energy_above_hull']
mask_main = df[main_targets].notna().all(axis=1)
print(f"主要目标(pce, band_gap, energy_above_hull)有效的样本: {mask_main.sum()}")
