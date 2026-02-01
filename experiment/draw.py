import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. 全局设置 ---
sns.set_theme(style="white") 
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False 

# --- 2. 数据 ---
models = [
    'PSC_Agent (Ours)',      
    'AutoGPT (Autonomous)',   
    'ReAct Agent (LangChain)',
    'Claude-3.5 (Zero-shot)', 
    'GPT-4o (Zero-shot)'      
]
scores = [9.6, 6.8, 5.5, 4.8, 4.5]
errors = [0.2, 0.4, 0.3, 0.5, 0.4]

# --- 3. 配色方案 (科技蓝渐变) ---
colors = ['#08306B', '#2171B5', '#6BAED6', '#9ECAE1', '#C6DBEF']

# --- 4. 绘图 ---
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图
bars = ax.barh(models, scores, xerr=errors, color=colors, 
               edgecolor='white', linewidth=1, height=0.65, 
               capsize=5, alpha=0.95, zorder=3)

# 误差棒
plt.errorbar(scores, models, xerr=errors, fmt='none', 
             ecolor='#444444', capsize=5, zorder=4)

# --- 5. 添加标签 ---
for bar, score, color in zip(bars, scores, colors):
    width = bar.get_width()
    ax.text(width + 0.6, # 稍微远一点
            bar.get_y() + bar.get_height()/2, 
            f'{score:.1f}', 
            ha='left', va='center', 
            fontsize=12, fontweight='bold', color=color) 

# --- 6. 装饰 ---
ax.set_xlabel('Expert Evaluation Score (0-10)', fontsize=12, fontweight='bold', color='#333333', labelpad=10)
ax.set_title('Comparative Output Quality of Scientific Workflows', fontsize=15, fontweight='bold', color='#111111', pad=20)

# 【关键修改 1】直接强制把X轴拉到12 (或者更大)，确保右边有巨大的空地
ax.set_xlim(0, 12.0)

ax.invert_yaxis() 
sns.despine(left=True, bottom=False)
ax.tick_params(axis='y', length=0, labelsize=12)
ax.tick_params(axis='x', colors='#555555')
ax.grid(axis='x', linestyle='--', alpha=0.3, color='gray', zorder=0)

# 【关键修改 2】手动调整边距，不再依赖自动布局
# right=0.85 意味着图表主体只占画布宽度的 85%，右边留 15% 给文字
plt.subplots_adjust(left=0.2, right=0.85, top=0.9, bottom=0.1)

plt.show()