#!/bin/bash
#SBATCH --job-name=psc_single
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

echo "============================================================"
echo "钙钛矿性质预测 - 单属性模型训练"
echo "============================================================"
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo ""

# 设置工作目录
cd "$(dirname "$0")"
echo "工作目录: $(pwd)"

# 创建目录
mkdir -p logs
mkdir -p data/csr
mkdir -p data/model/single_target
mkdir -p data/model/hyperopt
mkdir -p data/results

# 检查数据文件
if [ ! -f "data/raw/full_dataset.csv" ]; then
    echo "错误: 数据文件 data/raw/full_dataset.csv 不存在!"
    exit 1
fi

echo ""
echo "数据集信息:"
head -1 data/raw/full_dataset.csv
wc -l data/raw/full_dataset.csv

# 训练参数 - 优化版
N_TRIALS=200  # 增加到200次，更充分搜索
TARGETS="pce dft_band_gap energy_above_hull voc jsc ff"
FEATURES="comp_only cif_only"
MODELS="RF GBDT NN"

echo ""
echo "============================================================"
echo "训练配置:"
echo "  目标属性: $TARGETS"
echo "  特征模式: $FEATURES"
echo "  模型类型: $MODELS"
echo "  超参数优化试验次数: $N_TRIALS"
echo "============================================================"

# 清除旧的超参数数据库（可选）
# echo "清除旧的超参数优化数据库..."
# rm -f data/model/hyperopt/*.db

# ============================================================
# 方式一：按特征模式分组训练（推荐，减少特征加载次数）
# ============================================================
for FEATURE in $FEATURES; do
    echo ""
    echo "############################################################"
    echo "# 特征模式: $FEATURE"
    echo "############################################################"
    
    for MODEL in $MODELS; do
        echo ""
        echo "============================================================"
        echo "训练: $FEATURE + $MODEL (所有属性)"
        echo "开始时间: $(date)"
        echo "============================================================"
        
        python main_single_target.py \
            --target all \
            --feature $FEATURE \
            --model $MODEL \
            --hyperopt \
            --n_trials $N_TRIALS
        
        echo "完成: $FEATURE + $MODEL"
        echo "时间: $(date)"
    done
done

# ============================================================
# 方式二：单独训练某个模型（调试用）
# ============================================================
# python main_single_target.py --target pce --feature comp_only --model RF --hyperopt --n_trials 50

echo ""
echo "============================================================"
echo "检查生成的模型文件:"
echo "============================================================"
ls -lh data/model/single_target/*.pkl 2>/dev/null || echo "没有找到模型文件"

echo ""
echo "============================================================"
echo "生成的超参数优化数据库:"
echo "============================================================"
ls -lh data/model/hyperopt/*.db 2>/dev/null || echo "没有找到数据库文件"

echo ""
echo "============================================================"
echo "训练结果汇总:"
echo "============================================================"
if [ -d "data/results" ]; then
    ls -lht data/results/*.csv 2>/dev/null | head -5
fi

echo ""
echo "============================================================"
echo "全部训练完成!"
echo "结束时间: $(date)"
echo "============================================================"
