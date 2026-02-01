#!/bin/bash
#SBATCH --job-name=psc_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

echo "============================================================"
echo "钙钛矿性质预测模型训练"
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "============================================================"

mkdir -p logs
mkdir -p data/csr
mkdir -p data/model
mkdir -p data/model/hyperopt

cd "$(dirname "$0")"
echo "工作目录: $(pwd)"

if [ ! -f "data/raw/full_dataset.csv" ]; then
    echo "错误: 数据文件不存在!"
    exit 1
fi

echo ""
echo "数据集信息:"
wc -l data/raw/full_dataset.csv

echo ""
echo "============================================================"
echo "Step 1: 超参数优化 Composition-only 模型 (Random Forest)"
echo "============================================================"
python train_models.py --mode comp_only --model RF --hyperopt --n_trials 100 --clear_cache

echo ""
echo "============================================================"
echo "Step 2: 超参数优化 CIF-only 模型 (Random Forest)"
echo "============================================================"
python train_models.py --mode cif_only --model RF --hyperopt --n_trials 100

echo ""
echo "============================================================"
echo "Step 3: 超参数优化 Composition-only 模型 (Neural Network)"
echo "============================================================"
python train_models.py --mode comp_only --model NN --hyperopt --n_trials 100

echo ""
echo "============================================================"
echo "Step 4: 超参数优化 CIF-only 模型 (Neural Network)"
echo "============================================================"
python train_models.py --mode cif_only --model NN --hyperopt --n_trials 100

echo ""
echo "============================================================"
echo "Step 5: 超参数优化 Composition-only 模型 (GBDT)"
echo "============================================================"
python train_models.py --mode comp_only --model GBDT --hyperopt --n_trials 100

echo ""
echo "============================================================"
echo "Step 6: 超参数优化 CIF-only 模型 (GBDT)"
echo "============================================================"
python train_models.py --mode cif_only --model GBDT --hyperopt --n_trials 100

echo ""
echo "============================================================"
echo "所有训练完成!"
echo "结束时间: $(date)"
echo "============================================================"

echo ""
echo "生成的模型文件:"
ls -lh data/model/*.pkl 2>/dev/null || echo "没有找到模型文件"

echo ""
echo "============================================================"
echo "Step 7: 测试所有训练好的模型"
echo "============================================================"
python test_models.py

echo ""
echo "============================================================"
echo "全部流程完成!"
echo "结束时间: $(date)"
echo "============================================================"