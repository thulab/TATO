#!/bin/bash

# MOIRAI模型运行脚本
# 运行MOIRAI-small, MOIRAI-base, MOIRAI-large模型

set -e  # 遇到错误时退出

echo "========================================="
echo "MOIRAI模型运行脚本"
echo "========================================="

# 检查是否在正确的环境中运行
if [ -z "$VIRTUAL_ENV" ]; then
    echo "警告: 未检测到虚拟环境"
    echo "建议先激活MOIRAI模型环境:"
    echo "  source venv_moirai/bin/activate"
    read -p "是否继续? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "退出"
        exit 1
    fi
fi

# MOIRAI模型列表
MOIRAI_MODELS="MOIRAI-small MOIRAI-base MOIRAI-large"

# 数据集列表
DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 Electricity Exchange Traffic Weather"

# 预测长度列表
PRED_LENS="24 48 96 192"

# 设备设置
DEVICE="cuda:0"
if [ ! -x "$(command -v nvidia-smi)" ]; then
    echo "未检测到GPU，使用CPU"
    DEVICE="cpu"
fi

echo "设备: $DEVICE"
echo "MOIRAI模型: $MOIRAI_MODELS"
echo "数据集: $DATASETS"
echo "预测长度: $PRED_LENS"
echo "注意: MOIRAI模型使用uni2ts库，可能需要更多内存"
echo ""

# 运行计数器
total_runs=0
completed_runs=0
failed_runs=0

# 创建结果目录
RESULTS_DIR="results/moirai_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo "结果将保存到: $RESULTS_DIR"
echo ""

# 运行所有组合
for model in $MOIRAI_MODELS; do
    for dataset in $DATASETS; do
        for pred_len in $PRED_LENS; do
            total_runs=$((total_runs + 1))
            
            echo "运行 $total_runs: model=$model, dataset=$dataset, pred_len=$pred_len"
            
            # 创建日志文件
            LOG_FILE="$RESULTS_DIR/${model}_${dataset}_${pred_len}.log"
            
            # 运行实验
            if python experiment/run.py \
                --device "$DEVICE" \
                --dataset "$dataset" \
                --pred_len "$pred_len" \
                --model "$model" \
                > "$LOG_FILE" 2>&1; then
                
                echo "  ✅ 成功"
                completed_runs=$((completed_runs + 1))
                
                # 提取关键结果
                if grep -q "Test MSE" "$LOG_FILE"; then
                    mse=$(grep "Test MSE" "$LOG_FILE" | tail -1 | awk '{print $NF}')
                    echo "      Test MSE: $mse"
                fi
                
                # 检查patch_size信息
                if grep -q "patch_size" "$LOG_FILE"; then
                    patch_size=$(grep "patch_size" "$LOG_FILE" | tail -1 | awk '{print $NF}')
                    echo "      patch_size: $patch_size"
                fi
            else
                echo "  ❌ 失败"
                failed_runs=$((failed_runs + 1))
                
                # 显示错误信息
                echo "      错误信息:"
                tail -5 "$LOG_FILE" | sed 's/^/      /'
            fi
            
            echo ""
        done
    done
done

echo "========================================="
echo "运行完成"
echo "========================================="
echo "总运行数: $total_runs"
echo "成功: $completed_runs"
echo "失败: $failed_runs"
echo ""

if [ $failed_runs -eq 0 ]; then
    echo "✅ 所有运行都成功完成"
else
    echo "⚠️  有 $failed_runs 个运行失败"
    echo "   查看日志文件: $RESULTS_DIR"
fi

echo ""
echo "结果目录: $RESULTS_DIR"
echo "日志文件:"
ls -la "$RESULTS_DIR"/*.log 2>/dev/null | head -10

# 生成汇总报告
if [ $completed_runs -gt 0 ]; then
    echo ""
    echo "生成汇总报告..."
    cat > "$RESULTS_DIR/summary.md" << EOF
# MOIRAI模型实验结果汇总

## 实验信息
- 运行时间: $(date)
- 设备: $DEVICE
- 总运行数: $total_runs
- 成功: $completed_runs
- 失败: $failed_runs

## 模型配置
- 模型: $MOIRAI_MODELS
- 数据集: $DATASETS
- 预测长度: $PRED_LENS

## 结果表格

| 模型 | 数据集 | 预测长度 | 状态 | Test MSE | patch_size | 日志文件 |
|------|--------|----------|------|----------|------------|----------|
EOF
    
    # 添加每个运行的结果
    for model in $MOIRAI_MODELS; do
        for dataset in $DATASETS; do
            for pred_len in $PRED_LENS; do
                LOG_FILE="$RESULTS_DIR/${model}_${dataset}_${pred_len}.log"
                if [ -f "$LOG_FILE" ]; then
                    if grep -q "Test MSE" "$LOG_FILE"; then
                        mse=$(grep "Test MSE" "$LOG_FILE" | tail -1 | awk '{print $NF}')
                        status="✅ 成功"
                    else
                        mse="N/A"
                        status="❌ 失败"
                    fi
                    
                    if grep -q "patch_size" "$LOG_FILE"; then
                        patch_size=$(grep "patch_size" "$LOG_FILE" | tail -1 | awk '{print $NF}')
                    else
                        patch_size="N/A"
                    fi
                else
                    mse="N/A"
                    patch_size="N/A"
                    status="❌ 未运行"
                fi
                
                echo "| $model | $dataset | $pred_len | $status | $mse | $patch_size | [查看日志](${model}_${dataset}_${pred_len}.log) |" >> "$RESULTS_DIR/summary.md"
            done
        done
    done
    
    echo "汇总报告已保存到: $RESULTS_DIR/summary.md"
fi

echo ""
echo "使用说明:"
echo "1. 确保已激活MOIRAI模型环境: source venv_moirai/bin/activate"
echo "2. MOIRAI模型需要uni2ts库，确保已正确安装"
echo "3. 查看详细结果: less $RESULTS_DIR/summary.md"
echo "4. 重新运行特定配置: python experiment/run.py --device $DEVICE --dataset DATASET --pred_len LEN --model MODEL"
echo ""
echo "注意:"
echo "- MOIRAI模型支持多种patch_size配置"
echo "- 大型模型可能需要更多GPU内存"
echo "- 结果包含patch_size信息，用于分析模型性能"
