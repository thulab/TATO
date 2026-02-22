#!/bin/bash

# Sundial模型运行脚本
# 运行Sundial模型（需要transformers 4.40.1）

set -e  # 遇到错误时退出

echo "========================================="
echo "Sundial模型运行脚本"
echo "========================================="

# 检查是否在正确的环境中运行
if [ -z "$VIRTUAL_ENV" ]; then
    echo "警告: 未检测到虚拟环境"
    echo "重要: Sundial需要transformers 4.40.1，与其他模型不兼容"
    echo "建议先激活Sundial模型环境:"
    echo "  source venv_sundial/bin/activate"
    read -p "是否继续? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "退出"
        exit 1
    fi
fi

# 验证transformers版本
echo "验证transformers版本..."
TRANSFORMERS_VERSION=$(python -c "import transformers; print(transformers.__version__)")
if [ "$TRANSFORMERS_VERSION" != "4.40.1" ]; then
    echo "❌ 错误: Sundial需要transformers 4.40.1，当前版本: $TRANSFORMERS_VERSION"
    echo "请激活正确的虚拟环境: source venv_sundial/bin/activate"
    exit 1
fi
echo "✅ transformers版本正确: $TRANSFORMERS_VERSION"
echo ""

# Sundial模型列表
SUNDIAL_MODELS="Sundial"

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
echo "Sundial模型: $SUNDIAL_MODELS"
echo "数据集: $DATASETS"
echo "预测长度: $PRED_LENS"
echo "重要: Sundial使用自定义架构，需要trust_remote_code=True"
echo ""

# 运行计数器
total_runs=0
completed_runs=0
failed_runs=0

# 创建结果目录
RESULTS_DIR="results/sundial_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo "结果将保存到: $RESULTS_DIR"
echo ""

# 运行所有组合
for model in $SUNDIAL_MODELS; do
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
                
                # 检查input_token_len信息
                if grep -q "input_token_len" "$LOG_FILE"; then
                    input_token_len=$(grep "input_token_len" "$LOG_FILE" | tail -1 | awk '{print $NF}')
                    echo "      input_token_len: $input_token_len"
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
# Sundial模型实验结果汇总

## 实验信息
- 运行时间: $(date)
- 设备: $DEVICE
- 总运行数: $total_runs
- 成功: $completed_runs
- 失败: $failed_runs
- transformers版本: $TRANSFORMERS_VERSION

## 模型配置
- 模型: $SUNDIAL_MODELS
- 数据集: $DATASETS
- 预测长度: $PRED_LENS

## 结果表格

| 模型 | 数据集 | 预测长度 | 状态 | Test MSE | input_token_len | 日志文件 |
|------|--------|----------|------|----------|-----------------|----------|
EOF
    
    # 添加每个运行的结果
    for model in $SUNDIAL_MODELS; do
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
                    
                    if grep -q "input_token_len" "$LOG_FILE"; then
                        input_token_len=$(grep "input_token_len" "$LOG_FILE" | tail -1 | awk '{print $NF}')
                    else
                        input_token_len="N/A"
                    fi
                else
                    mse="N/A"
                    input_token_len="N/A"
                    status="❌ 未运行"
                fi
                
                echo "| $model | $dataset | $pred_len | $status | $mse | $input_token_len | [查看日志](${model}_${dataset}_${pred_len}.log) |" >> "$RESULTS_DIR/summary.md"
            done
        done
    done
    
    echo "汇总报告已保存到: $RESULTS_DIR/summary.md"
fi

echo ""
echo "使用说明:"
echo "1. 确保已激活Sundial模型环境: source venv_sundial/bin/activate"
echo "2. 验证transformers版本: python -c \"import transformers; print(transformers.__version__)\""
echo "3. 查看详细结果: less $RESULTS_DIR/summary.md"
echo "4. 重新运行特定配置: python experiment/run.py --device $DEVICE --dataset DATASET --pred_len LEN --model MODEL"
echo ""
echo "重要提示:"
echo "1. Sundial模型需要transformers 4.40.1版本，与其他模型不兼容"
echo "2. Sundial使用自定义的SundialForPrediction架构"
echo "3. 加载模型时需要trust_remote_code=True"
echo "4. 支持多步预测和概率预测"
echo ""
echo "版本冲突说明:"
echo "- 其他模型使用transformers 4.41.0"
echo "- Sundial使用transformers 4.40.1"
echo "- 因此必须使用独立虚拟环境"
