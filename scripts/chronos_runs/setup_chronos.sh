#!/bin/bash

# Chronos模型环境设置脚本
# 为Chronos模型创建独立的虚拟环境

set -e  # 遇到错误时退出

echo "========================================="
echo "Chronos模型环境设置脚本"
echo "========================================="

MODEL_NAME="chronos"
REQ_FILE="scripts/chronos_runs/chronos_requirements.txt"
MODEL_KEY="Chronos-tiny"

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>/dev/null | cut -d' ' -f2)
if [ -z "$PYTHON_VERSION" ]; then
    echo "错误: 未找到python3，请先安装Python 3.8+"
    exit 1
fi

echo "检测到Python版本: $PYTHON_VERSION"
echo "目标模型: Chronos (Chronos-tiny)"
echo "虚拟环境: venv_${MODEL_NAME}"
echo "依赖文件: ${REQ_FILE}"
echo ""

# 检查依赖文件是否存在
if [ ! -f "$REQ_FILE" ]; then
    echo "错误: 未找到依赖文件 $REQ_FILE"
    exit 1
fi

# 创建虚拟环境
echo "正在创建虚拟环境..."
python3 -m venv "venv_${MODEL_NAME}"

if [ ! -f "venv_${MODEL_NAME}/bin/activate" ]; then
    echo "错误: 虚拟环境创建失败"
    exit 1
fi

# 激活环境并安装依赖
echo "激活虚拟环境..."
source "venv_${MODEL_NAME}/bin/activate"
echo "已激活 venv_${MODEL_NAME} 环境"
echo ""

# 升级pip
echo "升级pip..."
pip install --upgrade pip
echo ""

# 安装基础依赖（可选）
echo "安装基础依赖..."
if [ -f "base_requirements.txt" ]; then
    pip install -r base_requirements.txt
    echo "基础依赖安装完成"
else
    echo "警告: 未找到基础依赖文件"
fi
echo ""

# 安装模型特定依赖
echo "安装Chronos模型特定依赖..."
echo "注意: Chronos需要chronos-forecasting库..."
pip install -r "$REQ_FILE"
echo "Chronos模型依赖安装完成"
echo ""

# 测试模型导入
echo "测试Chronos模型导入..."
cat > test_chronos.py << EOF
import sys
sys.path.append('.')
try:
    from model.model_factory import ModelFactory
    import argparse
    args = argparse.Namespace()
    model = ModelFactory.load_model('${MODEL_KEY}', 'cpu', args)
    print('✅ Chronos模型导入成功')
    print('模型名称:', model.model_name)
    print('num_samples:', model.num_samples)
except Exception as e:
    print(f'❌ Chronos模型导入失败: {e}')
    import traceback
    traceback.print_exc()
EOF

python test_chronos.py
rm -f test_chronos.py
echo ""

# 退出环境
deactivate
echo "已退出 venv_${MODEL_NAME} 环境"
echo ""

echo "========================================="
echo "Chronos模型环境设置完成"
echo "========================================="
echo ""
echo "使用说明:"
echo "1. 激活环境: source venv_${MODEL_NAME}/bin/activate"
echo "2. 运行Chronos模型: python your_script.py"
echo "3. 退出环境: deactivate"
echo ""
echo "示例代码:"
echo "```python"
echo "from model.model_factory import ModelFactory"
echo "import argparse"
echo "args = argparse.Namespace()"
echo "model = ModelFactory.load_model('Chronos-tiny', 'cpu', args)"
echo "```"
echo ""
echo "注意:"
echo "1. Chronos模型需要从CKPT目录加载预训练权重"
echo "2. Chronos使用概率预测，支持多样本生成"
echo "3. 使用torch.float16进行混合精度训练"
echo "4. 如果chronos-forecasting不可用，可能需要从源码安装"
