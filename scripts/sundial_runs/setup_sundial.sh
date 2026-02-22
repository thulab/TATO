#!/bin/bash

# Sundial模型环境设置脚本
# 为Sundial模型创建独立的虚拟环境（需要transformers 4.40.1）

set -e  # 遇到错误时退出

echo "========================================="
echo "Sundial模型环境设置脚本"
echo "========================================="

MODEL_NAME="sundial"
REQ_FILE="scripts/sundial_runs/sundial_requirements.txt"
MODEL_KEY="Sundial"

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>/dev/null | cut -d' ' -f2)
if [ -z "$PYTHON_VERSION" ]; then
    echo "错误: 未找到python3，请先安装Python 3.8+"
    exit 1
fi

echo "检测到Python版本: $PYTHON_VERSION"
echo "目标模型: Sundial"
echo "虚拟环境: venv_${MODEL_NAME}"
echo "依赖文件: ${REQ_FILE}"
echo "重要: Sundial需要transformers 4.40.1，与其他模型不兼容"
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
echo "安装Sundial模型特定依赖..."
echo "重要: Sundial需要transformers 4.40.1版本（与其他模型不同）"
pip install -r "$REQ_FILE"
echo "Sundial模型依赖安装完成"
echo ""

# 验证transformers版本
echo "验证transformers版本..."
python -c "import transformers; print(f'当前transformers版本: {transformers.__version__}'); assert transformers.__version__ == '4.40.1', 'Sundial需要transformers 4.40.1'"
echo "✅ transformers版本正确: 4.40.1"
echo ""

# 测试模型导入
echo "测试Sundial模型导入..."
cat > test_sundial.py << EOF
import sys
sys.path.append('.')
try:
    from model.model_factory import ModelFactory
    import argparse
    args = argparse.Namespace()
    model = ModelFactory.load_model('${MODEL_KEY}', 'cpu', args)
    print('✅ Sundial模型导入成功')
    print('模型名称:', model.model_name)
    print('input_token_len:', model.input_token_len)
except Exception as e:
    print(f'❌ Sundial模型导入失败: {e}')
    import traceback
    traceback.print_exc()
EOF

python test_sundial.py
rm -f test_sundial.py
echo ""

# 退出环境
deactivate
echo "已退出 venv_${MODEL_NAME} 环境"
echo ""

echo "========================================="
echo "Sundial模型环境设置完成"
echo "========================================="
echo ""
echo "使用说明:"
echo "1. 激活环境: source venv_${MODEL_NAME}/bin/activate"
echo "2. 运行Sundial模型: python your_script.py"
echo "3. 退出环境: deactivate"
echo ""
echo "示例代码:"
echo "```python"
echo "from model.model_factory import ModelFactory"
echo "import argparse"
echo "args = argparse.Namespace()"
echo "model = ModelFactory.load_model('Sundial', 'cpu', args)"
echo "```"
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
