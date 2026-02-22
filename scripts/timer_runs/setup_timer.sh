#!/bin/bash

# Timer模型环境设置脚本
# 为Timer模型创建独立的虚拟环境

set -e  # 遇到错误时退出

echo "========================================="
echo "Timer模型环境设置脚本"
echo "========================================="

MODEL_NAME="timer"
REQ_FILE="timer_runs/timer_requirements.txt"
MODEL_KEY="Timer-LOTSA"  # 也可以使用 Timer-UTSD

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>/dev/null | cut -d' ' -f2)
if [ -z "$PYTHON_VERSION" ]; then
    echo "错误: 未找到python3，请先安装Python 3.8+"
    exit 1
fi

echo "检测到Python版本: $PYTHON_VERSION"
echo "目标模型: Timer (Timer-UTSD, Timer-LOTSA)"
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
echo "安装Timer模型特定依赖..."
pip install -r "$REQ_FILE"
echo "Timer模型依赖安装完成"
echo ""

# 测试模型导入
echo "测试Timer模型导入..."
cat > test_timer.py << EOF
import sys
sys.path.append('.')
try:
    from model.model_factory import ModelFactory
    import argparse
    args = argparse.Namespace()
    model = ModelFactory.load_model('${MODEL_KEY}', 'cpu', args)
    print('✅ Timer模型导入成功')
    print('模型名称:', model.model_name)
except Exception as e:
    print(f'❌ Timer模型导入失败: {e}')
    import traceback
    traceback.print_exc()
EOF

python test_timer.py
rm -f test_timer.py
echo ""

# 退出环境
deactivate
echo "已退出 venv_${MODEL_NAME} 环境"
echo ""

echo "========================================="
echo "Timer模型环境设置完成"
echo "========================================="
echo ""
echo "使用说明:"
echo "1. 激活环境: source venv_${MODEL_NAME}/bin/activate"
echo "2. 运行Timer模型: python your_script.py"
echo "3. 退出环境: deactivate"
echo ""
echo "示例代码:"
echo "```python"
echo "from model.model_factory import ModelFactory"
echo "import argparse"
echo "args = argparse.Namespace()"
echo "model = ModelFactory.load_model('Timer-LOTSA', 'cpu', args)"
echo "# 或使用 Timer-UTSD"
echo "```"
echo ""
echo "注意: Timer模型需要从CKPT目录加载预训练权重"
