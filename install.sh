#!/bin/bash

echo "========================================"
echo "  对抗性图像生成项目 - 一键安装脚本"
echo "========================================"
echo

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python 3.8+"
    echo "下载地址: https://www.python.org/downloads/"
    exit 1
fi

# 显示Python版本
echo "检测到Python版本:"
python3 --version
echo

# 检查pip是否安装
if ! command -v pip3 &> /dev/null; then
    echo "错误: 未找到pip3，请确保Python安装完整"
    exit 1
fi

echo "开始安装项目依赖..."
echo

# 创建虚拟环境（可选）
read -p "是否创建虚拟环境？(y/n): " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "正在创建虚拟环境..."
    python3 -m venv adversarial-env
    echo "虚拟环境创建完成！"
    echo
    echo "请手动激活虚拟环境:"
    echo "   source adversarial-env/bin/activate"
    echo "然后执行: pip install -r requirements.txt"
    exit 0
fi

# 安装依赖
echo "正在安装依赖包..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo
    echo "========================================"
    echo "  依赖安装成功！"
    echo "========================================"
    echo
    echo "现在可以运行以下脚本:"
    echo "  - adversarial_image_generator.py"
    echo "  - large_size_adversarial_demo.py"  
    echo "  - test_image_recognition.py"
    echo
    echo "详细使用方法请参考 INSTALL.md"
else
    echo
    echo "========================================"
    echo "  依赖安装失败！"
    echo "========================================"
    echo
    echo "请检查:"
    echo "  1. 网络连接"
    echo "  2. Python版本兼容性"
    echo "  3. 系统权限"
fi

echo
read -p "按回车键退出..."
