# 环境安装指南

## 项目简介
这是一个对抗性图像生成项目，用于演示机器学习模型在面对精心设计的对抗性攻击时的脆弱性。

## 快速安装

### 方法一：使用 requirements.txt（推荐）
```bash
# 安装所有依赖
pip install -r requirements.txt
```

### 方法二：手动安装
```bash
# 安装核心依赖
pip install tensorflow numpy matplotlib Pillow

# 安装对抗性攻击工具包
pip install adversarial-robustness-toolbox

# 安装网络请求库（可选，用于下载ImageNet类别）
pip install requests
```

## 系统要求

### Python版本
- Python 3.8 或更高版本

### 硬件要求
- **最低配置**: 4GB RAM, 支持CPU运行
- **推荐配置**: 8GB+ RAM, NVIDIA GPU (支持CUDA)
- **GPU加速**: 如需GPU加速，请安装对应版本的TensorFlow GPU版本

### 操作系统
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 18.04+, CentOS 7+)

## 详细安装步骤

### 1. 创建虚拟环境（推荐）
```bash
# 创建虚拟环境
python -m venv adversarial-env

# 激活虚拟环境
# Windows:
adversarial-env\Scripts\activate
# Linux/Mac:
source adversarial-env/bin/activate
```

### 2. 安装依赖
```bash
# 在虚拟环境中安装
pip install -r requirements.txt
```

### 3. 验证安装
```bash
# 运行简单的测试脚本验证安装
python -c "import tensorflow as tf; import numpy as np; import matplotlib.pyplot as plt; from art.estimators.classification import KerasClassifier; print('所有依赖安装成功！')"
```

## 可选配置

### GPU支持（如适用）
```bash
# 如果使用NVIDIA GPU，安装GPU版本的TensorFlow
pip uninstall tensorflow
pip install tensorflow-gpu
```

### 中文字体支持
```bash
# Linux (Ubuntu/Debian):
sudo apt-get install fonts-wqy-microhei

# macOS:
brew install wqy-microhei

# Windows: 通常已内置中文字体支持
```

## 项目文件说明

- `adversarial_image_generator.py` - 主要的对抗性图像生成器
- `large_size_adversarial_demo.py` - 大尺寸图像对抗性攻击演示
- `test_image_recognition.py` - 图像识别测试脚本
- `imagenet_utils.py` - ImageNet工具类
- `imagenet_class_index.json` - ImageNet类别索引（英文）
- `imagenet_class_index_chinese.json` - ImageNet类别索引（中文）

## 常见问题

### Q: 安装过程中出现内存不足错误
A: 尝试使用较小的批次大小或减少训练轮数

### Q: 中文显示为方框
A: 安装中文字体包（见"中文字体支持"部分）

### Q: GPU无法使用
A: 确保已安装正确的CUDA和cuDNN版本

### Q: 依赖冲突
A: 使用虚拟环境隔离项目依赖

## 技术支持
如遇安装问题，请检查：
1. Python版本是否符合要求
2. 网络连接是否正常
3. 系统权限是否足够

或提交Issue到项目仓库。
