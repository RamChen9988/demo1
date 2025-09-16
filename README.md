# 对抗性图像生成项目

## 项目简介
这是一个演示机器学习模型在面对对抗性攻击时脆弱性的项目。通过生成精心设计的扰动，可以使模型对图像做出错误的分类判断。

## 主要功能

- **对抗性样本生成**: 使用FGSM、DeepFool、Carlini-Wagner等攻击方法
- **大尺寸图像处理**: 支持ResNet50等预训练模型
- **可视化展示**: 对比原始图像、扰动和对抗样本
- **性能评估**: 计算扰动统计信息和攻击成功率

## 快速开始

### 1. 安装依赖
```bash
# 使用一键安装脚本 (Windows)
install.bat

# 或使用一键安装脚本 (Linux/Mac)
chmod +x install.sh
./install.sh

# 或手动安装
pip install -r requirements.txt
```

### 2. 运行演示

#### 基础对抗性攻击演示
```bash
python adversarial_image_generator.py
```

#### 大尺寸图像对抗性攻击
```bash
python large_size_adversarial_demo.py
```

#### 图像识别测试
```bash
python test_image_recognition.py
```

## 项目结构

```
.
├── adversarial_image_generator.py  # 主要对抗性图像生成器
├── large_size_adversarial_demo.py  # 大尺寸图像对抗性演示
├── test_image_recognition.py       # 图像识别测试
├── imagenet_utils.py               # ImageNet工具类
├── imagenet_class_index.json       # ImageNet类别索引(英文)
├── imagenet_class_index_chinese.json # ImageNet类别索引(中文)
├── requirements.txt                # Python依赖列表
├── INSTALL.md                      # 详细安装指南
├── install.bat                     # Windows一键安装脚本
├── install.sh                      # Linux/Mac一键安装脚本
└── 1-5.jpg                         # 示例测试图片
```

## 支持的攻击方法

- **FGSM (Fast Gradient Sign Method)**: 快速梯度符号法
- **DeepFool**: 基于决策边界的攻击
- **Carlini-Wagner (L2)**: 强大的优化攻击
- **PGD (Projected Gradient Descent)**: 投影梯度下降法

## 系统要求

- Python 3.8+
- 4GB+ RAM (推荐8GB+)
- 支持的操作系统: Windows, macOS, Linux

## 文档

- [安装指南](INSTALL.md) - 详细的环境安装说明
- [requirements.txt](requirements.txt) - 项目依赖列表

## 技术支持

如遇问题，请参考：
1. 检查Python版本是否符合要求
2. 确认网络连接正常
3. 查看INSTALL.md中的常见问题解答

## 许可证

本项目仅供学习和研究使用。
