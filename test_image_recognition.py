import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

# 导入动态加载工具
from imagenet_utils import load_imagenet_classes, decode_predictions

# 简化字体设置
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']

def test_image_recognition(image_path):
    """测试图像识别"""
    if not os.path.exists(image_path):
        print(f"错误：图片路径不存在 - {image_path}")
        return
    
    # 加载预训练模型
    print("加载预训练的ResNet50模型...")
    model = ResNet50(weights='imagenet', include_top=True)
    
    # 预处理图片
    img = Image.open(image_path).convert('RGB')
    original_img = img.copy()
    img = img.resize((224, 224))
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # 预测
    print("进行预测...")
    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=3)[0]
    
    # 显示结果
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    # 创建预测结果文本
    pred_text = "\n".join([f"{cls}: {prob:.2f}%" for (_, cls, prob) in decoded_preds])
    plt.text(0.1, 0.5, f"Predictions:\n{pred_text}", fontsize=12, 
             transform=plt.gca().transAxes, verticalalignment='center')
    plt.axis('off')
    plt.title("Prediction Results")
    
    plt.tight_layout()
    plt.show()
    
    print("\n预测结果:")
    for i, (_, cls, prob) in enumerate(decoded_preds):
        print(f"{i+1}. {cls}: {prob*100:.2f}%")

if __name__ == "__main__":
    # 首先加载ImageNet类别
    print("正在加载ImageNet类别...")
    load_imagenet_classes()
    
    # 测试当前目录下的所有图片
    image_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("当前目录下没有找到图片文件")
    else:
        print(f"找到图片文件: {image_files}")
        
        for img_file in image_files:
            print(f"\n=== 测试图片: {img_file} ===")
            test_image_recognition(img_file)
