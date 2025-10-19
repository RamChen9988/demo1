"""
极简黑盒迁移攻击演示
目标：用替代模型生成的对抗样本攻击目标模型（ResNet50）
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod
import matplotlib.pyplot as plt
from PIL import Image

print("=== 黑盒迁移攻击演示 ===\n")

def load_and_preprocess_image(image_path):
    """加载并预处理图片"""
    print(f"1. 加载图片: {image_path}")
    img = Image.open(image_path)
    img = img.resize((224, 224))  # ResNet50输入尺寸
    img_array = np.array(img)
    
    # 如果图片是灰度图，转换为RGB
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # 如果有alpha通道，移除
        img_array = img_array[:, :, :3]
    
    # 预处理
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # 添加batch维度
    
    print(f"   图片形状: {img_array.shape}")
    return img_array, img

def create_models():
    """创建目标模型和替代模型（都是ResNet50但不同实例）"""
    print("2. 创建模型...")
    
    # 目标模型（模拟黑盒API）
    target_model = ResNet50(weights='imagenet')
    print("   ✓ 目标模型加载完成")
    
    # 替代模型（攻击者使用的模型）- 需要编译以支持ART
    substitute_model = ResNet50(weights='imagenet')
    substitute_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("   ✓ 替代模型加载完成")
    
    return target_model, substitute_model

def predict_and_show(model, image, model_name="模型"):
    """显示模型预测结果"""
    predictions = model.predict(image, verbose=0)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    
    print(f"   {model_name}预测结果:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"     {i+1}. {label}: {score:.4f}")
    
    return decoded_predictions[0][1]  # 返回最可能的标签

def whitebox_attack(substitute_model, original_image):
    """在白盒替代模型上生成对抗样本"""
    print("3. 在替代模型上发起白盒攻击...")
    
    # 使用ART包装替代模型
    classifier = KerasClassifier(model=substitute_model, use_logits=False)
    # 投影梯度下降法，更强的攻击
    # classifier = DeepFool(estimator=classifier, max_iter=50)


    # 创建FGSM攻击器
    # 修改eps值以调整攻击强度,0.9攻击成功，0.1失败
    attack = FastGradientMethod(estimator=classifier, eps=0.9)  # 攻击强度
    
    # 生成对抗样本
    adversarial_image = attack.generate(x=original_image)
    
    print("   ✓ 对抗样本生成完成")
    return adversarial_image, classifier

def blackbox_attack(target_model, original_image, adversarial_image):
    """测试黑盒迁移攻击效果"""
    print("4. 测试黑盒迁移攻击...")
    
    # 原始图片在目标模型上的预测
    print("   a. 原始图片预测:")
    original_label = predict_and_show(target_model, original_image, "目标模型")
    
    # 对抗样本在目标模型上的预测
    print("   b. 对抗样本预测:")
    adversarial_label = predict_and_show(target_model, adversarial_image, "目标模型")
    
    # 计算攻击成功率
    attack_success = original_label != adversarial_label
    print(f"   c. 攻击结果: {'成功' if attack_success else '失败'}")
    print(f"      原始分类: {original_label}")
    print(f"      攻击后分类: {adversarial_label}")
    
    return attack_success, original_label, adversarial_label

def visualize_attack(original_img, original_image, adversarial_image, 
                    original_label, adversarial_label):
    """可视化攻击效果"""
    print("5. 生成可视化结果...")
    
    plt.figure(figsize=(15, 5))
    # 简化字体设置，避免中文显示问题
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 原始图片 - 直接使用PIL加载的图片
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title(f'原始图片\n分类: {original_label}')
    plt.axis('off')
    
    # 对抗样本 - 正确反预处理以匹配原始图片颜色空间
    adv_img = adversarial_image[0].copy()
    
    # 反预处理：恢复ImageNet预处理
    # ResNet50使用caffe模式：RGB转BGR，然后减去ImageNet均值
    # ImageNet BGR均值：[103.939, 116.779, 123.68]
    mean = np.array([103.939, 116.779, 123.68])
    
    # 反预处理：加上均值，然后BGR转RGB
    adv_img = adv_img + mean
    # BGR转RGB
    adv_img = adv_img[:, :, ::-1]  # 反转通道顺序
    # 确保值在[0,255]范围内并转换为uint8
    adv_img = np.clip(adv_img, 0, 255).astype(np.uint8)
    
    plt.subplot(1, 3, 2)
    plt.imshow(adv_img)
    plt.title(f'对抗样本\n分类: {adversarial_label}')
    plt.axis('off')
    
    # 扰动可视化 - 使用相同的颜色空间
    perturbation = adversarial_image[0] - original_image[0]
    # 标准化扰动以便可视化
    perturbation = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min())
    
    plt.subplot(1, 3, 3)
    plt.imshow(perturbation)
    plt.title('攻击扰动\n(放大显示)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('blackbox_attack_result.png', dpi=150, bbox_inches='tight')
    print("   ✓ 结果已保存为 'blackbox_attack_result.png'")

def main():
    """主函数"""
    try:
        # 加载图片
        original_image, original_img = load_and_preprocess_image('1.jpg')
        
        # 创建模型
        target_model, substitute_model = create_models()
        
        # 白盒攻击：在替代模型上生成对抗样本
        adversarial_image, classifier = whitebox_attack(substitute_model, original_image)
        
        # 黑盒攻击：测试对抗样本对目标模型的效果
        attack_success, original_label, adversarial_label = blackbox_attack(
            target_model, original_image, adversarial_image)
        
        # 可视化结果
        visualize_attack(original_img, original_image, adversarial_image, 
                        original_label, adversarial_label)
        
        print(f"\n=== 演示完成 ===")
        print(f"黑盒迁移攻击: {'成功' if attack_success else '失败'}")
        print(f"模型被欺骗，将'{original_label}'识别为'{adversarial_label}'")
        
    except FileNotFoundError:
        print("错误: 找不到 '1.jpg' 文件")
        print("请确保在程序同目录下放置一张名为 '1.jpg' 的图片")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
