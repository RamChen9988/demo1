import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

# 导入动态加载工具
from imagenet_utils import load_imagenet_classes, decode_predictions

# 简化字体设置，避免中文显示问题
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class LargeImageAdversarialDemo:
    def __init__(self, target_size=(224, 224)):
        """
        初始化大尺寸图像对抗性攻击演示
        :param target_size: 图像尺寸，默认224x224，可改为320x320等
        """
        self.target_size = target_size
        self.model = self.load_pretrained_model()
        # 编译模型以设置损失函数
        self.model.compile(optimizer='adam', 
                          loss='categorical_crossentropy', 
                          metrics=['accuracy'])
        # 调整ART分类器以适应ResNet50的预处理
        self.classifier = KerasClassifier(
            model=self.model, 
            clip_values=(0, 255),  # ResNet50使用0-255范围
            preprocessing=(np.array([103.939, 116.779, 123.68]), np.array([1, 1, 1]))  # ImageNet均值和标准差
        )
        
    def load_pretrained_model(self):
        """加载预训练的ResNet50模型（在ImageNet上训练）"""
        print(f"加载预训练的ResNet50模型，输入尺寸{self.target_size}...")
        # 加载不包含顶部分类层的模型，然后添加适合的分类层
        base_model = ResNet50(weights='imagenet', include_top=True, 
                             input_shape=(*self.target_size, 3))
        return base_model
    
    def preprocess_image(self, image_path):
        """
        预处理本地图片，使其符合ResNet50的输入要求
        :param image_path: 本地图片路径
        :return: 预处理后的图像数组和原始图像（用于显示）
        """
        # 读取原始图片
        img = Image.open(image_path).convert('RGB')
        original_img = img.copy()  # 保存原始图像用于显示
        
        # 调整尺寸
        img = img.resize(self.target_size)
        
        # 转为数组并扩展维度
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 应用ResNet50的预处理
        img_array = preprocess_input(img_array)
        
        return img_array, original_img
    
    def generate_adversarial_samples(self, img_array, attack_method='fgsm', eps=2.0):
        """生成对抗性样本，支持多种攻击方法"""
        print(f"使用{attack_method}方法生成对抗性样本...")
        
        if attack_method == 'fgsm':
            # 快速梯度符号法
            attack = FastGradientMethod(estimator=self.classifier, eps=eps)
        elif attack_method == 'pgd':
            # 投影梯度下降法，更强的攻击
            attack = ProjectedGradientDescent(
                estimator=self.classifier, 
                eps=eps,
                eps_step=0.5,
                max_iter=20
            )
        else:
            raise ValueError(f"不支持的攻击方法: {attack_method}")
        
        # 生成对抗样本
        adv_array = attack.generate(x=img_array)
        return adv_array
    
    def predict_and_decode(self, img_array):
        """预测图像类别并解码结果"""
        preds = self.model.predict(img_array)
        # 使用动态加载的解码函数
        try:
            return decode_predictions(preds, top=3)[0]
        except Exception as e:
            print(f"解码预测结果时出错: {e}")
            print("使用默认类别标签...")
            # 返回默认的预测结果格式
            return [(i, f"类别_{i}", pred) for i, pred in enumerate(preds[0][:3])]
    
    def visualize_results(self, original_img, img_array, adv_array):
        """可视化原始图片、扰动和对抗样本"""
        # 获取预测结果
        pred_original = self.predict_and_decode(img_array)
        pred_adv = self.predict_and_decode(adv_array)
        
        # 计算扰动
        perturbation = adv_array - img_array
        
        # 调整图像用于显示
        original_display = original_img.resize((self.target_size[0], self.target_size[1]))
        
        # 修复对抗样本显示问题：需要正确转换回图像显示格式
        # 对抗样本经过了ResNet50的预处理：preprocess_input (BGR格式，减去ImageNet均值)
        # 需要反向处理：加上均值，转换回RGB，并限制到0-255范围
        adv_display_array = adv_array[0].copy()
        
        # ResNet50预处理反向操作：加上ImageNet均值（注意：ResNet50使用BGR顺序）
        imagenet_mean = np.array([103.939, 116.779, 123.68])  # BGR顺序的均值
        adv_display_array[..., 0] += imagenet_mean[0]  # B通道
        adv_display_array[..., 1] += imagenet_mean[1]  # G通道
        adv_display_array[..., 2] += imagenet_mean[2]  # R通道
        
        # 将BGR转换回RGB（ResNet50预处理使用BGR，但PIL需要RGB）
        adv_display_array = adv_display_array[..., ::-1]  # BGR -> RGB
        
        # 限制到0-255范围并转换为uint8
        adv_display_array = np.clip(adv_display_array, 0, 255).astype('uint8')
        adv_display = Image.fromarray(adv_display_array)
        
        # 绘图
        plt.figure(figsize=(18, 6))
        
        # 原始图片
        plt.subplot(1, 3, 1)
        plt.imshow(original_display)
        pred_text = "\n".join([f"{cls}: {prob:.2f}%" for (_, cls, prob) in pred_original[:2]])
        plt.title(f"Original Image\n{pred_text}", fontsize=12)
        plt.axis('off')
        
        # 扰动（放大显示）
        plt.subplot(1, 3, 2)
        perturb_vis = perturbation[0] * 10  # 放大扰动
        perturb_vis = np.clip(perturb_vis + 127, 0, 255).astype('uint8')  # 调整到可视范围
        plt.imshow(perturb_vis)
        plt.title("Perturbation (10x)", fontsize=12)
        plt.axis('off')
        
        # 对抗样本
        plt.subplot(1, 3, 3)
        plt.imshow(adv_display)
        pred_adv_text = "\n".join([f"{cls}: {prob:.2f}%" for (_, cls, prob) in pred_adv[:2]])
        plt.title(f"Adversarial Sample\n{pred_adv_text}", fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细预测结果
        print("\n原始图片预测结果:")
        for i, (_, cls, prob) in enumerate(pred_original):
            print(f"{i+1}. {cls}: {prob*100:.2f}%")
            
        print("\n对抗样本预测结果:")
        for i, (_, cls, prob) in enumerate(pred_adv):
            print(f"{i+1}. {cls}: {prob*100:.2f}%")
    
    def run_demo(self, image_path, attack_method='pgd', eps=2.0):
        """运行完整演示流程"""
        if not os.path.exists(image_path):
            print(f"错误：图片路径不存在 - {image_path}")
            return
        
        # 预处理图片
        img_array, original_img = self.preprocess_image(image_path)
        print(f"已加载图片：{image_path}，尺寸调整为{self.target_size}")
        
        # 测试原始图片的预测
        print("测试原始图片预测...")
        pred_original = self.predict_and_decode(img_array)
        print("原始图片预测结果:", pred_original)
        
        # 生成对抗样本
        adv_array = self.generate_adversarial_samples(img_array, attack_method, eps)
        
        # 展示结果
        self.visualize_results(original_img, img_array, adv_array)

if __name__ == "__main__":
    # 首先加载ImageNet类别
    print("正在加载ImageNet类别...")
    load_imagenet_classes()
    
    # 可以修改为320x320或其他更大尺寸
    target_size = (224, 224)  # 推荐224x224，这是ResNet50的标准输入尺寸
    #target_size = (320, 320)  # 也可以使用更大的尺寸
    
    # 初始化演示器
    demo = LargeImageAdversarialDemo(target_size=target_size)
    
    # 替换为你的本地图片路径
    # 推荐使用清晰的物体图片，如猫、狗、汽车、飞机等
    image_path = "5.jpg"  # 使用当前目录下的图片
    
    # 运行演示
    # attack_method可选'fgsm'（快速）或'pgd'（更强）
    # eps控制扰动强度，值越大效果越明显但扰动越可见（建议0.5-4之间）
    # 注意：较小的eps值会产生更隐形的对抗样本，但攻击效果可能减弱
    # 较大的eps值会产生更明显的扰动，但攻击效果更强
    
    # 选项1：强攻击（扰动可见，适合教学演示）
    demo.run_demo(image_path, attack_method='pgd', eps=2.0)
    
    # 选项2：中等攻击（平衡扰动和隐形性）
    #demo.run_demo(image_path, attack_method='pgd', eps=1.0)
    
    # 选项3：弱攻击（更隐形但攻击效果可能减弱）
    #demo.run_demo(image_path, attack_method='pgd', eps=0.1)
    
    # 选项3：弱攻击（更隐形但攻击效果可能减弱）
    # demo.run_demo(image_path, attack_method='pgd', eps=0.5)
    
    # 选项4：使用FGSM方法（通常产生更隐形的扰动）
    # demo.run_demo(image_path, attack_method='fgsm', eps=0.03)
    
    # 选项5：调整扰动可视化放大倍数（在visualize_results方法中修改）
    # 将 perturb_vis = perturbation[0] * 10 改为 perturb_vis = perturbation[0] * 5 或更小
