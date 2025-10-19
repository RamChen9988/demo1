"""
人工智能安全基础 - 第四课：黑盒攻击与对抗训练实战
环境要求：Python 3.8+, TensorFlow 2.x, Adversarial-Robustness-Toolbox (ART)
安装命令：pip install tensorflow adversarial-robustness-toolbox matplotlib
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod
from art.defences.trainer import AdversarialTrainer
from tensorflow.keras import layers, models
import os

# 设置随机种子保证结果可重现
np.random.seed(42)
tf.random.set_seed(42)

print("=== AI安全基础 - 黑盒攻击与对抗训练实战 ===\n")

class AISecurityDemo:
    def __init__(self):
        self.target_model = None  # 黑盒目标模型（模拟真实API）
        self.substitute_model = None  # 替代模型（攻击者自建）
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        
    def load_data(self):
        """
        加载MNIST数据集作为演示数据
        在实际应用中可替换为其他图像或数据
        """
        print("1. 正在加载MNIST数据集...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # 数据预处理
        x_train = x_train.astype('float32').reshape(-1, 28, 28, 1) / 255.0
        x_test = x_test.astype('float32').reshape(-1, 28, 28, 1) / 255.0
        
        # 转换为one-hot编码
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        
        # 只取部分数据加速演示
        x_train, y_train = x_train[:5000], y_train[:5000]
        x_test, y_test = x_test[:1000], y_test[:1000]
        
        print(f"   训练集: {x_train.shape[0]} 样本")
        print(f"   测试集: {x_test.shape[0]} 样本")
        return x_train, y_train, x_test, y_test
    
    def create_model(self, name="model"):
        """
        创建一个简单的CNN模型
        参数name: 模型名称，用于区分不同模型
        """
        print(f"2. 创建{name}...")
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def train_models(self):
        """训练目标模型和替代模型"""
        print("3. 训练模型...")
        
        # 训练目标模型（模拟真实部署的模型）
        print("   a. 训练目标模型（黑盒）...")
        self.target_model = self.create_model("目标模型")
        self.target_model.fit(self.x_train, self.y_train, 
                             epochs=3, batch_size=32, verbose=0)
        
        # 评估目标模型
        target_loss, target_acc = self.target_model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"      目标模型准确率: {target_acc:.4f}")
        
        # 训练替代模型（攻击者自己训练的模型）
        print("   b. 训练替代模型（白盒）...")
        self.substitute_model = self.create_model("替代模型")
        self.substitute_model.fit(self.x_train, self.y_train, 
                                 epochs=3, batch_size=32, verbose=0)
        
        # 评估替代模型
        substitute_loss, substitute_acc = self.substitute_model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"      替代模型准确率: {substitute_acc:.4f}")
    
    def whitebox_attack(self):
        """在替代模型上进行白盒攻击演示"""
        print("\n4. 白盒攻击演示（在替代模型上）...")
        
        # 使用ART包装替代模型
        classifier = KerasClassifier(model=self.substitute_model, use_logits=False)
        
        #调整eps参数（攻击强度），观察攻击效果变化
        # 创建FGSM攻击器
        attack = FastGradientMethod(estimator=classifier, eps=0.3)
        
        # 选择一些测试样本进行攻击
        sample_indices = np.random.choice(len(self.x_test), 100, replace=False)
        x_sample = self.x_test[sample_indices]
        y_sample = self.y_test[sample_indices]
        
        # 生成对抗样本
        print("   正在生成对抗样本...")
        x_adv = attack.generate(x=x_sample)
        
        # 评估攻击效果
        original_predictions = self.substitute_model.predict(x_sample)
        original_accuracy = np.mean(np.argmax(original_predictions, axis=1) == np.argmax(y_sample, axis=1))
        
        adv_predictions = self.substitute_model.predict(x_adv)
        adv_accuracy = np.mean(np.argmax(adv_predictions, axis=1) == np.argmax(y_sample, axis=1))
        
        print(f"   原始准确率: {original_accuracy:.4f}")
        print(f"   攻击后准确率: {adv_accuracy:.4f}")
        print(f"   攻击成功率: {1 - adv_accuracy:.4f}")
        
        return x_sample, x_adv, sample_indices
    
    def blackbox_attack(self, x_sample, x_adv, sample_indices):
        """黑盒迁移攻击演示"""
        print("\n5. 黑盒攻击演示（迁移攻击）...")
        
        # 使用白盒攻击生成的对抗样本攻击目标模型（黑盒）
        y_sample = self.y_test[sample_indices]
        
        # 评估对抗样本对目标模型的效果
        target_original_predictions = self.target_model.predict(x_sample)
        target_original_accuracy = np.mean(
            np.argmax(target_original_predictions, axis=1) == np.argmax(y_sample, axis=1))
        
        target_adv_predictions = self.target_model.predict(x_adv)
        target_adv_accuracy = np.mean(
            np.argmax(target_adv_predictions, axis=1) == np.argmax(y_sample, axis=1))
        
        print(f"   目标模型原始准确率: {target_original_accuracy:.4f}")
        print(f"   目标模型攻击后准确率: {target_adv_accuracy:.4f}")
        print(f"   黑盒迁移攻击成功率: {1 - target_adv_accuracy:.4f}")
        
        return target_adv_accuracy
    
    def adversarial_training(self):
        """对抗训练演示"""
        print("\n6. 对抗训练防御演示...")
        
        # 创建新模型进行对抗训练
        print("   a. 创建待加固模型...")
        robust_model = self.create_model("加固模型")
        
        # 使用ART包装模型
        robust_classifier = KerasClassifier(model=robust_model, use_logits=False)
        
        # 创建攻击器用于生成训练用的对抗样本
        # 调整eps参数（攻击强度），观察攻击效果变化
        attack = FastGradientMethod(estimator=robust_classifier, eps=0.3)
        
        # 创建对抗训练器
        # 尝试修改ratio参数（对抗样本比例），观察防御效果变化
        print("   b. 开始对抗训练...")
        trainer = AdversarialTrainer(robust_classifier, attack, ratio=0.5)
        
        # 进行对抗训练（简化版，实际需要更多epoch）
        trainer.fit(self.x_train, self.y_train, batch_size=32, nb_epochs=2)
        
        # 评估加固后模型
        print("   c. 评估加固模型...")
        robust_loss, robust_acc = robust_model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"      加固模型在干净数据上的准确率: {robust_acc:.4f}")
        
        return robust_model, robust_classifier
    
    def test_robustness(self, original_model, robust_model, robust_classifier):
        """测试模型鲁棒性"""
        print("\n7. 鲁棒性对比测试...")
        
        # 调整eps参数（攻击强度），观察攻击效果变化
        # 创建更强的攻击器
        strong_attack = FastGradientMethod(estimator=robust_classifier, eps=0.3)
        
        # 选择测试样本
        sample_indices = np.random.choice(len(self.x_test), 200, replace=False)
        x_sample = self.x_test[sample_indices]
        y_sample = self.y_test[sample_indices]
        
        # 生成对抗样本
        x_adv_strong = strong_attack.generate(x=x_sample)
        
        # 测试原始模型
        original_adv_predictions = original_model.predict(x_adv_strong)
        original_robust_accuracy = np.mean(
            np.argmax(original_adv_predictions, axis=1) == np.argmax(y_sample, axis=1))
        
        # 测试加固模型
        robust_adv_predictions = robust_model.predict(x_adv_strong)
        robust_robust_accuracy = np.mean(
            np.argmax(robust_adv_predictions, axis=1) == np.argmax(y_sample, axis=1))
        
        print(f"   原始模型在面对攻击时的准确率: {original_robust_accuracy:.4f}")
        print(f"   加固模型在面对攻击时的准确率: {robust_robust_accuracy:.4f}")
        print(f"   鲁棒性提升: {robust_robust_accuracy - original_robust_accuracy:.4f}")
    
    def visualize_results(self, x_sample, x_adv, sample_indices):
        """可视化攻击结果"""
        print("\n8. 生成可视化结果...")
        
        # 选择几个样本进行可视化
        n_display = 5
        display_indices = sample_indices[:n_display]
        
        plt.figure(figsize=(15, 4))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        for i, idx in enumerate(display_indices):
            # 原始图像
            plt.subplot(2, n_display, i + 1)
            plt.imshow(self.x_test[idx].squeeze(), cmap='gray')
            original_pred = np.argmax(self.target_model.predict(self.x_test[idx:idx+1]))
            true_label = np.argmax(self.y_test[idx])
            plt.title(f'原始\n真实:{true_label}, 预测:{original_pred}')
            plt.axis('off')
            
            # 对抗样本
            adv_idx = np.where(sample_indices == idx)[0][0]
            plt.subplot(2, n_display, i + n_display + 1)
            plt.imshow(x_adv[adv_idx].squeeze(), cmap='gray')
            adv_pred = np.argmax(self.target_model.predict(x_adv[adv_idx:adv_idx+1]))
            plt.title(f'对抗样本\n预测:{adv_pred}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('attack_comparison.png', dpi=150, bbox_inches='tight')
        print("   可视化结果已保存为 'attack_comparison.png'")
        
        # 显示图像差异
        plt.figure(figsize=(15, 3))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        for i, idx in enumerate(display_indices[:3]):
            adv_idx = np.where(sample_indices == idx)[0][0]
            perturbation = x_adv[adv_idx] - self.x_test[idx]
            
            plt.subplot(1, 3, i + 1)
            # 将扰动放大以便观察
            plt.imshow(perturbation.squeeze() * 10, cmap='RdBu_r', vmin=-1, vmax=1)
            plt.title(f'扰动 (放大10倍)')
            plt.colorbar()
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('perturbation.png', dpi=150, bbox_inches='tight')
        print("   扰动可视化已保存为 'perturbation.png'")

def main():
    """主函数：执行完整的攻防演示流程"""
    demo = AISecurityDemo()
    
    # 步骤1: 训练模型
    demo.train_models()
    
    # 步骤2: 白盒攻击
    x_sample, x_adv, sample_indices = demo.whitebox_attack()
    
    # 步骤3: 黑盒攻击
    target_adv_accuracy = demo.blackbox_attack(x_sample, x_adv, sample_indices)
    
    # 步骤4: 对抗训练防御
    robust_model, robust_classifier = demo.adversarial_training()
    
    # 步骤5: 鲁棒性测试
    demo.test_robustness(demo.target_model, robust_model, robust_classifier)
    
    # 步骤6: 可视化
    demo.visualize_results(x_sample, x_adv, sample_indices)
    
    print("\n=== 演示完成 ===")
    print("总结:")
    print("1. 白盒攻击成功率很高（在替代模型上）")
    print("2. 黑盒迁移攻击也有明显效果")
    print("3. 对抗训练显著提升了模型鲁棒性")
    print("4. 查看生成的PNG文件观察攻击效果")

if __name__ == "__main__":
    main()