import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod, DeepFool, CarliniL2Method
from art.utils import to_categorical as art_to_categorical

# 设置中文字体，确保中文正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class AdversarialImageGenerator:
    def __init__(self, dataset='mnist', attack_method='fgsm'):
        """初始化对抗性图像生成器"""
        self.dataset = dataset
        self.attack_method = attack_method
        self.model = None
        self.classifier = None
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None
        self.x_test_adv = None
        
    def load_dataset(self):
        """加载数据集（MNIST或CIFAR-10）"""
        print(f"加载{self.dataset}数据集...")
        if self.dataset == 'mnist':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
            # 归一化并添加通道维度
            self.x_train = self.x_train.astype(np.float32) / 255.0
            self.x_test = self.x_test.astype(np.float32) / 255.0
            self.x_train = np.expand_dims(self.x_train, axis=-1)
            self.x_test = np.expand_dims(self.x_test, axis=-1)
            self.class_names = [str(i) for i in range(10)]  # 数字0-9
        elif self.dataset == 'cifar10':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
            # 归一化
            self.x_train = self.x_train.astype(np.float32) / 255.0
            self.x_test = self.x_test.astype(np.float32) / 255.0
            self.y_train = self.y_train.flatten()
            self.y_test = self.y_test.flatten()
            # CIFAR-10类别名称
            self.class_names = ['飞机', '汽车', '鸟', '猫', '鹿', 
                               '狗', '青蛙', '马', '船', '卡车']
        
        print(f"数据集加载完成 - 训练集: {self.x_train.shape}, 测试集: {self.x_test.shape}")
        return self
    
    def build_model(self):
        """根据数据集构建相应的CNN模型"""
        print("构建模型...")
        if self.dataset == 'mnist':
            # MNIST模型
            self.model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(10, activation='softmax')
            ])
        else:
            # CIFAR-10模型
            self.model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(10, activation='softmax')
            ])
        
        # 编译模型
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self
    
    def train_model(self, epochs=5, batch_size=128):
        """训练模型"""
        print(f"开始训练模型，共{epochs}个epochs...")
        start_time = time.time()
        
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        train_time = time.time() - start_time
        print(f"模型训练完成，耗时: {train_time:.2f}秒")
        
        # 评估模型在测试集上的表现
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"模型在测试集上的准确率: {test_acc:.4f}")
        
        # 创建ART分类器
        self.classifier = KerasClassifier(model=self.model, clip_values=(0, 1))
        return self
    
    def generate_adversarial_samples(self, eps=0.1, max_iter=100):
        """生成对抗性样本"""
        print(f"使用{self.attack_method}方法生成对抗性样本...")
        start_time = time.time()
        
        # 根据选择的攻击方法创建攻击实例
        if self.attack_method == 'fgsm':
            # 快速梯度符号法
            attack = FastGradientMethod(estimator=self.classifier, eps=eps)
        elif self.attack_method == 'deepfool':
            # DeepFool攻击
            attack = DeepFool(estimator=self.classifier, max_iter=max_iter)
        elif self.attack_method == 'carlini':
            # Carlini-Wagner攻击
            attack = CarliniL2Method(classifier=self.classifier, max_iter=max_iter)
        else:
            raise ValueError(f"不支持的攻击方法: {self.attack_method}")
        
        # 生成对抗样本
        self.x_test_adv = attack.generate(x=self.x_test[:100])  # 只使用前100个测试样本以节省时间
        
        # 评估对抗样本的效果
        predictions_adv = self.classifier.predict(self.x_test_adv)
        accuracy_adv = np.sum(np.argmax(predictions_adv, axis=1) == self.y_test[:100]) / len(self.y_test[:100])
        attack_time = time.time() - start_time
        
        print(f"对抗样本生成完成，耗时: {attack_time:.2f}秒")
        print(f"模型在对抗样本上的准确率: {accuracy_adv:.4f}")
        return self
    
    def calculate_perturbation_stats(self):
        """计算扰动的统计信息"""
        if self.x_test_adv is None:
            raise ValueError("请先生成对抗性样本")
            
        # 计算扰动
        perturbations = self.x_test_adv - self.x_test[:100]
        avg_perturbation = np.mean(np.abs(perturbations))
        max_perturbation = np.max(np.abs(perturbations))
        
        print(f"扰动统计:")
        print(f"  平均扰动强度: {avg_perturbation:.6f}")
        print(f"  最大扰动强度: {max_perturbation:.6f}")
        return avg_perturbation, max_perturbation
    
    def visualize_results(self, num_samples=5):
        """可视化原始图像、对抗样本及其预测结果"""
        if self.x_test_adv is None:
            raise ValueError("请先生成对抗性样本")
            
        # 获取预测结果
        predictions_original = self.classifier.predict(self.x_test[:100])
        predictions_adv = self.classifier.predict(self.x_test_adv)
        
        # 创建图像
        plt.figure(figsize=(30, 5 * num_samples))
        
        for i in range(num_samples):
            # 选择样本
            idx = np.random.randint(0, len(self.x_test_adv))
            
            # 原始图像
            plt.subplot(num_samples, 3, i * 3 + 1)
            if self.dataset == 'mnist':
                plt.imshow(self.x_test[idx].squeeze(), cmap='gray')
            else:
                plt.imshow(self.x_test[idx])
            true_label = self.y_test[idx]
            pred_label = np.argmax(predictions_original[idx])
            plt.title(f"原始图像\n真实: {self.class_names[true_label]}\n预测: {self.class_names[pred_label]}")
            plt.axis('off')
            
            # 扰动图像
            plt.subplot(num_samples, 3, i * 3 + 2)
            perturbation = self.x_test_adv[idx] - self.x_test[idx]
            # 放大扰动以便可视化
            perturbation_vis = perturbation * 5  # 放大5倍显示
            if self.dataset == 'mnist':
                plt.imshow(perturbation_vis.squeeze(), cmap='gray')
            else:
                # 处理彩色图像的扰动显示
                perturbation_vis = np.clip(perturbation_vis + 0.5, 0, 1)  # 调整到[0,1]范围以便显示
                plt.imshow(perturbation_vis)
            plt.title("扰动 (放大显示)")
            plt.axis('off')
            
            # 对抗样本
            plt.subplot(num_samples, 3, i * 3 + 3)
            if self.dataset == 'mnist':
                plt.imshow(self.x_test_adv[idx].squeeze(), cmap='gray')
            else:
                plt.imshow(self.x_test_adv[idx])
            pred_adv_label = np.argmax(predictions_adv[idx])
            plt.title(f"对抗样本\n真实: {self.class_names[true_label]}\n预测: {self.class_names[pred_adv_label]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """主函数：演示对抗性图像生成流程"""
    try:
        # 可以修改这些参数来尝试不同的组合
        dataset = 'cifar10'  # 可选: 'mnist' 或 'cifar10'
        attack_method = 'fgsm'  # 可选: 'fgsm', 'deepfool', 'carlini'
        
        # 创建生成器实例
        generator = AdversarialImageGenerator(dataset=dataset, attack_method=attack_method)
        
        # 执行完整流程
        generator.load_dataset()\
                .build_model()\
                .train_model(epochs=5)\
                .generate_adversarial_samples(eps=0.1)\
                .calculate_perturbation_stats()
        
        # 可视化结果
        generator.visualize_results(num_samples=3)
        
    except Exception as e:
        print(f"运行过程中出错: {str(e)}")

if __name__ == "__main__":
    main()
