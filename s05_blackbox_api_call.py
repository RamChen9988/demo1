"""
极简模型窃取攻击演示
目标：通过查询目标模型的API来窃取其功能
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

print("=== 模型窃取攻击演示 ===\n")

class ModelStealingDemo:
    def __init__(self):
        self.target_model = None  # 要窃取的目标模型（黑盒）
        self.substitute_model = None  # 替代模型（攻击者训练的）
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        
    def load_data(self):
        """加载MNIST数据集"""
        print("1. 加载MNIST数据集...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # 数据预处理
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        
        # 重塑数据格式
        x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28, 1)
        x_test = np.expand_dims(x_test, -1)    # (10000, 28, 28, 1)
        
        # 转换为one-hot编码
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        # 为了加速演示，只使用部分数据
        x_train, y_train = x_train[:10000], y_train[:10000]
        x_test, y_test = x_test[:2000], y_test[:2000]
        
        print(f"   训练样本: {len(x_train)}")
        print(f"   测试样本: {len(x_test)}")
        return x_train, y_train, x_test, y_test
    
    def create_target_model(self):
        """创建并训练目标模型（模拟被窃取的商业模型）"""
        print("2. 创建目标模型...")
        
        model = keras.Sequential([
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax")
        ])
        
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )
        
        # 训练目标模型
        print("   训练目标模型中...")
        model.fit(self.x_train, self.y_train, 
                 batch_size=128, epochs=5, verbose=0, 
                 validation_split=0.1)
        
        # 评估目标模型
        target_loss, target_acc = model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"   目标模型测试准确率: {target_acc:.4f}")
        
        return model
    
    def blackbox_api_call(self, data):
        """
        模拟黑盒API调用
        在实际攻击中，攻击者只能通过这个接口获取预测结果
        """
        # 返回目标模型的预测概率（软标签）
        return self.target_model.predict(data, verbose=0)
    
    def create_substitute_model(self):
        """创建替代模型（攻击者使用的模型结构）"""
        print("3. 创建替代模型...")
        
        # 注意：替代模型的结构可以与目标模型不同
        model = keras.Sequential([
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])
        
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam", 
            metrics=["accuracy"]
        )
        
        return model
    
    def steal_model(self, query_budget=2000):
        """
        执行模型窃取攻击
        query_budget: 攻击者的查询预算（查询次数）
        """
        print(f"4. 开始模型窃取攻击 (查询预算: {query_budget})...")
        
        # 步骤1: 攻击者收集公共数据（这里使用测试集模拟）
        print("   a. 收集公共查询数据...")
        query_indices = np.random.choice(len(self.x_test), query_budget, replace=False)
        query_data = self.x_test[query_indices]
        
        # 步骤2: 查询目标模型API获取"知识"
        print("   b. 查询目标模型API...")
        stolen_knowledge = self.blackbox_api_call(query_data)
        
        # 步骤3: 用窃取的知识训练替代模型
        print("   c. 训练替代模型...")
        history = self.substitute_model.fit(
            query_data, stolen_knowledge,
            batch_size=128,
            epochs=10,
            verbose=0,
            validation_split=0.1
        )
        
        print("   ✓ 模型窃取完成")
        return history
    
    def evaluate_attack(self):
        """评估窃取效果"""
        print("5. 评估窃取效果...")
        
        # 评估目标模型
        target_loss, target_acc = self.target_model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # 评估替代模型（在真实标签上）
        substitute_loss, substitute_acc = self.substitute_model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # 评估功能相似性（用目标模型的输出来评估替代模型）
        target_predictions = self.target_model.predict(self.x_test, verbose=0)
        substitute_predictions = self.substitute_model.predict(self.x_test, verbose=0)
        
        # 计算预测一致性（两个模型预测相同的比例）
        target_labels = np.argmax(target_predictions, axis=1)
        substitute_labels = np.argmax(substitute_predictions, axis=1)
        agreement = np.mean(target_labels == substitute_labels)
        
        print(f"   目标模型准确率: {target_acc:.4f}")
        print(f"   替代模型准确率: {substitute_acc:.4f}")
        print(f"   模型间预测一致性: {agreement:.4f}")
        
        return target_acc, substitute_acc, agreement
    
    def compare_predictions(self, num_samples=5):
        """对比两个模型的预测结果"""
        print("6. 预测结果对比...")
        
        # 随机选择几个样本
        indices = np.random.choice(len(self.x_test), num_samples)
        
        plt.figure(figsize=(12, 3 * num_samples))
        
        for i, idx in enumerate(indices):
            image = self.x_test[idx]
            true_label = np.argmax(self.y_test[idx])
            
            # 获取两个模型的预测
            target_pred = self.target_model.predict(image[np.newaxis, ...], verbose=0)[0]
            substitute_pred = self.substitute_model.predict(image[np.newaxis, ...], verbose=0)[0]
            
            target_label = np.argmax(target_pred)
            substitute_label = np.argmax(substitute_pred)
            
            # 简化字体设置，避免中文显示问题
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
            plt.rcParams['axes.unicode_minus'] = False

            # 绘制图像和预测结果
            plt.subplot(num_samples, 2, 2*i + 1)
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title(f'样本 {i+1}\n真实标签: {true_label}\n目标模型预测: {target_label} (置信度: {target_pred[target_label]:.2f})')
            plt.axis('off')
            
            plt.subplot(num_samples, 2, 2*i + 2)
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title(f'样本 {i+1}\n真实标签: {true_label}\n替代模型预测: {substitute_label} (置信度: {substitute_pred[substitute_label]:.2f})')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        plt.savefig('model_stealing_comparison.png', dpi=150, bbox_inches='tight')
        print("   ✓ 对比结果已保存为 'model_stealing_comparison.png'")

def main():
    """主函数"""
    demo = ModelStealingDemo()
    
    # 步骤1: 创建目标模型（模拟商业API）
    demo.target_model = demo.create_target_model()
    
    # 步骤2: 创建替代模型
    demo.substitute_model = demo.create_substitute_model()
    
    # 步骤3: 执行模型窃取攻击
    demo.steal_model(query_budget=2000)
    
    # 步骤4: 评估攻击效果
    target_acc, substitute_acc, agreement = demo.evaluate_attack()
    
    # 步骤5: 可视化对比
    demo.compare_predictions()
    
    print(f"\n=== 攻击完成 ===")
    print(f"目标模型准确率: {target_acc:.4f}")
    print(f"替代模型准确率: {substitute_acc:.4f}") 
    print(f"模型功能相似度: {agreement:.4f}")
    print(f"窃取效果: {'成功' if agreement > 0.8 else '部分成功' if agreement > 0.6 else '失败'}")

if __name__ == "__main__":
    main()