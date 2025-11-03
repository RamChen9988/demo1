"""
极简成员推理攻击演示
目标：判断特定数据样本是否在模型的训练集中
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

print("=== 成员推理攻击演示 ===\n")

class MembershipInferenceDemo:
    def __init__(self):
        self.target_model = None  # 目标模型（被攻击的模型）
        self.attack_model = None  # 攻击模型（判断成员关系的模型）
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
        x_train, y_train = x_train[:8000], y_train[:8000]
        x_test, y_test = x_test[:2000], y_test[:2000]
        
        print(f"   训练样本: {len(x_train)}")
        print(f"   测试样本: {len(x_test)}")
        return x_train, y_train, x_test, y_test
    
    def create_target_model(self):
        """创建并训练目标模型"""
        print("2. 创建并训练目标模型...")
        
        model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.3),  # 添加dropout减少过拟合
            layers.Dense(10, activation="softmax")
        ])
        
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )
        
        # 训练目标模型（故意让它有点过拟合，这样成员推理效果更明显）
        print("   训练目标模型中...")
        model.fit(self.x_train, self.y_train, 
                 batch_size=128, epochs=8, verbose=0, 
                 validation_split=0.1)
        
        # 评估目标模型
        train_loss, train_acc = model.evaluate(self.x_train, self.y_train, verbose=0)
        test_loss, test_acc = model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"   目标模型训练集准确率: {train_acc:.4f}")
        print(f"   目标模型测试集准确率: {test_acc:.4f}")
        print(f"   过拟合程度: {train_acc - test_acc:.4f} (越大越容易遭受成员推理攻击)")
        
        return model
    
    def get_model_predictions(self, data):
        """获取目标模型对数据的预测置信度"""
        return self.target_model.predict(data, verbose=0)
    
    def prepare_attack_data(self):
        """准备攻击模型的数据"""
        print("3. 准备攻击模型数据...")
        
        # 成员数据：目标模型的训练集
        member_data = self.x_train
        member_predictions = self.get_model_predictions(member_data)
        member_labels = np.ones(len(member_data))  # 标签1表示成员
        
        # 非成员数据：目标模型的测试集
        non_member_data = self.x_test
        non_member_predictions = self.get_model_predictions(non_member_data)
        non_member_labels = np.zeros(len(non_member_data))  # 标签0表示非成员
        
        # 合并特征和标签
        X = np.vstack([member_predictions, non_member_predictions])
        y = np.hstack([member_labels, non_member_labels])
        
        print(f"   成员样本数: {len(member_data)}")
        print(f"   非成员样本数: {len(non_member_data)}")
        print(f"   攻击模型特征维度: {X.shape[1]}")
        
        return X, y
    
    def train_attack_model(self, X, y):
        """训练攻击模型"""
        print("4. 训练攻击模型...")
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 使用随机森林作为攻击模型
        self.attack_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        
        self.attack_model.fit(X_train, y_train)
        
        # 评估攻击模型
        train_score = self.attack_model.score(X_train, y_train)
        test_score = self.attack_model.score(X_test, y_test)
        
        print(f"   攻击模型训练集准确率: {train_score:.4f}")
        print(f"   攻击模型测试集准确率: {test_score:.4f}")
        
        return X_test, y_test
    
    def analyze_prediction_patterns(self):
        """分析模型对成员和非成员数据的预测模式差异"""
        print("5. 分析预测模式差异...")
        
        # 获取成员和非成员数据的预测
        member_pred = self.get_model_predictions(self.x_train[:500])
        non_member_pred = self.get_model_predictions(self.x_test[:500])
        
        # 计算置信度统计量
        member_max_conf = np.max(member_pred, axis=1)
        non_member_max_conf = np.max(non_member_pred, axis=1)
        
        member_entropy = -np.sum(member_pred * np.log(member_pred + 1e-8), axis=1)
        non_member_entropy = -np.sum(non_member_pred * np.log(non_member_pred + 1e-8), axis=1)
        
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 可视化置信度分布
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(member_max_conf, alpha=0.7, label='成员', bins=20, density=True)
        plt.hist(non_member_max_conf, alpha=0.7, label='非成员', bins=20, density=True)
        plt.xlabel('最大预测置信度')
        plt.ylabel('密度')
        plt.title('成员 vs 非成员: 最大置信度分布')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(member_entropy, alpha=0.7, label='成员', bins=20, density=True)
        plt.hist(non_member_entropy, alpha=0.7, label='非成员', bins=20, density=True)
        plt.xlabel('预测熵')
        plt.ylabel('密度')
        plt.title('成员 vs 非成员: 预测熵分布')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        plt.savefig('membership_inference_analysis.png', dpi=150, bbox_inches='tight')
        print("   ✓ 分析结果已保存为 'membership_inference_analysis.png'")
        
        # 打印统计信息
        print(f"   成员平均最大置信度: {np.mean(member_max_conf):.4f}")
        print(f"   非成员平均最大置信度: {np.mean(non_member_max_conf):.4f}")
        print(f"   成员平均预测熵: {np.mean(member_entropy):.4f}")
        print(f"   非成员平均预测熵: {np.mean(non_member_entropy):.4f}")
    
    def demo_attack_on_samples(self, num_samples=5):
        """在具体样本上演示攻击效果"""
        print("6. 具体样本攻击演示...")
        
        # 随机选择几个成员和非成员样本
        member_indices = np.random.choice(len(self.x_train), num_samples, replace=False)
        non_member_indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(15, 6))
        
        for i, idx in enumerate(member_indices):
            image = self.x_train[idx]
            true_label = np.argmax(self.y_train[idx])
            
            # 获取预测和攻击结果
            prediction = self.get_model_predictions(image[np.newaxis, ...])[0]
            attack_result = self.attack_model.predict(prediction[np.newaxis, ...])[0]
            attack_prob = self.attack_model.predict_proba(prediction[np.newaxis, ...])[0][1]
            
            predicted_label = np.argmax(prediction)
            confidence = prediction[predicted_label]
            
            plt.subplot(2, num_samples, i + 1)
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title(f'成员样本\n真实: {true_label}, 预测: {predicted_label}\n置信度: {confidence:.2f}\n攻击判断: {attack_prob:.2f}')
            plt.axis('off')
        
        for i, idx in enumerate(non_member_indices):
            image = self.x_test[idx]
            true_label = np.argmax(self.y_test[idx])
            
            # 获取预测和攻击结果
            prediction = self.get_model_predictions(image[np.newaxis, ...])[0]
            attack_result = self.attack_model.predict(prediction[np.newaxis, ...])[0]
            attack_prob = self.attack_model.predict_proba(prediction[np.newaxis, ...])[0][1]
            
            predicted_label = np.argmax(prediction)
            confidence = prediction[predicted_label]
            
            plt.subplot(2, num_samples, i + num_samples + 1)
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title(f'非成员样本\n真实: {true_label}, 预测: {predicted_label}\n置信度: {confidence:.2f}\n攻击判断: {attack_prob:.2f}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('membership_inference_demo.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("   ✓ 演示结果已保存为 'membership_inference_demo.png'")

def main():
    """主函数"""
    demo = MembershipInferenceDemo()
    
    # 步骤1: 创建目标模型
    demo.target_model = demo.create_target_model()
    
    # 步骤2: 准备攻击数据
    X, y = demo.prepare_attack_data()
    
    # 步骤3: 训练攻击模型
    X_test, y_test = demo.train_attack_model(X, y)
    
    # 步骤4: 分析预测模式
    demo.analyze_prediction_patterns()
    
    # 步骤5: 具体样本演示
    demo.demo_attack_on_samples()
    
    print(f"\n=== 攻击完成 ===")
    print(f"攻击模型准确率: {demo.attack_model.score(X_test, y_test):.4f}")
    print("攻击效果: 成功" if demo.attack_model.score(X_test, y_test) > 0.6 else "攻击效果: 有限")

if __name__ == "__main__":
    main()
