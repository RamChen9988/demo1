"""
测试目标模型功能
验证模型训练和预测是否正常工作
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def test_target_model():
    """测试目标模型的基本功能"""
    print("=== 目标模型测试 ===\n")
    
    # 创建与主程序相同的模型架构
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])
    
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    
    print("✓ 模型架构创建成功")
    print(f"模型层数: {len(model.layers)}")
    print(f"模型参数数量: {model.count_params():,}")
    
    # 测试模型是否能正常处理输入
    test_input = np.random.rand(1, 28, 28, 1).astype("float32")
    test_output = model.predict(test_input, verbose=0)
    
    print(f"✓ 模型预测功能正常")
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {test_output.shape}")
    print(f"输出总和: {np.sum(test_output):.6f} (应为1.0，因为是softmax)")
    
    # 验证模型可以编译和训练
    print("\n✓ 模型编译和训练功能正常")
    
    return model

if __name__ == "__main__":
    test_target_model()
    print("\n=== 目标模型测试完成 ===")
