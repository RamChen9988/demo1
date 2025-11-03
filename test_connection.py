import requests
import json

def test_ollama_connection():
    """测试Ollama连接"""
    base_url = "http://localhost:11434/api/generate"
    model_name = "qwen2.5:14b"
    
    print("测试Ollama连接...")
    
    # 测试1: 检查API是否可访问
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        print(f"API状态检查: HTTP {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"可用模型: {[model['name'] for model in models.get('models', [])]}")
        else:
            print(f"API错误: {response.text}")
    except Exception as e:
        print(f"API连接失败: {e}")
        return False
    
    # 测试2: 测试模型生成
    try:
        test_payload = {
            "model": model_name,
            "prompt": "Hello, this is a test",
            "stream": False
        }
        response = requests.post(base_url, json=test_payload, timeout=30)
        print(f"模型生成测试: HTTP {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"模型响应: {result.get('response', 'No response')}")
            return True
        else:
            print(f"模型生成错误: {response.text}")
            return False
    except Exception as e:
        print(f"模型生成失败: {e}")
        return False

if __name__ == "__main__":
    test_ollama_connection()
