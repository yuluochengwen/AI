import requests
import json

def call_zhipu_api(messages, model="deepseek-chat"):
    """
    调用智谱AI对话补全API
    
    Args:
        messages: 消息列表，格式如 [{"role": "user", "content": "你好"}]
        model: 模型ID，默认为 deepseek-chat
    
    Returns:
        API响应结果
    """
    # API端点
    url = f"https://open.bigmodel.cn/api/paas/v3/model-api/{model}/invoke"
    
    # 您的API Key
    api_key = "dff70369b03845608cc5653b337963d7.FPghIducrJpSXPjZ"
    
    # 请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 请求体
    payload = {
        "model": model,
        "messages": messages,
        # 可选参数
        "temperature": 0.7,  # 控制随机性 (0-1)
        "top_p": 0.9,       # 控制多样性 (0-1)
        "max_tokens": 1024  # 最大生成长度
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # 检查HTTP错误
        
        result = response.json()
        
        # 检查API调用是否成功
        if result.get("code") == 200 and result.get("success"):
            return result["data"]["choices"][0]["message"]["content"]
        else:
            print(f"API调用失败: {result.get('msg', '未知错误')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    # 构造对话消息
    messages = [
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ]
    
    # 调用API
    response = call_zhipu_api(messages)
    
    if response:
        print("AI回复:", response)
    else:
        print("调用失败")