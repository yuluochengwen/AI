#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API连接测试脚本
测试BigModel API和心脏病预测系统的连接状态
"""

import os
import sys
import requests
import json
from datetime import datetime

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ 成功加载 .env 文件")
except ImportError:
    print("⚠️ 未安装 python-dotenv，尝试直接读取环境变量")
except Exception as e:
    print(f"⚠️ 加载 .env 文件时出错: {e}")

def test_bigmodel_api():
    """测试BigModel对话API连接"""
    print("\n" + "="*50)
    print("🧪 测试 BigModel API 连接")
    print("="*50)
    
    # 获取API配置
    api_key = os.getenv("BIGMODEL_API_KEY")
    api_url = os.getenv("BIGMODEL_CHAT_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
    model = os.getenv("BIGMODEL_MODEL", "glm-4-flash")
    
    print(f"📋 API URL: {api_url}")
    print(f"📋 模型: {model}")
    print(f"📋 API Key: {'已配置' if api_key else '未配置'}")
    
    if not api_key:
        print("❌ BigModel API Key 未配置")
        return False
    
    # 构造测试请求
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': model,
        'messages': [
            {
                'role': 'system',
                'content': '你是一个AI助手，简洁地回答问题。'
            },
            {
                'role': 'user',
                'content': '请回复"API连接测试成功"'
            }
        ]
    }
    
    try:
        print("🚀 发送测试请求...")
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"📊 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                reply = data['choices'][0]['message']['content']
                print(f"✅ API调用成功！")
                print(f"🤖 AI回复: {reply}")
                return True
            else:
                print(f"⚠️ 响应格式异常: {data}")
                return False
        else:
            print(f"❌ API调用失败: HTTP {response.status_code}")
            print(f"📄 错误信息: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ 请求超时 - 网络连接可能有问题")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ 连接错误 - 无法连接到API服务器")
        return False
    except Exception as e:
        print(f"❌ 发生未知错误: {str(e)}")
        return False

def test_heart_prediction_api():
    """测试心脏病预测API连接"""
    print("\n" + "="*50)
    print("🧪 测试心脏病预测API连接")
    print("="*50)
    
    api_url = "http://localhost:5001/predict"
    
    # 构造测试数据
    test_data = {
        "Age": 45,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 120,
        "Cholesterol": 200,
        "FastingBS": "否",
        "RestingECG": "Normal",
        "MaxHR": 150,
        "ExerciseAngina": "否",
        "Oldpeak": 1.0,
        "ST_Slope": "Up"
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        print("🚀 发送心脏病预测测试请求...")
        print(f"📋 API URL: {api_url}")
        print(f"📋 测试数据: {json.dumps(test_data, ensure_ascii=False, indent=2)}")
        
        response = requests.post(
            api_url,
            headers=headers,
            json=test_data,
            timeout=30
        )
        
        print(f"📊 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ 心脏病预测API调用成功！")
                print(f"🎯 预测结果: {'心脏病风险' if data.get('prediction') == 1 else '健康'}")
                print(f"📊 置信度: {data.get('confidence', 0):.2%}")
                print(f"🏷️ 风险等级: {data.get('risk_level', '未知')}")
                return True
            else:
                print(f"⚠️ 预测失败: {data.get('error', '未知错误')}")
                return False
        else:
            print(f"❌ API调用失败: HTTP {response.status_code}")
            print(f"📄 错误信息: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 连接错误 - 心脏病预测服务可能未启动")
        print("💡 请先运行: python app_heart.py")
        return False
    except requests.exceptions.Timeout:
        print("❌ 请求超时")
        return False
    except Exception as e:
        print(f"❌ 发生未知错误: {str(e)}")
        return False

def test_heart_ai_chat_api():
    """测试心脏病预测系统的AI聊天功能"""
    print("\n" + "="*50)
    print("🧪 测试心脏病系统AI聊天API")
    print("="*50)
    
    api_url = "http://localhost:5001/ai_chat"
    
    test_message = {
        "message": "什么是心脏病？",
        "history": []
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        print("🚀 发送AI聊天测试请求...")
        print(f"📋 API URL: {api_url}")
        print(f"📋 测试消息: {test_message['message']}")
        
        response = requests.post(
            api_url,
            headers=headers,
            json=test_message,
            timeout=30
        )
        
        print(f"📊 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ AI聊天API调用成功！")
                print(f"🤖 AI回复: {data.get('reply', '无回复')[:200]}...")
                return True
            else:
                print(f"⚠️ AI聊天失败: {data.get('error', '未知错误')}")
                return False
        else:
            print(f"❌ API调用失败: HTTP {response.status_code}")
            print(f"📄 错误信息: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 连接错误 - 心脏病预测服务可能未启动")
        return False
    except Exception as e:
        print(f"❌ 发生未知错误: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("🔍 开始API连接测试")
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # 测试BigModel API
    bigmodel_success = test_bigmodel_api()
    results.append(("BigModel API", bigmodel_success))
    
    # 测试心脏病预测API
    heart_predict_success = test_heart_prediction_api()
    results.append(("心脏病预测API", heart_predict_success))
    
    # 测试AI聊天API
    heart_chat_success = test_heart_ai_chat_api()
    results.append(("AI聊天API", heart_chat_success))
    
    # 总结测试结果
    print("\n" + "="*50)
    print("📊 测试结果总结")
    print("="*50)
    
    for name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{name}: {status}")
    
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    print(f"\n📈 总体成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("🎉 所有API测试均通过！")
    else:
        print("⚠️ 部分API测试失败，请检查相关配置")

if __name__ == "__main__":
    main()
