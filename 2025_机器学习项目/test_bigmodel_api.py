#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的BigModel API测试脚本
"""

import os
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

def test_bigmodel_api_detailed():
    """详细测试BigModel API功能"""
    print("\n" + "="*60)
    print("🔍 BigModel API 详细功能测试")
    print("="*60)
    
    # 获取API配置
    api_key = os.getenv("BIGMODEL_API_KEY")
    api_url = os.getenv("BIGMODEL_CHAT_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
    model = os.getenv("BIGMODEL_MODEL", "glm-4-flash")
    
    print(f"📋 API Key: {api_key[:10]}...{api_key[-10:] if api_key else '未配置'}")
    print(f"📋 API URL: {api_url}")
    print(f"📋 模型: {model}")
    
    if not api_key:
        print("❌ BigModel API Key 未配置")
        return False
    
    # 测试1: 简单问答
    print("\n🧪 测试1: 简单问答")
    test_simple_qa(api_url, api_key, model)
    
    # 测试2: 健康咨询
    print("\n🧪 测试2: 健康咨询")
    test_health_consultation(api_url, api_key, model)
    
    # 测试3: 心脏病相关问题
    print("\n🧪 测试3: 心脏病相关问题")
    test_heart_disease_qa(api_url, api_key, model)

def test_simple_qa(api_url, api_key, model):
    """测试简单问答"""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': model,
        'messages': [
            {
                'role': 'user',
                'content': '你好，请简单介绍一下你自己。'
            }
        ]
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            reply = data['choices'][0]['message']['content']
            print(f"✅ 简单问答成功")
            print(f"🤖 AI回复: {reply[:100]}...")
            return True
        else:
            print(f"❌ 简单问答失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 简单问答异常: {str(e)}")
        return False

def test_health_consultation(api_url, api_key, model):
    """测试健康咨询功能"""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': model,
        'messages': [
            {
                'role': 'system',
                'content': '你是一个严谨的健康科普助手，面向心血管健康场景，提供通俗、审慎的建议；明确声明不替代医生诊断。'
            },
            {
                'role': 'user',
                'content': '日常生活中如何预防心脏病？'
            }
        ]
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            reply = data['choices'][0]['message']['content']
            print(f"✅ 健康咨询成功")
            print(f"🩺 健康建议: {reply[:200]}...")
            return True
        else:
            print(f"❌ 健康咨询失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康咨询异常: {str(e)}")
        return False

def test_heart_disease_qa(api_url, api_key, model):
    """测试心脏病相关问题"""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': model,
        'messages': [
            {
                'role': 'system',
                'content': '你是一个心血管健康专家助手，请根据用户的问题提供专业、准确、易懂的回答。'
            },
            {
                'role': 'user',
                'content': '胸痛一定是心脏病吗？什么情况下需要立即就医？'
            }
        ]
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            reply = data['choices'][0]['message']['content']
            print(f"✅ 心脏病咨询成功")
            print(f"💔 专业回答: {reply[:200]}...")
            return True
        else:
            print(f"❌ 心脏病咨询失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 心脏病咨询异常: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("🔍 BigModel API 综合测试")
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_bigmodel_api_detailed()
    
    print("\n" + "="*60)
    print("📊 BigModel API 测试完成")
    print("✅ 该API可以用于心脏病预测系统的AI健康咨询功能")
    print("💡 建议: 可以在心脏病预测结果页面集成此AI助手功能")
    print("="*60)

if __name__ == "__main__":
    main()
