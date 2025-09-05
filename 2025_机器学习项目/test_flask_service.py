#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask服务测试脚本
"""

import requests
import json
import time

def test_flask_service():
    """测试Flask服务是否可用"""
    print("🧪 测试Flask服务连接...")
    
    # 测试基础连接
    try:
        response = requests.get("http://localhost:5001/", timeout=5)
        print(f"✅ Flask服务基础连接成功 (状态码: {response.status_code})")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Flask服务未启动或连接失败")
        return False
    except Exception as e:
        print(f"❌ Flask服务测试异常: {e}")
        return False

def test_model_info():
    """测试模型信息接口"""
    print("🧪 测试模型信息接口...")
    
    try:
        response = requests.get("http://localhost:5001/model_info", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ 模型信息接口成功")
                print(f"📋 模型文件: {data.get('model_name', '未知')}")
                print(f"📋 模型类型: {data.get('model_type', '未知')}")
                return True
            else:
                print(f"⚠️ 模型加载失败: {data.get('error', '未知错误')}")
                return False
        else:
            print(f"❌ 模型信息接口失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 模型信息接口异常: {e}")
        return False

def test_prediction_simple():
    """测试简单的心脏病预测"""
    print("🧪 测试心脏病预测接口...")
    
    # 简化的测试数据
    test_data = {
        "Age": 50,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 140,
        "Cholesterol": 250,
        "FastingBS": "否",
        "RestingECG": "Normal",
        "MaxHR": 140,
        "ExerciseAngina": "是",
        "Oldpeak": 1.5,
        "ST_Slope": "Flat"
    }
    
    try:
        response = requests.post(
            "http://localhost:5001/predict",
            headers={'Content-Type': 'application/json'},
            json=test_data,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ 心脏病预测成功")
                print(f"🎯 预测结果: {'心脏病风险' if data.get('prediction') == 1 else '健康'}")
                print(f"📊 健康概率: {data.get('probability', {}).get('healthy', 0):.2%}")
                print(f"📊 心脏病概率: {data.get('probability', {}).get('heart_disease', 0):.2%}")
                print(f"🏷️ 风险等级: {data.get('risk_level', '未知')}")
                return True
            else:
                print(f"⚠️ 预测失败: {data.get('error', '未知错误')}")
                return False
        else:
            print(f"❌ 预测接口失败: HTTP {response.status_code}")
            print(f"错误内容: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 预测接口异常: {e}")
        return False

def main():
    """主测试函数"""
    print("🔍 Flask服务综合测试")
    print("="*50)
    
    results = []
    
    # 测试Flask基础服务
    flask_ok = test_flask_service()
    results.append(("Flask基础服务", flask_ok))
    
    if flask_ok:
        # 测试模型信息
        model_ok = test_model_info()
        results.append(("模型信息接口", model_ok))
        
        # 测试预测功能
        predict_ok = test_prediction_simple()
        results.append(("心脏病预测接口", predict_ok))
    
    # 汇总结果
    print("\n" + "="*50)
    print("📊 Flask服务测试结果")
    print("="*50)
    
    for name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{name}: {status}")
    
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    print(f"\n📈 成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")

if __name__ == "__main__":
    main()
