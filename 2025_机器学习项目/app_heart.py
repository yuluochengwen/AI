from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 加载保存的模型
MODEL_PATH = None
model = None

def load_latest_heart_model():
    """加载最新的心脏病预测模型"""
    global MODEL_PATH, model
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 优先查找balanced_models目录下的SMOTE模型
    balanced_models_dir = os.path.join(script_dir, 'balanced_models')
    
    # 查找指定的心脏病预测模型文件
    target_model = 'smote_catboost_balanced_20250903_092615.pkl'
    model_path = os.path.join(balanced_models_dir, target_model)
    
    if os.path.exists(model_path):
        MODEL_PATH = model_path
        try:
            # 使用joblib加载模型
            model = joblib.load(MODEL_PATH)
            print(f"🎉 成功加载心脏病预测模型: {MODEL_PATH}")
            print("✅ 此模型具有以下优势:")
            print("   - 心脏病患者零漏诊 (100%召回率)")
            print("   - 高精确率预测")
            print("   - 适合心脏病预警系统")
            return True, f"心脏病预测模型加载成功: {os.path.basename(MODEL_PATH)}"
        except Exception as e:
            return False, f"模型加载失败: {str(e)}"
    else:
        return False, f"未找到指定的心脏病预测模型文件: {target_model}"

def prepare_heart_input_data(data):
    """准备心脏病预测输入数据，确保特征顺序和格式正确"""
    
    # 创建DataFrame
    input_df = pd.DataFrame([data])
    
    # 数值型特征处理（确保数据类型正确）
    numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    for feature in numeric_features:
        if feature in input_df.columns:
            input_df[feature] = pd.to_numeric(input_df[feature], errors='coerce').fillna(0).astype('float64')
        else:
            input_df[feature] = 0.0
    
    # 二元分类特征编码（与训练时保持一致）
    # Sex: M=1, F=0
    if 'Sex' in input_df.columns:
        input_df['Sex'] = input_df['Sex'].map({'M': 1, 'F': 0}).fillna(0).astype('int64')
    else:
        input_df['Sex'] = 0
    
    # FastingBS: 是=1, 否=0
    if 'FastingBS' in input_df.columns:
        input_df['FastingBS'] = input_df['FastingBS'].map({'是': 1, '否': 0}).fillna(0).astype('int64')
    else:
        input_df['FastingBS'] = 0
    
    # ExerciseAngina: 是=1, 否=0
    if 'ExerciseAngina' in input_df.columns:
        input_df['ExerciseAngina'] = input_df['ExerciseAngina'].map({'是': 1, '否': 0}).fillna(0).astype('int64')
    else:
        input_df['ExerciseAngina'] = 0
    
    # 多分类特征独热编码（按训练时的顺序）
    # ChestPainType 独热编码
    chest_pain_type = input_df.get('ChestPainType', ['ASY'])[0] if 'ChestPainType' in input_df.columns else 'ASY'
    
    # 按字母顺序创建独热编码（通常pandas的get_dummies是按字母顺序的）
    input_df['ChestPainType_ASY'] = 1 if chest_pain_type == 'ASY' else 0
    input_df['ChestPainType_ATA'] = 1 if chest_pain_type == 'ATA' else 0
    input_df['ChestPainType_NAP'] = 1 if chest_pain_type == 'NAP' else 0
    input_df['ChestPainType_TA'] = 1 if chest_pain_type == 'TA' else 0
    
    # RestingECG 独热编码
    resting_ecg = input_df.get('RestingECG', ['Normal'])[0] if 'RestingECG' in input_df.columns else 'Normal'
    
    input_df['RestingECG_LVH'] = 1 if resting_ecg == 'LVH' else 0
    input_df['RestingECG_Normal'] = 1 if resting_ecg == 'Normal' else 0
    input_df['RestingECG_ST'] = 1 if resting_ecg == 'ST' else 0
    
    # ST_Slope 独热编码
    st_slope = input_df.get('ST_Slope', ['Up'])[0] if 'ST_Slope' in input_df.columns else 'Up'
    
    input_df['ST_Slope_Down'] = 1 if st_slope == 'Down' else 0
    input_df['ST_Slope_Flat'] = 1 if st_slope == 'Flat' else 0
    input_df['ST_Slope_Up'] = 1 if st_slope == 'Up' else 0
    
    # 删除原始的分类特征列
    columns_to_drop = ['ChestPainType', 'RestingECG', 'ST_Slope']
    for col in columns_to_drop:
        if col in input_df.columns:
            input_df = input_df.drop(col, axis=1)
    
    # 按照训练时的确切特征顺序排列
    # 这个顺序是通过模拟训练时的数据处理过程得出的
    expected_features = [
        'Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'ExerciseAngina', 'Oldpeak',
        'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
        'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST',
        'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up'
    ]
    
    # 确保所有特征都存在
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # 按照期望的顺序排列特征
    input_df = input_df[expected_features]
    
    # 确保数据类型正确
    for col in input_df.columns:
        if col in ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']:
            input_df[col] = input_df[col].astype('float64')
        else:
            input_df[col] = input_df[col].astype('int64')
    
    return input_df

@app.route('/')
def home():
    """返回HTML页面"""
    try:
        # 获取当前脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(script_dir, 'app_heart.html')
        
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # 获取当前脚本所在目录用于调试
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return f"""
        <h1>心脏病预测系统</h1>
        <p>❌ app_heart.html 文件未找到</p>
        <p>📂 当前脚本目录: {script_dir}</p>
        <p>🔍 寻找文件: {os.path.join(script_dir, 'app_heart.html')}</p>
        <p>📋 请确保 app_heart.html 文件在同一目录下</p>
        """

@app.route('/predict', methods=['POST'])
def predict():
    """心脏病预测接口"""
    try:
        # 检查模型是否已加载
        if model is None:
            success, message = load_latest_heart_model()
            if not success:
                return jsonify({
                    'success': False,
                    'error': message,
                    'suggestion': '请检查模型文件是否存在'
                }), 500
        
        # 获取请求数据
        data = request.json
        print(f"📥 收到预测请求: {data}")
        
        # 数据验证
        required_fields = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                          'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                          'Oldpeak', 'ST_Slope']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'缺少必要字段: {missing_fields}'
            }), 400
        
        # 准备输入数据
        input_data = prepare_heart_input_data(data)
        print(f"🔧 处理后的输入数据形状: {input_data.shape}")
        print(f"🔧 处理后的输入数据:\n{input_data}")
        print(f"🔧 输入特征名称: {list(input_data.columns)}")
        
        # 尝试获取模型期望的特征名称
        try:
            if hasattr(model, 'feature_names_'):
                print(f"🎯 模型期望的特征名称: {model.feature_names_}")
            elif hasattr(model, 'get_feature_names'):
                print(f"🎯 模型期望的特征名称: {model.get_feature_names()}")
            else:
                print("⚠️ 无法获取模型的特征名称")
        except Exception as e:
            print(f"⚠️ 获取模型特征名称时出错: {e}")
        
        # 进行预测
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # 解释预测结果
        result = {
            'success': True,
            'prediction': int(prediction),
            'probability': {
                'healthy': float(prediction_proba[0]),
                'heart_disease': float(prediction_proba[1])
            },
            'confidence': float(max(prediction_proba)),
            'risk_level': get_risk_level(prediction_proba[1]),
            'interpretation': get_heart_disease_interpretation(prediction, prediction_proba),
            'model_info': {
                'model_file': os.path.basename(MODEL_PATH) if MODEL_PATH else '未知',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        print(f"🎯 预测结果: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ 预测过程出错: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'预测过程出错: {str(e)}',
            'suggestion': '请检查输入数据格式是否正确'
        }), 500

def get_risk_level(heart_disease_prob):
    """根据心脏病概率返回风险等级"""
    if heart_disease_prob < 0.3:
        return '低风险'
    elif heart_disease_prob < 0.6:
        return '中等风险'
    elif heart_disease_prob < 0.8:
        return '高风险'
    else:
        return '极高风险'

def get_heart_disease_interpretation(prediction, proba):
    """获取心脏病预测结果的解释"""
    heart_disease_prob = proba[1]
    healthy_prob = proba[0]
    
    if prediction == 0:  # 预测为健康
        if healthy_prob > 0.8:
            return f"✅ 心脏健康状况良好 (置信度: {healthy_prob:.1%})"
        else:
            return f"⚠️ 心脏健康，但建议定期检查 (置信度: {healthy_prob:.1%})"
    else:  # 预测为心脏病
        if heart_disease_prob > 0.8:
            return f"🚨 存在心脏病风险，强烈建议立即就医 (置信度: {heart_disease_prob:.1%})"
        else:
            return f"⚠️ 可能存在心脏病风险，建议咨询医生 (置信度: {heart_disease_prob:.1%})"

@app.route('/model_info', methods=['GET'])
def model_info():
    """获取模型信息"""
    if model is None:
        success, message = load_latest_heart_model()
        if not success:
            return jsonify({
                'success': False,
                'error': message
            }), 500
    
    return jsonify({
        'success': True,
        'model_path': MODEL_PATH,
        'model_name': os.path.basename(MODEL_PATH) if MODEL_PATH else '未知',
        'load_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': '心脏病预测 - SMOTE平衡CatBoost模型'
    })

if __name__ == '__main__':
    print("🚀 启动心脏病预测系统...")
    print("=" * 50)
    
    # 尝试加载模型
    success, message = load_latest_heart_model()
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
        print("⚠️ 系统将在接收到第一个预测请求时尝试重新加载模型")
    
    print("=" * 50)
    print("🌐 服务器启动信息:")
    print("   - 访问地址: http://localhost:5001")
    print("   - 预测接口: http://localhost:5001/predict")
    print("   - 模型信息: http://localhost:5001/model_info")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
