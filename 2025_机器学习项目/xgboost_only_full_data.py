import numpy as np
import os
import sys
import pandas as pd

# 设置环境变量以处理Unicode编码问题
sys.setrecursionlimit(10000)

# 确保临时目录存在
temp_dir = 'C:\\tmp'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
os.environ['JOBLIB_TEMP_FOLDER'] = temp_dir  # 使用不包含中文字符的临时目录
os.environ['LOKY_MAX_CPU_COUNT'] = '2'  # 限制CPU数量以避免某些问题

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import xgboost as xgb  # XGBoost库

def load_data(file_path):
    """加载数据"""
    print("正在加载数据...")
    # 使用survey_code作为索引
    df = pd.read_csv(file_path, index_col='survey_code')
    
    print(f"数据集形状: {df.shape}")
    print("数据集前5行:")
    print(df.head())
    print("\n数据集基本信息:")
    print(df.info())
    
    # 检查缺失值
    print("\n缺失值情况:")
    print(df.isnull().sum())
    
    # 检查目标变量分布
    print("\n目标变量分布:")
    print(df['target'].value_counts())
    
    return df

def preprocess_data(df):
    """数据预处理"""
    print("\n开始数据预处理...")
    
    # 数据验证：检查必需的列是否存在
    required_columns = ['target', 'age', 'gender', 'bmi', 'blood_pressure']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"警告: 缺少必需的列: {missing_columns}")
    
    # 分离特征和目标变量
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 将目标变量转换为数值类型（XGBoost需要）
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # 检查目标变量的唯一值数量和分布
    unique_targets = np.unique(y)
    print(f"目标变量唯一值: {unique_targets}")
    print(f"目标变量分布:\n{Counter(y)}")
    
    # 分离数值型和分类型特征
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    print(f"\n数值型特征数量: {len(numeric_features)}")
    print(f"分类型特征数量: {len(categorical_features)}")
    
    # ========= 1. 异常值检测和处理 =========
    print("\n进行异常值检测和处理...")
    
    # 创建一个副本用于处理异常值
    X_clean = X.copy()
    
    # 对数值型特征进行异常值处理
    for col in numeric_features:
        if col in X_clean.columns:
            # 使用IQR方法检测异常值
            Q1 = X_clean[col].quantile(0.25)
            Q3 = X_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # 计算异常值边界
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 计算异常值数量
            outliers_count = ((X_clean[col] < lower_bound) | (X_clean[col] > upper_bound)).sum()
            
            if outliers_count > 0:
                # 使用截断方法处理异常值（限制在上下边界内）
                X_clean[col] = X_clean[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"  {col}: 处理了 {outliers_count} 个异常值 (占比: {outliers_count/len(X_clean)*100:.2f}%)")
    
    # ========= 2. 特征工程 =========
    print("\n进行特征工程...")
    
    # BMI分类
    if 'bmi' in X_clean.columns:
        X_clean['bmi_category'] = pd.cut(X_clean['bmi'], 
                                      bins=[0, 18.5, 25, 30, float('inf')],
                                      labels=['underweight', 'normal', 'overweight', 'obese'])
        print("  创建了特征: bmi_category")
    
    # 血压分类
    if 'blood_pressure' in X_clean.columns:
        X_clean['bp_category'] = pd.cut(X_clean['blood_pressure'], 
                                     bins=[0, 120, 140, float('inf')],
                                     labels=['normal', 'high', 'very_high'])
        print("  创建了特征: bp_category")
    
    # 健康风险评分
    risk_factors = []
    if 'bmi' in X_clean.columns:
        risk_factors.append((X_clean['bmi'] > 30).astype(int))  # 肥胖
    if 'blood_pressure' in X_clean.columns:
        risk_factors.append((X_clean['blood_pressure'] > 140).astype(int))  # 高血压
    if 'cholesterol' in X_clean.columns:
        risk_factors.append((X_clean['cholesterol'] > 240).astype(int))  # 高胆固醇
    
    if risk_factors:
        X_clean['health_risk_score'] = sum(risk_factors)
        print("  创建了特征: health_risk_score")
    
    # 睡眠质量指数（如果有相关特征）
    if 'sleep_hours' in X_clean.columns and 'sleep_quality' in X_clean.columns:
        # 将睡眠质量转换为数值
        sleep_quality_map = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
        X_clean['sleep_quality_numeric'] = X_clean['sleep_quality'].map(sleep_quality_map)
        
        # 计算睡眠质量指数（睡眠时长 × 睡眠质量）
        X_clean['sleep_score'] = X_clean['sleep_hours'] * X_clean['sleep_quality_numeric'].fillna(2)  # 默认为一般质量
        print("  创建了特征: sleep_score")
    
    # ========= 3. 缺失值处理策略 =========
    # 更新特征列表，因为我们添加了新特征
    numeric_features = X_clean.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_clean.select_dtypes(include=['object', 'category']).columns
    
    # 创建预处理管道
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # 使用稠密矩阵输出
    ])
    
    # 组合预处理步骤
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # ========= 4. 使用全部数据进行处理 =========
    # 直接使用全部数据，不进行采样
    X_clean_full = X_clean
    y_full = y
    print(f"\n使用全部数据进行处理，数据集大小: {len(X_clean_full)}")
    
    # 应用预处理
    X_preprocessed = preprocessor.fit_transform(X_clean_full)
    
    # 检查处理后的数据形状
    print(f"\n预处理后特征数量: {X_preprocessed.shape[1]}")
    
    # 处理不平衡数据
    print(f"\n原始类别分布: {Counter(y_full)}")
    
    # 应用SMOTE处理不平衡
    if len(Counter(y_full)) > 1:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y_full)
        print(f"SMOTE后类别分布: {Counter(y_resampled)}")
    else:
        X_resampled, y_resampled = X_preprocessed, y_full
        print("注意：目标变量只有一个类别，无需进行SMOTE处理")
    
    # ========= 5. 数据集分割 =========
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled if len(Counter(y_resampled)) > 1 else None
    )
    
    print(f"\n训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 数据验证：检查处理后的数据质量
    print(f"\n数据验证: 训练集缺失值总数: {np.isnan(X_train).sum()}")
    print(f"数据验证: 测试集缺失值总数: {np.isnan(X_test).sum()}")
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_and_evaluate_xgboost(X_train, X_test, y_train, y_test):
    """训练和评估XGBoost模型"""
    print("\n开始训练和评估XGBoost模型...")
    
    # 定义XGBoost模型
    xgboost_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, n_estimators=100, eval_metric='logloss')
    
    # 训练模型
    xgboost_model.fit(X_train, y_train)
    
    # 预测
    y_pred = xgboost_model.predict(X_test)
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted' if len(np.unique(y_test)) > 1 else 'binary')
    recall = recall_score(y_test, y_pred, average='weighted' if len(np.unique(y_test)) > 1 else 'binary')
    f1 = f1_score(y_test, y_pred, average='weighted' if len(np.unique(y_test)) > 1 else 'binary')
    
    # 计算AUC指标（需要概率预测）
    if len(np.unique(y_test)) > 1:  # 确保有多个类别
        y_pred_proba = xgboost_model.predict_proba(X_test)[:, 1]  # 获取正类概率
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc = None
        print("警告：目标变量只有一个类别，无法计算AUC")
    
    # 保存结果
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'model': xgboost_model
    }
    
    print(f"XGBoost 评估结果:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")
    if auc is not None:
        print(f"  AUC: {auc:.4f}")
    
    # 打印分类报告
    print(f"\nXGBoost 分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 超参数调优
    print(f"\n对XGBoost进行超参数调优...")
    
    # 简化参数网格，减少计算量
    param_grid = {
        'n_estimators': [100, 150],  # 减少为2个选项
        'learning_rate': [0.1, 0.2],  # 减少为2个选项
        'max_depth': [3, 5],  # 减少为2个选项
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # 限制并行任务数量为2，避免过多资源占用
    grid_search = GridSearchCV(xgboost_model, param_grid, cv=3, 
                              scoring='precision_weighted' if len(np.unique(y_test)) > 1 else 'precision', 
                              n_jobs=2,  # 限制为2个并行任务
                              verbose=1)  # 添加详细输出，方便查看进度
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    
    # 使用调优后的模型进行预测
    best_tuned_model = grid_search.best_estimator_
    y_pred_tuned = best_tuned_model.predict(X_test)
    
    # 评估调优后的模型
    tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
    tuned_precision = precision_score(y_test, y_pred_tuned, average='weighted' if len(np.unique(y_test)) > 1 else 'binary')
    tuned_recall = recall_score(y_test, y_pred_tuned, average='weighted' if len(np.unique(y_test)) > 1 else 'binary')
    tuned_f1 = f1_score(y_test, y_pred_tuned, average='weighted' if len(np.unique(y_test)) > 1 else 'binary')
    
    # 计算调优后的模型AUC
    if len(np.unique(y_test)) > 1:  # 确保有多个类别
        y_pred_proba_tuned = best_tuned_model.predict_proba(X_test)[:, 1]  # 获取正类概率
        tuned_auc = roc_auc_score(y_test, y_pred_proba_tuned)
    else:
        tuned_auc = None
    
    print(f"调优后 XGBoost 评估结果:")
    print(f"  准确率: {tuned_accuracy:.4f}")
    print(f"  精确率: {tuned_precision:.4f}")
    print(f"  召回率: {tuned_recall:.4f}")
    print(f"  F1分数: {tuned_f1:.4f}")
    if tuned_auc is not None:
        print(f"  AUC: {tuned_auc:.4f}")
    
    # 交叉验证
    print(f"\n对调优后的XGBoost进行交叉验证...")
    cv_scores = cross_val_score(best_tuned_model, X_train, y_train, cv=5, scoring='precision_weighted' if len(np.unique(y_test)) > 1 else 'precision', n_jobs=-1)
    print(f"交叉验证精确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 保存模型结果
    save_results(results, tuned_precision, tuned_auc)
    
    return best_tuned_model

def save_results(results, tuned_precision, tuned_auc):
    """保存模型结果"""
    # 创建结果目录
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 保存模型性能结果
    with open('results/xgboost_results.txt', 'w', encoding='utf-8') as f:
        f.write("XGBoost模型性能结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'评估指标':<20}{'分数':<10}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'准确率':<20}{results['accuracy']:.4f}\n")
        f.write(f"{'精确率':<20}{results['precision']:.4f}\n")
        f.write(f"{'召回率':<20}{results['recall']:.4f}\n")
        f.write(f"{'F1分数':<20}{results['f1']:.4f}\n")
        if results['auc'] is not None:
            f.write(f"{'AUC':<20}{results['auc']:.4f}\n")
        f.write("-" * 60 + "\n")
        f.write(f"调优后的精确率: {tuned_precision:.4f}\n")
        if tuned_auc is not None:
            f.write(f"调优后的AUC: {tuned_auc:.4f}\n")
    
    print("XGBoost模型结果已保存到results/xgboost_results.txt")

def main():
    """主函数"""
    # 加载数据
    file_path = 'health_lifestyle_classification.csv'
    df = load_data(file_path)
    
    # 数据预处理
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # 训练和评估XGBoost模型
    best_model = train_and_evaluate_xgboost(X_train, X_test, y_train, y_test)
    
    print("\nXGBoost模型训练任务已完成！")
    print("1. 数据处理过程包括了异常值检测、特征工程和类别不平衡处理")
    print("2. 已使用全部数据进行模型训练和评估")
    print("3. 模型结果已保存到results目录")

if __name__ == "__main__":
    main()