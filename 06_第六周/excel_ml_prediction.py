# -*- coding: utf-8 -*-
"""
Excel数据机器学习预测脚本
使用final_data_1.xlsx训练模型，预测test_features.xlsx
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("Excel机器学习预测系统")
print("="*80)

# 1. 加载训练数据
try:
    print("1. 加载训练数据...")
    train_data = pd.read_excel('data/final_data_1.xlsx')
    print(f"训练数据加载成功，形状: {train_data.shape}")
    print(f"列名: {list(train_data.columns)}")
    
    # 检查数据
    if train_data.empty:
        raise ValueError("训练数据集为空")
    
    print(f"前5行数据:")
    print(train_data.head())
    
except FileNotFoundError:
    print("错误：找不到训练文件 data/final_data_1.xlsx")
    print("请确保文件存在于data目录中")
    exit(1)
except Exception as e:
    print(f"加载训练数据时出错: {e}")
    exit(1)

# 2. 加载测试数据
try:
    print("\n2. 加载测试数据...")
    test_data = pd.read_excel('data/test_features.xlsx')
    print(f"测试数据加载成功，形状: {test_data.shape}")
    print(f"列名: {list(test_data.columns)}")
    
    if test_data.empty:
        raise ValueError("测试数据集为空")
        
    print(f"前5行测试数据:")
    print(test_data.head())
    
except FileNotFoundError:
    print("错误：找不到测试文件 data/test_features.xlsx")
    print("请确保文件存在于data目录中")
    exit(1)
except Exception as e:
    print(f"加载测试数据时出错: {e}")
    exit(1)

# 3. 数据预处理
print("\n3. 数据预处理...")

# 识别ID列和目标列
columns = train_data.columns.tolist()
print(f"所有列: {columns}")

# 假设第一列是ID，最后一列是目标
id_column = columns[0]
target_column = columns[-1]
feature_columns = columns[1:-1]  # 排除ID和目标列

print(f"ID列: {id_column}")
print(f"目标列: {target_column}")
print(f"特征列数量: {len(feature_columns)}")

# 分离特征和目标
X_train_full = train_data[feature_columns].copy()
y_train_full = train_data[target_column].copy()

# 处理测试数据特征
# 检查测试数据是否包含ID列
test_columns = test_data.columns.tolist()
if id_column in test_columns:
    # 如果测试数据包含ID列，提取特征列（排除ID列）
    test_feature_columns = [col for col in test_columns if col != id_column and col in feature_columns]
    X_test_full = test_data[test_feature_columns].copy()
    test_ids = test_data[id_column].copy()
else:
    # 如果测试数据不包含ID列，假设所有列都是特征
    test_feature_columns = [col for col in test_columns if col in feature_columns]
    X_test_full = test_data[test_feature_columns].copy()
    test_ids = range(len(test_data))  # 生成默认ID

print(f"训练特征形状: {X_train_full.shape}")
print(f"训练目标形状: {y_train_full.shape}")
print(f"测试特征形状: {X_test_full.shape}")

# 检查目标变量类型
print(f"\n目标变量信息:")
print(f"目标变量类型: {y_train_full.dtype}")
print(f"目标变量唯一值: {y_train_full.unique()}")
print(f"目标变量分布:")
print(y_train_full.value_counts())

# 处理目标变量（如果是字符串，进行编码）
label_encoder = None
if y_train_full.dtype == 'object':
    print("检测到字符串目标变量，进行标签编码...")
    label_encoder = LabelEncoder()
    y_train_full = label_encoder.fit_transform(y_train_full)
    print(f"编码后的目标变量: {np.unique(y_train_full)}")

# 检查缺失值
print(f"\n特征缺失值统计:")
missing_values = X_train_full.isnull().sum()
if missing_values.sum() > 0:
    print(missing_values[missing_values > 0])
else:
    print("没有缺失值")

# 检查数据类型
print(f"\n特征数据类型:")
print(X_train_full.dtypes.value_counts())

# 先识别数值列和分类列
numeric_columns = X_train_full.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = X_train_full.select_dtypes(include=['object']).columns.tolist()

print(f"数值列数量: {len(numeric_columns)}")
print(f"分类列数量: {len(categorical_columns)}")
if categorical_columns:
    print(f"分类列示例: {categorical_columns[:5]}")

# 处理缺失值
print("处理缺失值...")

# 数值列用均值填充
if numeric_columns:
    X_train_full[numeric_columns] = X_train_full[numeric_columns].fillna(
        X_train_full[numeric_columns].mean()
    )
    X_test_full[numeric_columns] = X_test_full[numeric_columns].fillna(
        X_train_full[numeric_columns].mean()  # 用训练集的均值填充测试集
    )

# 分类列用众数填充
if categorical_columns:
    for col in categorical_columns:
        mode_value = X_train_full[col].mode()
        if len(mode_value) > 0:
            X_train_full[col] = X_train_full[col].fillna(mode_value[0])
            X_test_full[col] = X_test_full[col].fillna(mode_value[0])
        else:
            # 如果没有众数，用字符串'missing'填充
            X_train_full[col] = X_train_full[col].fillna('missing')
            X_test_full[col] = X_test_full[col].fillna('missing')

# 处理非数值特征
if len(categorical_columns) > 0:
    print(f"编码分类特征: {list(categorical_columns)}")
    # 对分类特征进行标签编码
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        # 合并训练和测试数据来fit编码器
        combined_values = pd.concat([X_train_full[col], X_test_full[col]], ignore_index=True)
        le.fit(combined_values.astype(str))
        
        # 应用编码
        X_train_full[col] = le.transform(X_train_full[col].astype(str))
        X_test_full[col] = le.transform(X_test_full[col].astype(str))
        
        label_encoders[col] = le
    print("分类特征编码完成")

# 最终清理NaN值
print(f"编码前训练集NaN值: {X_train_full.isnull().sum().sum()}")
print(f"编码前测试集NaN值: {X_test_full.isnull().sum().sum()}")

# 确保所有特征都是数值型，用0填充任何剩余的NaN
X_train_full = X_train_full.fillna(0).astype(float)
X_test_full = X_test_full.fillna(0).astype(float)

print(f"最终训练集NaN值: {X_train_full.isnull().sum().sum()}")
print(f"最终测试集NaN值: {X_test_full.isnull().sum().sum()}")

print(f"预处理后的训练数据形状: {X_train_full.shape}")

# 4. 划分训练集和验证集
print("\n4. 划分训练集和验证集...")
# 由于是回归任务且目标变量的某些值只有一个样本，不使用分层抽样
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

print(f"训练集形状: {X_train.shape}")
print(f"验证集形状: {X_val.shape}")

# 5. 特征标准化
print("\n5. 特征标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("特征标准化完成")

# 6. 模型训练和比较
print("\n6. 模型训练和比较...")

# 定义多个回归模型（由于是连续目标变量）
# 为了避免高维特征导致的计算复杂度问题，选择适合高维数据的模型
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(random_state=42, alpha=1.0),
    'RandomForest': RandomForestRegressor(random_state=42, n_estimators=50, max_depth=10),
    'GradientBoosting': GradientBoostingRegressor(random_state=42, n_estimators=50, max_depth=5),
    'DecisionTree': DecisionTreeRegressor(random_state=42, max_depth=10)
    # 移除SVM和KNN，因为它们在高维数据上计算太慢
}

print(f"使用{len(models)}个适合高维数据的回归模型")

# 交叉验证设置
kfold = KFold(n_splits=3, shuffle=True, random_state=42)  # 减少fold数以加快速度
scoring = 'neg_mean_squared_error'  # 回归问题使用MSE

# 模型比较
results = []
model_names = []
best_score = -float('inf')  # MSE是负值，初始化为负无穷
best_model_name = ""

print("模型性能比较:")
print("-" * 50)

for name, model in models.items():
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    model_names.append(name)
    mean_score = cv_results.mean()
    std_score = cv_results.std()
    
    print(f"{name:20}: {mean_score:.4f} (±{std_score:.4f})")
    
    if mean_score > best_score:
        best_score = mean_score
        best_model_name = name

print(f"\n最佳模型: {best_model_name} (MSE: {-best_score:.4f})")

# 7. 超参数优化（针对最佳模型）
print(f"\n7. 对{best_model_name}进行超参数优化...")

if best_model_name == 'RandomForest':
    param_grid = {
        'n_estimators': [30, 50, 100],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5]
    }
    best_model = RandomForestRegressor(random_state=42)
    
elif best_model_name == 'GradientBoosting':
    param_grid = {
        'n_estimators': [30, 50, 100],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    best_model = GradientBoostingRegressor(random_state=42)
    
elif best_model_name == 'LinearRegression':
    param_grid = {}  # 线性回归没有超参数
    best_model = LinearRegression()
    
elif best_model_name == 'Ridge':
    param_grid = {
        'alpha': [0.1, 1, 10, 100]
    }
    best_model = Ridge(random_state=42)
    
elif best_model_name == 'DecisionTree':
    param_grid = {
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    best_model = DecisionTreeRegressor(random_state=42)
    
elif best_model_name == 'DecisionTree':
    param_grid = {
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    best_model = DecisionTreeRegressor(random_state=42)
    
else:
    # 对于其他模型，使用默认参数
    param_grid = {}
    best_model = models[best_model_name]

if param_grid:
    # 使用较少的cv折数和较简单的搜索来加快速度
    grid_search = GridSearchCV(
        best_model, param_grid, cv=3, scoring=scoring, n_jobs=-1, verbose=1
    )
    print("开始网格搜索...")
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"最优参数: {grid_search.best_params_}")
    print(f"最优交叉验证得分: {grid_search.best_score_:.4f}")
    
    # 使用最优模型
    final_model = grid_search.best_estimator_
else:
    # 使用默认参数
    print("使用默认参数...")
    final_model = best_model
    final_model.fit(X_train_scaled, y_train)

# 8. 模型评估
print("\n8. 模型评估...")

# 在验证集上评估
val_predictions = final_model.predict(X_val_scaled)
val_mse = mean_squared_error(y_val, val_predictions)
val_r2 = r2_score(y_val, val_predictions)

print(f"验证集MSE: {val_mse:.4f}")
print(f"验证集R²得分: {val_r2:.4f}")
print(f"验证集RMSE: {np.sqrt(val_mse):.4f}")
# 9. 处理测试数据并进行预测
print("\n9. 处理测试数据并进行预测...")

# 确保测试数据具有相同的特征
test_features = test_data.copy()

# 如果测试数据包含ID列，我们需要保留它用于最终输出
if id_column in test_features.columns:
    test_ids = test_features[id_column].copy()
    # 只保留特征列
    available_features = [col for col in feature_columns if col in test_features.columns]
    test_features = test_features[available_features]
    print(f"保留测试数据ID列，使用{len(available_features)}个特征进行预测")
else:
    # 假设测试数据只包含特征
    test_ids = range(len(test_features))
    available_features = [col for col in feature_columns if col in test_features.columns]
    test_features = test_features[available_features]
    print(f"测试数据不包含ID列，使用{len(available_features)}个特征进行预测")

# 处理测试数据的缺失值和分类特征（与训练数据相同的处理）
print("处理测试数据...")

# 分离数值和分类特征
test_numerical_cols = test_features.select_dtypes(include=[np.number]).columns.tolist()
test_categorical_cols = test_features.select_dtypes(include=['object']).columns.tolist()

print(f"测试数据 - 数值列数量: {len(test_numerical_cols)}")
print(f"测试数据 - 分类列数量: {len(test_categorical_cols)}")

# 处理缺失值
if test_features.isnull().sum().sum() > 0:
    print("处理测试数据缺失值...")
    # 数值特征用0填充
    test_features[test_numerical_cols] = test_features[test_numerical_cols].fillna(0)
    # 分类特征用'unknown'填充
    if test_categorical_cols:
        test_features[test_categorical_cols] = test_features[test_categorical_cols].fillna('unknown')

# 编码分类特征
if test_categorical_cols:
    print(f"编码测试数据分类特征: {test_categorical_cols}")
    for col in test_categorical_cols:
        le = LabelEncoder()
        test_features[col] = le.fit_transform(test_features[col].astype(str))

# 确保所有特征都是数值型并处理剩余的NaN值
test_features = test_features.astype(float)
test_features = test_features.fillna(0)

print(f"测试数据最终形状: {test_features.shape}")

# 标准化测试特征
test_features_scaled = scaler.transform(test_features)

# 进行预测
print("开始预测...")
test_predictions = final_model.predict(test_features_scaled)

print(f"预测完成！生成了{len(test_predictions)}个预测结果")

# 10. 保存预测结果
print("\n10. 保存预测结果...")

# 创建结果DataFrame
results_df = pd.DataFrame({
    'ID': test_ids,
    'Prediction': test_predictions
})

# 保存结果
output_file = 'data/predictions.xlsx'
results_df.to_excel(output_file, index=False)

print(f"预测结果已保存到: {output_file}")
print(f"结果预览:")
print(results_df.head(10))

# 11. 预测结果统计
print(f"\n11. 预测结果统计:")
print(f"预测值范围: {test_predictions.min():.4f} - {test_predictions.max():.4f}")
print(f"预测值均值: {test_predictions.mean():.4f}")
print(f"预测值标准差: {test_predictions.std():.4f}")

print(f"\n=== 预测完成！===")
print(f"使用模型: {best_model_name}")
print(f"验证集MSE: {val_mse:.4f}")
print(f"验证集R²得分: {val_r2:.4f}")
print(f"预测样本数: {len(test_predictions)}")
print(f"结果文件: {output_file}")
