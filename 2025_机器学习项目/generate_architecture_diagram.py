import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形和轴
fig, ax = plt.subplots(1, 1, figsize=(18, 14))
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.axis('off')

# 定义颜色
colors = {
    'data': '#E3F2FD',      # 浅蓝色 - 数据层
    'process': '#F3E5F5',   # 浅紫色 - 处理层
    'model': '#E8F5E8',     # 浅绿色 - 模型层
    'api': '#FFF3E0',       # 浅橙色 - API层
    'frontend': '#FCE4EC',  # 浅粉色 - 前端层
    'storage': '#F1F8E9'    # 浅绿色 - 存储层
}

def create_box(ax, x, y, width, height, text, color, fontsize=10):
    """创建带文本的方框"""
    box = FancyBboxPatch((x, y), width, height,
                        boxstyle="round,pad=0.05",
                        facecolor=color,
                        edgecolor='black',
                        linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
           ha='center', va='center', fontsize=fontsize, weight='bold')

def create_arrow(ax, start, end, style='->'):
    """创建箭头连接"""
    arrow = ConnectionPatch(start, end, "data", "data",
                          arrowstyle=style, shrinkA=5, shrinkB=5,
                          mutation_scale=20, fc="black")
    ax.add_patch(arrow)

# 标题
ax.text(6, 11.5, '通用疾病预测系统架构图', ha='center', va='center', 
        fontsize=22, weight='bold')

# 定义层级分隔线和标签函数
def draw_layer_separator(ax, y, label):
    """绘制层级分隔线和标签"""
    ax.plot([1.2, 10.8], [y, y], 'k-', linewidth=2, alpha=0.7)
    ax.text(0.5, y + 0.1, label, fontsize=13, weight='bold', 
            ha='left', va='bottom', rotation=90, color='#2E3B55')

# 6. 前端展示层 (最上层)
draw_layer_separator(ax, 10.2, '前端展示层')
create_box(ax, 1.5, 9.2, 2.8, 0.8, 'app_heart.html\n心脏病预测界面', colors['frontend'])
create_box(ax, 4.8, 9.2, 2.8, 0.8, 'Flask模板\n甲状腺预测界面', colors['frontend'])
create_box(ax, 8.1, 9.2, 2.4, 0.8, 'AI聊天组件', colors['frontend'])

# 5. API服务层
draw_layer_separator(ax, 8.4, 'API服务层')
create_box(ax, 1.2, 7.2, 3.2, 1, 'app_heart.py\n• 心脏病预测API\n• MySQL日志\n• AI助手集成', colors['api'])
create_box(ax, 4.8, 7.2, 3.2, 1, 'app.py\n• 甲状腺预测API\n• 模型自动加载\n• CORS支持', colors['api'])
create_box(ax, 8.4, 7.2, 2.4, 1, 'BigModel API\nAI对话服务', colors['api'])

# 4. 模型存储层
draw_layer_separator(ax, 6.4, '模型存储层')
create_box(ax, 1.5, 5.4, 2.8, 0.8, 'balanced_models/\nSMOTE平衡模型', colors['storage'])
create_box(ax, 4.8, 5.4, 2.8, 0.8, 'saved_models/\n训练好的模型', colors['storage'])
create_box(ax, 8.1, 5.4, 2.4, 0.8, '性能报告\nperformance_report', colors['storage'])

# 3. 模型训练层
draw_layer_separator(ax, 4.6, '模型训练层')
create_box(ax, 1.2, 3.4, 3.8, 1, 'A赵超文_模型优化与融合.ipynb\n• SMOTE过采样\n• CatBoost训练\n• 超参数优化', colors['model'])
create_box(ax, 5.4, 3.4, 2.8, 1, 'A成锦勋_网格优化.ipynb\n网格搜索调参', colors['model'])
create_box(ax, 8.6, 3.4, 2.2, 1, '通用疾病预测.ipynb\n模型集成', colors['model'])

# 2. 数据处理层
draw_layer_separator(ax, 2.6, '数据处理层')
create_box(ax, 1.2, 1.6, 3.2, 0.8, 'A谢焱宁_数据预处理3.0.ipynb\n数据清洗、特征工程', colors['process'])
create_box(ax, 4.8, 1.6, 3.2, 0.8, 'A庄璟武_数据探索.ipynb\n数据分析、可视化', colors['process'])
create_box(ax, 8.4, 1.6, 2.4, 0.8, '数据验证\n质量检查', colors['process'])

# 1. 数据层 (最底层)
draw_layer_separator(ax, 0.8, '数据层')
create_box(ax, 1.2, 0.2, 2.2, 0.5, 'heart.csv\n心脏病数据', colors['data'])
create_box(ax, 3.8, 0.2, 2.2, 0.5, 'thyroidDF.csv\n甲状腺数据', colors['data'])
create_box(ax, 6.4, 0.2, 2.2, 0.5, 'diabetes.csv\n糖尿病数据', colors['data'])
create_box(ax, 9.0, 0.2, 1.8, 0.5, '其他数据集', colors['data'])

# 添加连接箭头 - 重新调整坐标
# 数据到处理
create_arrow(ax, (2.3, 0.8), (2.8, 1.6))
create_arrow(ax, (4.9, 0.8), (6.4, 1.6))

# 处理到模型
create_arrow(ax, (2.8, 2.4), (3.1, 3.4))
create_arrow(ax, (6.4, 2.4), (6.8, 3.4))

# 模型到存储
create_arrow(ax, (3.1, 4.4), (2.9, 5.4))
create_arrow(ax, (6.8, 4.4), (6.2, 5.4))

# 存储到API
create_arrow(ax, (2.9, 6.2), (2.8, 7.2))
create_arrow(ax, (6.2, 6.2), (6.4, 7.2))

# API到前端
create_arrow(ax, (2.8, 8.2), (2.9, 9.2))
create_arrow(ax, (6.4, 8.2), (6.2, 9.2))

# 添加侧边说明
ax.text(0.2, 9.2, '系统特点:', fontsize=12, weight='bold')
ax.text(0.2, 8.9, '• 多疾病预测支持', fontsize=10)
ax.text(0.2, 8.6, '• SMOTE类不平衡处理', fontsize=10)
ax.text(0.2, 8.3, '• 零漏诊优化目标', fontsize=10)
ax.text(0.2, 8.0, '• REST API服务', fontsize=10)
ax.text(0.2, 7.7, '• AI助手集成', fontsize=10)

# 添加技术栈说明
ax.text(9.5, 9.2, '技术栈:', fontsize=12, weight='bold')
ax.text(9.5, 8.9, '• Python/Pandas', fontsize=9)
ax.text(9.5, 8.6, '• CatBoost/SMOTE', fontsize=9)
ax.text(9.5, 8.3, '• Flask/HTML', fontsize=9)
ax.text(9.5, 8.0, '• MySQL/PyMySQL', fontsize=9)
ax.text(9.5, 7.7, '• BigModel API', fontsize=9)

# 添加图例 - 重新定位
legend_y = 10.8
ax.text(1.2, legend_y, '图例:', fontsize=12, weight='bold')
legend_items = [
    ('数据层', colors['data']),
    ('处理层', colors['process']),
    ('模型层', colors['model']),
    ('存储层', colors['storage']),
    ('API层', colors['api']),
    ('前端层', colors['frontend'])
]

for i, (label, color) in enumerate(legend_items):
    x = 1.2 + (i % 6) * 1.4
    y = legend_y - 0.3
    create_box(ax, x, y, 0.3, 0.15, '', color, fontsize=8)
    ax.text(x + 0.35, y + 0.075, label, fontsize=9, va='center')

plt.tight_layout()
plt.savefig('系统架构图.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print("✅ 系统架构图已生成并保存为 '系统架构图.png'")
