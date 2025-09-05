import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形和轴 - 增大画布和坐标范围
fig, ax = plt.subplots(1, 1, figsize=(20, 16))
ax.set_xlim(0, 16)
ax.set_ylim(0, 14)
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

# 定义层级分隔线和标签函数
def draw_layer_separator(ax, y, label):
    """绘制层级分隔线和标签"""
    ax.plot([1.5, 14.5], [y, y], 'k-', linewidth=2, alpha=0.7)
    ax.text(0.8, y + 0.1, label, fontsize=14, weight='bold', 
            ha='center', va='bottom', rotation=0, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# 标题
ax.text(8, 13, '通用疾病预测系统架构图', ha='center', va='center', 
        fontsize=24, weight='bold')

# 6. 前端展示层 (最上层)
draw_layer_separator(ax, 11.5, '前端展示层')
create_box(ax, 2, 10.5, 3.5, 0.8, 'app_heart.html\n心脏病预测界面', colors['frontend'], 11)
create_box(ax, 6, 10.5, 3.5, 0.8, 'Flask模板\n甲状腺预测界面', colors['frontend'], 11)
create_box(ax, 10, 10.5, 2.5, 0.8, 'AI聊天组件', colors['frontend'], 11)

# 5. API服务层
draw_layer_separator(ax, 9.8, 'API服务层')
create_box(ax, 2, 8.5, 3.5, 1.2, 'app_heart.py\n心脏病预测API\nMySQL日志\nAI助手集成', colors['api'], 10)
create_box(ax, 6, 8.5, 3.5, 1.2, 'app.py\n甲状腺预测API\n模型自动加载\nCORS支持', colors['api'], 10)
create_box(ax, 10, 8.5, 2.5, 1.2, 'BigModel API\nAI对话服务', colors['api'], 10)

# 4. 模型存储层
draw_layer_separator(ax, 7.8, '模型存储层')
create_box(ax, 2, 6.8, 3, 0.8, 'balanced_models/\nSMOTE平衡模型', colors['storage'], 11)
create_box(ax, 5.5, 6.8, 3, 0.8, 'saved_models/\n训练好的模型', colors['storage'], 11)
create_box(ax, 9, 6.8, 3.5, 0.8, '性能报告\nperformance_report', colors['storage'], 11)

# 3. 模型训练层
draw_layer_separator(ax, 6.1, '模型训练层')
create_box(ax, 2, 4.8, 4.5, 1.2, 'A赵超文_模型优化与融合.ipynb\nSMOTE过采样\nCatBoost训练\n超参数优化', colors['model'], 10)
create_box(ax, 7, 4.8, 3, 1.2, 'A成锦勋_网格优化.ipynb\n网格搜索调参', colors['model'], 10)
create_box(ax, 10.5, 4.8, 2.5, 1.2, '通用疾病预测.ipynb\n模型集成', colors['model'], 10)

# 2. 数据处理层
draw_layer_separator(ax, 4.1, '数据处理层')
create_box(ax, 2, 3, 3.5, 0.8, 'A谢焱宁_数据预处理3.0.ipynb\n数据清洗、特征工程', colors['process'], 10)
create_box(ax, 6, 3, 3.5, 0.8, 'A庄璟武_数据探索.ipynb\n数据分析、可视化', colors['process'], 10)
create_box(ax, 10, 3, 2.5, 0.8, '数据验证\n质量检查', colors['process'], 10)

# 1. 数据层 (最底层)
draw_layer_separator(ax, 2.3, '数据层')
create_box(ax, 2, 1.2, 2.5, 0.8, 'heart.csv\n心脏病数据', colors['data'], 11)
create_box(ax, 5, 1.2, 2.5, 0.8, 'thyroidDF.csv\n甲状腺数据', colors['data'], 11)
create_box(ax, 8, 1.2, 2.5, 0.8, 'diabetes.csv\n糖尿病数据', colors['data'], 11)
create_box(ax, 11, 1.2, 2, 0.8, '其他数据集', colors['data'], 11)

# 添加连接箭头 - 调整坐标
# 数据到处理
create_arrow(ax, (3.25, 2.0), (3.75, 3.0))
create_arrow(ax, (6.25, 2.0), (7.75, 3.0))

# 处理到模型
create_arrow(ax, (3.75, 3.8), (4.25, 4.8))
create_arrow(ax, (7.75, 3.8), (8.5, 4.8))

# 模型到存储
create_arrow(ax, (4.25, 6.0), (3.5, 6.8))
create_arrow(ax, (8.5, 6.0), (7, 6.8))

# 存储到API
create_arrow(ax, (3.5, 7.6), (3.75, 8.5))
create_arrow(ax, (7, 7.6), (7.75, 8.5))

# API到前端
create_arrow(ax, (3.75, 9.7), (3.75, 10.5))
create_arrow(ax, (7.75, 9.7), (7.75, 10.5))

# 移动系统特点和技术栈到右侧，避免重叠
# 系统特点
ax.text(14.5, 12, '系统特点', fontsize=14, weight='bold', ha='left')
features = [
    '多疾病预测支持',
    'SMOTE类不平衡处理', 
    '零漏诊优化目标',
    'REST API服务',
    'AI助手集成'
]
for i, feature in enumerate(features):
    ax.text(14.5, 11.5 - i*0.4, f'• {feature}', fontsize=11, ha='left')

# 技术栈
ax.text(14.5, 9, '技术栈', fontsize=14, weight='bold', ha='left')
tech_stack = [
    'Python/Pandas',
    'CatBoost/SMOTE',
    'Flask/HTML', 
    'MySQL/PyMySQL',
    'BigModel API'
]
for i, tech in enumerate(tech_stack):
    ax.text(14.5, 8.5 - i*0.4, f'• {tech}', fontsize=11, ha='left')

# 添加图例 - 移动到底部中央
legend_y = 0.3
ax.text(8, legend_y + 0.8, '图例', fontsize=14, weight='bold', ha='center')
legend_items = [
    ('数据层', colors['data']),
    ('处理层', colors['process']), 
    ('模型层', colors['model']),
    ('存储层', colors['storage']),
    ('API层', colors['api']),
    ('前端层', colors['frontend'])
]

for i, (label, color) in enumerate(legend_items):
    x = 4 + (i % 3) * 2.5
    y = legend_y + 0.4 - (i // 3) * 0.3
    create_box(ax, x, y, 0.4, 0.15, '', color, fontsize=8)
    ax.text(x + 0.5, y + 0.075, label, fontsize=11, va='center', ha='left')

plt.tight_layout()
plt.savefig('系统架构图_优化版.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print("✅ 优化版系统架构图已生成并保存为 '系统架构图_优化版.png'")
print("主要改进:")
print("- 增大了画布尺寸和元素间距")
print("- 将系统特点和技术栈移至右侧，避免重叠")
print("- 优化了层级标签的显示方式")
print("- 调整了图例位置到底部中央")
print("- 增加了文字大小提高可读性")
