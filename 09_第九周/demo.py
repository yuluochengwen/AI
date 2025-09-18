# 最终优化版本：多策略轮廓线检测（修复版）
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

def find_multi_strategy_contours(img):
    """
    多策略轮廓线检测：结合边缘检测、形态学操作和统计分析
    """
    height, width = img.shape
    
    # 策略1: 基于Sobel梯度的检测
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # 策略2: Canny边缘检测
    edges = cv2.Canny(img, 80, 150)
    
    # 寻找左侧垂直线（白→黑）
    left_line = None
    max_edge_count = 0
    
    # 扫描左半部分，寻找显著的垂直边缘
    for x in range(50, width//2):
        edge_count = 0
        intensity_changes = 0
        
        for y in range(height//4, 3*height//4):
            # 检查边缘检测结果
            if edges[y, x] > 0:
                edge_count += 1
            
            # 检查强度变化
            if x < width - 1:
                current = img[y, x]
                next_px = img[y, x + 1]
                if int(current) - int(next_px) > 100:
                    intensity_changes += 1
        
        # 综合判断
        total_score = edge_count + intensity_changes * 2
        if total_score > max_edge_count and total_score > 80:
            max_edge_count = total_score
            left_line = x
    
    # 寻找右侧垂直线（白→灰）
    right_line = None
    max_edge_count = 0
    
    for x in range(width - 50, width//2, -1):
        edge_count = 0
        intensity_changes = 0
        
        for y in range(height//4, 3*height//4):
            if edges[y, x] > 0:
                edge_count += 1
                
            if x > 0:
                current = img[y, x]
                prev_px = img[y, x - 1]
                if abs(int(current) - int(prev_px)) > 80:  # 较小的阈值用于灰色变化
                    intensity_changes += 1
        
        total_score = edge_count + intensity_changes * 2
        if total_score > max_edge_count and total_score > 20:
            max_edge_count = total_score
            right_line = x
    
    # 寻找水平线（白→深）- 专门检测从白色进入深色的边缘
    horizontal_line = None
    max_edge_count = 0
    WHITE_THRESHOLD = 180  # 白色阈值
    DARK_THRESHOLD = 100   # 深色阈值
    
    for y in range(50, height - 50):
        edge_count = 0
        white_to_dark_transitions = 0
        
        for x in range(width//4, 3*width//4):
            if edges[y, x] > 0:
                edge_count += 1
                
            # 检测从白色到深色的特定转换
            if y < height - 1:
                current = img[y, x]
                next_px = img[y + 1, x]
                
                # 只计算从白色进入深色的转换
                if current >= WHITE_THRESHOLD and next_px <= DARK_THRESHOLD:
                    white_to_dark_transitions += 3  # 给予更高权重
                elif abs(int(current) - int(next_px)) > 50 and current > next_px:
                    # 其他从亮到暗的变化（权重较低）
                    white_to_dark_transitions += 1
        
        # 综合评分：边缘密度 + 白色到深色转换的权重
        total_score = edge_count + white_to_dark_transitions * 2
        if total_score > max_edge_count and total_score > 25:
            max_edge_count = total_score
            horizontal_line = y
    
    return left_line, right_line, horizontal_line, edges, gradient_magnitude

def create_final_visualization(img, left_line, right_line, horizontal_line, edges, gradient_mag):
    """创建最终的可视化结果"""
    
    # 创建结果图像
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    height, width = img.shape
    
    # 绘制检测到的线条
    line_thickness = 5
    detected_lines = []
    
    if left_line is not None:
        cv2.line(result, (left_line, 0), (left_line, height - 1), (0, 0, 255), line_thickness)  # 红色
        cv2.putText(result, f'L:{left_line}', (left_line - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        detected_lines.append(f"左垂直线: x={left_line}")
    
    if right_line is not None:
        cv2.line(result, (right_line, 0), (right_line, height - 1), (0, 0, 255), line_thickness)  # 红色
        cv2.putText(result, f'R:{right_line}', (right_line - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        detected_lines.append(f"右垂直线: x={right_line}")
    
    if horizontal_line is not None:
        cv2.line(result, (0, horizontal_line), (width - 1, horizontal_line), (0, 0, 255), line_thickness)  # 红色
        cv2.putText(result, f'H:{horizontal_line}', (width//2 - 50, horizontal_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        detected_lines.append(f"水平线: y={horizontal_line}")
    
    return result, detected_lines

# 读取图像
img = cv2.imread('OpenCV_test/Cam0-6.jpg', cv2.IMREAD_GRAYSCALE)
# 去噪
img = cv2.medianBlur(img, 3)


# 执行最终的多策略检测
left_final, right_final, horizontal_final, edges_final, grad_mag_final = find_multi_strategy_contours(img)
result_final, detected_list = create_final_visualization(img, left_final, right_final, horizontal_final, edges_final, grad_mag_final)

# 显示最终结果
plt.figure(figsize=(20, 15))

# 主结果显示
plt.subplot(2, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('原始图像\n(1944×2592)', fontsize=12)
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(cv2.cvtColor(result_final, cv2.COLOR_BGR2RGB))
plt.title('最终检测结果\n(红线标注)', fontsize=12, weight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()

