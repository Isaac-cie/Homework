import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

### 绘制柱状图

plt.rcParams['font.family'] = 'Times New Roman'

kernel_functions = ['Linear', 'Polynomial', 'RBF', 'Sigmoid']
pixel_accuracy = [0.837, 0.8755, 0.8836, 0.7074]
hog_accuracy = [0.8755, 0.8971, 0.9003, 0.7652]

# 设置宽度和位置
bar_width = 0.20
index = np.arange(len(kernel_functions))

# 设置颜色
color_pixel = '#ff850f'  # 橙色
color_hog = '#3789be'    # 蓝色

# 绘制柱状图
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')  # 设置背景为白色
rects1 = ax.bar(index, pixel_accuracy, bar_width,
                label='Pixel+SVM',
                color=color_pixel,
                edgecolor='white', linewidth=0.7)
rects2 = ax.bar(index + bar_width, hog_accuracy, bar_width,
                label='HOG+SVM',
                color=color_hog,
                edgecolor='white', linewidth=0.7)

# 添加标题和标签
ax.set_xlabel('Kernel Function Type', fontsize=22)
ax.set_ylabel('Accuracy', fontsize=22)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(kernel_functions, fontsize=18)

# 详细图例说明，包括准确率数值
legend_handles = [plt.Rectangle((0,0),1,1, color=color_pixel), 
                  plt.Rectangle((0,0),1,1, color=color_hog)]
legend_labels = ['Pixel : {:.2f}%'.format(max(pixel_accuracy)*100), 
                 'HOG : {:.2f}%'.format(max(hog_accuracy)*100)]
ax.legend(legend_handles, legend_labels, fontsize=18, title='Max Accuracy', title_fontsize=16)

# 网格线
ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7, color='gray')

# y轴字体大小+刻度百分比显示
plt.yticks(fontsize=18)
ax.yaxis.set_major_formatter('{:.0%}'.format)

# 紧凑布局
plt.tight_layout()

# 显示图形
plt.show()