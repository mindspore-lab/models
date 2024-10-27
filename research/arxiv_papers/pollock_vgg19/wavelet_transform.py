import numpy as np
from matplotlib import pyplot as plt
import pywt
from PIL import Image
import matplotlib.font_manager as fm

# 打开图像并转换为灰度图
img = Image.open("pollock_0000.png").convert('L')  # 'L' 表示转换为灰度图
img = np.array(img)

# 使用 Haar 小波对图像进行二维离散小波变换
coeffs2 = pywt.dwt2(img, 'haar')
LLY, (LHY, HLY, HHY) = coeffs2

# 计算 turbulence_mean_power
turbulence_mean_power = np.mean(np.abs(LHY)**2 + np.abs(HLY)**2 + np.abs(HHY)**2)

# 计算 turbulence_variance_power
turbulence_variance = np.var(np.abs(LHY)**2 + np.abs(HLY)**2 + np.abs(HHY)**2)
turbulence_variance_power = np.sqrt(turbulence_variance)

# 计算功率谱
power_spectrum = np.abs(LHY)**2 + np.abs(HLY)**2 + np.abs(HHY)**2

# 计算功率谱的平均值
power_spectrum_mean = np.mean(power_spectrum)

# 计算功率谱的方差
power_spectrum_variance = np.var(power_spectrum)

print(f"湍流平均功率: {turbulence_mean_power}")
print(f"湍流方差功率: {turbulence_variance_power}")
print(f"功率谱平均值: {power_spectrum_mean}")
print(f"功率谱方差: {power_spectrum_variance}")

# 设置高分辨率的图像显示
plt.figure(figsize=(12, 12), dpi=300)  # figsize 调整图像大小，dpi 增加分辨率

# 创建一个2x2的子图并优化图像显示
titles = ['LLY (low-frequency component)', 'LHY (horizontal high-frequency component)', 'HLY (vertical high-frequency component)', 'HHY (diagonal high-frequency component)']
components = [LLY, LHY, HLY, HHY]

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(components[i], cmap="gray")
    plt.title(titles[i], fontsize=14)  # 字体大小可以调节
    plt.axis('off')  # 去掉坐标轴

# 调整布局，避免子图重叠
plt.tight_layout()

# 显示图像
plt.show()
