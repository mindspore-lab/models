import numpy as np
import matplotlib.pyplot as plt
from skimage import color, filters
from osgeo import gdal


def differential_box_counting(img, min_box_size=5, num_boxes=20):
    assert len(img.shape) == 2
    img = (img - img.min()) / (img.max() - img.min())
    N = img.shape[0]
    sizes = np.logspace(np.log2(min_box_size), np.log2(N), num=num_boxes, base=2).astype(int)
    counts = []

    for size in sizes:
        reduced_image = measure_block_counting(img, size)
        counts.append(reduced_image)

    counts = np.array(counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0], sizes, counts


def measure_block_counting(Z, box_size):
    block_count = 0
    for i in range(0, Z.shape[0], box_size):
        for j in range(0, Z.shape[1], box_size):
            block = Z[i:i + box_size, j:j + box_size]
            if np.any(block):
                block_count += 1
    return block_count


def read_tif_image(image_path):
    dataset = gdal.Open(image_path)
    band = dataset.GetRasterBand(1)
    image = band.ReadAsArray()
    return image


def main():
    img_path = "pollock.png"  # 替换为你的TIF文件路径
    img = read_tif_image(img_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 3:
        img = color.rgb2gray(img)

    # 保存原图用于显示
    original_img = img.copy()

    # 边缘提取
    img = filters.sobel(img)

    # 直接使用整个图像进行处理
    selected_img = original_img

    # 使用预处理后的图像进行分形维数计算
    selected_processed_img = filters.sobel(selected_img)

    fd, sizes, counts = differential_box_counting(selected_processed_img, min_box_size=5, num_boxes=20)

    print(f'Fractal Dimension: {fd}')

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(selected_img, cmap='gray')  # 显示整张图
    axs[0].set_title(f'Entire Image\nFractal Dimension: {fd}')

    axs[1].plot(np.log(sizes), np.log(counts), 'o-', label='Data')
    axs[1].plot(np.log(sizes), np.polyval(np.polyfit(np.log(sizes), np.log(counts), 1), np.log(sizes)), 'r--', label=f'Fit: D={fd:.2f}')
    axs[1].set_xlabel('log(Box Size)')
    axs[1].set_ylabel('log(Count)')
    axs[1].legend()
    axs[1].set_title('Fractal Dimension Fitting')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
