import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from scipy.stats import entropy
# from skimage.feature import greycomatrix, greycoprops

# 计算UCIQE（简化版）
def calculate_UCIQE(img12, img2):
    mse = np.mean((img12 - img2) ** 2)
    ssim_value = ssim(img12, img2, multichannel=True)
    uciqe = 1 - (0.0448 * mse + 0.2856 * (1 - ssim_value))
    return uciqe
def greycomatrix(image, distances, angles, levels, symmetric=False, normed=False):
    # 将图像量化为指定的灰度级别
    image = np.round((image / 255.0) * (levels - 1)).astype(int)
    h, w = image.shape
    glcm = np.zeros((levels, levels, len(distances), len(angles)))

    for d_idx, d in enumerate(distances):
        for a_idx, a in enumerate(angles):
            # 根据距离和角度计算偏移
            x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
            x_offset = int(round(d * np.cos(a)))
            y_offset = int(round(d * np.sin(a)))

            # 创建一个偏移的图像
            x = np.clip(x_coords + x_offset, 0, w - 1)
            y = np.clip(y_coords + y_offset, 0, h - 1)
            offset_image = image[y, x]

            # 计算GLCM
            for i in range(levels):
                for j in range(levels):
                    glcm[i, j, d_idx, a_idx] = np.sum((image == i) & (offset_image == j))

    if symmetric:
        # 使GLCM对称
        for d_idx in range(len(distances)):
            for a_idx in range(len(angles)):
                glcm[:, :, d_idx, a_idx] = (glcm[:, :, d_idx, a_idx] + glcm[:, :, d_idx, a_idx].T) / 2

    if normed:
        # 归一化GLCM
        for d_idx in range(len(distances)):
            for a_idx in range(len(angles)):
                glcm[:, :, d_idx, a_idx] /= glcm[:, :, d_idx, a_idx].sum()

    return glcm


def calculate_psnr(img1, img2, max_pixel_value=255):
    print(img1.shape)
    print(img2.shape)
    # 确保图像是浮点数类型
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 计算 MSE
    mse = np.mean((img1 - img2) ** 2)

    # 计算 PSNR
    if mse == 0:
        return float('inf')
    else:
        return 10 * np.log10((max_pixel_value ** 2) / mse)
# 计算UIQM（简化版）
def calculate_UIQM(img):
    gray_img = rgb2gray(img)
    # 计算灰度图像的对比度
    contrast = greycoprops(greycomatrix(gray_img, [1], levels=256, symmetric=True, normed=False), 'contrast')[0, 0]
    # 计算灰度图像的均匀性
    homogeneity = greycoprops(greycomatrix(gray_img, [1], levels=256, symmetric=True, normed=False), 'homogeneity')[
        0, 0]
    # 计算灰度图像的清晰度（锐度）
    # 这里使用Laplacian算子作为示例
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    sharpness = np.mean(np.abs(np.convolve(gray_img, kernel, mode='same')))

    # 计算UIQM值，这里使用简单的加权和作为示例
    # 实际的UIQM计算可能需要更复杂的方法和更多的特征
    uiqm_value = 0.4 * contrast + 0.4 * homogeneity + 0.2 * sharpness

    return uiqm_value


# 读取图像并计算指标
def calculate_metrics(img_path1, img_path2, output_path):
    files1 = os.listdir(img_path1)
    files2 = os.listdir(img_path2)
    files1.sort()
    files2.sort()
    results = []
    for f1, f2 in zip(files1, files2):
        if f1 != f2:
            continue
        img1 = cv2.imread(os.path.join(img_path1, f1))
        img2 = cv2.imread(os.path.join(img_path2, f2))
        cv2.imshow('img_eq', img1)
        plt.show()
        cv2.waitKey(0)
        cv2.imshow('img_eq', img2)
        plt.show()
        cv2.waitKey(0)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # 计算PSNR
        img1_resized = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
        # 确保两个图像的尺寸相同
        if img1_resized.shape != img2.shape:
            raise ValueError("Images do not have the same dimensions after resizing.")
        print(img1_resized.shape)
        print(img2.shape)
        psnr_value = calculate_psnr(img1_resized,img2)

        # 计算UCIQE
        uciqe_value = calculate_UCIQE(img1, img2)

        # 计算UIQM
        uiqm_value = calculate_UIQM(img2)

        results.append([f1, "blurry", psnr_value, uciqe_value, uiqm_value])

    # 将结果写入Excel文件
    df = pd.DataFrame(results, columns=['Image File Name', 'Degraded Image Classification', 'PSNR', 'UCIQE', 'UIQM'])
    df.to_excel(output_path, index=False)


# 图像文件夹路径
img_folder1 = "F:\\AMPC\\AMPC\\picture\\result\\blurry"
img_folder2 = "F:\\AMPC\\AMPC\\picture\\result\\blurry_st"

# 输出Excel文件路径
output_excel = 'Answer.xls'

# 计算评价指标并写入Excel
calculate_metrics(img_folder1, img_folder2, output_excel)