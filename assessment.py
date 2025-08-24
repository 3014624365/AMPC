import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray

import math


# 读取图像并计算PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1e-10:
        return 100
    return 20 * (math.log10(1.0 / math.sqrt(mse)))
def calculate_UCIQE(img1, img2):
    # 将图像转换为浮点数
    img1 = img_as_float(img1)
    img2 = img_as_float(img2)

    # 计算MSE
    mse = np.mean((img1 - img2) ** 2)

    # 计算SSIM
    from skimage.metrics import structural_similarity as ssim
    ssim_value = ssim(img1, img2, multichannel=True)

    # 计算UCIQE，这里只是一个示例，实际计算更复杂
    uciqe = 1 - (0.0448 * mse + 0.2856 * (1 - ssim_value))
    return uciqe

# 计算评价指标
def calculate_metrics(img_path1, img_path2, output_path):
    # 读取两个文件夹中的图像文件名
    files1 = os.listdir(img_path1)
    files2 = os.listdir(img_path2)
    # 确保文件名排序一致
    files1.sort()
    files2.sort()

    # 初始化结果列表
    results = []

    # 遍历文件名列表
    for f1, f2 in zip(files1, files2):
        # 读取图像
        img1 = cv2.imread(os.path.join(img_path1, f1))
        img2 = cv2.imread(os.path.join(img_path2, f2))

        # 假设图像需要是相同的尺寸
        img1 = cv2.resize(img1, (256, 256))
        img2 = cv2.resize(img2, (256, 256))

        # 计算PSNR
        psnr_value = calculate_psnr(img1, img2)

        # 计算UCIQE
        uciqe_value = calculate_UCIQE(img1, img2)

        # 计算UIQM
        uiqm_value = calculate_UIQM(img1, img2)

        # 将结果添加到列表中
        results.append([f1, psnr_value, uciqe_value, uiqm_value])

    # 将结果写入Excel文件
    df = pd.DataFrame(results, columns=['image file name', 'PSNR', 'UCIQE', 'UIQM'])
    df.to_excel(output_path, index=False)

def calculate_UIQM(img):
    gray_img = rgb2gray(img)
    clarity = np.mean(gray_img) ** 2 / np.var(gray_img)
    information = -np.sum([np.sum(np.histogram(gray_img, i, density=True)[0] * np.log2(np.histogram(gray_img, i, density=True)[0] + 1e-10)) for i in range(256)]) / 256
    color_info = np.std(img, axis=2).mean()
    uiqm = clarity + information + color_info
    return uiqm
    # path = "F:\\AMPC\\AMPC\\picture\\图片\\input"
    # sec_path="F:\\AMPC\\AMPC\\picture\\图片\\output"
img_folder1 = "F:\\AMPC\\AMPC\\picture\\result\\output"
img_folder2 = "F:\\AMPC\\AMPC\\picture\\result\\input"
# 输出Excel文件路径
output_excel = 'Answer.xls'
# 计算评价指标并写入Excel
calculate_metrics(img_folder1, img_folder2, output_excel)