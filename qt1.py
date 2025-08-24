import argparse
import cv2
import numpy as np
import os
import shutil
# # 定义一个函数，算图像的拉普拉斯方差，用于评估图像的清晰度
# def variance_of_laplacian(image):
#     # 使用cv2.Laplacian计算图像的拉普拉斯变换，并返回变换后的方差
#     return cv2.Laplacian(image, cv2.CV_64F).var()
#定义阈值
threshold_Blurry=100
threshold_color=1.5
threshold_dark=60.0
Blurry_path="F:\\AMPC\\result\\pt_test\\blurry"
color_path="F:\\AMPC\\result\\pt_test\\color"
dark_path="F:\\AMPC\\result\\pt_test\\dark"
# 定义一个函数，计算图像的Sobel梯度均值
# 定义一个函数，判断图像是否模糊
def compute_sobel(image):
    # 将图像从BGR转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算水平和垂直方向的Sobel梯度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    # 计算梯度的合成幅度
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 打印梯度的均值
    print(np.mean(sobel))
    # 返回梯度的均值
    return np.mean(sobel)

# 定义一个函数，判断图像是否模糊
def is_blurry(image):
    # 将图像从BGR转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算拉普拉斯方差
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 计算Sobel梯度均值
    sobel_mean = compute_sobel(image)
    # 计算模糊度量值，这里结合了拉普拉斯方差和Sobel均值255是参数，根据你场景的图片进行调整
    blur_measure = (0.5 * fm) + (0.5 * (sobel_mean-255))
    # 如果模糊度量值小于阈值，则返回True，表示图像模糊
    return blur_measure

def color_test(image):
    b, g, r = cv2.split(image)
    print(image.shape)
    m, n, z = image.shape
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    d_a, d_b, M_a, M_b = 0, 0, 0, 0
    for i in range(m):
        for j in range(n):
            d_a = d_a + a[i][j]
            d_b = d_b + b[i][j]
    d_a, d_b = (d_a / (m * n)) - 128, (d_b / (n * m)) - 128
    D = np.sqrt((np.square(d_a) + np.square(d_b)))
    for i in range(m):
        for j in range(n):
            M_a = np.abs(a[i][j] - d_a - 128) + M_a
            M_b = np.abs(b[i][j] - d_b - 128) + M_b
    M_a, M_b = M_a / (m * n), M_b / (m * n)
    M = np.sqrt((np.square(M_a) + np.square(M_b)))
    k = D / M
    print('偏色值:%f' % k)
    return k
def tei(img_path,file_path):
    os.makedirs(file_path, exist_ok=True)
    destination_path = os.path.join(file_path, os.path.basename(img_path))
    shutil.copy(img_path, destination_path)
    return
def is_dark(image_path):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算平均亮度
    mean_brightness = np.mean(gray_image)
    # 计算亮度直方图
    hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])
    # 计算亮度标准差（可选）
    std_brightness = np.std(gray_image)
    return mean_brightness
def judge(image,img_path):
    # 种类判断常数
    image_dark_1=is_dark(image)
    image_color=color_test(image)
    image_Blurry=is_blurry(image)
    print(is_blurry(image))
    if image_Blurry<threshold_Blurry:
        tei(img_path,Blurry_path)
    if image_color>threshold_color:
        tei(img_path, color_path)
    if image_dark_1<threshold_dark:
        tei(img_path, dark_path)
    return
if __name__ == '__main__':
    img_files = []
    path = "F:\\AMPC\\AMPC\\picture\\result\\input"
    #sec_path = "F:\\AMPC\\AMPC\\picture\\图片\\output"
    kp_full = []
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.svg']
    # 遍历目录中的所有文件
    for root, dirs, files in os.walk(path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                img_files.append(file)
    for name_file in img_files:
        full_path = os.path.join(path, name_file)
        image = cv2.imread(full_path)
        judge(image,full_path)
