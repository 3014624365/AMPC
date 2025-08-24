import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def not_Blurry(img):
    img = cv2.resize(img, (640,480))
    # cv2.imshow('img', img)
    kernel_1 = np.ones((3,3))
    for i in range(len(kernel_1)):
        for j in range(len(kernel_1)):
            kernel_1[i][j]=-1
    m=len(kernel_1)//2
    kernel_1[m][m]=9
    img1 = cv2.filter2D(img, -1, kernel_1)
    gauss = cv2.GaussianBlur(img, (9,9), 0,)
    img2 = cv2.addWeighted(img, 1.5, gauss, -0.5, 3)
    img = cv2.addWeighted(img1, 0.4, img2, 0.6, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img)
    hist = cv2.calcHist([v], [0], None, [256], [0,155])
    plt.plot(hist)
    v = cv2.equalizeHist(v)
    hist = cv2.calcHist([v], [0], None, [256], [0,256])
    plt.plot(hist)
    img_merge = cv2.merge((h, s, v))
    img_eq = cv2.cvtColor(img_merge, cv2.COLOR_HSV2BGR)
    return img_eq
# cv2.imshow('img_eq', img_eq)
# plt.show()
# cv2.waitKey(0)
if __name__ == '__main__':
    img_files = []
    path = "pc/blurry"
    save_path="pc/blurry_st"
    kp_full = []
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.svg']
    # 遍历目录中的所有文件
    for root, dirs, files in os.walk(path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                img_files.append(file)
    for name_file in img_files:
        full_path = os.path.join(path, name_file)
        full_save_path = os.path.join(save_path, name_file)
        image = cv2.imread(full_path)
        finally_img = not_Blurry(image)
        # sceneRadiance_sec=RecoverHE(sceneRadiance)
        cv2.imwrite(full_save_path,finally_img)