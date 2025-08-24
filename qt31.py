import os
import numpy as np
import cv2
import natsort
from skimage import exposure

def RecoverHE(sceneRadiance):
    for i in range(3):
        sceneRadiance[:, :, i] = cv2.equalizeHist(sceneRadiance[:, :, i])
    return sceneRadiance

def RecoverCLAHE(sceneRadiance):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    for i in range(3):
        sceneRadiance[:, :, i] = clahe.apply(sceneRadiance[:, :, i])
    return sceneRadiance

def tei(img_path,file_path):
    os.makedirs(file_path, exist_ok=True)
    destination_path = os.path.join(file_path, os.path.basename(img_path))
    shutil.copy(img_path, destination_path)
    return

np.seterr(over='ignore')
if __name__ == '__main__':
    img_files = []
    path = "pc/dark"
    save_path="pc/dark_st"
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
        sceneRadiance = RecoverHE(image)
        # sceneRadiance_sec=RecoverHE(sceneRadiance)
        cv2.imwrite(full_save_path,sceneRadiance)


