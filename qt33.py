import numpy as np
import cv2
import os

def auto_white_balance(image_input):
    # 读取图像
    if image_input is not None:
        # 分离各个通道
        image_bgr = cv2.split(image_input)

        # 求RGB分量的均值(opencv中排列顺序是B,G,R)
        b, g, r = image_bgr
        b_mean = np.mean(b)
        g_mean = np.mean(g)
        r_mean = np.mean(r)

        # 计算各分量的增益
        k_r = (r_mean + g_mean + b_mean) / (3 * r_mean)*0.45
        k_g = (r_mean + g_mean + b_mean) / (3 * g_mean)
        k_b = (r_mean + g_mean + b_mean) / (3 * b_mean)
        # 计算各通道变换后的灰度值
        blue_channel = b * k_b
        green_channel = g * k_g
        red_channel = r * k_r

        # 确保值在0-255范围内
        blue_channel = np.clip(blue_channel, 0, 255).astype(np.uint8)
        green_channel = np.clip(green_channel, 0, 255).astype(np.uint8)
        red_channel = np.clip(red_channel, 0, 255).astype(np.uint8)

        image_source = cv2.merge((blue_channel, green_channel, red_channel))

        # 显示图像
        # cv2.imshow("原始图像", image_input)
        # cv2.imshow("白平衡调整后", image_source)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return image_source

if __name__ == "__main__":
    img_files = []
    path = "pc/color"
    save_path = "pc/color_st"
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
        finally_img = auto_white_balance(image)
        # sceneRadiance_sec=RecoverHE(sceneRadiance)
        cv2.imwrite(full_save_path, finally_img)