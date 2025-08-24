import cv2
import matplotlib.pyplot as plt

# 读取图像
image_path = 'image_004.png'  # 替换为你的图片路径
image = cv2.imread(image_path)

# 转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 计算直方图
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])  # 256是直方图的bin数量

# 绘制直方图
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
plt.plot(hist)
plt.xlim([0, 256])
plt.show()