import cv2
import numpy as np
# 读取两张图像
image_path1 ='test_002.png'  # 替换为你的第一张图片路径
image_path2 = 'test_002_1.png'  # 替换为你的第二张图片路径
image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)

# 调整第二张图像的尺寸以匹配第一张图像的高度
height1, width1, _ = image1.shape
height2, width2, _ = image2.shape
image2_resized = cv2.resize(image2, (width2, height1))

# 横向拼接两张图像
cv2.vconcat([image1, image2])

# 显示对比图
cv2.imshow('对比图',imp)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存对比图
cv2.imwrite('对比图.jpg', 对比图)