import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.io import imread

# โหลดภาพจากไฟล์ในเครื่อง
img_path = "F:\\My_Works_Programer\\My_Coding_Practice\\Computer_Vision_Project\\images\\test_cv_project.jpg"  # แก้ไขเส้นทางไฟล์ภาพ
img = imread(img_path)
img = rgb2gray(img)

# กำหนดค่าเริ่มต้นของ contour (วงกลมรอบวัตถุ)
s = np.linspace(0, 2 * np.pi, 400)
r = 270 + 200 * np.sin(s)
c = 500 + 300 * np.cos(s)
init = np.array([r, c]).T

print(init)

# ใช้ active contour model
snake = active_contour(
    gaussian(img, sigma=3, preserve_range=False),
    init,
    alpha=0.001,
    beta=0.000005,
    gamma=0.001,
)

# แสดงผลลัพธ์
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', label="Initial Contour")
ax.plot(snake[:, 1], snake[:, 0], '-b', label="Snake Contour")
# ax.set_xticks([]), ax.set_yticks([])
# ax.axis([0, img.shape[1], img.shape[0], 0])
ax.legend()
plt.show()
