import numpy as np
import cv2
from skimage import io, color
from skimage.filters import sobel
from skimage.segmentation import active_contour

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from scipy.interpolate import splprep, splev

from skimage.filters import gaussian

# โหลดรูปภาพ
image = io.imread(
    "F:\\My_Works_Programer\\My_Coding_Practice\\Computer_Vision_Project\\images\\test_cv_project.jpg",
    as_gray=True,
)

# ใช้ Gaussian Blur เพื่อลด noise
blurred_image = cv2.GaussianBlur((image * 255).astype(np.uint8), (3, 3), 0)

sharpened = cv2.addWeighted(blurred_image, 1.5, blurred_image, -0.5, 0)

edges = cv2.Canny(sharpened, threshold1=50, threshold2=200)
edges = (edges / edges.max() * 255).astype(np.uint8)

# สร้างตัวแปรเก็บจุดที่ผู้ใช้เลือก
polygon_points = []

# ฟังก์ชัน callback เมื่อผู้ใช้เลือกเส้นรอบวัตถุ
def on_select(verts):
    global polygon_points
    polygon_points = np.array(verts)
    plt.close()  # ปิดหน้าต่างเมื่อเลือกเสร็จ

# แสดงรูปภาพเพื่อให้ผู้ใช้ลากเส้นรอบวัตถุ
fig, ax = plt.subplots()
ax.imshow(image, cmap="gray")
ax.set_title("Select Object And Enter")
polygon_selector = PolygonSelector(ax, on_select)
plt.show()

# ถ้ามีจุดที่เลือกแล้ว นำไปใช้เป็นเส้นเริ่มต้น
if len(polygon_points) > 0:
    polygon_points = np.array(polygon_points)
    tck, u = splprep(polygon_points.T, s=0, per=True)
    u_new = np.linspace(u.min(), u.max(), 400)
    polygon_points = np.array(splev(u_new, tck)).T

    print(polygon_points)

    snake = active_contour(
        gaussian(image, sigma=3, preserve_range=False),
        polygon_points,
        alpha=0.001,  # ลดการเปลี่ยนแปลงของเส้นโค้ง
        beta=0.000005,   # ลดความแข็งของเส้นโค้ง (ให้ยืดหยุ่นมากขึ้น)
        gamma=0.001,   # เพิ่มอัตราการเปลี่ยนแปลงของเส้น
    )

    # แสดงผลลัพธ์
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.plot(polygon_points[:, 0], polygon_points[:, 1], "--r", label="Initial Contour")
    ax.plot(snake[:, 0], snake[:, 1], "-b", label="Final Contour")
    ax.legend()
    plt.show()
else:
    print("No points were selected!")
