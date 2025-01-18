import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.io import imread

# ตรวจสอบสไตล์ที่ใช้ได้
available_styles = plt.style.available

# เลือกสไตล์ที่มีอยู่
style_name = "seaborn-darkgrid" if "seaborn-darkgrid" in available_styles else "ggplot"
plt.style.use(style_name)

# โหลดภาพ
img_path = "F:\\My_Works_Programer\\My_Coding_Practice\\Computer_Vision_Project\\images\\test_cv_project_2.jpg"
img = imread(img_path)
img = rgb2gray(img)

# ค่าเริ่มต้นของวงกลม
center_x, center_y = 500, 270
radius_x, radius_y = 300, 200
num_points = 400

# ฟังก์ชันสำหรับสร้าง contour วงกลม
def create_contour(cx, cy, rx, ry, points):
    s = np.linspace(0, 2 * np.pi, points)
    r = cy + ry * np.sin(s)
    c = cx + rx * np.cos(s)
    return np.array([r, c]).T

# ค่าจุดเริ่มต้น
init = create_contour(center_x, center_y, radius_x, radius_y, num_points)

# สร้างหน้าต่างแสดงภาพ
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left=0.1, bottom=0.35)  # เว้นที่ด้านล่างให้ Slider
ax.imshow(img, cmap="gray")
contour_line, = ax.plot(init[:, 1], init[:, 0], '--r', linewidth=2, label="Initial Contour")
ax.legend(loc="upper right", fontsize=12)

# -------------------------------
# 🎨 ปรับแต่ง UI ของ Slider
# -------------------------------
slider_color = "lightblue"

ax_center_x = plt.axes([0.25, 0.25, 0.55, 0.03], facecolor=slider_color)
ax_center_y = plt.axes([0.25, 0.2, 0.55, 0.03], facecolor=slider_color)
ax_radius_x = plt.axes([0.25, 0.15, 0.55, 0.03], facecolor=slider_color)
ax_radius_y = plt.axes([0.25, 0.1, 0.55, 0.03], facecolor=slider_color)

slider_center_x = Slider(ax_center_x, "Center X", 0, img.shape[1], valinit=center_x, valfmt="%0.0f")
slider_center_y = Slider(ax_center_y, "Center Y", 0, img.shape[0], valinit=center_y, valfmt="%0.0f")
slider_radius_x = Slider(ax_radius_x, "Radius X", 10, img.shape[1] // 2, valinit=radius_x, valfmt="%0.0f")
slider_radius_y = Slider(ax_radius_y, "Radius Y", 10, img.shape[0] // 2, valinit=radius_y, valfmt="%0.0f")

# ฟังก์ชันอัปเดตเส้น contour ตาม Slider
def update(val):
    new_center_x = slider_center_x.val
    new_center_y = slider_center_y.val
    new_radius_x = slider_radius_x.val
    new_radius_y = slider_radius_y.val

    new_contour = create_contour(new_center_x, new_center_y, new_radius_x, new_radius_y, num_points)
    contour_line.set_data(new_contour[:, 1], new_contour[:, 0])
    fig.canvas.draw_idle()

# เชื่อม Slider กับฟังก์ชัน update
slider_center_x.on_changed(update)
slider_center_y.on_changed(update)
slider_radius_x.on_changed(update)
slider_radius_y.on_changed(update)

# -------------------------------
# 🎨 ปุ่ม "Start" สไตล์โมเดิร์น
# -------------------------------
ax_button = plt.axes([0.78, 0.02, 0.15, 0.06])
button = Button(ax_button, "▶ Start", color="skyblue", hovercolor="deepskyblue")

def apply_active_contour(event):
    global init
    init = create_contour(slider_center_x.val, slider_center_y.val, slider_radius_x.val, slider_radius_y.val, num_points)

    # ใช้ active contour model
    snake = active_contour(
        gaussian(img, sigma=3, preserve_range=False),
        init,
        alpha=0.001,
        beta=0.000005,
        gamma=0.001,
    )

    # แสดงผลลัพธ์ใหม่
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.imshow(img, cmap="gray")
    ax2.plot(init[:, 1], init[:, 0], '--r', linewidth=2, label="Initial Contour")
    ax2.plot(snake[:, 1], snake[:, 0], '-b', linewidth=2, label="Snake Contour")
    ax2.legend(loc="upper right", fontsize=12)
    plt.show()

button.on_clicked(apply_active_contour)

plt.show()