import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# ====== 1. 读取 npz 文件 ======
data = np.load("/home/yanggq/project/Few-shot/H-SAM-main/datasets/Synapse/train_npz/case0010_slice084.npz")
image = data["image"]      # (H, W) 或 (H, W, 3)
label = data["label"]      # (H, W)

# ====== 2. 转换为 3 通道 ======
if image.ndim == 2:  
    image = np.stack([image]*3, axis=-1)   # (H, W, 3)

# ====== 3. resize 图像和标签 ======
target_size = 224
image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
label_resized = cv2.resize(label, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

# ====== 4. 可视化：叠加 mask ======
organ_names = [
    "spleen", "right_kidney", "left_kidney", "gallbladder", 
    "esophagus", "liver", "stomach", "aorta"
]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_resized)
plt.title("Resized Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image_resized)

for i, cls in enumerate(range(1, 9)):  # 1~8 类器官
    mask = (label_resized == cls)
    if mask.sum() == 0:
        continue
    plt.imshow(mask, cmap="jet", alpha=0.4)  # 半透明叠加
    y, x = np.where(mask)
    plt.text(x.mean(), y.mean(), organ_names[i], color="white",
             bbox=dict(facecolor="black", alpha=0.5), fontsize=8)

plt.title("Resized Label + Masks")
plt.axis("off")
plt.show()