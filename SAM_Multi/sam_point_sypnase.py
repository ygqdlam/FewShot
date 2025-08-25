import torch
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import cv2

# -------------------------
# 1. 加载模型
# -------------------------
sam_checkpoint = "sam_vit_h_4b8939.pth"   # 模型权重路径
model_type = "vit_h"                      # 可选: vit_h, vit_l, vit_b
device = "cuda" if torch.cuda.is_available() else "cpu"

#sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam = sam_model_registry['vit_b'](checkpoint='/home/yanggq/project/Few-shot/checkpoints/sam_vit_b_01ec64.pth')

sam.to(device=device)
predictor = SamPredictor(sam)

# -------------------------
# 2. 加载图片
# -------------------------
"""image_path = '/home/yanggq/project/Few-shot/segment-anything-main/test.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

# -------------------------
# 3. 定义不同类别的提示
# 例如：猫、狗、人
# 每个类别用不同的点提示
# -------------------------
categories = {
    "cat": np.array([[200, 500]]),
    "dog": np.array([[600, 500]])
}"""


# ========== 2. 加载图片 ==========
data = np.load("/home/yanggq/project/Few-shot/H-SAM-main/datasets/Synapse/train_npz/case0010_slice084.npz")
image = data["image"]      # (H, W) 或 (H, W, 3)
label = data["label"]      # (H, W)
if image.ndim == 2:  
    image = np.stack([image]*3, axis=-1)   # (H, W, 3)
target_size = 224
image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

predictor.set_image(image_resized)

label_resized = cv2.resize(label, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
image_tensor = torch.tensor(image_resized).permute(2, 0, 1).unsqueeze(0).float().cuda()  # [1, 3, H, W]
categories = {}
organ_names = [
    "spleen", "right_kidney", "left_kidney", "gallbladder", 
    "esophagus", "liver", "stomach", "aorta"
]

for i, cls in enumerate(range(1, 9)):  # 1~8
    coords = np.argwhere(label_resized == cls)
    if coords.size > 0:
        y, x = coords[len(coords)//2]   # 中心点
        categories[organ_names[i]] = np.array([[x, y]])


category_masks = {}

for cls_name, input_point in categories.items():
    input_label = np.array([1])  # 1=正样本点
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True  # 返回多个候选 mask
    )
    best_mask = masks[np.argmax(scores)]  # 取置信度最高的一个
    category_masks[cls_name] = best_mask
# -------------------------
# 4. 可视化结果
# -------------------------
plt.figure(figsize=(8,8))
plt.imshow(image)
for i, (cls, mask) in enumerate(category_masks.items()):
    color = np.random.rand(3,)
    plt.imshow(mask, cmap="jet", alpha=0.4)  # 每个 mask 叠加
    y, x = np.where(mask)
    if len(x) > 0:
        plt.text(x.mean(), y.mean(), cls, color="white",
                 bbox=dict(facecolor="black", alpha=0.5))
plt.axis("off")
plt.show()