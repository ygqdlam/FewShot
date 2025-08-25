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
image_path = '/home/yanggq/project/Few-shot/segment-anything-main/test.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_h, image_w = image.shape[:2]


predictor.set_image(image)

# -------------------------
# 3. 定义不同类别的提示
# 例如：猫、狗、人
# 每个类别用不同的点提示
# -------------------------
categories = {
    "cat": np.array([[200, 500]]),
    "dog": np.array([[600, 500]])
}

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

# -------------------------
# 关键修改：合并掩码并保存（cat=1，dog=2，背景=0）
# -------------------------
# 1. 创建空白标签图（背景为0，尺寸与原图一致）
merged_label = np.zeros((image_h, image_w), dtype=np.uint8)  # 用uint8节省内存，支持0/1/2

# 2. 赋值：cat区域设为1，dog区域设为2（若有重叠，后赋值的会覆盖前一个，可根据需求调整顺序）
merged_label[category_masks["cat"]] = 1  # cat 对应值1
merged_label[category_masks["dog"]] = 2  # dog 对应值2

# 3. 保存合并后的标签图（支持png/jpg/npz格式，推荐png避免压缩失真）
save_path = "/home/yanggq/project/Few-shot/segment-anything-main/merged_segmentation_label.png"
cv2.imwrite(save_path, merged_label)
print(f"✅ 合并标签图已保存至：{save_path}")