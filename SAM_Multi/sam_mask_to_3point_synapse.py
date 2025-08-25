import torch
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import cv2
import random

# -------------------------
# 1. 加载模型
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry['vit_b'](checkpoint='/home/yanggq/project/Few-shot/checkpoints/sam_vit_b_01ec64.pth')
sam.to(device=device)
predictor = SamPredictor(sam)

# -------------------------
# 2. 工具函数：从掩码随机选3个点
# -------------------------
def get_3_random_points(coords):
    if len(coords) < 3:
        selected = coords
    else:
        np.random.shuffle(coords)
        selected = coords[:2]
    return np.array([[x, y] for y, x in selected])

# -------------------------
# 3. 加载图片与预处理
# -------------------------
data = np.load("/home/yanggq/project/Few-shot/H-SAM-main/datasets/Synapse/train_npz/case0010_slice084.npz")
image = data["image"]
label = data["label"]
target_size = 224

# 灰度转RGB+Resize
if image.ndim == 2:  
    image = np.stack([image]*3, axis=-1)
image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
label_resized = cv2.resize(label, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

# 初始化SAM图像特征
predictor.set_image(image_resized)

# -------------------------
# 4. 生成3个随机点Prompt
# -------------------------
categories = {}
organ_names = [
    "spleen", "right_kidney", "left_kidney", "gallbladder", 
    "esophagus", "liver", "stomach", "aorta"
]

for i, cls in enumerate(range(1, 9)):
    organ_name = organ_names[i]
    coords = np.argwhere(label_resized == cls)
    if coords.size == 0:
        print(f"⚠️ {organ_name}：当前切片无该器官，跳过")
        continue
    random_3_points = get_3_random_points(coords)
    categories[organ_name] = random_3_points
    print(f"✅ {organ_name} 3个随机点坐标：\n{random_3_points}")

# -------------------------
# 5. SAM分割
# -------------------------
category_masks = {}
for cls_name, input_points in categories.items():
    input_labels = np.array([1]*len(input_points))
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )
    best_mask = masks[np.argmax(scores)]
    category_masks[cls_name] = best_mask
    print(f"✅ {cls_name} 分割完成，置信度：{scores[np.argmax(scores)]:.4f}")

print("实际分割出的类别：")
for organ_name in category_masks.keys():
    print(f"- {organ_name}")

    
# -------------------------
# 6. 可视化：白色背景 + 深色目标
# -------------------------
plt.figure(figsize=(10, 10))

# 白色背景
white_bg = np.ones_like(image_resized, dtype=np.float32)  # 全 1 = 白色
plt.imshow(white_bg)

# 深色调色板
dark_organ_colors = {
    "spleen": (139/255, 0, 0),        # 深红
    "right_kidney": (0, 100/255, 0),   # 深绿
    "left_kidney": (0, 0, 139/255),    # 深蓝
    "gallbladder": (184/255, 134/255, 11/255),  # 暗金黄
    "esophagus": (139/255, 0, 139/255), # 深紫
    "liver": (0, 139/255, 139/255),    # 深青
    "stomach": (139/255, 69/255, 19/255), # 巧克力棕
    "aorta": (85/255, 107/255, 47/255)   # 暗橄榄绿
}

# 绘制器官掩码
for cls_name, mask in category_masks.items():
    dark_color = dark_organ_colors[cls_name]
    
    # 生成器官掩码层
    organ_mask_layer = np.ones_like(white_bg, dtype=np.float32)  # 初始全白
    organ_mask_layer[mask] = dark_color  # 掩码区域涂成深色
    
    plt.imshow(organ_mask_layer, alpha=0.9)
    
    # 绘制随机点
    input_points = categories[cls_name]
    for idx, (x, y) in enumerate(input_points):
        plt.scatter(
            x=x, y=y,
            color="#FFA500",
            s=50 + idx*50,
            marker="o",
            edgecolor="#000000",
            linewidth=2.5,
            zorder=10
        )
        plt.text(
            x+6, y+6, str(idx+1),
            color="black",
            fontsize=6,
            weight="bold",
            bbox=dict(facecolor="white", pad=1.5, edgecolor="black", linewidth=1)
        )
    
    # 类别名称（文字颜色与类别颜色一致，无背景框）
    y_coords, x_coords = np.where(mask)
    if len(x_coords) > 0:
        center_x = x_coords.mean()
        center_y = y_coords.mean()
        plt.text(
            center_x, center_y,
            cls_name,
            color=dark_color,   # 文本颜色与类别颜色保持一致
            fontsize=10,
            weight="bold",
            bbox=dict(facecolor="white", alpha=0.6, pad=1, edgecolor="none")  # 可选白底提高可读性
        )

# 隐藏坐标轴
plt.axis("off")
plt.title("SAM 医学分割结果（白色背景+深色器官）", fontsize=16, pad=20, color="#333333")

# 图例
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black")
    for color in dark_organ_colors.values()
] + [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFA500', markersize=12, 
               markeredgecolor='#000000', markeredgewidth=2, label='随机选点')]

plt.legend(
    legend_elements, list(dark_organ_colors.keys()) + ["随机选点"],
    loc="center left", bbox_to_anchor=(1.02, 0.5),
    fontsize=11,
    frameon=True,
    facecolor="#FFFFFF",
    edgecolor="#CCCCCC"
)

plt.savefig(
    "medical_segmentation_white_bg.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.3,
    facecolor="white"
)
plt.show()