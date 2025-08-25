import torch
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import cv2
import random

# -------------------------
# 1. 基础配置
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry['vit_b'](checkpoint='/home/yanggq/project/Few-shot/checkpoints/sam_vit_b_01ec64.pth')
sam.to(device=device)
predictor = SamPredictor(sam)

# 路径配置
saved_segment_path = "/home/yanggq/project/Few-shot/segment-anything-main/merged_segmentation_label.png"
image_path = '/home/yanggq/project/Few-shot/segment-anything-main/test.jpg'
target_size = 1024


# -------------------------
# 2. 从掩码中随机选择3个点（核心函数）
# -------------------------
def get_3_random_points_from_mask(mask, num_points=3):
    """从掩码的非零区域随机选择3个点，确保点之间有一定距离"""
    # 获取所有有效坐标 (y, x)
    valid_coords = np.argwhere(mask > 0)
    if len(valid_coords) < num_points:
        raise ValueError(f"掩码有效区域过小，无法选择{num_points}个点！")
    
    # 随机打乱并选择前3个点（保证一定随机性）
    np.random.shuffle(valid_coords)
    selected_coords = valid_coords[:num_points]
    
    # 转换为SAM要求的格式 (N, 2)，其中N=3，每个点为(x, y)
    points = np.array([[x, y] for y, x in selected_coords])
    return points


# -------------------------
# 3. 加载数据
# -------------------------
def load_data(image_path, saved_mask_path, target_size):
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像：{image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    
    # 加载历史掩码
    saved_mask = cv2.imread(saved_mask_path, cv2.IMREAD_GRAYSCALE)
    if saved_mask is None:
        raise ValueError(f"无法读取历史分割结果：{saved_mask_path}")
    saved_mask_resized = cv2.resize(saved_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    
    # 按类别拆分掩码
    cat_mask = (saved_mask_resized == 1).astype(np.float32)
    dog_mask = (saved_mask_resized == 2).astype(np.float32)
    
    return image_resized, {"cat": cat_mask, "dog": dog_mask}


# 执行加载
image_resized, class_masks = load_data(image_path, saved_segment_path, target_size)
predictor.set_image(image_resized)


# -------------------------
# 4. 选择3个随机点 + 分割
# -------------------------
random_points = {}  # 存储每个类别的3个点
category_masks = {}  # 存储分割结果

for cls_name, mask in class_masks.items():
    # 选择3个随机点
    points = get_3_random_points_from_mask(mask)
    random_points[cls_name] = points
    print(f"✅ {cls_name} 3个选点坐标：\n{points}")
    
    # 分割（使用3个点作为提示）
    # 点标签全部设为1（正样本）
    input_labels = np.array([1]*len(points))
    
    masks, scores, _ = predictor.predict(
        point_coords=points,  # 传入3个点
        point_labels=input_labels,
        multimask_output=True
    )
    
    # 选择置信度最高的掩码
    best_mask = masks[np.argmax(scores)].astype(np.bool_)
    category_masks[cls_name] = best_mask


# -------------------------
# 5. 可视化（显示3个点和分割结果）
# -------------------------
plt.figure(figsize=(12, 12))
plt.imshow(image_resized)

# 类别颜色配置
cls_colors = {
    "cat": (1, 0, 0),    # 红色
    "dog": (0, 1, 0)     # 绿色
}

for cls_name, mask in category_masks.items():
    color = cls_colors[cls_name]
    points = random_points[cls_name]
    
    # 1. 显示分割掩码
    colored_mask = np.zeros_like(image_resized, dtype=np.float32)
    colored_mask[mask] = color
    plt.imshow(colored_mask, cmap="gray", alpha=0.4)
    
    # 2. 显示3个随机点（不同大小区分）
    for i, (x, y) in enumerate(points):
        # 点的大小逐渐增大，便于区分
        plt.scatter(
            x=x, y=y,
            color="yellow", 
            s=80 + i*40,  # 第一个点80，第二个120，第三个160
            marker="o", 
            edgecolor="black", 
            linewidth=2,
            label=f"{cls_name} 点{i+1}" if i == 0 else ""  # 只在第一个点显示标签
        )
    
    # 3. 显示类别标签（放在3个点的中心位置）
    points_center = np.mean(points, axis=0).astype(int)  # 计算3个点的中心
    offset = 30  # 偏移量，避免覆盖点
    label_x = min(points_center[0] + offset, target_size - 50)
    label_y = min(points_center[1] + offset, target_size - 50)
    
    plt.text(
        label_x, label_y,
        f"{cls_name}\n(3个点)",
        color="white", 
        fontsize=11,
        bbox=dict(facecolor=color, alpha=0.8, pad=3)
    )

plt.legend(loc="upper right")
plt.axis("off")
plt.title("SAM 分割结果（3个随机点作为提示）", fontsize=14)
plt.show()


# 保存结果
new_merged_label = np.zeros((target_size, target_size), dtype=np.uint8)
new_merged_label[category_masks["cat"]] = 1
new_merged_label[category_masks["dog"]] = 2
save_path = "/home/yanggq/project/Few-shot/segment-anything-main/three_points_segmentation.png"
cv2.imwrite(save_path, new_merged_label)
print(f"✅ 结果已保存至：{save_path}")
