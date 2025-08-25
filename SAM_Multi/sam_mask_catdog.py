import torch
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import cv2

# -------------------------
# 1. 基础配置（模型+路径）
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载SAM模型（与之前一致）
sam = sam_model_registry['vit_b'](checkpoint='/home/yanggq/project/Few-shot/checkpoints/sam_vit_b_01ec64.pth')
sam.to(device=device)
predictor = SamPredictor(sam)

# 关键路径：替换为你之前保存的“分割结果标签图”路径（如png或npz）
# 假设之前保存的是png格式（cat=1，dog=2，背景=0）
saved_segment_path = "/home/yanggq/project/Few-shot/segment-anything-main/merged_segmentation_label.png"
# 原始图像路径
image_path = '/home/yanggq/project/Few-shot/segment-anything-main/test.jpg'
# 目标尺寸（与之前分割时一致，确保掩码尺寸对齐）
target_size = 1024


# -------------------------
# 2. 加载2个核心数据：原始图像 + 历史分割结果（作为新Prompt）
# -------------------------
def load_image_and_saved_mask(image_path, saved_mask_path, target_size):
    """
    加载原始图像和历史分割结果
    返回：resize后的图像、resize后的单通道掩码（按类别提取）
    """
    # 加载并预处理原始图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取原始图像：{image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR→RGB
    image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

    # 加载历史分割结果（标签图：cat=1，dog=2，背景=0）
    saved_mask = cv2.imread(saved_mask_path, cv2.IMREAD_GRAYSCALE)  # 以灰度图加载（单通道）
    if saved_mask is None:
        raise ValueError(f"无法读取历史分割结果：{saved_mask_path}")
    # Resize掩码到目标尺寸（与图像一致），用最近邻避免标签值失真
    saved_mask_resized = cv2.resize(saved_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

    # 按类别提取单通道掩码（cat的掩码=1的区域，dog的掩码=2的区域）
    cat_mask = (saved_mask_resized == 1).astype(np.float32)  # cat：1→1.0，其他→0.0
    dog_mask = (saved_mask_resized == 2).astype(np.float32)  # dog：2→1.0，其他→0.0

    return image_resized, {"cat": cat_mask, "dog": dog_mask}


# 执行加载：获取resize后的图像 + 按类别拆分的历史分割掩码（新Prompt）
image_resized, history_mask_prompts = load_image_and_saved_mask(
    image_path=image_path,
    saved_mask_path=saved_segment_path,
    target_size=target_size
)

# 验证加载结果
for cls_name, mask in history_mask_prompts.items():
    print(f"✅ 加载的{cls_name}历史掩码：形状{mask.shape}，非零像素数{np.sum(mask>0)}")  # 应输出(1024,1024) + 合理像素数


# -------------------------
# 3. 初始化SAM图像特征 + 计算Mask Input尺寸
# -------------------------
predictor.set_image(image_resized)  # 缓存图像特征
sam_input_size = predictor.input_size  # SAM内部处理尺寸（通常1024x1024）
mask_input_hw = (sam_input_size[0]//4, sam_input_size[1]//4)  # Mask Input必须是1/4尺寸（256x256）
print(f"✅ SAM内部输入尺寸：{sam_input_size}，Mask Input要求尺寸：{mask_input_hw}")


# -------------------------
# 4. 用历史分割掩码作为Prompt，执行SAM分割
# -------------------------
category_masks = {}  # 存储新的分割结果
for cls_name, raw_mask in history_mask_prompts.items():
    # 步骤1：将历史掩码Resize到256x256（SAM要求的Mask Input尺寸）
    mask_resized = cv2.resize(
        raw_mask,
        dsize=(mask_input_hw[1], mask_input_hw[0]),  # OpenCV要求dsize=(W,H)
        interpolation=cv2.INTER_LINEAR
    )
    print(f"✅ {cls_name} 历史掩码Resize后：形状{mask_resized.shape}")  # 应输出(256,256)

    # 步骤2：维度转换为SAM要求的4维（1,1,256,256）
    mask_input = torch.from_numpy(mask_resized)  # (256,256)→2维
    mask_input = mask_input.unsqueeze(0)        # 加通道维度→(1,256,256)（3维）
    mask_input = mask_input.to(device)
    print(f"✅ {cls_name} 最终Mask Input：形状{mask_input.shape}")  # 必须是(1,1,256,256)

    # 步骤3：SAM预测（用历史掩码作为Prompt）
    masks, scores, logits = predictor.predict(
        point_coords=None,    # 不使用点提示
        point_labels=None,    # 不使用点标签
        mask_input=mask_input,  # 核心：用历史分割结果作为Mask Prompt
        multimask_output=True  # 返回3个候选掩码，选最优
    )

    # 步骤4：保存置信度最高的掩码（转为布尔型，便于后续处理）
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx].astype(np.bool_)
    category_masks[cls_name] = best_mask
    print(f"✅ {cls_name} 新分割完成，置信度：{scores[best_mask_idx]:.4f}\n")


# -------------------------
# 5. 可视化：原始图像 + 新分割结果（叠加）
# -------------------------
plt.figure(figsize=(10, 10))
plt.imshow(image_resized)  # 底图：原始图像

# 固定类别颜色（与之前一致，便于对比）
cls_colors = {"cat": (1, 0, 0), "dog": (0, 1, 0)}
for cls_name, mask in category_masks.items():
    # 生成彩色掩码（仅目标区域显色）
    colored_mask = np.zeros_like(image_resized, dtype=np.float32)
    colored_mask[mask] = cls_colors[cls_name]
    # 叠加显示（半透明）
    plt.imshow(colored_mask, cmap="gray", alpha=0.4)

    # 添加类别标签（在掩码中心）
    y_coords, x_coords = np.where(mask)
    if len(x_coords) > 0:
        plt.text(
            x_coords.mean(), y_coords.mean(),
            cls_name, color="white", fontsize=12,
            bbox=dict(facecolor="black", alpha=0.8, pad=3)
        )

plt.axis("off")
plt.title("SAM 分割结果（用历史分割掩码作为Prompt）", fontsize=14)
plt.show()


# -------------------------
# （可选）保存新的分割结果（cat=1，dog=2，背景=0）
# -------------------------
new_merged_label = np.zeros((target_size, target_size), dtype=np.uint8)
new_merged_label[category_masks["cat"]] = 1  # cat=1
new_merged_label[category_masks["dog"]] = 2  # dog=2
# 保存为png（无损）
new_save_path = "/home/yanggq/project/Few-shot/segment-anything-main/new_merged_segmentation.png"
cv2.imwrite(new_save_path, new_merged_label)
print(f"✅ 新分割结果已保存至：{new_save_path}")