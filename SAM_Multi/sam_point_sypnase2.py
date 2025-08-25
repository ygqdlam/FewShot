import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor

# -------------------------
# 1. 基础配置（与你的代码保持一致，确保模型匹配）
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载 vit_b 模型（与你的代码一致，模型+权重匹配）
sam = sam_model_registry['vit_b'](checkpoint='/home/yanggq/project/Few-shot/checkpoints/sam_vit_b_01ec64.pth')
sam.to(device=device)
predictor = SamPredictor(sam)

# 器官名称与固定颜色（用于合并图，避免随机色，便于对比）
organ_names = [
    "spleen", "right_kidney", "left_kidney", "gallbladder",
    "esophagus", "liver", "stomach", "aorta"
]
organ_colors = [  # 与之前一致的颜色，确保对比清晰
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
]
target_size = 224  # 与你的代码一致，避免尺寸适配问题


# -------------------------
# 2. 工具函数：生成合并图（仅保留核心功能）
# -------------------------
def generate_merged_mask_plot(mask_dict, plot_title, save_path):
    """生成合并掩码图（所有器官掩码叠加+图例）"""
    # 创建黑色背景（突出掩码颜色）
    bg = np.zeros((target_size, target_size, 3), dtype=np.float32)
    # 叠加所有器官掩码
    for organ_name, mask in mask_dict.items():
        color_idx = organ_names.index(organ_name)
        color = np.array(organ_colors[color_idx]) / 255.0  # 归一化到0-1
        bg[mask] = color  # 填充当前器官颜色
    
    # 绘制图像
    plt.figure(figsize=(10, 10))
    plt.imshow(bg)
    plt.title(plot_title, fontsize=16, pad=20)
    plt.axis("off")  # 隐藏坐标轴
    
    # 添加颜色图例（右侧布局，不遮挡掩码）
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=np.array(c)/255.0, edgecolor="black")
        for c in organ_colors
    ]
    plt.legend(
        legend_elements, organ_names,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=12, frameon=True, fancybox=True
    )
    
    # 保存高分辨率图片
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",  # 自动裁剪空白
        pad_inches=0.3  # 留边距避免图例截断
    )
    plt.show()
    print(f"✅ {plot_title} 已保存至 {save_path}")


# -------------------------
# 3. 加载数据 + 提取点提示（完全沿用你的逻辑）
# -------------------------
# 读取 npz 数据
data = np.load("/home/yanggq/project/Few-shot/H-SAM-main/datasets/Synapse/train_npz/case0010_slice084.npz")
image = data["image"]  # (H,W) 灰度图
label = data["label"]  # (H,W) 标签图

# 图像预处理（与你的代码完全一致）
if image.ndim == 2:
    image = np.stack([image] * 3, axis=-1)  # 灰度转 RGB
image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
label_resized = cv2.resize(label, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

# 初始化 SAM 图像特征（必须步骤）
predictor.set_image(image_resized)

# 提取每个器官的中心点（点提示核心，与你的代码一致）
categories = {}  # 器官名: 中心点坐标
for i, cls in enumerate(range(1, 9)):
    organ_name = organ_names[i]
    # 找到当前类别的所有像素坐标
    coords = np.argwhere(label_resized == cls)
    if coords.size > 0:
        # 取中心点（避免边缘点，提高SAM识别准确率）
        y, x = coords[len(coords) // 2]
        categories[organ_name] = np.array([[x, y]])  # SAM 要求 (N,2) 格式（x,y）
        print(f"✅ {organ_name} 中心点：({x}, {y})")
    else:
        print(f"⚠️ {organ_name}：当前切片无该器官")


# -------------------------
# 4. SAM 分割（沿用你的点提示逻辑）
# -------------------------
print("\n=== 开始 SAM 分割 ===")
sam_segment_masks = {}  # SAM 分割结果掩码
original_label_masks = {}  # 原始标签掩码（用于合并图对比）

for organ_name, input_point in categories.items():
    # 1. 提取原始标签掩码（用于对比）
    original_mask = (label_resized == (organ_names.index(organ_name) + 1)).astype(np.bool_)
    original_label_masks[organ_name] = original_mask
    
    # 2. SAM 点提示分割（与你的代码一致）
    input_label = np.array([1])  # 1=正样本点（表示“这是目标区域”）
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True  # 生成3个候选掩码
    )
    
    # 3. 选置信度最高的掩码
    best_mask_idx = np.argmax(scores)
    sam_segment_masks[organ_name] = masks[best_mask_idx].astype(np.bool_)
    print(f"✅ {organ_name} 分割完成，置信度：{scores[best_mask_idx]:.4f}")


# -------------------------
# 5. 生成两张核心合并图（仅合并图，无单个子图）
# -------------------------
print("\n=== 生成合并图 ===")
# 背景：黑色（突出掩码颜色，便于对比）
black_bg = np.zeros((target_size, target_size, 3), dtype=np.float32)


# 图1：原始标签合并图（点提示对应的原始标签）
generate_merged_mask_plot(
    mask_dict=original_label_masks,
    plot_title="原始标签合并图（Point Prompt 对应标签）",
    save_path="merged_original_label.png"
)

# 图2：SAM 分割结果合并图
generate_merged_mask_plot(
    mask_dict=sam_segment_masks,
    plot_title="SAM 分割结果合并图（基于 Point Prompt）",
    save_path="merged_sam_segmentation.png"
)
