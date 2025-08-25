import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor

# -------------------------
# 1. 基础配置
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
target_size = 1024  # SAM适配尺寸
sam_checkpoint = "/home/yanggq/project/Few-shot/checkpoints/sam_vit_b_01ec64.pth"
model_type = "vit_b"

# 器官名称与颜色（统一颜色，便于对比）
organ_names = [
    "spleen", "right_kidney", "left_kidney", "gallbladder",
    "esophagus", "liver", "stomach", "aorta"
]
organ_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
]

# -------------------------
# 2. 工具函数：图像缩放至0-255 + 合并图生成
# -------------------------
def scale_to_0_255(image):
    """线性拉伸图像至0-255范围"""
    img_min, img_max = np.min(image), np.max(image)
    if img_max == img_min:
        return np.zeros_like(image, dtype=np.uint8)
    return ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)

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
# 3. 加载SAM模型
# -------------------------
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# -------------------------
# 4. 读取与处理数据
# -------------------------
data_path = "/home/yanggq/project/Few-shot/H-SAM-main/datasets/Synapse/train_npz/case0010_slice084.npz"
data = np.load(data_path)
image_gray = data["image"]
label = data["label"]

# 图像处理（仅缩放0-255+转3通道+调整尺寸）
#image_scaled = scale_to_0_255(image_gray)


image_scaled = image_gray

image_rgb = np.stack([image_scaled] * 3, axis=-1)
image_resized = cv2.resize(image_rgb, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
label_resized = cv2.resize(label, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

predictor.set_image(image_resized)

# -------------------------
# 5. 提取原始Prompt Mask（合并用）
# -------------------------
print("\n=== 提取原始标签掩码（Prompt Mask） ===")
prompt_masks = {}
for i, cls in enumerate(range(1, 9)):
    organ_name = organ_names[i]
    org_mask = (label_resized == cls).astype(np.bool_)
    if org_mask.sum() == 0:
        print(f"⚠️ {organ_name}：当前切片无该器官，跳过")
        continue
    prompt_masks[organ_name] = org_mask

# -------------------------
# 6. SAM分割：生成分割结果Mask（合并用）
# -------------------------
print("\n=== SAM分割生成结果掩码 ===")
seg_masks = {}
for organ_name in prompt_masks.keys():
    # 准备SAM的mask输入
    prompt_mask_float = prompt_masks[organ_name].astype(np.float32)
    mask_input = torch.from_numpy(prompt_mask_float).unsqueeze(0).unsqueeze(0).to(device)
    mask_input = F.interpolate(
        mask_input, size=(target_size//4, target_size//4),
        mode="bilinear", align_corners=False
    ).squeeze(0)
    
    # SAM预测
    masks, scores, _ = predictor.predict(
        point_coords=None, point_labels=None,
        mask_input=mask_input, multimask_output=False
    )
    
    seg_masks[organ_name] = masks[0].astype(np.bool_)
    print(f"✅ {organ_name} 分割完成，置信度：{scores[0]:.4f}")

# -------------------------
# 7. 生成两张核心合并图（仅合并图，无单个子图）
# -------------------------
print("\n=== 生成合并掩码图 ===")
# 图1：原始Prompt Mask合并图
generate_merged_mask_plot(
    mask_dict=prompt_masks,
    plot_title="原始标签掩码合并图（Prompt Mask）",
    save_path="merged_prompt_mask.png"
)

# 图2：SAM分割结果合并图
generate_merged_mask_plot(
    mask_dict=seg_masks,
    plot_title="SAM分割结果掩码合并图（Segmentation Mask）",
    save_path="merged_segmentation_mask.png"
)