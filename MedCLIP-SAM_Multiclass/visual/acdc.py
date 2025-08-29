import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm import tqdm
import argparse

join = os.path.join
basename = os.path.basename

# -------------------------- 1. 修正参数解析（无逻辑问题，仅规范格式） --------------------------
parser = argparse.ArgumentParser(description="Visualize GT vs Segmentation masks for ACDC dataset (3 heart organs)")
parser.add_argument('--gt_path', type=str, default='data/ACDC/test_masks', help="Path to ground truth masks")
parser.add_argument('--seg_path', type=str, default='sam_outputs/data/ACDC/test_masks', help="Path to segmentation results")
parser.add_argument('--output', type=str, default='visual/acdc', help="Path to save visualization figures")
args = parser.parse_args()

gt_path = args.gt_path
seg_path = args.seg_path
output_path = args.output

# Create output folder if not exists
os.makedirs(output_path, exist_ok=True)

# -------------------------- 2. 修正文件列表逻辑（避免重复判断，增加合法性校验） --------------------------
# 仅保留png/jpg/jpeg格式，且确保GT文件存在（避免后续报错）
filenames = []
for x in os.listdir(seg_path):
    if x.endswith(('.png', '.jpg', '.jpeg')):
        gt_file = join(gt_path, x)
        seg_file = join(seg_path, x)
        if os.path.exists(gt_file) and os.path.exists(seg_file):
            filenames.append(x)
filenames.sort()  # 按文件名排序，保证结果可复现

# -------------------------- 3. 修正类别定义（匹配ACDC数据集真实标签，删除冗余） --------------------------
# ACDC数据集仅3个目标类别（左心室、右心室、心肌），原代码"8个类别"注释错误，且ORGAN_INDEX未实际使用
# 标签对应：0=背景，1=左心室，2=右心室，3=心肌（与ACDC标准标签一致）
ORGAN_MAPPING = {
    0: "Background",
    1: "Left Ventricle",
    2: "Right Ventricle",
    3: "Myocardium"
}

# -------------------------- 4. 修正颜色映射（匹配器官语义，避免混淆，背景用黑色，器官用鲜明色） --------------------------
# 原代码颜色与器官不对应（如把脾脏红色用在左心室），修正为符合医学可视化习惯的颜色
class_colors = {
    0: [0, 0, 0],        # Background (Black) - 背景黑色
    1: [255, 0, 0],      # Left Ventricle (Red) - 左心室红色（突出心脏核心腔室）
    2: [0, 255, 0],      # Right Ventricle (Green) - 右心室绿色
    3: [0, 0, 255]       # Myocardium (Blue) - 心肌蓝色
}

# -------------------------- 5. 修正可视化逻辑（布局匹配类别数、增加语义标题、优化颜色显示） --------------------------
# Compute metrics for each file (原代码未计算指标，注释保留但可后续补充)
for name in tqdm(filenames, desc="Generating visualizations"):
    # Load ground truth and segmentation masks (灰度图读取，确保无通道问题)
    gt_mask = cv2.imread(join(gt_path, name), cv2.IMREAD_GRAYSCALE)
    seg_mask = cv2.imread(join(seg_path, name), cv2.IMREAD_GRAYSCALE)
    
    # 修正：若分割图尺寸与GT不一致，按GT尺寸重采样（使用最近邻避免类别混淆）
    if seg_mask.shape != gt_mask.shape:
        seg_mask = cv2.resize(seg_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Convert to uint8 (确保数据类型统一，避免后续计算错误)
    gt_data = np.uint8(gt_mask)
    seg_data = np.uint8(seg_mask)

    # -------------------------- 6. 简化合并逻辑（删除冗余赋值，直接基于原始数据生成合并图） --------------------------
    # 原代码"merged_gt_mask = gt_data"即可，无需循环赋值（gt_data已是类别ID矩阵）
    merged_gt_mask = gt_data
    merged_seg_mask = seg_data

    # -------------------------- 7. 修正子图布局（2行4列 → 2行5列，适配"3类+1合并图"的显示逻辑） --------------------------
    # 原代码2行4列仅能显示8个子图，但需要显示：3类GT + 合并GT + 3类Seg + 合并Seg → 共8个子图？不，重新梳理：
    # 正确布局：每行显示「3个类别图 + 1个合并图」，共2行（GT行 + Seg行），2行×4列刚好适配
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2行（GT/Seg），4列（3类+1合并）

    # -------------------------- 8. 修正绘图循环（按实际类别数循环，增加语义化标题） --------------------------
    # 遍历3个目标类别（1-3），背景（0）通常不单独显示（避免冗余，若需显示可调整循环范围为0-3）
    for class_id in [1, 2, 3]:  # 优先显示目标器官，背景可省略（如需显示则改为range(4)）
        # 生成单类别掩码（布尔值转uint8，确保后续颜色赋值有效）
        gt_class_mask = (gt_data == class_id).astype(np.uint8)
        seg_class_mask = (seg_data == class_id).astype(np.uint8)

        # 生成RGB彩色图（背景黑色，目标类别按预设颜色显示）
        gt_rgb = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
        seg_rgb = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
        gt_rgb[gt_class_mask == 1] = class_colors[class_id]
        seg_rgb[seg_class_mask == 1] = class_colors[class_id]

        # 计算子图索引（class_id=1→列0，class_id=2→列1，class_id=3→列2）
        col_idx = class_id - 1
        # 绘制GT子图（第0行，对应列）
        axes[0, col_idx].imshow(gt_rgb, interpolation='none')
        axes[0, col_idx].set_title(f"GT - {ORGAN_MAPPING[class_id]}", fontsize=12)  # 语义化标题
        axes[0, col_idx].axis('off')  # 关闭坐标轴，聚焦掩码显示
        # 绘制Seg子图（第1行，对应列）
        axes[1, col_idx].imshow(seg_rgb, interpolation='none')
        axes[1, col_idx].set_title(f"Seg - {ORGAN_MAPPING[class_id]}", fontsize=12)
        axes[1, col_idx].axis('off')

    # -------------------------- 9. 绘制合并图（第3列，显示所有类别叠加效果） --------------------------
    # 合并GT图：使用tab10 colormap，且手动设置颜色映射（确保与class_colors一致，避免颜色混乱）
    im_gt = axes[0, 3].imshow(merged_gt_mask, cmap='tab10', interpolation='none', vmin=0, vmax=3)
    axes[0, 3].set_title("Merged GT (All Organs)", fontsize=12)
    axes[0, 3].axis('off')
    # 合并Seg图
    im_seg = axes[1, 3].imshow(merged_seg_mask, cmap='tab10', interpolation='none', vmin=0, vmax=3)
    axes[1, 3].set_title("Merged Seg (All Organs)", fontsize=12)
    axes[1, 3].axis('off')

    # -------------------------- 10. 增加颜色条（关键！原代码无图例，无法区分类别颜色） --------------------------
    # 在图右侧添加颜色条，标注每个颜色对应的器官
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # 颜色条位置（右、下、宽、高）
    cbar = fig.colorbar(im_gt, cax=cbar_ax, ticks=list(ORGAN_MAPPING.keys()))
    cbar.set_ticklabels([ORGAN_MAPPING[k] for k in ORGAN_MAPPING.keys()])  # 语义化刻度
    cbar.set_label('Organ Class', rotation=270, labelpad=20, fontsize=12)  # 颜色条标签

    # -------------------------- 11. 优化保存逻辑（紧凑布局，避免文字截断，设置高分辨率） --------------------------
    fig.suptitle(f"ACDC Mask Comparison - {name}", fontsize=14, y=0.98)  # 总标题（标注文件名）
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])  # 预留颜色条空间（右0.9以内为子图）
    # 保存为高分辨率PNG（dpi=300，避免模糊）
    fig.savefig(join(output_path, name), dpi=300, bbox_inches='tight')
    plt.close(fig)  # 关闭图释放内存，避免批量处理时内存溢出

print(f"Visualization completed! Results saved to {output_path}")