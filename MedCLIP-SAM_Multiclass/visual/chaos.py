import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

join = os.path.join
basename = os.path.basename

# -------------------------- 1. 优化参数解析（增加描述，提升可读性） --------------------------
parser = argparse.ArgumentParser(description="Visualize GT vs Segmentation masks for Multi-Organ Dataset (Liver/Kidneys/Spleen)")
parser.add_argument('--gt_path', type=str, default='data/CHAOS/test_masks', help="Path to ground truth masks (grayscale)")
parser.add_argument('--seg_path', type=str, default='sam_outputs/data/CHAOS/test_masks', help="Path to segmentation result masks (grayscale)")
parser.add_argument('--output', type=str, default='visual/chaos', help="Path to save visualization figures")
args = parser.parse_args()

gt_path = args.gt_path
seg_path = args.seg_path
output_path = args.output

# 创建输出文件夹（exist_ok=True避免重复创建报错）
os.makedirs(output_path, exist_ok=True)

# -------------------------- 2. 修正文件列表逻辑（双重校验GT+Seg存在，避免后续报错） --------------------------
filenames = []
for x in os.listdir(seg_path):
    # 仅保留图像格式，且确保GT和Seg文件都存在（避免加载时找不到文件）
    if x.endswith(('.png', '.jpg', '.jpeg')):
        gt_file = join(gt_path, x)
        seg_file = join(seg_path, x)
        if os.path.exists(gt_file) and os.path.exists(seg_file):
            filenames.append(x)
filenames.sort()  # 按文件名排序，保证结果可复现（如按序号1.png、2.png顺序处理）

# -------------------------- 3. 修正类别定义（解决语义-像素值不匹配，删除错误注释） --------------------------
# 原代码"8个类别"注释错误，实际仅4个目标器官+1背景（共5类）；且ORGAN_INDEX语义与像素值颠倒（如Liver对应1，但class_colors[1]是Spleen）
# 统一：像素值0=背景，1=肝脏，2=右肾，3=左肾，4=脾脏（与class_colors颜色映射对应）
ORGAN_MAPPING = {
    0: "Background",       # 背景
    1: "Liver",            # 肝脏（像素值1）
    2: "Right Kidney",     # 右肾（像素值2）
    3: "Left Kidney",      # 左肾（像素值3）
    4: "Spleen"            # 脾脏（像素值4）
}

# -------------------------- 4. 修正颜色映射（语义-颜色对应，避免混淆） --------------------------
# 原代码颜色与器官不匹配（如像素值1是Liver，但class_colors[1]标为Spleen），修正后每个颜色对应明确器官，便于人工区分
class_colors = {
    0: [0, 0, 0],          # Background (Black) - 背景黑色（不干扰目标器官）
    1: [255, 165, 0],      # Liver (Orange) - 肝脏橙色（医学可视化中常用橙色标注肝脏）
    2: [0, 255, 0],        # Right Kidney (Green) - 右肾绿色
    3: [0, 0, 255],        # Left Kidney (Blue) - 左肾蓝色（左右肾用对比色区分）
    4: [255, 0, 0]         # Spleen (Red) - 脾脏红色（与肾脏颜色差异大，避免混淆）
}

# -------------------------- 5. 修正可视化布局（解决子图越界，适配5类+1合并图） --------------------------
for name in tqdm(filenames, desc="Generating multi-organ visualizations"):
    # 加载GT和Seg掩码（灰度图模式，避免cv2默认BGR通道干扰）
    gt_mask = cv2.imread(join(gt_path, name), cv2.IMREAD_GRAYSCALE)
    seg_mask = cv2.imread(join(seg_path, name), cv2.IMREAD_GRAYSCALE)
    
    # 修正：若Seg尺寸与GT不一致，按GT尺寸重采样（最近邻插值避免类别值被修改）
    if seg_mask.shape != gt_mask.shape:
        seg_mask = cv2.resize(seg_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 统一数据类型为uint8（避免后续布尔运算出错）
    gt_data = np.uint8(gt_mask)
    seg_data = np.uint8(seg_mask)

    # -------------------------- 6. 简化合并图逻辑（删除冗余循环，直接复用原始类别矩阵） --------------------------
    # 原代码循环赋值merged_gt_mask[gt_class_mask==1] = class_id，等价于merged_gt_mask = gt_data（gt_data已是类别ID矩阵）
    merged_gt_mask = gt_data
    merged_seg_mask = seg_data

    # 修正子图布局：2行（GT行 + Seg行）×6列（5类+1合并图），避免原代码2行4列导致的"axes[0,4]越界"错误
    fig, axes = plt.subplots(2, 6, figsize=(24, 10))  # 列数=类别数（5）+合并图（1），宽度设为24适配多列

    # -------------------------- 7. 循环绘制单类别图（语义化标题，避免"Class 1"这种无意义标注） --------------------------
    for class_id in range(5):  # 0=背景，1=肝脏，2=右肾，3=左肾，4=脾脏（共5类）
        # 生成单类别二值掩码（True=当前类别，False=其他）
        gt_class_mask = (gt_data == class_id).astype(np.uint8)
        seg_class_mask = (seg_data == class_id).astype(np.uint8)

        # 生成RGB彩色图（背景黑色，当前类别按预设颜色填充）
        gt_rgb = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
        seg_rgb = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
        gt_rgb[gt_class_mask == 1] = class_colors[class_id]
        seg_rgb[seg_class_mask == 1] = class_colors[class_id]

        # 绘制GT子图（第0行，对应列=class_id）
        axes[0, class_id].imshow(gt_rgb, interpolation='none')
        axes[0, class_id].set_title(f"GT - {ORGAN_MAPPING[class_id]}", fontsize=12)  # 标题带器官名，如"GT - Liver"
        axes[0, class_id].axis('off')  # 关闭坐标轴，聚焦掩码本身

        # 绘制Seg子图（第1行，对应列=class_id）
        axes[1, class_id].imshow(seg_rgb, interpolation='none')
        axes[1, class_id].set_title(f"Seg - {ORGAN_MAPPING[class_id]}", fontsize=12)
        axes[1, class_id].axis('off')

    # -------------------------- 8. 绘制合并图（所有类别叠加，便于整体对比） --------------------------
    # 合并GT图（用tab10色表，vmin/vmax固定为0-4确保颜色映射一致）
    im_gt = axes[0, 5].imshow(merged_gt_mask, cmap='tab10', interpolation='none', vmin=0, vmax=4)
    axes[0, 5].set_title("Merged GT (All Organs)", fontsize=12)
    axes[0, 5].axis('off')

    # 合并Seg图
    im_seg = axes[1, 5].imshow(merged_seg_mask, cmap='tab10', interpolation='none', vmin=0, vmax=4)
    axes[1, 5].set_title("Merged Seg (All Organs)", fontsize=12)
    axes[1, 5].axis('off')

    # -------------------------- 9. 增加颜色条（关键！解决"不知道颜色对应哪个器官"的问题） --------------------------
    # 在图右侧添加颜色条，标注每个颜色对应的器官名称
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # 颜色条位置：右(0.93)、下(0.15)、宽(0.02)、高(0.7)
    cbar = fig.colorbar(im_gt, cax=cbar_ax, ticks=list(ORGAN_MAPPING.keys()))  # 刻度对应类别ID
    cbar.set_ticklabels([ORGAN_MAPPING[k] for k in ORGAN_MAPPING.keys()])  # 刻度替换为器官名
    cbar.set_label('Organ Class', rotation=270, labelpad=20, fontsize=12)  # 颜色条标签（垂直显示）

    # -------------------------- 10. 优化保存逻辑（避免标题截断，提高分辨率） --------------------------
    fig.suptitle(f"Multi-Organ Mask Comparison - {name}", fontsize=14, y=0.98)  # 总标题（标注当前文件名，便于定位问题）
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])  # 预留颜色条空间（右边界0.92以内为子图）
    # 保存为高分辨率PNG（dpi=300，避免模糊；bbox_inches='tight'防止颜色条被截断）
    fig.savefig(join(output_path, name), dpi=300, bbox_inches='tight')
    plt.close(fig)  # 关闭图释放内存，避免批量处理时内存溢出

print(f"Visualization completed! All figures saved to: {output_path}")