import numpy as np
import cv2
import os
import argparse
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import SimpleITK as sitk

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks"
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--mask-input",
    type=str,
    required=True,
    help="Path to either a single crf mask image or folder of crf mask images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument(
    "--prompts",
    type=str,
    required=True,
    help="The type of prompts to use, in ['points', 'boxes', 'both']",
)

parser.add_argument(
    "--num-points",
    type=int,
    required=False,
    default=10,
    help="Number of points when using point prompts, default is 8",
)

parser.add_argument(
    "--negative",
    action="store_true",
    help="Whether to sample points in the background. Default is False.",
)

parser.add_argument(
    "--neg-num-points",
    type=int,
    required=False,
    default=10,
    help="Number of negative points when using negative mode, default is 20",
)

parser.add_argument(
    "--pos-margin",
    type=float,
    required=False,
    default=10,
    help="controls the sampling margin for the positive point prompts, default is 2, for large structures use above 15, but \
    for smaller objects use 2-5",
)

parser.add_argument(
    "--neg-margin",
    type=float,
    required=False,
    default=5,
    help="controls the sampling margin for the negative point prompts, default is 5",
)


parser.add_argument(
    "--multimask",
    action="store_true",
    help="Whether to output multimasks in SAM. Default is False.",
)

parser.add_argument(
    "--multicontour",
    action="store_true",
    help="Whether to output multiple bounding boxes for each contour. Default is False.",
)

parser.add_argument(
    "--voting",
    type=str,
    default="AVERAGE",
    help="['MRM','STAPLE','AVERAGE']",
)

parser.add_argument(
    "--plot",
    action="store_true",
    help="Whether to plot the points and boxes in the contours. Default is False.",
)

parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Whether to plot the points and boxes in the contours. Default is False.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")




def write_mask_to_folder(mask , t_mask, path: str,num_contours) -> None:
    file = t_mask.split("/")[-1]
    filename = f"{file}"
    mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1]
    mask = mask.astype(np.uint8)*255
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(mask)
    sizes = stats[:, cv2.CC_STAT_AREA]
    sorted_sizes = sorted(sizes[1:], reverse=True) 

    # Determine the top K sizes
    top_k_sizes = sorted_sizes[:num_contours]
    
    im_result = np.zeros_like(im_with_separated_blobs)
    
    for index_blob in range(1, nb_blobs):
        if sizes[index_blob] in top_k_sizes:
            im_result[im_with_separated_blobs == index_blob] = 255
    mask = im_result
    cv2.imwrite(os.path.join(path, filename), mask)

    return

def scoremap2bbox(scoremap, multi_contour_eval=False):
    height, width = scoremap.shape
    scoremap_image = (scoremap * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        image=scoremap_image,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE)
    
    num_contours = len(contours)

    if len(contours) == 0:
        return np.asarray([[0, 0, width, height]]), 1
    

    if not multi_contour_eval:
        # contours = [max(contours, key=cv2.contourArea)]
        contours = [np.concatenate(contours)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return estimated_boxes, contours,num_contours

# -------------------------
# 1. 从掩码随机选 N 个点
# -------------------------
def get_random_points(mask, num_points=50, seed=None):
    rng = np.random.default_rng(seed)
    coords_yx = np.argwhere(mask > 0)  # (y, x)
    if coords_yx.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    if len(coords_yx) <= num_points:
        chosen = coords_yx
    else:
        idx = rng.choice(len(coords_yx), size=num_points, replace=False)
        chosen = coords_yx[idx]

    points_xy = np.stack([chosen[:, 1], chosen[:, 0]], axis=1).astype(np.float32)
    return points_xy


# -------------------------
# 2. 从掩码生成点提示
# -------------------------
def get_prompts(mask, args):
    pos_random_points = get_random_points(mask, num_points=args.num_points)
    if pos_random_points.shape[0] == 0:
        return np.empty((0,2)), np.array([])

    all_random_points = pos_random_points
    all_input_labels = np.ones(len(pos_random_points), dtype=np.int32)

    # 如果需要负点，可以启用
    if args.negative:
        neg_random_points = get_random_points((mask == 0).astype(np.uint8), num_points=args.neg_num_points)
        if neg_random_points.shape[0] > 0:
            all_random_points = np.concatenate([pos_random_points, neg_random_points])
            all_input_labels = np.concatenate([
                np.ones(len(pos_random_points), dtype=np.int32),
                np.zeros(len(neg_random_points), dtype=np.int32)
            ])

    return all_random_points, all_input_labels


# -------------------------
# 3. 只用 points 提示生成 mask
# -------------------------
def get_final_mask(predictor, all_random_points, all_input_labels, image, args):
    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=all_random_points,
        point_labels=all_input_labels,
        multimask_output=args.multimask,
    )

    if args.multimask:
        best_idx = np.argmax(scores)
        final_mask = masks[best_idx].astype(float)
    else:
        final_mask = np.squeeze(masks).astype(float)

    return final_mask



# Mulitclass
# 你的 8 个固定类别（保持原样大小写与空格）
import re
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
# 8 个类别的固定顺序
ORGAN_SYNAPSE_INDEX = {
    'Spleen': 1,
    'Right kidney': 2,
    'Left kidney': 3,
    'Gallbladder': 4,
    'Esophagus': 5,
    'Liver': 6,
    'Stomach': 7,
    'Pancreas': 8
}

ORGAN_ACDC_INDEX = {
    'Left Ventricle': 1,
    'Right Ventricle': 2,
    'Myocardium': 3
}

ORGAN_CHAOS_INDEX = {
    'Liver': 1,
    'Right Kidney': 2,
    'Left Kidney': 3,
    'Spleen': 4,
}


def write_merged_mask(all_masks: dict, save_path: str, ORGAN_INDEX) -> None:
    """
    将 8 个类别的 mask 合并为一个单通道 mask：
      - 输入: all_masks = {organ_name: mask_bool 或 mask_float}
      - 输出: 单通道 uint8 图像，每个像素值 1~8 表示类别编号
    """
    # 假设所有 mask 尺寸一致
    H, W = list(all_masks.values())[0].shape
    merged = np.zeros((H, W), dtype=np.uint8)

    for organ, mask in all_masks.items():
        if organ not in ORGAN_INDEX:
            continue
        cls_id = ORGAN_INDEX[organ]
        # 二值化（阈值化后得到 bool mask）
        mask_bin = (mask > 0.5).astype(np.uint8)
        # 将该器官区域写入 cls_id
        merged[mask_bin == 1] = cls_id

    cv2.imwrite(save_path, merged)

def list_files(path):
    if os.path.isdir(path):
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        files.sort()
        return files
    else:
        return [path]
def stem(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]
def organ_regex(base, organ):
    """
    构造一个忽略大小写、允许空格/下划线互换的正则：
      ^image001_(Right[ _]kidney)$
    """
    organ_pattern = re.sub(r"\s+", r"[ _]", organ)   # 'Right kidney' -> 'Right[ _]kidney'
    return re.compile(rf"^{re.escape(base)}_{organ_pattern}$", re.IGNORECASE)    


def main(args: argparse.Namespace) -> None:
    print("Loading SAM...")

    if args.dataset == "synapse":
        ORGAN_INDEX = ORGAN_SYNAPSE_INDEX
    elif args.dataset == "acdc":
        ORGAN_INDEX = ORGAN_ACDC_INDEX
    elif args.dataset == "chaos":
        ORGAN_INDEX = ORGAN_CHAOS_INDEX
    
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=device)
    predictor = SamPredictor(sam)

    # 图像与掩码列表
    image_files = list_files(args.input)
    mask_files = list_files(args.mask_input)
    os.makedirs(args.output, exist_ok=True)

    print("Segmenting images with multi-class prompts...")
    for img_path in tqdm(image_files):
        if not img_path.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[x] Could not load image: {img_path}, skip.")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        base_name = stem(img_path)

        # 为该图像匹配 8 个器官掩码（支持空格或下划线）
        organ_to_mask = {}
        # 先把候选掩码按前缀过滤，提高效率
        prefix_candidates = [m for m in mask_files if stem(m).lower().startswith((base_name + "_").lower())]

        for organ in ORGAN_INDEX.keys():
            rx = organ_regex(base_name, organ)
            chosen = None
            for m in prefix_candidates:
                if rx.match(stem(m)):
                    chosen = m
                    break
            if chosen is None:
                print(f"[!] {base_name}: no prompt mask for organ '{organ}', skip this organ.")
            else:
                organ_to_mask[organ] = chosen

        if len(organ_to_mask) == 0:
            print(f"[!] {base_name}: no organ masks found, skip this image.")
            continue

        # ---- 对该图像的每个器官掩码做一次 SAM 推理，并合并结果 ----
        all_masks = {}  # 存储每个器官的分割结果
        for organ, mask_path in organ_to_mask.items():
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[x] Could not load mask: {mask_path}, skip {organ}.")
                continue
            if (mask.shape[0] != H) or (mask.shape[1] != W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)


            all_random_points, all_input_labels = get_prompts(mask, args)
            final_mask = get_final_mask(predictor, all_random_points, all_input_labels, img_rgb, args)

            # 保存单个器官的分割结果
            all_masks[organ] = final_mask  # 保存每个器官的最终 mask

        # 合并所有器官的 mask，并保存
        merged_mask_path = os.path.join(args.output, f"{base_name}.png")
        write_merged_mask(all_masks, merged_mask_path, ORGAN_INDEX)

    print("SAM Segmentation Done!")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
