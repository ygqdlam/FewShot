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

def get_prompts(mask, args):
    # List to store bounding boxes and random points
        bounding_boxes = []
        all_random_points = []
        all_input_labels = []

        bounding_boxes, contours, num_contours = scoremap2bbox(mask, multi_contour_eval=args.multicontour)
        
        if(args.prompts == "boxes"):
            bounding_boxes = np.array(bounding_boxes)
            return np.zeros_like(bounding_boxes), np.zeros_like(bounding_boxes), bounding_boxes, num_contours
    
        pos_num_points = args.num_points  # number of positive random points to get per contour
        neg_num_points = args.neg_num_points  # number of negative random points to get per contour
        pos_random_points = []
        neg_random_points = []
        candidate_points = np.argwhere(mask.transpose(1,0) > 0)
        h,w = mask.shape
        random_index = np.random.choice(len(candidate_points), pos_num_points, replace=False)
        pos_random_points = candidate_points[random_index]

        if(args.negative):
            # Filter some points
            candidate_points = np.argwhere(mask.transpose(1,0) == 0)
            random_index = np.random.choice(len(candidate_points), neg_num_points, replace=False)
            neg_random_points = candidate_points[random_index]
            all_random_points = np.concatenate([pos_random_points,neg_random_points])
            all_input_labels = [1]*len(pos_random_points) + [0]*len(neg_random_points)
        
        elif(not args.negative):
            all_random_points = pos_random_points
            all_input_labels = [1]*len(pos_random_points)

        # Convert the lists to NumPy arrays
        all_random_points = np.array(all_random_points)
        all_input_labels = np.array(all_input_labels)
        bounding_boxes = np.array(bounding_boxes)

        return all_random_points, all_input_labels, bounding_boxes, num_contours
    
def get_final_mask(predictor,all_random_points, all_input_labels, 
                   bounding_boxes, image, args):
    
    input_boxes = torch.tensor(bounding_boxes, device=args.device) 
    input_points = torch.tensor(all_random_points, device=args.device)
    input_labels = torch.tensor(all_input_labels, device=args.device)
    input_points = input_points.repeat((len(bounding_boxes),1,1))
    input_labels = input_labels.repeat((len(bounding_boxes),1))

    predictor.set_image(image)

    if(args.prompts == "both"):
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2]) 
        transformed_points = predictor.transform.apply_coords_torch(input_points, image.shape[:2]) 
        masks, scores, _ = predictor.predict_torch(
            point_coords=transformed_points,
            point_labels=input_labels,
            boxes=transformed_boxes,
            multimask_output=args.multimask
        )
        masks = masks.cpu().numpy()
        scores = scores.cpu().numpy()

    elif(args.prompts == "points"):
        masks, scores, _ = predictor.predict(
            point_coords=all_random_points,
            point_labels=all_input_labels,
            multimask_output=args.multimask,
        )

    elif(args.prompts == "boxes"):
        input_boxes = torch.tensor(bounding_boxes, device=args.device)  
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])  
        masks, scores, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=args.multimask
        )
        masks = masks.cpu().numpy()
        scores = scores.cpu().numpy()

    if(args.multimask):
        if(args.voting == "MRM"):
            scores = np.expand_dims(scores, axis=-1)
            scores = np.expand_dims(scores, axis=-1)
            final_mask = (masks * scores).sum(axis=0)
            final_mask = (final_mask - final_mask.min()) / (final_mask.max() - final_mask.min())
        elif(args.voting == "STAPLE"):
            final_mask = []
            for i in range(masks.shape[0]):
                seg_sitk = sitk.GetImageFromArray(masks[i].astype(np.int16)) # STAPLE requires we cast into int16 arrays
                final_mask.append(seg_sitk)

            # Run STAPLE algorithm
            final_mask_img = sitk.STAPLE(final_mask, 1.0 ) # 1.0 specifies the foreground value

            # convert back to numpy array
            final_mask = sitk.GetArrayFromImage(final_mask_img)

        elif(args.voting == "AVERAGE"):
            final_mask = np.mean(masks, axis=0)
    else:
        final_mask = np.squeeze(masks).astype(float)

    if(final_mask.ndim == 3):
        final_mask = final_mask.sum(axis=0).clip(0, 1)

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

    if args.dataset == "sysnapse":
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

            # 第二段代码接口：由掩码生成点/框等提示
            all_random_points, all_input_labels, bounding_boxes, num_contours = get_prompts(mask, args)

            # 第二段代码接口：基于 predictor + 提示 + 图像 生成最终分割
            final_mask = get_final_mask(
                predictor,
                all_random_points,
                all_input_labels,
                bounding_boxes,
                img_rgb,
                args
            )

            # 保存单个器官的分割结果
            all_masks[organ] = final_mask  # 保存每个器官的最终 mask

        # 合并所有器官的 mask，并保存
        merged_mask_path = os.path.join(args.output, f"{base_name}.png")
        write_merged_mask(all_masks, merged_mask_path, ORGAN_INDEX)

    print("SAM Segmentation Done!")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)