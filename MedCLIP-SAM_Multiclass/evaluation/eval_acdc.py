import numpy as np
import cv2
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from tqdm import tqdm
import argparse

join = os.path.join
basename = os.path.basename

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, default='data/breast/test_masks')
parser.add_argument('--seg_path', type=str, default='crf_outputs/breast/test_CRF')

args = parser.parse_args()
gt_path = args.gt_path
seg_path = args.seg_path

# Get list of filenames
filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')]
filenames = [x for x in filenames if os.path.exists(join(seg_path, x))]
filenames.sort()

# Initialize metrics dictionary
seg_metrics = OrderedDict(
    Name = list(),
    DSC = list(),
    NSD = list(),
)

# 3 个类别的固定顺序

ORGAN_INDEX = {
    'Left Ventricle': 1,
    'Right Ventricle': 2,
    'Myocardium': 3
}
# Compute metrics for each file
for name in tqdm(filenames):
    seg_metrics['Name'].append(name)
    gt_mask = cv2.imread(join(gt_path, name), cv2.IMREAD_GRAYSCALE)
    seg_mask = cv2.imread(join(seg_path, name), cv2.IMREAD_GRAYSCALE)
    seg_mask = cv2.resize(seg_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Convert to uint8
    gt_data = np.uint8(gt_mask)
    seg_data = np.uint8(seg_mask)

    # Initialize DSC and NSD arrays for each class
    dsc_per_class = []
    nsd_per_class = []

    # For each class from 0 to 8, calculate DSC and NSD
    for class_id in range(4):  # 0-8 classes
        gt_class_mask = (gt_data == class_id).astype(np.uint8)
        seg_class_mask = (seg_data == class_id).astype(np.uint8)

        if np.sum(gt_class_mask) == 0 and np.sum(seg_class_mask) == 0:
            DSC_i = 1
            NSD_i = 1
        elif np.sum(gt_class_mask) == 0 and np.sum(seg_class_mask) > 0:
            DSC_i = 0
            NSD_i = 0
        else:
            # Compute Dice coefficient
            DSC_i = compute_dice_coefficient(gt_class_mask, seg_class_mask)

            # Compute NSD
            case_spacing = [1, 1, 1]  # assuming isotropic spacing
            surface_distances = compute_surface_distances(gt_class_mask[..., None], seg_class_mask[..., None], case_spacing)
            NSD_i = compute_surface_dice_at_tolerance(surface_distances, 2)

        dsc_per_class.append(DSC_i)
        nsd_per_class.append(NSD_i)

    # Compute the average DSC and NSD for the current image
    DSC_avg = np.mean(dsc_per_class)
    NSD_avg = np.mean(nsd_per_class)

    # Append to the overall metrics
    seg_metrics['DSC'].append(round(DSC_avg, 4))
    seg_metrics['NSD'].append(round(NSD_avg, 4))

    # Print per-class DSC and NSD
    print(f"Image: {name}")
    for class_id in range(4):
        print(f"Class {class_id} - DSC: {dsc_per_class[class_id]:.4f}, NSD: {nsd_per_class[class_id]:.4f}")
    
    print(f"Average DSC for {name}: {DSC_avg:.4f}")
    print(f"Average NSD for {name}: {NSD_avg:.4f}")
    print("-" * 40)

# Save metrics to CSV
dataframe = pd.DataFrame(seg_metrics)

# Calculate and print average and std deviation for metrics
case_avg_DSC = dataframe['DSC'].mean()
case_avg_NSD = dataframe['NSD'].mean()

print(20 * '>')
print(f'Average DSC for {basename(seg_path)}: {case_avg_DSC:.4f}')
print(f'Average NSD for {basename(seg_path)}: {case_avg_NSD:.4f}')
print(20 * '<')