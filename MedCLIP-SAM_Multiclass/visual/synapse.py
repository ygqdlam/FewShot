import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm import tqdm
import argparse

join = os.path.join
basename = os.path.basename

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, default='data/Synapse/test_masks')
parser.add_argument('--seg_path', type=str, default='sam_outputs/data/Synapse/test_masks')
parser.add_argument('--output', type=str, default='visual/synapse')
args = parser.parse_args()
gt_path = args.gt_path
seg_path = args.seg_path
output_path = args.output

# Create output folder if not exists
os.makedirs(output_path, exist_ok=True)

# Get list of filenames
filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')]
filenames = [x for x in filenames if os.path.exists(join(seg_path, x))]
filenames.sort()


# 8 个类别的固定顺序
ORGAN_INDEX = {
    'Spleen': 1,
    'Right kidney': 2,
    'Left kidney': 3,
    'Gallbladder': 4,
    'Esophagus': 5,
    'Liver': 6,
    'Stomach': 7,
    'Pancreas': 8
}

# Define colors for each class (RGB)
class_colors = {
    0: [0, 0, 0],  # Background (Black)
    1: [255, 0, 0],  # Spleen (Red)
    2: [0, 255, 0],  # Right kidney (Green)
    3: [0, 0, 255],  # Left kidney (Blue)
    4: [255, 255, 0],  # Gallbladder (Yellow)
    5: [0, 255, 255],  # Esophagus (Cyan)
    6: [255, 0, 255],  # Liver (Magenta)
    7: [128, 128, 0],  # Stomach (Olive)
    8: [128, 0, 128],  # Pancreas (Purple)
}

# Compute metrics for each file
for name in tqdm(filenames):
    # Load ground truth and segmentation masks
    gt_mask = cv2.imread(join(gt_path, name), cv2.IMREAD_GRAYSCALE)
    seg_mask = cv2.imread(join(seg_path, name), cv2.IMREAD_GRAYSCALE)
    seg_mask = cv2.resize(seg_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Convert to uint8
    gt_data = np.uint8(gt_mask)
    seg_data = np.uint8(seg_mask)

    # Initialize the combined masks for GT and Segmentation
    merged_gt_mask = np.zeros_like(gt_data)
    merged_seg_mask = np.zeros_like(seg_data)

    # Create a figure for visualization (2 rows and 9 columns)
    fig, axes = plt.subplots(2, 9, figsize=(20, 10))

    # Titles for the subplots
    titles = ['GT Class 0', 'GT Class 1', 'GT Class 2', 'GT Class 3', 'GT Class 4', 
              'GT Class 5', 'GT Class 6', 'GT Class 7', 'GT Class 8', 'Merged GT', 
              'Seg Class 0', 'Seg Class 1', 'Seg Class 2', 'Seg Class 3', 'Seg Class 4', 
              'Seg Class 5', 'Seg Class 6', 'Seg Class 7', 'Seg Class 8', 'Merged Seg']

    # Plot each class's GT and Segmentation
    for class_id in range(9):
        gt_class_mask = (gt_data == class_id).astype(np.uint8)
        seg_class_mask = (seg_data == class_id).astype(np.uint8)

        # Color the ground truth and segmentation masks for visualization
        gt_rgb = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
        seg_rgb = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)

        # Color the masks according to the predefined colors
        gt_rgb[gt_class_mask == 1] = class_colors[class_id]
        seg_rgb[seg_class_mask == 1] = class_colors[class_id]

        # Merge masks for visualization (value 1 to 8 for each class)
        merged_gt_mask[gt_class_mask == 1] = class_id  # Set class ID for ground truth
        merged_seg_mask[seg_class_mask == 1] = class_id  # Set class ID for segmentation

        # Plot GT and Segmentation for each class in the first two rows
        axes[0, class_id].imshow(gt_rgb, interpolation='none')
        axes[0, class_id].set_title(f"Class {class_id} GT")
        axes[0, class_id].axis('off')

        axes[1, class_id].imshow(seg_rgb, interpolation='none')
        axes[1, class_id].set_title(f"Class {class_id} Seg")
        axes[1, class_id].axis('off')

    # Plot the merged GT and merged Segmentation in the last columns
    axes[0, 8].imshow(merged_gt_mask, cmap='tab10', interpolation='none')
    axes[0, 8].set_title("Merged GT")
    axes[0, 8].axis('off')

    axes[1, 8].imshow(merged_seg_mask, cmap='tab10', interpolation='none')
    axes[1, 8].set_title("Merged Seg")
    axes[1, 8].axis('off')

    # Save the figure as an image
    fig.tight_layout()
    fig.savefig(join(output_path, name))
    plt.close(fig)  # Close the plot to avoid memory issues

print("Visualization completed and saved!")