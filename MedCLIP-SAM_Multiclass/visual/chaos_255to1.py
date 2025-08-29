import os
import cv2
import numpy as np

# 输入和输出文件夹
input_dir = "data/CHAOS/test_masks"        # 修改为你的mask文件夹路径
output_dir = "data/CHAOS/test_masks2"
os.makedirs(output_dir, exist_ok=True)

# 映射字典
organ_dict = {
    126: 3,   # Left Ventricle
    189: 2,    # Right Ventricle
    63: 1,    # Myocardium
    252: 4    # Myocardium

}
# 遍历文件夹
for filename in os.listdir(input_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):  # 根据实际格式调整
        filepath = os.path.join(input_dir, filename)
        
        # 读取灰度图
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        # 创建新mask
        new_mask = np.zeros_like(mask, dtype=np.uint8)
        
        # 替换像素值
        for old_val, new_val in organ_dict.items():
            new_mask[mask == old_val] = new_val
        
        # 保存新mask
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, new_mask)
        print(f"[✓] Processed {filename}")