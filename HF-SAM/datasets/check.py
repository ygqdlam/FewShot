import os
import shutil

# 路径自行修改
list_file = "/home/yanggq/project/semi/data/ACDC/test.list"
src_dir = "/home/yanggq/project/semi/data/ACDC/data/"
dst_dir = "datasets/ACDC/test_vol/"

# 确保目标文件夹存在
os.makedirs(dst_dir, exist_ok=True)

# 逐行读取 list 文件
with open(list_file, "r") as f:
    for line in f:
        filename = line.strip()
        if not filename:
            continue
        src_path = os.path.join(src_dir, filename + '.h5')
        dst_path = os.path.join(dst_dir, filename + '.h5')
        print(src_path)
        if os.path.exists(src_path):    
            # 确保目标目录存在（防止有子文件夹层级）
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)  # copy2 保留时间戳信息
            print(f"Copied: {filename}")
        else:
            print(f"Not found: {filename}")