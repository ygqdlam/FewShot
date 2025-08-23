
# 介绍HF-SAM的使用
---
## HF-SAM
```bash
HF-SAM：使用的是ACDC和Synapse。
```

数据集的目录结构：
```bash
HF-SAM-main
--datasets
----Synapse
------train_npz
---------case0005_slice000.npz
---------xxxx
------test_vol_h5
---------case0001.npy.h5
----ACDC
------ACDC_training_slices
---------patient001_frame01_slice_0.h5
---------xxxx
------ACDC_training_volumes
---------patient001_frame01.h5
```
运行命令
```bash
sh train_synapse.sh
```


---
## Citation
### HF-SAM
链接：
```bash
https://github.com/1683194873xrn/HF-SAM/tree/main
```
```bash
https://github.com/HuCaoFighting/Swin-Unet/tree/main
```
引用：
```bash
@InProceedings{swinunet,
author = {Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
title = {Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation},
booktitle = {Proceedings of the European Conference on Computer Vision Workshops(ECCVW)},
year = {2022}
}

@misc{cao2021swinunet,
      title={Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation}, 
      author={Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
      year={2021},
      eprint={2105.05537},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
