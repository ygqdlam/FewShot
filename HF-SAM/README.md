# 介绍HF-SAM的使用
---
## HF-SAM
```bash
HF-SAM：使用的是ACDC和Synapse。

```
```bash
python scripts/main_autosam_seg.py --src_dir ./data/ACDC \
--data_dir ./data/ACDC/imgs/ --save_dir ./results/ACDC  \
--b 4 --dataset ACDC --gpu 0 \
--fold 0 --tr_size 1  --model_type vit_h --num_classes 4
```
```bash
PYTHONPATH=$(pwd) python scripts/main_autosam_seg.py \
  --src_dir ./data/ACDC \
  --data_dir ./data/ACDC/imgs/ \
  --save_dir ./results/ACDC \
  --b 4 --dataset ACDC --gpu 0 --fold 0 --tr_size 1 \
  --model_type vit_b --num_classes 4
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
