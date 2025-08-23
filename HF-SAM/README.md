
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

## Result
ACDC数据集的第549epoch结果：
```bash
Namespace(config=None, dataset='ACDC', volume_path='datasets/ACDC/test_vol', num_classes=4, list_dir='./lists/ACDC/', output_dir='./output_acdc', lora_ckpt='model_acdc_out/epoch_549.pth', img_size=224, input_size=224, seed=2345, is_savenii=False, deterministic=1, ckpt='/home/yanggq/project/Few-shot/checkpoints/sam_vit_b_01ec64.pth', vit_name='vit_b', rank=6, module='sam_lora_image_encoder_mask_decoder', stage=3, mode='test')
total 10 samples
10 test iterations per epoch
0it [00:00, ?it/s]idx 0 case patient001_frame02 mean_dice 0.228317 mean_hd95 16.125079 mean_jaccard 0.155435 mean_asd 6.990136 mean_recall 0.295565 mean_specificity 0.295565 mean_precision 0.194057
1it [00:01,  1.62s/it]idx 1 case patient001_frame01 mean_dice 0.274764 mean_hd95 10.425546 mean_jaccard 0.201739 mean_asd 5.848075 mean_recall 0.324755 mean_specificity 0.324755 mean_precision 0.249687
2it [00:03,  1.49s/it]idx 2 case patient007_frame01 mean_dice 0.212626 mean_hd95 11.322577 mean_jaccard 0.144950 mean_asd 5.880211 mean_recall 0.285438 mean_specificity 0.285438 mean_precision 0.199473
3it [00:04,  1.45s/it]idx 3 case patient007_frame02 mean_dice 0.200385 mean_hd95 11.206908 mean_jaccard 0.132833 mean_asd 5.987901 mean_recall 0.283459 mean_specificity 0.283459 mean_precision 0.178417
4it [00:05,  1.42s/it]idx 4 case patient008_frame02 mean_dice 0.264597 mean_hd95 10.689675 mean_jaccard 0.191229 mean_asd 6.252786 mean_recall 0.322322 mean_specificity 0.322322 mean_precision 0.238831
5it [00:07,  1.42s/it]idx 5 case patient008_frame01 mean_dice 0.297297 mean_hd95 10.231578 mean_jaccard 0.224678 mean_asd 5.634892 mean_recall 0.342020 mean_specificity 0.342020 mean_precision 0.286100
6it [00:08,  1.44s/it]idx 6 case patient011_frame02 mean_dice 0.106948 mean_hd95 20.462239 mean_jaccard 0.062259 mean_asd 12.362891 mean_recall 0.135595 mean_specificity 0.135595 mean_precision 0.088543
7it [00:10,  1.42s/it]idx 7 case patient011_frame01 mean_dice 0.120640 mean_hd95 20.182138 mean_jaccard 0.072183 mean_asd 11.604600 mean_recall 0.144429 mean_specificity 0.144429 mean_precision 0.103590
8it [00:11,  1.41s/it]idx 8 case patient013_frame02 mean_dice 0.265458 mean_hd95 8.432078 mean_jaccard 0.184448 mean_asd 5.030990 mean_recall 0.325124 mean_specificity 0.325124 mean_precision 0.280385
9it [00:12,  1.41s/it]idx 9 case patient013_frame01 mean_dice 0.270612 mean_hd95 8.868531 mean_jaccard 0.196521 mean_asd 4.713448 mean_recall 0.291712 mean_specificity 0.291712 mean_precision 0.299215
10it [00:14,  1.43s/it]
Mean class 1 name spleen mean_dice 0.012949 mean_hd95 30.134542 mean_jaccard 0.006541 mean_asd 20.589474 mean_recall 0.007028 mean_specificity 0.007028 mean_precision 0.089679
Mean class 2 name right kidney mean_dice 0.263541 mean_hd95 11.630884 mean_jaccard 0.155416 mean_asd 4.041971 mean_recall 0.287338 mean_specificity 0.287338 mean_precision 0.249367
Mean class 3 name left kidney mean_dice 0.620168 mean_hd95 9.413114 mean_jaccard 0.464553 mean_asd 3.490927 mean_recall 0.805802 mean_specificity 0.805802 mean_precision 0.508274
Mean class 4 name gallbladder mean_dice 0.000000 mean_hd95 0.000000 mean_jaccard 0.000000 mean_asd 0.000000 mean_recall 0.000000 mean_specificity 0.000000 mean_precision 0.000000
Testing performance in best val model: mean_dice : 0.224164 mean_hd95 : 12.794635 mean_jaccard 0.156628 mean_asd 7.030593 mean_recall 0.275042 mean_specificity 0.275042 mean_precision 0.211830
Testing Finished!
```
Nypsnase数据集的第549epoch结果：
```bash
Namespace(config=None, dataset='Synapse', volume_path='datasets/Synapse/test_vol_h5', num_classes=9, list_dir='./lists/Synapse/', output_dir='datasets/Synapse/test_vol_h5', lora_ckpt='model_synapse_out/epoch_1999.pth', img_size=224, input_size=224, seed=2345, is_savenii=False, deterministic=1, ckpt='/home/yanggq/project/Few-shot/checkpoints/sam_vit_b_01ec64.pth', vit_name='vit_b', rank=6, module='sam_lora_image_encoder_mask_decoder', stage=3, mode='test')
12 test iterations per epoch
0it [00:00, ?it/s]
idx 0 case case0008 mean_dice 0.005594 mean_hd95 24.223751 mean_jaccard 0.002869 mean_asd 3.809940 mean_recall 0.003116 mean_specificity 0.003116 mean_precision 0.027315
idx 1 case case0022 mean_dice 0.003694 mean_hd95 19.926098 mean_jaccard 0.001878 mean_asd 2.803683 mean_recall 0.002048 mean_specificity 0.002048 mean_precision 0.018809
idx 2 case case0038 mean_dice 0.008284 mean_hd95 12.511723 mean_jaccard 0.004302 mean_asd 2.197134 mean_recall 0.004663 mean_specificity 0.004663 mean_precision 0.037072
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

