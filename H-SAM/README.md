
# H-SAM
## result
### Synapse
自己训练的
```bash
(HFSAM) ygq@4090:/home/sanx/FewShot/H-SAM-main$ CUDA_VISIBLE_DEVICES="0" python test.py --is_savenii --lora_ckpt /home/sanx/FewShot/H-SAM-main/output/Synapse_224_pretrain_vit_b_epo300_bs8_lr0.0026_s2345/epoch_299.pth --vit_name='vit_b' --ckpt=/home/sanx/FewShot/checkpoints/sam_vit_b_01ec64.pth --img_size=224 --stage=3
Namespace(config=None, volume_path='datasets/Synapse/test_vol_h5', dataset='Synapse', num_classes=8, list_dir='./lists/lists_Synapse/', output_dir='./outputs', img_size=224, input_size=224, seed=2345, is_savenii=True, deterministic=1, ckpt='/home/sanx/FewShot/checkpoints/sam_vit_b_01ec64.pth', lora_ckpt='/home/sanx/FewShot/H-SAM-main/output/Synapse_224_pretrain_vit_b_epo300_bs8_lr0.0026_s2345/epoch_299.pth', vit_name='vit_b', rank=5, module='sam_lora_image_encoder', stage=3, mode='test')
11 test iterations per epoch
0it [00:00, ?it/s]idx 0 case case0008 mean_dice 0.584672 mean_hd95 22.748644
1it [02:54, 174.60s/it]idx 1 case case0022 mean_dice 0.858642 mean_hd95 10.189292
2it [04:35, 131.36s/it]idx 2 case case0038 mean_dice 0.722493 mean_hd95 12.231314
3it [06:27, 122.48s/it]idx 3 case case0036 mean_dice 0.800167 mean_hd95 15.804154
4it [10:15, 164.11s/it]idx 4 case case0002 mean_dice 0.833974 mean_hd95 6.011691
5it [12:52, 161.44s/it]idx 5 case case0029 mean_dice 0.750571 mean_hd95 7.895119
6it [14:40, 143.22s/it]idx 6 case case0003 mean_dice 0.623429 mean_hd95 45.822354
7it [18:38, 174.35s/it]idx 7 case case0001 mean_dice 0.728313 mean_hd95 28.910191
8it [21:26, 172.46s/it]idx 8 case case0004 mean_dice 0.725630 mean_hd95 16.732736
9it [24:04, 167.74s/it]idx 9 case case0025 mean_dice 0.846449 mean_hd95 5.999244
10it [25:39, 145.39s/it]idx 10 case case0035 mean_dice 0.808287 mean_hd95 6.779104
11it [27:09, 148.18s/it]
Mean class 1 name spleen mean_dice 0.782422 mean_hd95 13.966583
Mean class 2 name right kidney mean_dice 0.550000 mean_hd95 10.993355
Mean class 3 name left kidney mean_dice 0.819134 mean_hd95 18.762610
Mean class 4 name gallbladder mean_dice 0.817748 mean_hd95 16.789096
Mean class 5 name liver mean_dice 0.926443 mean_hd95 14.537816
Mean class 6 name stomach mean_dice 0.510218 mean_hd95 20.953818
Mean class 7 name aorta mean_dice 0.878457 mean_hd95 12.241165
Mean class 8 name pancreas mean_dice 0.739306 mean_hd95 22.027442
Testing performance in best val model: mean_dice : 0.752966 mean_hd95 : 16.283986
Testing Finished!
```
他们给的10%的训练结果
```bash
(HFSAM) ygq@4090:/home/sanx/FewShot/H-SAM-main$ CUDA_VISIBLE_DEVICES="0" python test.py --is_savenii --lora_ckpt 220_epoch_299.pth --vit_name='vit_b' --ckpt=/home/sanx/FewShot/checkpoints/sam_vit_b_01ec64.pth --stage=3 --img_size=512
Namespace(config=None, volume_path='datasets/Synapse/test_vol_h5', dataset='Synapse', num_classes=8, list_dir='./lists/lists_Synapse/', output_dir='./outputs', img_size=512, input_size=224, seed=2345, is_savenii=True, deterministic=1, ckpt='/home/sanx/FewShot/checkpoints/sam_vit_b_01ec64.pth', lora_ckpt='220_epoch_299.pth', vit_name='vit_b', rank=5, module='sam_lora_image_encoder', stage=3, mode='test')
11 test iterations per epoch
0it [00:00, ?it/s]idx 0 case case0008 mean_dice 0.121077 mean_hd95 85.711712
1it [02:58, 178.08s/it]idx 1 case case0022 mean_dice 0.095601 mean_hd95 120.592880
2it [04:43, 135.57s/it]idx 2 case case0038 mean_dice 0.074279 mean_hd95 103.062899
3it [06:39, 126.40s/it]idx 3 case case0036 mean_dice 0.091230 mean_hd95 117.284362
4it [10:34, 169.30s/it]idx 4 case case0002 mean_dice 0.116032 mean_hd95 99.098309
5it [13:14, 166.07s/it]idx 5 case case0029 mean_dice 0.058383 mean_hd95 114.931074
6it [15:05, 147.32s/it]idx 6 case case0003 mean_dice 0.046036 mean_hd95 131.597184
7it [19:08, 178.49s/it]


```






## Citation
```bash
https://github.com/Cccccczh404/H-SAM/tree/main
```
```bash
@inproceedings{cheng2024unleashing,
  title={Unleashing the Potential of SAM for Medical Adaptation via Hierarchical Decoding},
  author={Cheng, Zhiheng and Wei, Qingyue and Zhu, Hongru and Wang, Yan and Qu, Liangqiong and Shao, Wei and Zhou, Yuyin},
  booktitle={CVPR},
  year={2024}
}
```
