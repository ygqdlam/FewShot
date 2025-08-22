# 介绍MedCLIP-SAMv2的使用
---
## MedCLIP-SAMv2
```bash
AutoSAM：使用的是ACDC和Synapse，本项目需要使用ACDC进行实验。后续可以添加一个器官分割的数据集。
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
### MedCLIP-SAMv2
链接：
```bash
github:https://github.com/HealthX-Lab/MedCLIP-SAMv2
```
引用：
```bash
@article{koleilat2024medclipsamv2,
  title={MedCLIP-SAMv2: Towards Universal Text-Driven Medical Image Segmentation},
  author={Koleilat, Taha and Asgariandehkordi, Hojat and Rivaz, Hassan and Xiao, Yiming},
  journal={arXiv preprint arXiv:2409.19483},
  year={2024}
}

@inproceedings{koleilat2024medclip,
  title={MedCLIP-SAM: Bridging text and image towards universal medical image segmentation},
  author={Koleilat, Taha and Asgariandehkordi, Hojat and Rivaz, Hassan and Xiao, Yiming},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={643--653},
  year={2024},
  organization={Springer}
}
```
}
```
