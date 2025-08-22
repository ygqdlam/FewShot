# FewShot
此项目需要包含两个部分：sam和clip

---

## 1. 确定数据集
使用以下数据集：
###  多类别数据集
必须要先训练一下
```bash
HF-SAM：使用的是Synapse数据集
AutoSAM：使用的是ACDC和Synapse，本项目需要使用ACDC进行实验。后续可以添加一个器官分割的数据集。
```

python scripts/main_autosam_seg.py --src_dir ./data/ACDC \
--data_dir ./data/ACDC/imgs/ --save_dir ./results/ACDC  \
--b 4 --dataset ACDC --gpu 0 \
--fold 0 --tr_size 1  --model_type vit_h --num_classes 4




---
## Citation
### AutoSAM
链接：
```bash
github:https://github.com/xhu248/AutoSAM
```
引用：
```bash
@article{hu2023efficiently,
  title={How to Efficiently Adapt Large Segmentation Model (SAM) to Medical Images},
  author={Hu, Xinrong and Xu, Xiaowei and Shi, Yiyu},
  journal={arXiv preprint arXiv:2306.13731},
  year={2023}
}
```


