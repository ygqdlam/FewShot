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
