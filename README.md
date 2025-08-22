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
