# MedCLIP-SAMv2
在该项目下介绍如何使用M饿到CLIP-SAMv2

---

## 1.数据路径
```bash
MedCLIP-SAM<br>
  data<br>
    brain_tumors<br>
      train_images<br>
        000.png<br>
      train_masks<br>
        000.png<br>
      test_images<br>
      test_masks<br>
      val_images<br>
      val_masks<br>
    breast_tumors<br>
```
---

## 2.运行命令
### Zero-shot Segmentation

首先，需要下载 generate_saliency_maps.py 文件中所需要的 AutoModel 权重，并放在./saliency_maps/model路径下。
```bash
https://huggingface.co/ZiyueWang/biomedclip
https://pan.baidu.com/s/1ONkJTR4nAqa7yk5c_Kl2Ig?pwd=v44v
```
第二，需要设置好 SAM 权重，在zeroshot_brain_tumors.sh文件里设置。

第三，必须去掉zeroshot_brain_tumors.sh文件里的 --finetuned \ 参数,下载下来的权重肯定是不对的。

最后，You can run the whole zero-shot framework with the following:
```bash
bash zeroshot.sh <path/to/dataset>
bash zeroshot_scripts/zeroshot_brain_tumors.sh data/brain_tumors
```

--- 

## 3.运行结果
--finetuned  参数运行结果：
```bash
(medclipsamv2) yanggq@yanggq-Dlam:~/project/Few-shot/MedCLIP-SAM$ bash zeroshot_scripts/zeroshot_brain_tumors.sh data/brain_tumors
Average DSC for test_masks: 0.017322166666666666
Average NSD for test_masks: 0.019587
(medclipsamv2) yanggq@yanggq-Dlam:~/project/Few-shot/MedCLIP-SAM$ bash zeroshot_scripts/zeroshot_breast_tumors.sh data/breast_tumors/
Average DSC for test_masks: 0.03918761061946902
Average NSD for test_masks: 0.04334778761061947
```
去掉--finetuned  参数运行结果：
```bash
(medclipsamv2) yanggq@yanggq-Dlam:~/project/Few-shot/MedCLIP-SAM$ bash zeroshot_scripts/zeroshot_brain_tumors.sh data/brain_tumors
Average DSC for test_masks: 0.7381393333333334
Average NSD for test_masks: 0.7953011666666666
(medclipsamv2) yanggq@yanggq-Dlam:~/project/Few-shot/MedCLIP-SAM$ bash zeroshot_scripts/zeroshot_breast_tumors.sh data/breast_tumors/
Average DSC for test_masks: 0.8020929203539823
Average NSD for test_masks: 0.8457584070796459
```

使用 selected_descriptions_from_best.json 作为text_prompts 的测试结果：
```bash
Average DSC for test_masks: 0.7510429999999999
Average NSD for test_masks: 0.8032523333333332
```


---
## Citation
链接：
```bash
https://github.com/HealthX-Lab/MedCLIP-SAMv2
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


