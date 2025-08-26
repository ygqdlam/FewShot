# MedCLIP-SAMv2
在该项目下介绍如何使用MedCLIP-SAMv2是如何进行多类别分割的

---

## 1.数据路径
MedCLIP-SAM
  data
    brain_tumors
      train_images
        000.png
      train_masks
        000.png
      test_images
      test_masks
      val_images
      val_masks
    breast_tumors

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
### 3.单个类别到多个类别拓展

#### 修改 saliency_maps/generate_saliency_maps.py 文件

新增函数 text_prompts_index 
```bash
def text_prompts_index(dataset):
    # 顺序应与json里的顺序一样
    if dataset == "chaos":
        index = ['Left Kidney','Right Kidney','Liver','Spleen']
    elif dataset == "acdc":
        index = ['Left Ventricle','Right Ventricle','Myocardium']
    elif dataset == "prostate":
        index = ['Prostate']
    return index
```

命令行参数新增 --dataset
```bash
parser.add_argument('--dataset', required=True, default="chaos", type=str, help='select dataset')
```

显著性图生成逻辑修改
```bash
# 处理每个器官
for organ, prompts in text.items():
    if organ in prompts_index:
        # 1. 图像预处理,已经处理过了
        # 2. 文本编码
        text_ids = torch.tensor([tokenizer.encode(prompts, add_special_tokens=True)]).to(args.device)

        # 3. 生成热力图
        vmap = vision_heatmap_iba(
            text_ids, image_feat,
            model, args.vlayer, args.vbeta, args.vvar,
            ensemble=args.ensemble,
            progbar=False
        )
        # 4. 调整热力图大小并保存
        img = np.array(image)
        vmap_resized = cv2.resize(np.array(vmap), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        output_filename = f"{os.path.splitext(image_id)[0]}_{organ}.png"
        cv2.imwrite(os.path.join(args.output_path, output_filename), vmap_resized * 255)
```

模型加载部分
```bash
prompts_index = text_prompts_index(args.dataset)
```

运行的地方加上参数
```bash
--dataset synapse
```

#### 修改 segment-anything/prompt_sam.py 文件
该文件从以下代码开始，后面的代码都经过修改。与我写的SAM_Multi项目不同的是，我只使用3个point，这里貌似用了allpoint。
```bash
import re
ORGAN_NAMES = ['Spleen', 'Right kidney', 'Left kidney', 'Gallbladder', 'Esophagus', 'Liver', 'Stomach', 'Pancreas']
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
```

#### 修改 evaluation/eval.py 文件
该文件修改内容过多，已重新写一个文件，eval_synapse.py:


### 3.可视化
新增了visual文件夹，运行以下命令进行可视化：
```bash
visual/synapse.py
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


