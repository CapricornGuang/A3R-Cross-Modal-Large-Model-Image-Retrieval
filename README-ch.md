# 跨模态大模型图像检索比赛
![PyTorch 1.12.1](https://img.shields.io/badge/PyTorch-1.12.1-green?style=plastic)
![OpenClipTorch 2.17.1](https://img.shields.io/badge/OpenClipTorch-2.17.1-orange?style=plastic)
![CVPR 2023](https://img.shields.io/badge/CVPR-2023-red?style=plastic)

交通场景中高性能的图像检索能力对于交通执法、治安治理具有十分重要的作用，传统的图像检索方式通常使用先对图像进行属性识别再通过与期望属性的对比实现检索能力。随着多模态大模型技术的发展，文本与图像的表征统一和模态转换已有广泛应用，使用该能力可以进一步提升图像检索的精度和灵活性。


🎉🎉🎉我们的相关工作[自增强改进基础视觉语言模型中的文本图像检索](https://arxiv.org/abs/2306.06691) 已在CVPR 2023上发表，如果这个工作对你有帮助，请按照[以下格式](#Citations)引用我们。

🎉🎉🎉本工作获得[CVPR2023 大模型多模态检索挑战](https://foundation-model.com/) Top-10 Award奖项。


# 1 介绍

[CVPR2023 大模型多模态检索挑战](https://foundation-model.com/)旨在提升交通场景中文本图像检索的精度。因此我们将多种公开数据集以及网络数据中的交通参与者图像进行了文本描述标注从而构建了多对多的图像-文本对，选手可以在此基础上进行多模态技术的研究工作，提升文本检索图像的精度。

# 2 代码复现
Tip: 由于赛后资料整理时改变了Infer代码的执行逻辑，可能复现结果会有微小差异

## 2.1 赛制审核
1. 可一键复现的 Pytorch 算法代码：```notebook-reproduce.ipynb``` 提供了用于复现的一键运行 Jupyter Notebook, ```notebook-quick-review.ipynb``` 提供了用于快速得到最优结果文件的 Jupyter Notebook, ```notebook-quick-review-last.ipynb``` 提供了用于快速得到最后一次提交的结果文件的 Jupyter Notebook
2. 提交模型文件对应的 checkpoint：日志保存在 best-result-review 中，模型需要另外下载
   下载链接：https://pan.baidu.com/s/17P6nzWl9PnVH42DFQsCd_w 提取码：067e
3. 代码内容说明：在 ```notebook-reproduce.ipynb``` 与 ```notebook-reproduce.ipynb``` 中提供了详细说明
4. 模型构建思路

   （1）完整算法结构框图、思路步骤详述、代码组织结构介绍：见如下介绍

   （2）数据增强/清洗策略：见如下介绍

   （3）调参优化策略（若多次迭代，还需说明迭代的具体策略）：见如下介绍

   （4）训练脚本/代码，最好包含训练一个 epoch 的运行日志: 在"可一键复现的 Pytorch 算法代码, ```notebook-reproduce.ipynb```"部分提供. best-result-review 提供了得到 A 榜最好训练结果的运行日志

   （5）测试脚本/代码，必须包含评估得到最终精度的运行日志: 在"可一键复现的 Pytorch 算法代码, ```notebook-quick-review*.ipynb```"部分提供，其中 notebook-quick-review 表示最优的提交结果，notebook-quick-review-last 表示最后一次提交结果

## 2.2 预下载内容
1. 模型 checkpoint 文件: https://pan.baidu.com/s/17P6nzWl9PnVH42DFQsCd_w 提取码：067e
2. 数据集文件：https://pan.baidu.com/s/1RXR7q_LAxRusSlamfmoG9A 提取码：tjz2 
3. 数据增强文件：https://pan.baidu.com/s/1EFiY6dt5v0Na1SgGZByHvw 提取码：922a

## 2.3 代码结构
The tree below illustrates the organization of this project.
```bash
├── data
│   ├── augmented*.txt #数据增强文件
│   ├── car_attrbute*.json #汽车属性文件
│   ├── dataset #数据集（ImageRetrival，提供在百度盘需要自己下载）
├── best-result-review #与log一致，但该文件夹单独保存了最优提交的运行记录
│   │   ├──out.log
│   │   ├──params.txt
│   │   ├──checkpoints #需要自行下载
│   │   ├──tensorboard
├── log
│   ├── * #日志文件夹（每次训练都会生成一个专有的文件夹）
│   │   ├──out.log #输出训练日志
│   │   ├──params.txt #超参数
│   │   ├──checkpoints #模型权重
│   │   ├──tensorboard #cd 到 *目录后运行 tensorboard --logdir = ./tensorboard --host localhost --port 20421 会在localhost:20421打开当前训练的tensorboard
├── script #运行脚本，建议在notebook-reproduce.ipynb中看
├── src 
│   ├── infer
│   │   ├──merge_json.py
│   │   ├──open_clip_infer.py #推理代码
│   │   ├──open_clip_infer_prompt.py #推理代码+Prompt增强
│   ├── preprocess
│   │   ├──parse_attr.py #得到汽车属性文件
│   │   ├──parse_augment.py #解析数据增强后的文件格式为CLIP训练需要的格式
│   │   ├──parse_split.py #解析*_person_label.txt, *_car_label.txt(分离后的data/datasets/*_label.txt)的文件格式为CLIP训练需要的格式
│   ├── train
│   │   ├──* #参考https://github.com/mlfoundations/open_clip
│   │   ├──data #存放了open_clip用于推理的csv文件
│   │   ├──training
│   │   │   ├──*
│   │   │   ├──params.py #script/run_model.sh中对于各个参数的说明
│   │   │   ├──main.py #模型训练时运行的主文件 
```

# 3 模型设计
本比赛主要利用 CLIP 方法进行多模态对比学习训练, 整体的训练思路如下：
<!-- <p align="center">
<img src="framework.png" height = "240" alt="" align=center />
<br><br>
<b>图1.整体思路</b>
</p> -->

![图1.整体思路](./framework.png)

## 3.1 优化策略
以下描述的均为能够提高 A 榜单得分的策略。

### 3.1.1 模型选择
大模型内部存在大量的隐式知识，模型的网络通路也具备更强的鲁棒性。在本项目发现大模型对于光照具有很好的鲁棒性，具备很好的白平衡能力
选取 ViT-G-14 大模型作为主干网络

### 3.1.2 图像样本大小平衡
考虑到车与人之间的图片大小比例区别较大，因此将车与人的图片都 Padding 为正方向（填充0），然后 resize 到 224

### 3.1.3 零样本数据增强
**(a) 分析**

使用 ViT-G-14 针对汽车样本进行零样本数据增强，从颜色、品牌、车型三个方面进行数据讨论
- 颜色：ViT-G-14 对于颜色的识别具有很强的鲁棒性，可以进行数据增强
- 车型：ViT-G-14 的对比学习版本在互联网数据上进行训练，互联网对车型这种大类的数据很充分，可以进行数据增强
- 品牌：实测大模型对于品牌这种细分类的效果很不好，不进行数据增强

**(b) Prompt 构造**

Prompt 的构造要求如下：
- 对于缺少类型的，Prompt构造为："{prefix} + 颜色 + 品牌 + {Prompt-Type}"
- 对于缺少颜色和品牌的，不必要去zero-shot品牌，Prompt构造为： "{prefix} + {Prompt-Color} + Type"

此处 {prefix} 是指属性文本 label 中原先的自带前缀

### 3.1.4 数据分布差异大
1. 训练集与测试集车辆图像分布差异较大，导致测试集上的精度提升无异于测试集精度提升，使用小学习率 $4e-7$，只微调 5 个 epoch。
2. 训练集是网络数据集，测试集是监控数据集，本质上是属于跨域问题，在 Query 的时候做了一个 Prompt 增强，也即对于汽车的数据 + "image taken by traffic surveillance cameras"。



<a id="Citations"></a>

# 4 引用
请用以下信息引用我们的工作:
```bash
Yang Y, Wang Y, Geng S, et al. Self-Enhancement Improves Text-Image Retrieval in Foundation Visual-Language Models[J]. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

@inproceedings{yang2023self,
  title={Self-Enhancement Improves Text-Image Retrieval in Foundation Visual-Language Models},
  author={Yang, Yuguang and Wang, Yiming and Geng, Shupeng and Wang, Runqi and Wang, Yimi and Wu, Sheng and Zhang, Baochang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

# 4 致谢
感谢由 Ilharco, Gabriel 等人提供的 CLIP 对比学习训练代码 [OpenCLIP](https://github.com/mlfoundations/open_clip)
