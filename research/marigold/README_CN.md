# Marigold 的 Mindspore 实现

本仓库是 Marigold 的 Mindspore 实现，论文题为 [Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation](https://arxiv.org/abs/2312.02145) 被 CVPR 2024 接收。代码基于[官方实现](https://github.com/prs-eth/Marigold)修改而来。

![demo](doc/demo.png)

## 模型介绍
Marigold 是一个基于扩散的图像生成模型，用于单目深度估计，其核心思想是利用现代生成式图像模型中储存的丰富视觉知识。该模型源自 Stable Diffusion，并使用合成数据进行微调，能够在未见数据集上进行零样本迁移。

## 环境要求
训练和推理代码的测试环境：

- **1*ascend-snt9b|ARM: 24核 192GB**
- **python3.9, mindspore-2.2.14, cann7.0.0.beta1**

请先安装上述软件包和 [mindone](https://github.com/mindspore-lab/mindone)，然后使用以下命令安装剩余环境：

```bash
pip install -r requirements.txt
```

## 数据集处理

### 评估数据集下载

下载评估数据集到相应的子文件夹，请使用以下命令：

```bash
bash script/download_eval_data.sh
```

如果无法访问[官方存储](https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset)，你可以使用以下脚本下载我上传到[mindspore平台](https://xihe.mindspore.cn/datasets/Braval/Marigold-Eval)的评估数据集 NYUv2 和 KITTI：

```bash
bash script/download_eval_data_mindspore.sh
```

下载后，评估数据集路径应如下所示：

```
marigold
|——marigold-data
|   |——nyuv2
|   |   ㇗nyu_labeled_extracted.tar
|   |——kitti
|   |   ㇗kitti_eigen_split_test.tar
...
```

### 训练数据集下载

由于训练数据集之一 Hypersim 过大，代码仅验证了在数据集 [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) 上的训练。

你可以运行以下脚本下载 Virtual KITTI 2 的 RGB 和深度的 zip 文件，获取所需的训练格式：

```bash
bash script/download_vkitti.sh
```

如果无法访问 VKITTI，可以使用以下脚本下载我上传到 [mindspore平台](https://xihe.mindspore.cn/datasets/Braval/Marigold-Train) 的训练数据集：

```bash
bash script/download_vkitti_mindspore.sh
```

下载后，训练数据集路径应如下所示：

```
marigold
|——marigold-data
|   |——vkitti
|   |   ㇗vkitti.tar
...
```

## 快速入门

我们建议使用官方发布的 [checkpoint](https://huggingface.co/prs-eth/marigold-v1-0) 进行推理。你可以使用以下脚本下载 checkpoint 权重或自行从 hugging face 获取：

```bash
bash script/download_weights.sh marigold-v1-0
```

如果无法访问 hugging face 或[官方存储](https://share.phys.ethz.ch/~pf/bingkedata/marigold/checkpoint/marigold-v1-0.tar)，可以使用以下脚本下载我上传到[mindspore平台](https://xihe.mindspore.cn/models/Braval/Marigold-Model) 的 checkpoint。(请在运行前确保你已经安装了 `git-lfs`, 如果没有你可以参考 [git-lfs](https://github.com/git-lfs/git-lfs) 来安装。)

```bash
bash script/download_weights_mindspore.sh
```

下载后，checkpoint 路径应如下所示：

```
marigold
|——marigold-checkpoint
|   |——marigold-v1-0
...
```

### 推理过程与示例

#### 准备图像

使用论文中的样例：

```bash
bash script/download_sample_data.sh
```

如果无法访问官方存储，可以使用以下脚本下载我上传到[mindspore平台](https://xihe.mindspore.cn/datasets/Braval/Marigold-Example)的样例：

```bash
bash script/download_sample_data_mindspore.sh
```

你也可以把自己要进行深度估计的图像放在 `input/in-the-wild_example` 目录中，然后运行以下推理命令：

```bash
python run.py --fp16
```

这将使用如下默认设置对你的图像进行推理：

```bash
python run.py \
    --fp16 \
    --checkpoint marigold-checkpoint/marigold-v1-0 \
    --denoise_steps 50 \
    --ensemble_size 2 \
    --input_rgb_dir input/in-the-wild_example \
    --output_dir output/in-the-wild_example
```

所有结果将保存在 `output/in-the-wild_example` 目录中。

#### 推理设置调整

默认设置已针对推理效果进行优化调整。你也可以自行调整以下参数来改变代码行为：

1\. 准确性与速度的权衡：

- ensemble_size：集成推理的次数，更多结果集成的最终结果一般更准确。默认：2（run.py）、10（infer.py）。

- denoise_steps：每次推理的去噪步骤数。推荐使用 10-50 步。

2\. 处理分辨率：

- processing_res：推理时的分辨率。设置为 0 时直接对输入分辨率进行处理，不进行调整。默认：768。

- output_processing_res：是否按照推理时的分辨率输出，设置为 False 则按原输入分辨率输出。默认：False。

- resample_method：调整输入图像和深度估计结果的重采样方法。可以是 bilinear、bicubic 或 nearest。默认：bilinear。

3\. 是否使用半精度：

- half_precision 或 fp16：以半精度（16 位浮点）运行，减少内存使用，速度更快，但可能导致结果略有偏差。

4\. 色彩映射：

- color_map：色彩映射方式的设置，用于对深度估计结果进行着色。默认：Spectral。设置为 None 会跳过深度图的着色。

### 评估过程与示例

参考[数据集处理](#数据集处理)下载评估数据集后，可以使用以下脚本分别运行推理和评估：

```bash
# 运行推理
bash script/eval/11_infer_nyu.sh --fp16
# 评估推理结果
bash script/eval/12_eval_nyu.sh
```

**注意**：尽管已经设置了随机种子，不同硬件上的结果可能略有不同。

#### 参考评估结果

在[环境要求](#环境要求)一节给出的环境下，运行 `bash script/eval/11_infer_nyu.sh --fp16` 和 `bash script/eval/12_eval_nyu.sh` 在 NYUV2 上推理评估之后，结果保存在 `output/eval/nyu_test/eval_metric/eval_metrics-least_square.txt` 中，我的测试结果是：

<div style="overflow-x: auto;">
  <table>
    <tr>
      <th>abs_relative_difference</th>
      <th>squared_relative_difference</th>
      <th>rmse_linear</th>
      <th>rmse_log</th>
      <th>log10</th>
      <th>delta1_acc</th>
      <th>delta2_acc</th>
      <th>delta3_acc</th>
      <th>i_rmse</th>
      <th>silog_rmse</th>
    </tr>
    <tr>
      <td>0.07632677976141465</td>
      <td>0.034922071355487985</td>
      <td>0.2666904163644838</td>
      <td>0.1054675378854218</td>
      <td>0.03196998397227432</td>
      <td>0.9409628818883776</td>
      <td>0.9861820843519824</td>
      <td>0.9965493818191595</td>
      <td>0.060883635098728725</td>
      <td>10.48842097980554</td>
    </tr>
  </table>
</div>

运行 `bash script/eval/21_infer_kitti.sh --fp16` 和 `bash script/eval/22_eval_kitti.sh` 在 KITTI 上推理评估之后，结果保存在 `output/eval/kitti_eigen_test/eval_metric/eval_metrics-least_square.txt` 中，我的测试结果是：

<div style="overflow-x: auto;">
    <table>
        <tr>
            <th>abs_relative_difference</th>
            <th>squared_relative_difference</th>
            <th>rmse_linear</th>
            <th>rmse_log</th>
            <th>log10</th>
            <th>delta1_acc</th>
            <th>delta2_acc</th>
            <th>delta3_acc</th>
            <th>i_rmse</th>
            <th>silog_rmse</th>
        </tr>
        <tr>
            <td>0.1320127714469808</td>
            <td>0.5493009202080718</td>
            <td>3.5613749612471963</td>
            <td>0.16613638565750502</td>
            <td>0.054278594441513485</td>
            <td>0.8580591877961016</td>
            <td>0.9786774559119611</td>
            <td>0.9953850826046079</td>
            <td>0.016776495173318865</td>
            <td>16.43052654338995</td>
        </tr>
    </table>
</div>

### 训练过程与示例

参考[数据集处理](#数据集处理)下载训练数据后即可开始训练，建议在图模式下进行训练，以获得更快的性能。请先下载 Mindspore 实现的 Stable Diffusion v2 的 [checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_768_v-e12e3a9b.ckpt)，并将其保存到 `marigold-checkpoint` 目录中，可以运行如下命令：

```bash
mkdir -p marigold-checkpoint
wget -nv --show-progress -P marigold-checkpoint https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_768_v-e12e3a9b.ckpt
```

然后运行以下命令，在图模式下训练你的深度估计模型：

```bash
python train_graph.py \
    --train_config "config/train_marigold.yaml" \
    --output_path "output/graph-train" \
    --pretrained_model_path "marigold-checkpoint/sd_v2_768_v-e12e3a9b.ckpt"
```

目前不支持断点续训，建议不中断完成训练。

目前训练结果还有待优化，欢迎相关代码的优化建议。

## 引用

感谢官方的 [Marigold](https://github.com/prs-eth/Marigold) 仓库。引用他们的论文：

```bibtex
@InProceedings{ke2023repurposing,
      title={Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation},
      author={Bingxin Ke and Anton Obukhov and Shengyu Huang and Nando Metzger and Rodrigo Caye Daudt and Konrad Schindler},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2024}
}
```

## License

This work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

By downloading and using the code and model you agree to the terms in the  [LICENSE](LICENSE.txt).

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)
