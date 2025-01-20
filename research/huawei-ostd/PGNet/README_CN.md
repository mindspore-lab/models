# PGNet 的 Mindspore 实现

本仓库是 PGNet 的 Mindspore 实现，论文题为 [PGNet: Real-time Arbitrarily-Shaped Text Spotting with Point Gathering Network](https://arxiv.org/abs/2104.05458) 被 AAAI 2021 接收。代码基于[官方实现](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/algorithm_e2e_pgnet.md)修改而来。

## 模型介绍
PGNet是OCR领域实时任意形状文本检测与识别的新模型，避免了NMS、ROI等耗时操作，通过点聚集CTC损失学习像素级字符分类，使用图强化模块优化识别结果，提升端到端性能。

## 环境要求
训练和推理代码的测试环境：

- **Ascend: 1*ascend-d910b|CPU: 24核 192GB**
- **python3.9, mindspore-2.3.1, cann8.0.RC1**

请先安装上述软件包，然后使用以下命令安装剩余环境：

```bash
pip install -e .
```

## 数据集处理
下载评估数据集到相应的子文件夹，下载链接如下：
(https://paddleocr.bj.bcebos.com/dataset/total_text.tar)
(https://paddleocr.bj.bcebos.com/dataset/Groundtruth.tar)

下载后，数据集路径应如下所示：

```
train_data
|——total_text
|   |——train
|   |   ㇗rgb
        ㇗train.txt
|   |——test
|   |   ㇗gt
|   |   ㇗rgb
|   |   ㇗test.txt
```

## 快速入门

### 推理过程与示例

模型的最好权重文件保存在 [checkpoint](https://download-mindspore.osinfra.cn/model_zoo/research/cv/pgnet/pgnet_best_weight.ckpt) ，您需要自行下载后放入weight目录下，之后执行以下命令：

```bash
python tools/infer/predict_e2e.py \
    --image_dir train_data/total_text/train/rgb/img11.jpg \
    --e2e_algorithm PG \
    --e2e_model_config configs/pgnet_r50.yaml \
    --e2e_model_dir weight/pgnet_best_weight.ckpt
```
所有结果将保存在 `inference_results/` 目录中。

### 评估过程与示例

参考[数据集处理]下载数据集后，可以使用以下命令进行评估：

```bash
python tools/eval.py \
    -c configs/pgnet_r50.yaml \
    --opt eval.ckpt_load_path=weight/pgnet_best_weight.ckpt \
```

**注意**：尽管已经设置了随机种子，不同硬件上的结果可能略有不同。

#### 参考评估结果
运行 [评估过程与示例] 中的命令，在 Ascend 910b 上推理评估之后，我的测试结果是：

<div style="overflow-x: auto;">
    <table>
        <tr>
            <th>total_num_gt</th>
            <th>total_num_det</th>
            <th>global_accumulative_recall</th>
            <th>hit_str_count</th>
            <th>recall</th>
            <th>precision</th>
            <th>f_score</th>
            <th>seqerr</th>
            <th>recall_e2e</th>
            <th>precision_e2e</th>
            <th>f_score_e2e</th>
        </tr>
        <tr>
            <td>2204</td>
            <td>2070</td>
            <td>1819.9999999999966</td>
            <td>1266</td>
            <td>0.8257713248638823</td>
            <td>0.8765217391304333</td>
            <td>0.8503900216750798</td>
            <td>0.3043956043956031</td>
            <td>0.574410163339383</td>
            <td>0.6115942028985507</td>
            <td>0.5924192793635938</td>
        </tr>
    </table>
</div>

### 训练过程与示例

参考[数据集处理]下载训练数据后即可开始训练，可以运行如下命令：

```bash
python tools/train.py \
    -c configs/pgnet_r50.yaml \
```

如果想使用多卡训练，可以运行如下命令（num_card指卡数）：

```bash
bash scripts/train.sh configs/pgnet_r50.yaml [num_card]
```

例如，使用4卡训练，可以运行以下命令：

```bash
bash scripts/train.sh configs/pgnet_r50.yaml 4
```

目前已支持断点续训，只需要将configs/pgnet_r50.yaml中的resume属性置为True。

目前训练结果还有待优化，欢迎相关代码的优化建议。

## 引用

感谢官方的 [PGNet](https://github.com/PaddlePaddle/PaddleOCR) 仓库。引用他们的论文：

```bibtex
@InProceedings{wang2021pgnet,
      title={Pgnet: Real-time arbitrarily-shaped text spotting with point gathering network},
      author={Wang, Pengfei and Zhang, Chengquan and Qi, Fei and Liu, Shanshan and Zhang, Xiaoqiang and Lyu, Pengyuan and Han, Junyu and Liu, Jingtuo and Ding, Errui and Shi, Guangming},
      booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
      year={2021}
}
```

## License

This work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

By downloading and using the code and model you agree to the terms in the  [LICENSE](LICENSE.txt).

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)
