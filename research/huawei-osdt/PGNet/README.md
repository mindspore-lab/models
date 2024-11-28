#  Mindspore Implementation of PGNet

This repository is the mindspore implementation of PGNet, which paper titled "PGNet: Real-time Arbitrarily-Shaped Text Spotting with Point Gathering Network" and is accepted by AAAI 2021. The code is based on [official implementation](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/algorithm_e2e_pgnet.md).

## Model Introduction

PGNet is a new model for real-time arbitrary-shape text detection and recognition in the OCR field. It avoids time-consuming operations such as NMS and ROI, learns pixel-level character classification through point aggregation CTC loss, and uses a graph enhancement module to optimize recognition results and improve end-to-end performance.

## Dependencies

The train and inference code was tested on:

- **Ascend: 1*ascend-d910b|CPU: 24Core 192GB**
- **python3.9, mindspore-2.3.1, cann8.0.RC1**

The other dependent libraries has been recorded in **requirements.txt**, please first install above package, then use command below to install the environment.

```bash
pip install -e .
```

## Prepare images

Download the dataset to the corresponding subfolder. The download link is as follows：
(https://paddleocr.bj.bcebos.com/dataset/total_text.tar)
(https://paddleocr.bj.bcebos.com/dataset/Groundtruth.tar)

After download, the checkpoint path should be like this:

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

## Quick Start

### Inference

The best weight file of the model is saved in [checkpoint](https://download-mindspore.osinfra.cn/model_zoo/research/cv/pgnet/pgnet_best_weight.ckpt). You need to download it and put it in the weight directory, then execute the following command:

```bash
python tools/infer/predict_e2e.py \
    --image_dir train_data/total_text/train/rgb/img11.jpg \
    --e2e_algorithm PG \
    --e2e_model_config configs/pgnet_r50.yaml \
    --e2e_model_dir weight/pgnet_best_weight.ckpt
```

All results will be saved in the `inference_results/` directory.

### Evaluation

After downloading the dataset as described in [Prepare images], you can use the following command to evaluate it:

```bash
python tools/eval.py \
    -c configs/pgnet_r50.yaml \
    --opt eval.ckpt_load_path=weight/pgnet_best_weight.ckpt \
```

**Note**: Even though the random seed has been set, results may vary slightly on different hardware.

#### Evaluation Results

After running the commands in [Evaluation] and performing inference evaluation on the Ascend 910b, my test results are：

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

### Training

Refer to [Prepare images] to download the training data and start training. You can run the following command：

```bash
python tools/train.py \
    -c configs/pgnet_r50.yaml \
```

If you want to use multi-card training, you can run the following command：

```bash
bash scripts/train.sh configs/pgnet_r50.yaml [num_card]
```

For example, to train with 4 cards, you can run the following command：

```bash
bash scripts/train.sh configs/pgnet_r50.yaml 4
```

**Note**: The training code is still being updated to ensure the training result which not performs good now.

## Citation

Thanks to the official [PGNet](https://github.com/PaddlePaddle/PaddleOCR) repository. And cite their paper here:

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
