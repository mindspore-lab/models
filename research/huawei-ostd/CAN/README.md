# Handwriting mathematical formula recognition algorithm-CAN

- [Handwriting mathematical formula recognition algorithm-CAN](#handwriting-mathematical-formula-recognition-algorithm-can)
  - [1. Algorithm introduction](#1-algorithm-introduction)
  - [2. Environment configuration](#2-environment-configuration)
  - [3. Model prediction, evaluation and training](#3-model-prediction-evaluation-and-training)
    - [3.1 Prediction](#31-prediction)
    - [3.2 Dataset preparation](#32-dataset-preparation)
      - [Prepare the training set](#prepare-the-training-set)
      - [Prepare the validation set](#prepare-the-validation-set)
    - [3.3 Training](#33-training)
    - [3.4 Evaluate](#34-evaluate)
  - [4. FAQ](#4-faq)
  - [Quote](#quote)
  - [Reference](#reference)



## 1. Algorithm introduction
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->
> [CAN: When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition](https://arxiv.org/pdf/2207.11463.pdf)

CAN is an attention-mechanism encoder-decoder handwriting mathematical formula recognition algorithm with a weakly supervised counting module. In this paper, the author studies most of the existing handwritten mathematical formula recognition algorithms and finds that they basically adopt the encoder-decoder structure based on the attention mechanism. This structure can make the model pay attention to the corresponding position region of the symbol in the image when recognizing each symbol. When recognizing conventional text, the movement law of attention is relatively simple (usually from left to right or from right to left), and this mechanism has high reliability in this scenario. However, when recognizing mathematical formulas, the movement of attention in the image has more possibilities. Therefore, when decoding complex mathematical formulas, the model is prone to inaccurate attention, leading to repeated recognition of a symbol or missing recognition of a symbol.

To this end, the authors design a weakly supervised count module that can predict the number of each symbol class without symbol-level location annotations, and then plug this into a typical attention-based HMER codec model. This approach is mainly based on the following two considerations: 1. Symbol counting can implicitly provide symbol location information, which can make attention more accurate. 2. Symbol counting results can be used as additional global information to improve the accuracy of formula recognition.

<p align="center">
  <img src="https://raw.githubusercontent.com/zhangjunlongtech/Material/refs/heads/main/CAN/miss_word.png" width=640 />
</p>
<p align="center">
  <em> Comparison of handwritten mathematical formula recognition algorithms [<a href="#参考文献">1</a>] </em>
</p>

The CAN model consists of a backbone feature extraction network, a multi-scale counting module (MSCM) and an attentional decoder (CCAD) that combines counting. Trunk feature extraction uses DenseNet to obtain the feature map, and inputting the feature map into MSCM to obtain a Counting Vector. The dimension of the counting vector is 1*C, where C is the size of the formula word list. Then, the counting vector and the feature map are input into CCAD together. Final output formula of latex.
<p align="center">
  <img src="https://raw.githubusercontent.com/zhangjunlongtech/Material/refs/heads/main/CAN/total_process.png" width=640 />
</p>
<p align="center">
  <em> Global model structure [<a href="#参考文献">1</a>] </em>
</p>

The multi-scale counting module MSCM block is designed to predict the number of each symbol class, which consists of multi-scale feature extraction, channel attention and pooling operators. Due to differences in writing habits, formula images often contain symbols of various sizes. The size of a single convolution kernel cannot handle scale changes efficiently. To do this, two parallel convolution branches are first utilized to extract multi-scale features by using different kernel sizes (set to 3×3 and 5×5). After the convolution layer, channel attention is used to further enhance the feature information.

<p align="center">
  <img src="https://raw.githubusercontent.com/zhangjunlongtech/Material/refs/heads/main/CAN/MSCM.png" width=640 />
</p>
<p align="center">
  <em> MSCM multi-scale counting module [<a href="#参考文献">1</a>] </em>
</p>

Attention decoder combined with counting: In order to enhance the perception of spatial position of the model, location coding is used to represent different spatial positions in the feature map. In addition, unlike most previous formula recognition methods which only use local features for symbol prediction, symbol counting results are introduced as additional global information to improve the recognition accuracy.

<p align="center">
  <img src="https://raw.githubusercontent.com/zhangjunlongtech/Material/refs/heads/main/CAN/CCAD.png" width=640 />
</p>
<p align="center">
  <em> Combined with counting attention decoder CCAD [<a href="#参考文献">1</a>] </em>
</p>

<a name="model"></a>
`CAN` is trained using the CROHME handwriting formula dataset with the following accuracy on the corresponding test set:

| model | backbone network | configuration file |ExpRate| Download link |
| ----- | ----- | ----- | ----- | ----- |
|CAN|DenseNet|[rec_d28_can.yml](./configs/can_d28.yaml)|52.84%|[Training model](https://paddleocr.bj.bcebos.com/contribution/rec_d28_can_train.tar)|

<a name="2"></a>
## 2. Environment configuration
Please install `MindSpore = 2.4.0` and go to the project directory and execute `pip install -e .`


<a name="3"></a>
## 3. Model prediction, evaluation and training

<a name="3-1"></a>
### 3.1 Prediction

First prepare the model weight file, here the best weight file provided by the project as an example ([weight file download address](https://download-mindspore.osinfra.cn/model_zoo/research/cv/can/)), run the following command to deduce:

```shell

Python  /tools/predict_can.py   --image_dir {path_to_img} \
                                --rec_algorithm CAN \
                                --rec_model_dir {path_to_ckpt} \
                                --rec_char_dict_path {path_to_dict} \

```
**Note:**
- Where `--image_dir` is the image address, `rec_model_dir` is the weight file address, `rec_char_dict_path` is the address of the identification dictionary, which needs to be modified according to the actual address during model inference
- It should be noted that the predicted image is **white characters** on black background, that is, the handwritten formula part is white and the background is black.

![Sample test picture](https://raw.githubusercontent.com/zhangjunlongtech/Material/refs/heads/main/CAN/101_user0.jpg)

After executing the command, the predicted result of the above image (the recognized text) is printed on the screen, as shown in the following example:
```shell
All rec res: ['S = ( \\sum _ { i = 1 } ^ { n } \\theta _ { i } - ( n - 2 ) \\pi ) r ^ { 2 }']
```

<a name="3-2"></a>

### 3.2 Dataset preparation

This model provides the data set, the [` CROHME dataset `](https://paddleocr.bj.bcebos.com/dataset/CROHME.tar) will be stored as a handwritten formula black white format, if you prepare data sets, by contrast, please unified treatment before training data set.


#### Prepare the training set
Please put all training pictures in the same folder, and specify a txt file in the upper path to label all training pictures and corresponding labels. txt file examples are as follows

```
# Filename # corresponds to tag
word_421.png	k ^ { 3 } + 1 4 k ^ { 2 } - 1 3 2 k + 1 7 8 9
word_1657.png	 x _ { x } ^ { x } + y _ { y } ^ { y } + z _ { z } ^ { z } - x - y - z
word_1814.png	\sqrt { a } = 2 ^ { - n } \sqrt { 4 ^ { n } a }
```
**Note** : Please separate the image name and label with `\tab `and avoid using Spaces or other delimiters.

The final training set store will take the following form:

```
|-data
    |- gt_training.txt
    |- training
        |- word_001.png
        |- word_002.jpg
        |- word_003.jpg
        | ...
```

#### Prepare the validation set
Also, put all verified images in the same folder, and specify a txt file in the upper path to label all verified images and corresponding labels. The final verification set will be stored in the following form:

```
|-data
    |- gt_validation.txt
    |- validation
        |- word_001.png
        |- word_002.jpg
        |- word_003.jpg
        | ...
```
<a name="3-3"></a>
### 3.3 Training

After completing the data preparation, you can start the training, the training command is as follows:
```shell
python tools/train.py --config configs/rec_d28_can.yaml
```
The command for training multiple compute cards is as follows:
```shell
mpirun --allow-run-as-root -n {card_nums} python tools/train.py --config configs/can_d28.yaml
```
**Note:**
- You need to enter the `configs/rec_d28_can.yaml` configuration file and configure `character_dict_path` and `ckpt_load_path` as the current address
- Need to adjust `is_train` to `True` under configuration file `model-head` entry
- Need to adjust `dataset_root`, `data_dir`, `label_file` to the actual address of the training dataset under the configuration file `train-dataset` entry
- The `eval.py` and `rec_d28_can.yaml` files can be replaced with absolute file addresses on the command line

<a name="3-4"></a>
### 3.4 Evaluate

Evaluate the trained model file using the following command:

```shell
python tools/eval.py --config configs/rec_d28_can.yaml
```
**Note:**
- You need to enter the `configs/rec_d28_can.yaml` configuration file and configure `character_dict_path` and `ckpt_load_path` as the current address
- Need to adjust `is_train` to `True` under configuration file `model-head` entry
- Need to adjust `dataset_root`, `data_dir`, `label_file` to the actual address of the training dataset under the configuration file `train-dataset` entry
- The `eval.py` and `rec_d28_can.yaml` files can be replaced with absolute file addresses on the command line






<a name="4"></a>
## 4. FAQ

1. CROHME data set from the source repo [CAN](https://github.com/LBH1024/CAN).

## Quote

```bibtex
@misc{https://doi.org/10.48550/arxiv.2207.11463,
  doi = {10.48550/ARXIV.2207.11463},
  url = {https://arxiv.org/abs/2207.11463},
  author = {Li, Bohan and Yuan, Ye and Liang, Dingkang and Liu, Xiao and Ji, Zhilong and Bai, Jinfeng and Liu, Wenyu and Bai, Xiang},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```


## Reference
<!--- Guideline: Citation format GB/T 7714 is suggested. -->
[1] Xiaoyu Yue, Zhanghui Kuang, Chenhao Lin, Hongbin Sun, Wayne Zhang. RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition. arXiv:2007.07542, ECCV'2020
