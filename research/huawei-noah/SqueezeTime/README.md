# SqueezeTime
This is an official mindspore implementation of our paper "**No Time to Waste: Squeeze Time into Channel for Mobile Video Understanding**". In this paper, we propose to squeeze the time axis of a video sequence into the channel dimension and present a lightweight video recognition network, term as **SqueezeTime**, for mobile video understanding. To enhance the temporal modeling capability of the model, we design a Channel-Time Learning (CTL) Block to capture temporal dynamics of the sequence. This module has two complementary branches, in which one branch is for temporal importance learning and another branch with temporal position restoring capability is to enhance inter-temporal object modeling ability. The proposed SqueezeTime is much lightweight and fast with high accuracies for mobile video understanding. Extensive experiments on various benchmarks, i.e., Kinetics400, Kinetics600, SomethingSomethingV2, HMDB51, AVA2.1, and THUMOS14, demonstrate the superiority of our model. For example, our SqueezeTime achieves **+1.2%** accuracy and **+80%** GPU throughput gain on Kinetics400 than prior methods.

<p align="center">
    <img src="figure/motivation.PNG" width="80%"/> <br />
 <em> 
    Figure 1: Pipeline of the SqueezeTime.
    </em>
</p>

##  1️⃣ Requirements
Install Package
```Shell
conda create -n SqueezeTime python=3.7.10 -y
conda activate SqueezeTime
pip install --upgrade pip   
pip install -r requirements.txt
```

## 2️⃣ Data Preparation

All dataset are organized using mmaction2 format. Please organize the `data` directory as follows after downloading all of them: 
  - <details>
    <summary> Data Structure Tree </summary>

    ```
    ├── data
        ├── kinetics400
        │   ├── kinetics400_val_list_videos.txt
        │   └── kinetics_videos/
        ├── kinetics600
        │   ├── kinetics400_val_list_videos.txt
        │   └── videos/
        ├── hmdb51
        │   ├── hmdb51_val_split_1_videos.txt
        │   ├── hmdb51_val_split_2_videos.txt
        │   ├── hmdb51_val_split_3_videos.txt
        │   └── videos/
        ├── sthv2
        │   ├── sthv2_val_list_videos.txt
        │   └── videos/
        └── ava
            ├── ava_val_v2.1.csv
            └── rawframes/
    ```
    </details>
	  
## 3️⃣ Training & Testing

Please download the mindspore checkpoint from [here](https://github.com/xinghaochen/SqueezeTime/releases/download/ckpts/SqueezeTime_K400_71.64_mindspore.ckpt).
Take the Kinectics400 dataset for an example:
- Test the SqueezeTime on K400:

`python test.py config/SqueezeTime_K400.py ckpts/SqueezeTime_K400_71.64_mindspore.ckpt `

## 4️⃣ Evaluation
The following results are on the Kinetics400 dataset. Please see the paper for the results on other datasets.
<p align="center">
    <img src="figure/k400cut.PNG" width="80%"/> <br />
 <em> 
    Figure2. Performace comparison of multiple lightweight methods on K400 dataset.
    </em>
</p>
<p align="center">
    <img src="figure/figure1.PNG" width="80%"/> <br />
 <em> 
    Figure3. Speed comparison of methods on K400 dataset. (a) CPU and GPU speed, (b) Mobile Phone CPU Latency.
    </em>
</p>


## ✏️ Reference
If you find SqueezeTime useful in your research or applications, please consider giving a star ⭐ and citing using the following BibTeX:
```
@article{zhai2024SqueezeTime,
  title={No Time to Waste: Squeeze Time into Channel for Mobile Video Understanding},
  author={Zhai, Yingjie and Li, Wenshuo and Tang, Yehui and Chen, Xinghao and Wang, Yunhe},
  journal={arXiv preprint},
  year={2024}
}
```