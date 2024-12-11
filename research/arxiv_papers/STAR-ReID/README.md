# 🌟 基于视频的跨模态行人重识别的骨架引导时空特征学习方法

该仓库为论文：“**Skeleton-Guided Spatial-Temporal Feature Learning for Video-Based Visible-Infrared Person Re-Identification**” 的代码仓库。该版本代码的框架基于 **MindSpore**。

![框架图](./Fig/framework.jpg)

🎥 **Video-based visible-infrared person re-identification (VVI-ReID)** is challenging due to significant modality feature discrepancies. Spatial-temporal information in videos is crucial, but the accuracy of spatial-temporal information is often influenced by issues like low quality and occlusions in videos. Existing methods mainly focus on reducing modality differences, but pay limited attention to improving spatial-temporal features, particularly for infrared videos.

To address this, we propose a novel **Skeleton-guided spatial-Temporal feAture leaRning (STAR)** method for VVI-ReID. By using skeleton information, which is robust to issues such as poor image quality and occlusions, STAR improves the accuracy of spatial-temporal features in videos of both modalities.

Specifically:

1. 🖼️ **Frame level**: The robust structured skeleton information refines the visual features of individual frames.
2. 🔄 **Sequence level**: A feature aggregation mechanism based on a skeleton key points graph learns the contribution of different body parts to spatial-temporal features, further enhancing global features.

📊 **Experiments on benchmark datasets** demonstrate that STAR outperforms state-of-the-art methods.

---

## ⚙️ 配置

### 示例代码

```sh
msrun --worker_num=2 --local_worker_num=2 --master_port=8118 --log_dir=msrun_log --join=True --cluster_time_out=300 code_new/model_ms.py
```

### 环境要求

- MindSpore >= 2.3.0
- Python >= 3.8
- Ascend: 2\*ascend-snt9b3|ARM: 48 核 384GB

### 安装依赖

在本代码中使用到了非官方版本 einops：

```sh
pip install git+https://github.com/lvyufeng/einops
```

---

## 📂 数据集

### 数据集下载

我们采用以下公开数据集进行实验：

- 🌌 **SYSU-MM01**: 下载链接 [SYSU-MM01](https://github.com/link-to-sysumm01)
- 🔥 **RegDB**: 下载链接 [RegDB](https://github.com/link-to-regdb)

将数据集下载后，按以下结构组织：

````
datasets/
├── SYSU-MM01/
│   ├── visible/
│   ├── infrared/
├── RegDB/
    ├── visible/
    ├── infrared/

---

## 🧪 实验运行

### 训练

```sh
python train.py --config config.yaml
````

### 测试

```sh
python test.py --model_path checkpoints/best_model.pth
```

---

## 📖 引用

如果您在研究中使用了我们的代码或方法，请引用以下论文：

```bibtex
@inproceedings{Jiang2024SkeletonGuidedSF,
  title={Skeleton-Guided Spatial-Temporal Feature Learning for Video-Based Visible-Infrared Person Re-Identification},
  author={Wenjia Jiang and Xiaoke Zhu and Jiakang Gao and Di Liao},
  year={2024},
  url={https://arxiv.org/abs/}
}
```

---

## 🙏 致谢

感谢 **MindSpore 社区** 提供的支持。更多信息请访问 [MindSpore 官网](https://www.mindspore.cn)。

---
