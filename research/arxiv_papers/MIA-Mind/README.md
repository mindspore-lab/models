# MIA-Mind: 基于CBAM注意力机制的轻量级多任务学习平台

## 项目简介
本项目实现了基于 MindSpore 框架的轻量级深度学习平台，集成了三类典型任务：
- **图像分类**（基于 CIFAR-10 数据集，模型：ResNet-50 + MIA）
- **医学图像分割**（基于 ISBI2012 数据集，模型：U-Net + MIA）
- **网络流量异常检测**（基于 MachineLearningCVE 数据集，模型：CNN + MIA）

所有模型均集成了 MIA-Mind 注意力模块，显著提升特征表达能力。

---

## 项目结构
```
mindd/
├── model/
│   ├── ResNet-50.py              # ResNet-50 + MIA-Mind
│   ├── U-Net.py                  # U-Net + MIA-Mind
│   └── cnn.py                    # CNN + MIA-Mind
├── data/
│   ├── cifar-10-batches-bin      # ResNet-50 + MIA-Mind
│   ├── isbi2012                  # U-Net + MIA-Mind
│   └── MachineLearningCVE        # CNN + MIA-Mind
├── MIA-Mind.py                   # MIA-Mind 注意力模块
├── data.py                       # 数据集加载与预处理
├── metrics.py                    # 统一评价指标
├── train.py                      # 通用训练脚本
├── README.md                     # 项目说明
├── requirements.txt              # 依赖包清单
```

---

## 环境要求
- Python >= 3.9
- MindSpore >= 2.5.0
- numpy
- pandas
- scikit-learn
- tifffile
- tqdm

安装依赖：
```bash
pip install -r requirements.txt
```

---

## 快速开始

### 训练 CIFAR-10 分类模型
```bash
python train.py --task cifar10 --epochs 20 --batch_size 32 --lr 0.001
```

### 训练 ISBI2012 分割模型
```bash
python train.py --task isbi2012 --epochs 30 --batch_size 4
```

### 训练 MachineLearningCVE 流量检测模型
```bash
python train.py --task cic --epochs 15 --batch_size 64
```

---

## 说明
- MIA-Mind模块已封装在 `MIA-Mind.py` 中，所有模型统一调用。
- 不同任务使用了适配的损失函数和评价指标，详见 `metrics.py`。
- 数据集路径可在 `data.py` 中指定。

---

## TODO
- 支持更多注意力机制（如 SE、ECA）
- 增加推理部署（MindIR导出）
- 支持更多数据集与任务

---

## 致谢
- MindSpore 官方团队
- 公开数据集提供者（CIFAR、ISBI2012、CIC）

🚀 Enjoy learning with MIA-Mind!
