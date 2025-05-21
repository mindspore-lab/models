# 🧠 Aspect-Based Sentiment Analysis with TextCNN (MindSpore)

本项目基于 **MindSpore 框架**，实现了一个轻量级的 **情感极性分类系统**，支持多个典型情感分析数据集的加载、处理、训练与评估，并采用 **TextCNN** 架构进行建模。

## ✅ 特性 Highlights

- 📦 **支持数据集**：
  - SemEval-2014 Task 4 (Laptops)
  - MAMS-ATSA
  - SST-2（自动兼容 `.parquet` 转 JSON）

- 🔨 **统一数据预处理模块**：
  - 多格式支持（CSV / XML / JSONL / Parquet）
  - 处理结果统一转为 JSON 格式

- ⚙️ **模型架构**：
  - 轻量化 TextCNN
  - 支持 Dropout / Weight Decay 正则化
  - 可快速在 CPU 上训练

- 📊 **完整评估指标输出**：
  - 准确率 / 精确率 / 召回率 / F1
  - 每个数据集独立日志 & 最终汇总为 CSV

## 📁 项目结构

```
├── data/                       # 数据预处理（多格式 → JSON）
│   ├── MAMS                    # MAMS数据集
│   ├── SemEval_2014_Task_4     # SemEval数据集
│   └── sst-2                   # sst-2数据集
├── processed/                  # 存储处理好的 JSON 格式数据集
│   ├── mams                    # MAMS数据集
│   ├── semeval                 # SemEval数据集
│   └── sst2                    # sst-2数据集
├── train.py                    # TextCNN 模型训练入口
├── dataset.py                  # 数据加载器 + 词表构建
└── README.md
```

## 🚀 使用说明

### ✅ 1. 数据预处理

请将原始数据放入指定目录，然后运行：

```bash
python dataset.py
```

将自动处理以下数据集并转换为 JSON 格式：

- `data/SemEval_2014_Task_4/`
- `data/MAMS/MAMS-ATSA/raw/`
- `data/SST2/`

### ✅ 2. 模型训练

运行主训练脚本：

```bash
python train.py
```

将依次训练以下数据集：

- `semeval`
- `mams`
- `sst2`

每个数据集：

- 自动构建词表；
- 划分训练/验证集（比例 8:1:1）；
- 输出日志至 `log_*.txt`；
- 将评估指标汇总至 `results_summary.csv`。

## 📊 示例评估结果（`results_summary.csv`）

| 数据集  | Acc  | Prec | Recall | F1   |
| ------- | ---- | ---- | ------ | ---- |
| semeval | 0.71 | 0.69 | 0.68   | 0.69 |
| mams    | 0.61 | 0.59 | 0.60   | 0.59 |
| sst2    | 0.86 | 0.85 | 0.86   | 0.85 |

## 🛠️ 环境依赖

- Python ≥ 3.9
- MindSpore ≥ 2.5.0 (CPU)
- numpy, pandas, scikit-learn, tqdm
- pyarrow（用于读取 `.parquet`）

安装依赖：

```bash
pip install -r requirements.txt
```

## 📌 注意事项

- 如果不使用 `sst2` 数据集，可在 `train.py` 中移除对应行；
- SST-2 默认使用 `train.json` 划分训练/验证集

## 📚 参考文献

- Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification*.
- Socher, R. et al. (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank*.