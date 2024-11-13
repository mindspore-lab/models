
# CGL-MHA：集成GCN、LSTM和多头注意力的讽刺检测模型

论文“An Innovative CGL-MHA Model for Sarcasm Sentiment Recognition Using the MindSpore Framework”源码
这是我们的论文：https://arxiv.org/abs/2411.01264

## 项目结构

- **headlines/**：包含不同模型及配置的Python脚本。
  - `CNN.py`：卷积神经网络模型。
  - `GRU.py`：门控循环单元模型。
  - `LSTM.py`：长短期记忆网络模型。
  - `LSTM_GRU_CNN_MulA_Per.py`：使用LSTM、GRU、CNN和多头注意力的混合模型。
  - `LSTM_GRU_MulA.py`：使用LSTM、GRU和多头注意力的组合模型。
  - `SVM.py`：支持向量机模型，用于对比实验。

- **requirements.txt**：列出运行代码所需的Python依赖库。

## 安装步骤

1. **克隆仓库**  
   下载项目文件或克隆此仓库。

   ```bash
   git clone https://github.com/bitbitlemon/CGL-MHA.git
   cd CGL-MHA-main
   ```

2. **安装依赖**  
   使用 `pip` 安装 `requirements.txt` 中列出的依赖包。

   ```bash
   pip install -r requirements.txt
   ```

## 使用说明

根据实验需求，在 `headlines` 目录中运行相应的模型脚本。例如，要运行LSTM-GRU-多头注意力模型，使用以下命令：

```bash
python headlines/LSTM_GRU_MulA.py
```

## 模型描述

- **GCN**：图卷积网络帮助建模图结构中节点之间的依赖关系。
- **LSTM**：LSTM网络用于捕捉序列数据中的时间依赖性。
- **多头注意力**：注意力机制允许模型关注输入序列的不同部分，提升了可解释性。

每个模型脚本都提供了不同的方法来解决谎言检测问题，以便进行灵活的实验。

## 感谢MindSpore社区提供的支持


