# LSGTIME：集成GCN、LSTM和多头稀疏注意力的犯罪时空预测模型
 
论文“Innovative LSGTime Model for Crime Spatiotemporal Prediction Based on MindSpore Framework”源码
这是我们的论文：https://arxiv.org/abs/2503.20136

## 项目结构

- **model/**：包含不同模型及配置的Python脚本。
  - `CNN.py`：卷积神经网络模型。
  - `GRU.py`：门控循环单元模型。
  - `LSTM.py`：长短期记忆网络模型。
  - `LSTM_GRU_SparseMultiHeadAttention.py`：使用LSTM、GRU、CNN和多头稀疏注意力的混合模型。
  - `LSTM_GRU.py`：使用LSTM、GRU的组合模型。
  - `RNN.py`：循环神经模型，用于对比实验。

## 安装步骤

1. **克隆仓库**  
   下载项目文件或克隆此仓库。

   ```bash
   git clone https://github.com/weibaozhong/LGSTime.git
   cd model
   ```

2. **安装依赖**  
   使用 `pip` 安装 `requirements.txt` 中列出的依赖包。

   ```bash
   pip install -r requirements.txt
   ```

## 使用说明

根据实验需求，在 `model` 目录中运行相应的模型脚本。例如，要运行LSTM-GRU-多头稀疏注意力模型，使用以下命令：

```bash
python model/LSTM_GRU_SparseMultiHeadAttention.py
```

## 感谢MindSpore社区提供的支持