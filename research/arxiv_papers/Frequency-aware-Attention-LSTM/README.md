# 基于频率感知的注意力机制-LSTM模型在PM2.5时间序列预测中的应用
 
论文“FREQUENCY-AWARE ATTENTION-LSTM FOR PM2.5 TIME SERIES FORECASTING”源码
这是我们的论文：

## 项目结构

- **Frequency-aware-Attention-LSTM/**：包含不同模型及配置的Python脚本。
  - `main.py`：项目启动文件。
  - `models.py`：模型文件。
  - `pm25_preprocessor.py`：数据处理文件。
  - `trainer.py`：模型训练文件。
  - `utils.py`：日志处理文件。

## 安装步骤

1. **克隆仓库**  
   下载项目文件或克隆此仓库。

   ```bash
   git clone https://github.com/ikun-szh666/Frequency-aware-Attention-LSTM.git
   cd model
   ```

2. **安装依赖**  
   使用 `pip` 安装 `requirements.txt` 中列出的依赖包。

   ```bash
   pip install -r requirements.txt
   ```

## 使用说明

要运行项目，使用以下命令：

```bash
python main.py
```

## 感谢MindSpore社区提供的支持
