# 项目名称：基于MACD指标的交易策略分析与回测

## 代码功能解释
raw_draw.py 为MACD买入卖出图
little_wave.py 为小波分析结果
divergence.py 为背离分析结果
end_1.py 为小波分析和背离分析
end2_mindspore.py 为遗传算法测定买入点
end_2_draw.py 为遗传算法测定买入点最终结果的画图程序
trading_strategies.xlsx 为小波分析和背离分析的运行结果
x1.xlsx 为原始策略交易量化指标
x2.xlsx 为小波分析后交易量化指标
x3.xlsx 为背离分析后交易量化指标

## 项目简介
本项目旨在通过分析MACD指标（移动平均收敛散度）来构建一个简单的交易策略，并使用历史数据进行回测。项目结合了技术分析中的MACD指标、小波变换（Wavelet Transform）以及MindSpore框架，对股票价格数据进行分析，生成买卖信号，并计算策略的表现指标。

## 主要功能
1. **数据加载与预处理**：从Excel文件中加载股票价格数据，并进行简单的预处理。
2. **MACD指标计算**：计算短期和长期的指数移动平均线（EMA），并生成MACD指标（DIF、DEA、MACD柱状图）。
3. **小波变换**：对DIF信号进行小波变换，提取近似信号以平滑数据。
4. **买卖信号生成**：基于MACD指标生成买卖信号，并结合局部极值点检测技术识别牛市和熊市背离信号。
5. **策略回测**：根据生成的买卖信号进行回测，计算策略的收益率、胜率、夏普比率等关键指标。
6. **可视化**：绘制股票价格、MACD指标、买卖信号以及背离信号的图表。

## 依赖库
- `numpy`
- `pandas`
- `matplotlib`
- `pywt`
- `mindspore`

**安装依赖库**：
确保已安装所需的Python库。可以通过以下命令安装：
```bash
pip install -r requirements.txt
```

## 文件结构
- `data/510300.SH.xlsx`：包含股票价格数据的Excel文件。
- `end2 mindspore.py`：主程序文件，包含数据加载、指标计算、信号生成、回测和可视化代码。

## 使用方法
1. **准备数据**：
   将股票价格数据保存为Excel文件，并放置在`data/`目录下。文件应包含至少以下列：
   - `日期`：日期列。
   - `收盘价(元)`：股票收盘价。

2. **运行代码**：
   运行`end2_mindspore.py`文件，程序将自动加载数据、计算指标、生成买卖信号并进行回测。最终结果将显示在图表中，并打印出策略的表现指标。

   ```bash
   python end2_mindspore.py
   ```

3. **查看结果**：
   - 图表将展示股票价格、MACD指标、买卖信号以及背离信号。
   - 控制台将输出策略的表现指标，包括胜率、总收益、年化收益、夏普比率等。

## 关键代码说明
- **MACD指标计算**：
  ```python
  ema_short = data['收盘价(元)'].ewm(span=9, adjust=False).mean()
  ema_long = data['收盘价(元)'].ewm(span=22, adjust=False).mean()
  dif = ema_short - ema_long
  dea = dif.ewm(span=25, adjust=False).mean()
  macd_histogram = dif - dea
  ```

- **小波变换**：
  ```python
  coeffs = pywt.wavedec(dif.asnumpy(), 'coif5', level=4)
  approximation = coeffs[0]
  reconstructed_signal = pywt.waverec([approximation] + [np.zeros_like(coeff) for coeff in coeffs[1:]], 'coif5')
  ```

- **买卖信号生成**：
  ```python
  buy_signals = (data['DIF'] > data['MACD']) & (data['DIF'].shift(1) <= data['MACD'].shift(1))
  sell_signals = (data['DIF'] < data['MACD']) & (data['DIF'].shift(1) >= data['MACD'].shift(1))
  ```

- **策略回测**：
  ```python
  for i in range(len(data) - 1):
      if (buy_signals.iloc[i] or data['bear_divergence'][i]) and capital > 0:
          # 买入逻辑
      elif (sell_signals.iloc[i] or data['bull_divergence'][i]) and stocks_held > 0:
          # 卖出逻辑
  ```

## 输出结果
- **图表**：
  - 股票价格与买卖信号图。
  - MACD指标图，包含DIF、DEA以及MACD柱状图。
- **策略表现指标**：
  - 胜率（Win Rate）
  - 盈亏比（Odds Ratio）
  - 交易频率（Trade Frequency）
  - 总收益（Total Return）
  - 年化收益（Annual Return）
  - 夏普比率（Sharpe Ratio）
  - 最大回撤（Max Drawdown）

## 注意事项
- 本项目使用的数据为历史数据，回测结果仅供参考，实际交易中可能存在滑点、手续费等因素，需谨慎使用。
- 代码中的参数（如EMA的周期、小波变换的类型等）可以根据实际需求进行调整。

## 未来改进
- 增加更多的技术指标（如RSI、布林带等）来优化策略。
- 引入机器学习模型来预测买卖信号。
- 考虑交易成本、滑点等实际交易中的因素。

## 许可证
本项目采用MIT许可证。详情请参阅LICENSE文件。

## **MindSpore 的核心功能**
在本项目中，MindSpore 主要用于以下两个部分：

### 2.1 **数据转换为 MindSpore Tensor**
MindSpore 的核心数据结构是 `Tensor`，它类似于 NumPy 的 `ndarray`，但支持更高效的数值计算和自动微分。项目中通过 `convert_to_mindspore_tensor` 函数将 Pandas DataFrame 中的数据转换为 MindSpore Tensor。

```python
import mindspore as ms

def convert_to_mindspore_tensor(data):
    return ms.Tensor(data.values)
```

- **输入**：Pandas Series 或 DataFrame 中的数据。
- **输出**：MindSpore Tensor 格式的数据。
- **作用**：将 MACD 指标（DIF、DEA、MACD 柱状图）转换为 MindSpore Tensor，以便后续计算。

### 2.2 **使用 MindSpore 进行数值计算**
在项目中，MindSpore 的 `mnp` 模块（MindSpore NumPy）被用于一些数值计算任务。例如，计算 MACD 柱状图中的红色和蓝色部分：

```python
import mindspore.numpy as mnp

red_bar = mnp.where(macd_histogram > 0, macd_histogram, 0)
blue_bar = mnp.where(macd_histogram < 0, macd_histogram, 0)
```

- **`mnp.where`**：类似于 NumPy 的 `np.where`，用于条件筛选。
- **作用**：将 MACD 柱状图分为红色（正值）和蓝色（负值）两部分，便于后续可视化。

---

## **MindSpore 的优势**
在本项目中使用 MindSpore 的主要优势包括：

1. **高效计算**：
   - MindSpore Tensor 支持高效的数值计算，尤其是在大规模数据集上表现优异。
   - 与 NumPy 相比，MindSpore 的计算速度更快，尤其是在 GPU 或 Ascend 硬件上。

2. **自动微分**：
   - 虽然本项目未涉及深度学习模型，但 MindSpore 的自动微分功能为未来扩展（如引入神经网络模型）提供了便利。

3. **跨平台支持**：
   - MindSpore 支持 CPU、GPU 和 Ascend 等多种硬件平台，便于在不同环境中运行代码。

---

## **MindSpore 的扩展性**
如果需要进一步扩展本项目，可以利用 MindSpore 的以下功能：

1. **引入神经网络模型**：
   - 使用 MindSpore 构建深度学习模型，预测股票价格或生成买卖信号。
   - 示例代码：
     ```python
     import mindspore.nn as nn

     class StockPredictionModel(nn.Cell):
         def __init__(self):
             super(StockPredictionModel, self).__init__()
             self.fc1 = nn.Dense(10, 64)
             self.fc2 = nn.Dense(64, 1)

         def construct(self, x):
             x = self.fc1(x)
             x = self.fc2(x)
             return x
     ```

2. **优化计算性能**：
   - 使用 MindSpore 的图模式（Graph Mode）加速计算。
   - 示例代码：
     ```python
     ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
     ```

3. **分布式训练**：
   - 如果需要处理更大规模的数据，可以使用 MindSpore 的分布式训练功能。

---

## **注意事项**
- **硬件支持**：确保运行环境支持 MindSpore（如安装正确版本的 CUDA 或 Ascend 驱动）。
- **数据格式**：MindSpore Tensor 与 NumPy 数组可以互相转换，但需要注意数据类型的兼容性。
- **性能调优**：对于大规模数据，建议使用 MindSpore 的图模式或 GPU 加速。

---

## **参考文档**
- [MindSpore 官方文档](https://www.mindspore.cn/docs)
- [MindSpore GitHub 仓库](https://github.com/mindspore-ai/mindspore)