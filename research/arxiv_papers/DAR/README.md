# 基于情感增强因子的多因子动态资产定价研究

本项目致力于构建一个**将文本情感因子纳入多因子资产定价模型**的完整分析框架。通过整合金融新闻与社交媒体情感信息，结合 Fama–French 五因子模型，评估日度股票超额收益的变动与市场波动性之间的动态关系。

## 🔍 项目简述

* 📉 获取并处理 S\&P 500 成分股的日度对数收益率
* 📊 合并 Fama–French 因子、VIX 指数与十年期美债收益率
* 🧠 使用 **MindSpore NLP** 的 FinancialBERT 模型提取文本情绪得分
* 📈 执行带情感因子的回归分析，包含交互项和滞后变量
* 🔄 实施滚动回归以捕捉情绪因子系数的时间变动
* 📅 对 2022 年 6 月 15 日美联储加息事件进行事件研究分析
* 📦 最终输出标准化数据集，支持进一步实证研究或可视化

## 📂 数据说明与下载

请从以下ModelScope链接下载本项目所需的完整数据集：
[lastxuanshendian/DAR_data](https://www.modelscope.cn/datasets/lastxuanshendian/DAR_data/summary)

下载完成后，请将解压后的所有文件放入项目根目录下的 `data/` 文件夹中。目录结构如下：

```
project_root/
│
├── data/
│   ├── stock_returns.csv
│   ├── ff_factors.csv
│   ├── sentiment_scores.csv
│   └── ...
├── scripts/
├── main.py
└── README.md
```

所有程序依赖的数据读取路径均默认指向 `data/` 目录，无需额外修改代码即可运行。

## 🧠 情绪分析模型（MindSpore 框架加载）

我们使用基于金融领域微调的 BERT 模型来执行句子级别的情绪判断，借助 **MindSpore NLP 工具包**，完成了从分词、推理到 softmax 概率估计的一体化流程。

```python
from mindnlp.transformers import AutoTokenizer, AutoModelForSequenceClassification
```

### 示例输出：

`daily_sentiment.csv` 文件格式如下：

```
Date,S_t
20200101,0.0899
20200102,0.0759
...
```

## 📦 依赖环境

请先确保安装以下依赖：

```bash
pip install yfinance statsmodels matplotlib pandas numpy mindspore mindnlp
```

关于 MindSpore 的安装，请参考官网指南：
👉 [https://www.mindspore.cn/install](https://www.mindspore.cn/install)

## 🙏 致谢

特别感谢 **MindSpore 团队** 以及 **MindNLP 项目**的开源贡献，使我们能够将深度学习与资产定价模型无缝集成，极大地提升了研究的效率与质量。

