# 基于情感增强因子的多因子动态资产定价研究

本项目致力于构建一个**将文本情感因子纳入多因子资产定价模型**的完整分析框架。通过整合金融新闻与社交媒体情感信息，结合 Fama–French 五因子模型，评估日度股票超额收益的变动与市场波动性之间的动态关系。

## 🔍 项目简述

- 📉 获取并处理 S&P 500 成分股的日度对数收益率
- 📊 合并 Fama–French 因子、VIX 指数与十年期美债收益率
- 🧠 使用 **MindSpore NLP** 的 FinancialBERT 模型提取文本情绪得分
- 📈 执行带情感因子的回归分析，包含交互项和滞后变量
- 🔄 实施滚动回归以捕捉情绪因子系数的时间变动
- 📅 对 2022 年 6 月 15 日美联储加息事件进行事件研究分析
- 📦 最终输出标准化数据集，支持进一步实证研究或可视化


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

建议先安装以下库：

```bash
pip install yfinance statsmodels matplotlib pandas numpy mindspore mindnlp
```

MindSpore 安装指南请参考官网：
👉 [https://www.mindspore.cn/install](https://www.mindspore.cn/install)

## 🙏 致谢

特别感谢 **MindSpore 团队** 以及 **MindNLP 项目**的开源贡献，使我们能够将深度学习与资产定价模型无缝集成，极大地提升了研究的效率与质量。
