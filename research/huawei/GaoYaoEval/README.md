# GaoYaoEval - 皋陶多语言大模型评测框架

> 🌐 一站式多语言、多文化、多题型大模型能力评测框架，支持客观题、主观题、翻译题等丰富评测场景  
>   
> 多语言评测技术报告：[./GaoYao_Multilingual_Benchmark_Technical_Report.pdf](./GaoYao_Multilingual_Benchmark_Technical_Report.pdf)
>
> 数据集开源仓库：https://github.com/zhaocorey/GaoYaoEvalDataset.git

---

![](./assets/cover_iamge.png)

## 项目简介
GaoYao是一个系统化的多语言多文化评测基准，构建了涵盖通用多语言能力、跨文化能力和单一文化能力的三维度九子项评估框架。该基准将指令遵循与多轮对话测试集通过语言专家人工精译扩展至19种语言，语言覆盖较现有工作提升111%。针对文化能力，采用专家参与的数据合成方法覆盖34种文化，文化代表性提升88%。评测整合高质量人工校验数据，避免纯机器翻译的质量缺陷。最终对20余个主流开源与商业大模型开展分层评估，为多语言多文化能力提供全面可靠的衡量标准。

---

## 📁 代码架构

```
GaoYaoEval/
├── data/                          # 数据层
│   ├── original/                  # 原始评测数据集
│   │   └── belebele/{language}.jsonl
│   ├── inference_result/          # 模型推理结果
│   │   └── Qwen3-32B/belebele/infer.jsonl
│   └── evaluation_result/         # 评测分析结果
│       └── Qwen3-32B/belebele/
│           ├── eval_report.jsonl  # 指标报告
│           ├── eval.jsonl         # 评测详情
│           ├── bad_cases.jsonl    # 异常用例
│           └── not_pass.jsonl     # 未通过用例
│
├── src/                           # 核心代码
│   ├── evaluation/                # 评测引擎
│   │   ├── base_eval.py           # 评测基类
│   │   └── {dataset}_eval.py      # 各数据集评测实现
│   ├── tools/                     # 通用工具方法
│   │   ├── file_operation.py
│   │   ├── judger_algorithm.py
│   │   ├── llm_request.py
│   │   ├── metrics_and_report_operator.py
│   │   ├── prompt_templates/
│   │   └── text_processing.py
│   ├── inference/                 # 推理模块（开发中）
│   └── pipeline/                  # 评测流水线（开发中）
├── requirements.txt               # 依赖清单
├── README.md                      # 本文件
└── LICENSE                        # 许可证
```

---

## 📊 数据规范

### 📋 推理结果字段 (`inference_result/*.jsonl`)

| 编号 | 字段名    | 定义                 | 必选 | 示例值          |
|------|-----------|----------------------|------|-----------------|
| 1    | `uuid`    | 唯一标识             | ✅   | `"uuid-001"`    |
| 2    | `prompt`  | 问题（提示词）       | ✅   | `"法国首都是?"` |
| 3    | `response`| 模型推理结果         | ✅   | `"巴黎"`        |
| 4    | `gt`      | 参考答案             | ✅   | `"Paris"`       |
| 5    | `language`| 语种                 | ❌   | `"fr"`          |
| 6    | `country` | 地区                 | ❌   | `"France"`      |

### 📈 评测结果扩展字段 (`evaluation_result/*.jsonl`)

| 编号 | 字段名         | 定义                                      | 适用题型   | 必选 |
|------|----------------|-------------------------------------------|------------|------|
| +1   | `judge_score`  | 评测得分                                  | 客观题     | 客观题✅   |
| +2   | `prediction`   | 评测抽取后的标准化结果                    | 客观题     | 客观题✅   |
| +3   | `winner`       | 胜率结果 (`win`/`lose`/`tie`)            | 主观题     | 主观题 ✅   |
| +4   | `bad_case`     | 异常响应原始记录                          | 全类型     | ❌   |
| +5   | `not_pass`     | 未通过用例原始响应                        | 全类型     | ❌   |

> 💡 评测结果自动继承推理结果全部字段，并追加上述扩展字段

---

## ⚙️ 快速开始

### 环境准备
```bash
# 克隆仓库
git https://github.com/mindspore-lab/models
cd research/huawei/GaoYaoEval

# 安装依赖
pip install -r requirements.txt
```

### 执行评测
```shell
# 启动评测（示例：belebele 数据集）
python belebele_eval.py
```

### 自定义评测
```python
# 1. 继承基类实现新数据集评测
from src.evaluation.base_eval import BaseEval

class MyDatasetEval(BaseEval):
    def evaluate(self, sample: dict) -> dict:
        # 实现单条用例评测逻辑
        pass
```

---

## 📋 评测集说明

| 编号 | 评测集          | 题型            | 说明                   |       评测维度(层级)        |
|:----:|:---------------:|:---------------:|:-----------------------:|:---------------------:|
| 1    | `belebele`      | 客观题          | 多语言阅读理解          | Reading Comprehension |
| 2    | `mgsm`          | 客观题          | 多语言数学推理          |         Math          |
| 3    | `mmmlu`         | 客观题          | 多学科知识              |       Reasoning       |
| 4    | `superblend`    | 客观题          | 混合领域综合能力        |     Cross-Culture     |
| 5    | `include`       | 客观题          | 文化包容性评测          |       Knowledge       |
| 6    | `culture_scope` | 客观题+主观题   | 单文化场景深度评测      |     Mono-Culture      |
| 7    | `sage`          | 客观题+主观题   | 跨文化理解与适应能力    |     Cross-Culture     |
| 8    | `s_alpaca_eval`   | 主观题          | 指令遵循能力            |   Instructi Follow    |
| 9    | `s_mt_bench`      | 主观题          | 多轮对话质量            |       Dialogue        |
| 10   | `flores`        | 翻译题          | 高质量机器翻译          |      Translation      |

---

## 💡 核心设计

### 扩展性设计
- **插件式评测器**：通过继承 `BaseEval` 快速接入新数据集  
- **统一数据契约**：标准化输入/输出字段，降低集成成本  
