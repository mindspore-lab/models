# GaoYaoEval - 皋陶多语言大模型评测框架

> 🌐 一站式多语言、多文化、多题型大模型能力评测框架，支持客观题、主观题、翻译题等丰富评测场景  
>   
> 多语言评测技术报告：`./GaoYao_Multilingual_Benchmark_Technical_Report.pdf`
>
> 数据集开源仓库：https://github.com/zhaocorey/GaoYaoEvalDataset.git
---

## 📁 项目架构

```
GaoYaoEval/
├── data/                          # 数据层
│   ├── original/                  # 原始评测数据集
│   │   └── m3exam/{language}.jsonl
│   ├── inference_result/          # 模型推理结果
│   │   └── Qwen3-32B/m3exam/infer.jsonl
│   └── evaluation_result/         # 评测分析结果
│       └── Qwen3-32B/m3exam/
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
# 启动评测（示例：m3exam 数据集）
python m3exam_eval.py
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

| 编号 | 评测集          | 题型               | 说明                     |
|------|-----------------|------------------|------------------------|
| 1    | `m3exam`        | 客观题             | 覆盖100+语言基础能力     |
| 2    | `belebele`      | 客观题             | 多语言阅读理解           |
| 3    | `mgsm`          | 客观题             | 多语言数学推理           |
| 4    | `mmmlu`         | 客观题             | 多学科知识               |
| 5    | `superblend`    | 客观题             | 混合领域综合能力         |
| 6    | `include`       | 客观题             | 文化包容性评测           |
| 7    | `mono_culture`  | 客观题+主观题       | 单文化场景深度评测       |
| 8    | `cross_culture` | 客观题+主观题       | 跨文化理解与适应能力     |
| 9    | `alpaca_eval`   | 主观题             | 指令遵循能力             |
| 10   | `mt_bench`      | 主观题             | 多轮对话质量             |
| 11   | `flores`        | 翻译题             | 高质量机器翻译           |

---

## 💡 核心设计

### 扩展性设计
- **插件式评测器**：通过继承 `BaseEval` 快速接入新数据集  
- **统一数据契约**：标准化输入/输出字段，降低集成成本  
