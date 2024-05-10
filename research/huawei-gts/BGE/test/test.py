from datasets import load_dataset
from C_MTEB.tasks import *
from flag_dres_model import FlagDRESModel
from mteb import MTEB

# 创建数据集
dataset = load_dataset("./path/to/data", split="dev")
ds = dataset.train_test_split(test_size=0.1, seed=1)

model_name_or_path = "/path/to/model"
model = FlagDRESModel(model_name_or_path,
                      query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章",
                      pooling_method="cls")

# 自定义评估类
class CustomReranking(AbsTaskReranking):
    @property
    def description(self):
        return {
            "name": "CustomReranking",
            "hf_hub_name": "/path/to/data",
            "description": "T2Reranking:",
            "reference": "",
            "type": "Reranking",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "map"
        }

    def evaluate(self, model, split="test", **kwargs):
        self.dataset = ds
        data_split = self.dataset[split]
        evaluator = ChineseRerankingEvaluator(data_split, **kwargs)
        scores = evaluator(model)

        return dict(scores)

# 评估
evaluation = MTEB(task=[CustomReranking()], task_langs=['zh', 'zh-CN'])
evaluation.run(model, output_folder=f"zh_result/{model_name_or_path.split('/')[-1]}")
