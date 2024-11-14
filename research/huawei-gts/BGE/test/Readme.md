# 下游任务训练和评测



BGE模型可以应用与检索、分类、聚类等下游任务，通常训练后的模型需要结合下游任务评测效果。

本文我们使用重排数据集T2Reranking，实验BGE模型的微调和评测



## 数据集

T2Reranking下载地址：www.huggingface.co/datasets/C-MTEB/T2Reranking

Reranking数据集格式为：

```
{
	"query": "xxxx",
	"positive": ["text", "text"],
	"negative": ["text", "text"]
}
```

需要先将数据集格式调整为BGE微调需要的格式，并保存切分好的训练数据train.jsonl：

```
from datasets import load_dataset

# 创建数据集
dataset = load_dataset("./path/to/data", split="dev")
ds = dataset.train_test_split(test_size=0.1, seed=1) # 固定生成数据

# 转化数据集并保存
ds_train = ds["train"].rename_column("positive", "pos").rename_column("negative", "neg").filter(lambda x: len(x["pos"]) > 0 and len(x["neg"]) > 0)
ds_train.to_json("train.jsonl", force_ascii=False)
```



## 微调

微调使用bge-small-zh-v1.5作为基模型，查看模型微调效果，执行命令（微调详细介绍见主页readme）：

```
python run.py \
--model_name_or_path ./bge-small-zh-v1.5 \
--train_data ./train.jsonl \
--epoch 5 \
--batch_size 12 \
--train_group_size 2 \
--query_max_len 512 \
--passage_max_len 512 \
--save_steps 5000 \
--output_dir ./checkpoint \
--lr 3e-5 \
--query_instruction_for_retrieval "为这个句子生成表示以用于检索相关文章" \
--temperature 0.02
```



## 评测

微调后可以评测模型在下游任务中的效果，评测依赖FlagEmbedding中的C_MTEB和MTEB两个评估套件，需要在GPU上运行。所以需要将微调后的ckpt转化为bin文件，然后再去GPU上进行评测。

1. 环境依赖：

   mteb[beir]==1.1.1（通过pip安装）

   [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) (在GPU上下载原仓代码)

   

2. 模型转化：将ckpt模型转化为bin文件，然后将转化后的bin文件替换原先baai-small-zh-1.5中的bin文件，这样就生成了一份torch可以使用的模型文件

   ```
   from mindspore import load_checkpoint
   import torch
   ckpt = load_checkpoint("./path/to/ckpt")
   
   torch_bin = {}
   for key, value in ckpt.items():
       if "adam_m" not in key and "adam_v" not in key:
           try:
               torch_bin[key.replace("model.", "")] = torch.Tensor(value.numpy())
           except ValueError:
               print(key)
   
   torch.save(torch_bin, "./path/to/pytorch_model.bin")
   ```

   

3. 评测

   在GPU环境上（该环境已经安装FlagEmbedding代码仓相关依赖）,进入都C_MTEB评估模块文件夹下

   ```
   cd C_MTEB
   ```

   执行以下脚本进行评测：

   ```
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
   ```

   评估结果可以在zh_result下找到，类似如下结果

   ```
   {
   	"dataset_revision": null,
   	"mteb_dataset_name": "CustomReranking",
   	"mteb_version": "1.1.1",
   	"test":{
   		"map": 0.6950280580025318,
   		"mrr": 0.8031592637054822
   	}
   }
   ```

   原始模型评估结果如下：

   ```
   {
   	"dataset_revision": null,
   	"mteb_dataset_name": "CustomReranking",
   	"mteb_version": "1.1.1",
   	"test":{
   		"map": 0.6674649850614085,
   		"mrr": 0.7556635987728424
   	}
   }
   ```

   

   可以看出经过微调，模型map有约3%提升，mrr约有5%提升。