# FlagEmbedding-GTS



## BAAI-general-embedding (BGE)概述

语义向量模型（Embedding Model）被广泛应用于搜索、推荐、数据挖掘等重要领域，将自然形式的数据样本（如语言、代码、图片、音视频）转化为向量（即连续的数字序列），并用向量间的“距离”衡量数据样本之间的“相关性” 。

BAAI-general-embedding (**BGE**) 是智源研究院发布的一款开源的中英文语义**向量模型**。在2023/08/02发布后，在中英文榜单均获得最好性能。

本仓库将原始Pytorch实现，迁移到**Mindspore框架**，提供模型推理和微调能力。



**参考论文**：C-Pack: Packaged Resources To Advance General Chinese Embedding

  

## 环境要求

本工程测试验证配套关系

| 环境 | 驱动&固件            | CANN    | mindspore    | mindspore_lite |
| ---- | -------------------- | ------- | ------------ | -------------- |
| 910B | 23.0.RC3/6.4.0.4.220 | 7.0.RC1 | 2.2.11       | 2.2.11         |
| 310P | 23.0.0/7.1.0.3.220   | 7.0.0   | 2.2.0/2.2.11 | 2.2.0/2.2.11   |

本工程运行有部分依赖：

```
mindnlp>=0.2.4
mindspore>=2.2.11
datasets==2.18.0
numpy
faiss-cpu
...
```

可以直接将requiremens.txt下的依赖安装在环境中

  

## 预训练模型

本工程分析验证了以下BGE中文向量模型，可以通过HuggingFace超链接下载对应的权重文件。

| **Model**                                                    | **Embedding dimension** | **language** |
| ------------------------------------------------------------ | ----------------------- | ------------ |
| [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) | 1024                    | Chinese      |
| [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) | 768                     | Chinese      |
| [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) | 512                     | Chinese      |

  

## **快速入门：**

  

### **推理：**

#### 	**在线推理：**

- 推理脚本可参考如下样例，详细见inference.py

```
# import mindspore as ms
# ms.set_context(mode=0)
# from mindnlp.transformers import MSBertModel
from mindspore.ops import L2Normalize
from mindnlp.transformers import AutoTokenizer, AutoModel


# pytorch 原始模型权重文件
model_path = './bge-large-zh-v1.5'

# 样例数据
sentences_1 = ["样例数据-1", "样例数据-2"]

# 加载模型
model = AutoModel.from_pretrained(model_path)
# model = MSBertModel.from_pretrained(model_path)

# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 推理
encoded_input = tokenizer(sentences_1, padding=True, truncation=True, return_tensors="ms")
# pad to max sequence length 512
# encoded_input = tokenizer(sentences_1, padding="max_length", truncation=True, return_tensors="ms", max_length=512)
model_output = model(**encoded_input)
sentence_embeddings = model_output[0][:, 0]
l2_normalize = L2Normalize(axis=1, epsilon=1e-12)
final_outputs_1 = l2_normalize(sentence_embeddings)

# 计算相似度
similarity = final_outputs_1 @ final_outputs_1.T
print(similarity)

```

使用以上脚本推理后，显示以下相似度矩阵，句子相似度为：

```
[[0.9999996, 0.87915164]
 [0.87915164, 1.       ]]
```

**注意：**

- 若需要加载mindspore训练好的**ckpt文件**，可以参考gen_mindir_from_ckpt.py中的写法。
- 若需要使用**静态图**推理，可以参考inference.py中注释的代码，设置mode=0并通过MSBertModel加载模型可以启动静态图推理模式（静态图因需要编译，首次推理较慢，但后续推理时间比动态图有大幅提升。）

  

   

#### 	**lite推理：**

**按顺序执行如下操作**

1. 在训练服务(如910B)上导出ckpt格式的权重，参考gen_ckpt.py

```
python gen_ckpt.py --model-path ./model
```

参数解释:
/--model-path：原仓模型的路径，如./bge-large-zh-v1.5

2. 在训练服务器(如910B)上将ckpt转化为mindir格式的文件，参考gen_mindir_from_ckpt.py

```
python gen_mindir_from_ckpt.py --model-path=./model/model.ckpt --config-path=./bge-large-zh-v1.5
```

参数解释:
/--model-path：ckpt模型路径

/--config-path:   模型配置文件，tokenizer配置文件与词表路径，默认为bge-large-zh-v1.5

3. 在推理服务器(如310P)上使用生成好的mindir文件进行推理，参考bge-cloud-lite.py

```
python bge-cloud-lite.py --model-path=./model/model.mindir --config-path=./bge-large-zh-v1.5
```

参数解释:
/--model-path：mindir模型路径

/--config-path:   模型配置文件，tokenizer配置文件与词表路径，默认为bge-large-zh-v1.5

**注意**：

- 如果要使用自己训练好的ckpt模型进行推理，可以在训练侧使用callback方式导出，或者参考gen_ckpt.py脚本的导出方式。

- 只有步骤2脚本依赖bge_injection模块。
- 步骤2生成mindir路径默认为./model 文件夹。
- 生成mindir文件过大时，可能会生成model_graph.mindir与data文件夹，推理时应将model_graph.mindir作为输入模型，且保证data文件夹在模型同一目录下。






### **微调：**

**数据集**

训练数据集为json格式，每一行为字典格式，如下所示：

```
{"query": str, "pos": List[str], "neg":List[str]}
```

query是查询语句，pos为与查询语句相关的文本，neg是一个列表的负相关语句。样例数据可以查看example/finetune/toy_finetune_data.jsonl



**单卡微调启动命令**

```
python run.py \
--model_name_or_path ./bge-large-zh-v1.5 \
--train_data ./toy_finetune_data.jsonl \
--epoch 2 \
--batch_size 2 \
--train_group_size 2 \
--query_max_len 256 \
--passage_max_len 256 \
--save_steps 1000 \
--output_dir ./checkpoint \
--lr 1e-5 \
--query_instruction_for_retrieval "" \
--temperature 0.02
# 注意：mindspore静态图训练，编译时间较长，请耐心等待
```

参数介绍：

--model_name_or_path: 预训练模型地址

--train_data: 训练数据集地址

--epoch: 迭代轮次

--batch_size: 每次训练样本数量

--train_group_size: 设置passage的数据大小，正样本和负样本总数量

--query_max_len: query的最大长度（为解决静态图中的动态shape问题，query_max_len需要等于passage_max_len）

--passage_max_len: passage的最大长度（为解决静态图中的动态shape问题，query_max_len需要等于passage_max_len）

--save_steps: 每多少个step保存一次checkpoint**

--output_dir: 模型保存路径

--lr: 学习率，默认1e-5

--query_instruction_for_retrieval: query 指令语句，默认为空

--temperature: 影响相似度分数的分布

  

**分布式微调启动命令**

- 启动分布式训练，需要先运行baai_general_embedding下的hccl_tools.py，手动生成**RANK_TABLE_FILE**，生成命令如下：

```
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./hccl_tools.py --device_num "[0,8)"
```

RANK_TABLE_FILE 单机8卡参考样例:

```
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

- 执行分布式微调命令

```
bash run_singlenode.sh "python run.py \
--model_name_or_path ./bge-large-zh-v1.5 \
--train_data ./toy_finetune_data.jsonl \
--epoch 2 \
--batch_size 2 \
--train_group_size 2 \
--query_max_len 256 \
--passage_max_len 256 \
--save_steps 1000 \
--output_dir ./checkpoint \
--use_parallel" /path/to/hccl_8p_01234567_127.0.1.1.json [0,8] 8
# 注意：mindspore静态图训练，编译时间较长，请耐心等待
```

**注意**：分布式训练需要设置use_parallel

- 日志会生成在./output/log/目录下，例如8卡微调，会生成rank_0 ~ rank_7 共计8个目录，分别存放不同卡上的训练过程，查看日志命令如下：

```
tail -f output/log/rank_0/mindformer.log
```

 

 

### 评估

#### 数据集

以[msmarco-corpus](https://huggingface.co/datasets/namespace-Pt/msmarco-corpus)与[msmarco](https://huggingface.co/datasets/namespace-Pt/msmarco)为例，需要在multi_infer.py与eval_msmarco.py中分别指定对应数据集的路径


#### 分布式评估启动命令

运行评估时需要**先分布式生成faiss向量库，再启动评估脚本进行评估。**

生成向量库

```
bash pre_infer.sh
```

评估

```
bash evaluate.sh
```

#### 分布式生成faiss向量库

```
rm -rf device
mkdir device
echo "start training"
RANK_SIZE=8
for((i=0;i<${RANK_SIZE};i++));
do
    export MS_WORKER_NUM=${RANK_SIZE}  # 设置集群中Worker进程数量为8
    export MS_SCHED_HOST=127.0.0.1     # 设置Scheduler IP地址为本地环路地址
    export MS_SCHED_PORT=8118          # 设置Scheduler端口
    export MS_ROLE=MS_WORKER           # 设置启动的进程为MS_WORKER角色
    export MS_NODE_ID=$i               # 设置进程id，可选
    python multi_infer.py \
        --encoder ./bge-large-zh-v1.5 \
        --fp16 \
        --add_instruction \
        --k 100 \
    > device/worker_$i.log 2>&1 &
done

export MS_WORKER_NUM=${RANK_SIZE}   # 设置集群中Worker进程数量为8
export MS_SCHED_HOST=127.0.0.1      # 设置Scheduler IP地址为本地环路地址
export MS_SCHED_PORT=8118           # 设置Scheduler端口
export MS_ROLE=MS_SCHED             # 设置启动的进程为MS_SCHED角色
python multi_infer.py > device/scheduler.log 2>&1 &
```

参数介绍：
\--encoder encoder：推理模型地址，建议设置为绝对路径
\--fp16：推理时启用fp16
\--add_instruction：添加询问侧指令
\--max_passenage_length：passage的最大长度
\--batch_size：单次推理样本数量
\--corpus_embeddings：corpus_embeddings输出路径，建议设置为绝对路径

输出：
在corpus_embeddings文件夹中生成如corpus_embeddings_rankx.npy形式的文件，其中rankx表示第几张卡计算生成的数据，单卡时为rank0。

**注意**：

- 分布式生成faiss向量库脚本使用mindspore动态组网方式实现并行。

##### 评估

```
python eval_msmarco.py \
--encoder ./bge-large-zh-v1.5 \
--fp16 \
--add_instruction \
--k 100 \
--corpus_embeddings ./corpus_embeddingsw
```

参数介绍：
\--encoder encoder：推理模型地址，建议设置为绝对路径
\--corpus_embeddings：对应分布式推理脚本中corpus_embeddings的输出路径
\--fp16：推理时启用fp16
\--add_instruction：添加询问侧指令
\--max_query_length：query的最大长度
\--batch_size：单次推理样本数量
\--index_factory：faiss参数
\--k：召回数

使用评估模型评估baai-base-en-v1.5模型，可以得到如下输出:

```
{
    'MRR@1': 0.2329512893982808, 
    'MRR@10': 0.35789756446991444, 
    'MRR@100': 0.36928145881558194, 
    'Recall@1': 0.22584765998089776, 
    'Recall@10': 0.6415830945558741, 
    'Recall@100': 0.900704393505253
}
```

**注意**

- 执行推理脚本时日志会生成在./device目录下。以8卡为例，会生成9个日志：scheduler.log worker_0.log 到 worker_7.log，可以用如下命令查看日志。

```
tail -f ./device/worker_0.log
```

​	






## **FAQ**

