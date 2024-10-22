## CRF 简介

`CRF`（条件随机场 - Conditional Random Field）模型是一种用于序列预测任务的统计模型，适用于需要考虑序列中元素之间依赖关系的标注和解析任务，如`词性标注`、`命名实体识别`等。

当前`package`中，我们基于`MindSpore`实现了`CRF`模型。

## 代码架构

**主要组成函数**

1. **初始化函数**：用于构建`CRF`模型

2. **解码函数（viterbi_decode）**：为了在给定观测序列的情况下找到最优的标签序列，`CRF`使用`维特比算法`（Viterbi Algorithm），来解码最可能的标签路径。
   - 这里的解码函数与`TorchCRF`略有差异，解码函数返回的是最佳序列得分`score`与历史最佳标签候选者的转换位置（用于回溯），所以需要继续调用`post_decode`函数用于标签序列解码（`TorchCRF`会在计算完得分与历史后，直接进行标签序列解码）。
   - 该函数所依赖的`mask`由`Mask生成器`提供。
   
3. **标签序列解码函数（post_decode）**：使用Score和History计算最佳预测序列。

4. **Mask生成器（sequence_mask）**：根据序列实际长度和最大长度生成`mask`矩阵。

5. **损失函数**：对数似然损失函数（Negative Log Likelihood, NLL），它衡量了模型预测的标签序列与真实标签序列之间的差异，他的计算依赖于`Normalizer函数`与`Score函数`。

6. **Normalizer函数（compute_normalizer）**：用于归一化分数，以便于模型在不同标签路径之间进行比较和选择。

   - 该函数所依赖的`mask`由`Mask生成器`提供。

7. **Score函数（compute_score）**：用于计算给定观测序列的标签序列的分数。

   - **发射分数（Emission Scores）**：这部分分数表示观测序列与标签序列的匹配程度。
   - **转移分数（Transition Scores）**：这部分分数表示标签序列中标签之间的转移概率。

   - `Score函数`将这两部分分数结合起来，为整个标签序列生成一个总分数。
   - 该函数所依赖的`mask`由`Mask生成器`提供。

## 使用场景案例

这里简单介绍一下，基于BiLSTM+CRF模型结构：

> nn.Embedding -> nn.BiLSTM -> nn.Dense -> CRF
>
> *`nn`由MindSpore`提供`*

1. **nn.Embedding（词嵌入层）**

   - **作用：**将词汇映射为高维空间中的向量，通过向量（称为词向量或嵌入）捕捉词汇的语义和语法特性。

   - **输入**：通常是文本序列中的单词或字符的索引。

   - **输出**：每个索引对应的词向量，形成序列化的词向量矩阵。

2. **nn.LSTM（长短期记忆网络）**

   - **作用：**`BiLSTM`由两个`LSTM`组成，分别处理序列的正向和反向，以捕获序列中每个元素的全面上下文信息。
   - **输入：**`nn.Embedding`层的输出，即序列化的词向量。
   - **输出：**每个时间步的隐藏状态向量，这些向量包含了输入序列的正向和反向上下文信息。

3. **nn.Dense（全连接层）**

  - **作用：**将`BiLSTM`层的输出映射到标签空间，为每个时间步生成一个关于所有可能标签的得分。
  - **输入：**`BiLSTM`层的最后一个时间步的输出，或者拼接所有时间步的输出以获得更丰富的特征表示。
  - **输出：**每个时间步的标签得分向量，其维度是标签的数量。

4. **CRF（条件随机场）**

  - **作用：**在给定观测序列（`BiLSTM`的输出）的条件下，对标签序列进行建模，同时考虑标签之间的转移概率，以找到最优的标签序列。
  - **输入：**`nn.Dense`层的输出，即每个时间步的标签得分向量。

  - **输出：**最可能的标签序列，通过维特比算法解码得到。

##  数据集

数据验证基于`BiLSTM-CRF`模型，使用`CoNLL-2003`数据集，下载地址：[CoNLL-2003](https://data.deepai.org/conll2003.zip)

## 环境要求

- 硬件(GPU/CPU/Ascend)

  - 准备Ascend或GPU处理器搭建硬件环境。

- 框架

  - [MindSpore 2.2.11](https://www.mindspore.cn/install/)

- 软件

  - Python 3.9.x
  - CANN：7.0.RC1
  - 驱动与固件：23.0.RC1

- Python依赖

  - |           |           |
    | --------- | --------- |
    | mindspore | = 2.2.11  |
    | numpy     | >= 1.26.4 |
    | tqdm      | >= 4.66.2 |

## 快速开始

### 初始化模型

#### 参数说明

> `num_tags: int` -- 模型在序列标注任务中需要识别的标签数量
>
> 这个参数对于模型的构建和功能至关重要，具体作用如下：

1. **定义标签空间大小**：`num_tags`参数定义了`CRF`模型中标签空间的大小，即模型需要预测的标签种类数。这对于模型的训练和预测都是必要的，因为它决定了模型输出层的神经元数量。
2. **确定转移矩阵的维度**：在`CRF`模型中，一个重要的组成部分是转移矩阵（transition matrix），它表示不同标签之间的转移概率。`num_tags`参数用于确定这个转移矩阵的维度，因为转移矩阵的形状通常是`(num_tags, num_tags)`。
3. **解码算法的输入**：在模型的解码阶段，`CRF`模型使用`维特比算法`（Viterbi Algorithm）等解码算法来找到最优的标签序列。解码算法需要知道有多少个可能的标签，这也是通过`num_tags`参数来指定的。
4. **损失函数计算**：在训练CRF模型时，需要计算损失函数以进行反向传播和参数更新。`num_tags`参数帮助定义了损失函数的结构，因为它涉及到对每个标签的预测概率的计算。
5. **模型输出解释**：在模型的预测阶段，`num_tags`参数对于解释模型输出至关重要。模型输出的每一个向量通常会被解释为对应于`num_tags`中某个标签的概率分布。

>  `batch_first: bool = False` -- 通常用于指定输入数据的维度顺序
>
> 这个参数决定了是将批次大小（batch size）作为张量的第一个维度，还是将序列长度（sequence length）作为第一个维度。

- **`batch_first=True`**：输入数据的维度顺序是 `(batch_size, sequence_length, feature_size)`。这意味着每个输入样本的批次大小（即序列的数量）是张量的第一个维度，序列长度是第二个维度，而特征或标签的数量是最后一个维度。

- **`batch_first=False`**：输入数据的维度顺序是 `(sequence_length, batch_size, feature_size)`。在这种情况下，序列长度是张量的第一个维度，批次大小是第二个维度，特征或标签数量是最后一个维度。

#### 样例代码

```python
from CRF.src.model.lstm_crf_model import CRF


# 需要使用的实体索引，可以根据需要使用BIO或者BIOES标注模式，当然也可以给予实际使用场景进行定义
tag_to_idx = {"B": 0, "I": 1, "O": 2}

# 初始化模型
crf = CRF(len(tag_to_idx), batch_first=True)
```



### 计算损失函数

创建完成`CRF`模型后，可以通过`emission`与`tags`计算损失函数（Negative Log Likelihood, NLL）

**提别注意：`MindSpore`会自动添加mask，无需像`TorchCRF`做额外的构造**

#### 参数说明

> `emissions: mindspore.common.tensor.Tensor` -- 模型的输入特征，在计算损失函数阶段，`必填`

- 这个参数通常是从前面的网络层（如`BiLSTM`）传递过来的。

- 它代表了每个时间步可能的标签的未归一化得分或概率。这些特征将被用来计算每个标签序列的得分。

> `tags: mindspore.common.tensor.Tensor` -- 数据的真实标签序列，在计算损失函数阶段，`必填`

- 在模型训练阶段用于计算损失函数。
- 在CRF中，这些真实标签用于计算模型输出和真实情况之间的差异，进而通过反向传播算法更新模型的权重。

> `seq_length: mindspore.common.tensor.Tensor = None` -- 每个输入序列的实际长度，`非必填`

- `seq_length`用于指示每个序列的最后一个有效时间步，以避免对填充（padding）元素进行计算。有利于保持模型的计算效率和准确性。

#### 样例代码

```python
import mindspore.numpy as mnp

from CRF.src.model.lstm_crf_model import CRF


# 需要使用的实体索引，可以根据需要使用BIO或者BIOES标注模式，当然也可以给予实际使用场景进行定义
tag_to_idx = {"B": 0, "I": 1, "O": 2}

# 构建模型
crf = CRF(len(tag_to_idx))

# emissions需要3个参数描述shape，seq_length, batch_size, num_tags
seq_length = 3
batch_size = 2
emissions = mnp.randn(seq_length, batch_size, len(tag_to_idx))

tags = mnp.array([[0, 1], [2, 4], [3, 1]])

# loss计算函数
loss = crf(emissions, tags)
```

### Decode

可以使用`CRF`中的`decode`函数，获取当前的`score`与`history`，通过这两个参数，再结合seq_length，可以获得最终结果`best_tags_list`

#### 参数说明

> `decode`函数接受一个`emission`，参见上面的介绍

> `post_decode`函数，需要额外传入一个`seq_length`，参见上面的介绍

#### 样例代码

```python
import mindspore.numpy as mnp

from CRF.src.model.lstm_crf_model import CRF

# 需要使用的实体索引，可以根据需要使用BIO或者BIOES标注模式，当然也可以给予实际使用场景进行定义
tag_to_idx = {"B": 0, "I": 1, "O": 2}

# 构建模型
crf = CRF(len(tag_to_idx))

# emissions需要3个参数描述shape，seq_length, batch_size, num_tags
seq_length = 3
batch_size = 2
emissions = mnp.randn(seq_length, batch_size, len(tag_to_idx))

# decode
score, history = crf.decode(emissions)
best_tags_list = crf.post_decode(score, history, mnp.full((batch_size,), seq_length))
```



### 推理与训练样例

可以参考如下代码，进行推理与训练

```python
import mindspore as ms
import mindspore.nn as nn
from tqdm import tqdm
from CRF.src.model.lstm_crf_model import BiLSTM_CRF


def train_step(data, seq_length, label):
    loss, grads = grad_fn(data, seq_length, label)
    optimizer(grads)
    return loss


def prepare_sequence(seqs, word_to_idx, tag_to_idx):
    seq_outputs, label_outputs, seq_length = [], [], []
    max_len = max([len(i[0]) for i in seqs])

    for seq, tag in seqs:
        seq_length.append(len(seq))
        idxs = [word_to_idx[w] for w in seq]
        labels = [tag_to_idx[t] for t in tag]
        idxs.extend([word_to_idx['<pad>'] for i in range(max_len - len(seq))])
        labels.extend([tag_to_idx['O'] for i in range(max_len - len(seq))])
        seq_outputs.append(idxs)
        label_outputs.append(labels)

    return ms.Tensor(seq_outputs, ms.int64), \
        ms.Tensor(label_outputs, ms.int64), \
        ms.Tensor(seq_length, ms.int64)


def sequence_to_tag(sequences, idx_to_tag):
    outputs = []
    for seq in sequences:
        outputs.append([idx_to_tag[i] for i in seq])
    return outputs


if __name__ == '__main__':
    ms.set_context(device_target="Ascend")

    embedding_dim = 16
    hidden_dim = 32

    training_data = [(
        "清 华 大 学 坐 落 于 首 都 北 京".split(),
        "B I I I O O O O O B I".split()
    ), (
        "重 庆 是 一 个 魔 幻 城 市".split(),
        "B I O O O O O O O".split()
    )]

    word_to_idx = {}
    word_to_idx['<pad>'] = 0
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

    tag_to_idx = {"B": 0, "I": 1, "O": 2}

    model = BiLSTM_CRF(len(word_to_idx), embedding_dim, hidden_dim, len(tag_to_idx))
    optimizer = nn.SGD(model.trainable_params(), learning_rate=0.01, weight_decay=1e-4)

    grad_fn = ms.value_and_grad(model, None, optimizer.parameters)

    data, label, seq_length = prepare_sequence(training_data, word_to_idx, tag_to_idx)

    steps = 300
    with tqdm(total=steps) as t:
        for i in range(steps):
            loss = train_step(data, seq_length, label)
            t.set_postfix(loss=loss)
            t.update(1)

    score, history = model(data, seq_length)
    predict = model.crf.post_decode(score, history, seq_length)

idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
outputs = sequence_to_tag(predict, idx_to_tag)
print('predict to BIO', outputs)
```

### 导出CKPT

我们提供了统一脚本，方便使用者导出`ckpt`文件。但是导出`ckpt`需要手动将数据集放入指定位置，具体操作如下：

**步骤1：** 参照步骤四，下载数据集

**步骤2：** 解压文件

**步骤3：** 将解压后的 `train.txt`，`test.txt`，`vaild.txt`文件，放入到`conll2003`目录下

**步骤4：** 进入`script`目录，执行命令

```bash
bash export_ascend.sh 0 ../conll2003 lstm_crf.ckpt
```

**步骤5：** 完成后，会在当前目录下生成`crf-model.ckpt`文件

**End**

### 导出MINDIR

我们提供了统一脚本，方便使用者导出`mindir`文件。但是导出ckpt需要手动将数据集放入指定位置，具体操作如下：

**步骤1：** 参照步骤四，下载数据集

**步骤2：** 解压文件

**步骤3：** 将解压后的 `train.txt`，`test.txt`，`vaild.txt`文件，放入到`conll2003`目录下

**步骤4：** 进入`script`目录，执行命令

```bash
bash export_mindir.sh 0 ../conll2003 lstm_crf.mindir
```

**步骤5：** 完成后，会在当前目录下生成`crf-model.ckpt`文件

**End**

### 离线推理
在`310P`机器上，我们可以采用`mindir`的方式，进行离线推理

#### 获取`MINDIR`
通过上一章节的脚本，我们可以得到`MINDIR`文件

#### 进行离线推理

- 在使用离线推理时，需要通过`pip`安装`mindspore_lite`(2.2.11版本)
我们可以参考如下代码，进行离线推理

```python
import numpy as np
import mindspore_lite as mslite

context = mslite.Context()
# 这里设置在CPU或者Ascend进行推理
context.target = ["Ascend"]
# 这里可以设置使用哪个NPU进行推理，一般0是第一块NPU
context.ascend.device_id = 1

# 这里可以设置使用哪个CPU进行推理
# context.cpu.thread_num = 1
# context.cpu.thread_affinity_mode = 2

# 指定mindir文件所在位置
MODEL_PATH = "../../scripts/crf-lite-Ascend.mindir"

model = mslite.Model()
model.build_from_file(MODEL_PATH, mslite.ModelType.MINDIR, context)
inputs = model.get_inputs()

# 特别注意，这里emission的shape需要与mindir一致
emissions = np.ones((16, 113), np.int32)

# 这里的shape需要与mindir保持一致
inputs[0] = mslite.Tensor(emissions.astype(dtype=np.int32))

# 进行预测
outputs = model.predict(inputs)
score, history = outputs[0], outputs[1:]
```