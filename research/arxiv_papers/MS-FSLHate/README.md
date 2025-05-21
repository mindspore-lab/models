# MS-FSLHate

# 概述
本项目旨在构建一个MindSpore框架的仇恨言论检测模型，能够对用户发布的文本内容进行分类，识别是否包含仇恨、冒犯或正常言论。

# 数据集
本项目使用的数据集有HateXplain数据集和Hate Speech and Offensive Language Dataset (HSOL)数据集。
HateXplain数据集包含20,148条来自Twitter和Gab的帖子，每条帖子都从三个不同角度进行标注：基本的三类分类（仇恨、攻击性或正常）、目标社区（帖子中仇恨言论/攻击性言论的受害者社区）以及理由（标注决策所依据的帖子部分）。
HSOL数据集 是一个用于仇恨言论检测的数据集。作者从包含单词和 被互联网用户识别为仇恨言论的短语，由 Hatebase.org 汇编。他们使用 Twitter API 搜索 对于包含词典中术语的推文，从而产生来自 33,458 个 Twitter 用户的推文样本。他们提取了 每个用户的时间线，从而产生一组 8540 万条推文。他们从这个语料库中随机抽取了 25k 条包含词典术语的推文样本，并由 CrowdFlower （CF） 工作人员手动编码。员工被要求将每条推文标记为以下三类之一：仇恨言论、冒犯性但不是仇恨言论，或既不冒犯也不是仇恨言论。
# 模型架构
我们设计了一个仇恨言论检测模型，主要特点包括：
1. Prompt Embedding：引入可学习的提示向量，增强模型在短文本场景中的上下文理解能力；

2. 卷积层（CNN）：提取局部特征，捕捉短语或组合词的信息；

3. 双向LSTM：建模长距离依赖，获取文本语义的上下文信息；

4. 注意力机制：对关键信息赋予更高权重，提高模型解释性；

5. Dropout & LayerNorm：提升模型泛化能力。


# 训练流程
1. 训练框架：基于 MindSpore 实现整个训练、评估、推理流程；
2. 学习率策略：使用余弦退火衰减（cosine decay）调整学习率；
3. 梯度裁剪：控制训练稳定性，避免梯度爆炸；
4. 指标评估：用精准率、召回率、f1-score 综合评估模型性能；
5. 对抗数据增强：通过 nltk.wordnet 同义词替换策略对训练数据进行扰动，增强模型鲁棒性；如果缺少 WordNet 资源，可手动下载到对应目录。
> 🔧 如果缺少 WordNet 资源，可手动下载：

```python
import nltk

# 下载 WordNet 数据集到本地指定路径
nltk.download('wordnet', download_dir='./nltk_data')

# 添加数据路径（确保代码可以找到资源）
nltk.data.path.append('./nltk_data')


# 代码结构解析
一、MS-FSLHate_HateXplain.py

1. compute_majority_label 函数
从多个标注者的注释中提取标签。
将标签统一转为小写，统计出现频率。
返回出现次数最多的标签，若无有效标签则返回 "normal"。

2. synonym_replacement 函数
用于对抗数据增强，基于 NLTK 的 WordNet 替换部分单词为其同义词。
遍历词列表，以设定概率随机替换词汇为其同义词。
返回增强后的词列表。

3. build_vocab 函数
从多个分词后的文本中构建词汇表。
特别处理以 HASHTAG_ 开头的词。
限制最大词汇数量为 max_vocab_size，并加入特殊词如 <pad>、<unk>。
返回构建好的词典（词到索引的映射）。

4. tokens_to_ids 函数
将单个文本的词序列转换为对应的 ID 序列。
遇到未知词使用 <unk> ID，补齐或截断为固定长度 max_length。
返回转换后的 NumPy 数组（int32 类型）。

5. load_dataset 函数
从 JSON 文件加载标注数据。
对每条样本调用 compute_majority_label 获取最终标签。
将标签映射为整数：normal=0, offensive=1, hatespeech=2。
返回分词后的文本和对应标签的 NumPy 数组。

6. create_datasets 函数
加载训练、验证、测试集数据。
使用 build_vocab 构建词表。
对训练数据执行同义词增强，并将原始与增强数据合并。
使用 tokens_to_ids 将文本转换为 ID 序列。
使用 mindspore.dataset 创建训练和测试数据集对象。
计算类别权重以缓解类别不平衡问题。
返回构建好的训练集、测试集、词表及类别权重。

7. EnhancedClassifier 类
一个增强型文本分类器，集成词嵌入、卷积层、LSTM 和注意力机制。
包括嵌入层、提示向量、Conv1D、双向 LSTM、注意力、归一化、Dropout 和最终全连接层。
支持 prompt 拼接、通道变换、特征提取和聚合。
用于三分类任务（normal、offensive、hatespeech）。

8. evaluate_model 函数
对模型进行评估，输出预测结果、Softmax 概率和真实标签。
使用 classification_report 输出精确率、召回率、F1 分数。
计算加权 F1 值和 ROC AUC 值（One-vs-Rest 多分类模式）。
返回评估结果。

9. CustomTrainOneStep 类
自定义训练步骤单元。
包含梯度计算与裁剪功能。
用于训练时控制梯度爆炸。

10. main 函数
主流程控制函数，执行整个训练、验证、测试与保存。
加载配置、构建数据集与模型。
设置优化器、损失函数和学习率调度器。
使用训练步骤类执行多轮训练，并在每轮后评估模型。
最后保存模型检查点。


二、MS-FSLHate_HSOL.py

1. synonym_replacement 函数
实现基于 WordNet 的对抗数据增强（同义词替换）。
遍历文本 tokens，并以设定概率将其中部分词替换为其同义词。
返回增强后的 token 序列。

2. build_vocab 函数
构建训练语料词表，支持最多 max_size 个词。
特别处理 HASHTAG_ 前缀词项，并加入 <pad>、<unk> 等特殊符号。
返回词汇到索引的映射字典。

3. tokens_to_ids 函数
将 token 序列转换为固定长度的 ID 序列。
若不足长度，则填充 <pad>；超过则截断。
返回 ID 的 NumPy 数组（int32 类型）。

4. create_datasets 函数
从 CSV 文件中读取文本与标签，并进行分层采样构建训练/测试集。
构建词表并执行同义词替换数据增强，扩展训练样本。
将文本 token 转换为 ID 序列，并构建 MindSpore 格式的数据集。
计算类别权重以平衡训练。
返回训练集、测试集、词表和类别权重。

5. EnhancedClassifier 类
多模块文本分类模型，包含以下结构：
词嵌入层（Embedding）；
可学习的 prompt 向量；
卷积特征提取模块（Conv1D + ReLU + MaxPool）；
双向 LSTM；
注意力机制（Dense 实现）；
层归一化和 Dropout；
输出层为全连接分类器。
支持前向推理过程，返回分类 logits。

6. CustomTrainOneStep 类
自定义训练单元，支持梯度裁剪（Clip by Global Norm）。
封装损失计算、反向传播与优化器更新。

7. evaluate_model 函数
用于在测试集上评估模型性能。
收集预测结果、真实标签和 Softmax 概率。
输出详细分类报告（Precision、Recall、F1）、加权 F1 分数和 ROC AUC。
支持多分类（OvR）ROC AUC 计算。

8. main 函数
配置参数并执行主训练流程，包括：
加载和预处理数据集；
构建模型与词表；
配置损失函数、优化器与学习率调度；
按轮次训练模型并评估；
保存最终训练好的模型参数。
输出模型评估结果及训练损失信息。

## 运行环境要求

1. **Python 版本**
   - 建议使用 Python 3.10 及以上版本。

2. **MindSpore 框架**
   - 需要安装 MindSpore 框架，可通过以下命令安装：
     ```
     pip install mindspore
     ```

3. **其他依赖库**
   - 需要安装以下依赖库：
     ```
     pip install numpy pandas scikit-learn tqdm nltk
     ```

4. **运行代码**
运行脚本：python MS-FSLHate_HateXplain.py
运行脚本：python MS-FSLHate_HSOL.py

 ## 感谢MindSpore社区提供的支持
