# GRAPH mode
# import mindspore as ms
# ms.set_context(mode=0)
from mindspore.ops import L2Normalize
from mindnlp.transformers import AutoTokenizer, AutoModel
from mindnlp.transformers import MSBertModel

# pytorch original model
model_path = './bge-large-zh-v1.5'

# sample data
sentences_1 = ["样例数据-1", "样例数据-2"]

# PYNATIVE mode
model = AutoModel.from_pretrained(model_path)
# GRAPH mode
# model = MSBertModel.from_pretrained(model_path)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Inference
encoded_input = tokenizer(sentences_1, padding=True, truncation=True, return_tensors="ms")
# pad to max sequence length 512
# encoded_input = tokenizer(sentences_1, padding="max_length", truncation=True, return_tensors="ms", max_length=512)
model_output = model(**encoded_input)
sentence_embeddings = model_output[0][:, 0]
l2_normalize = L2Normalize(axis=1, epsilon=1e-12)
final_outputs_1 = l2_normalize(sentence_embeddings)

# Similarity
similarity = final_outputs_1 @ final_outputs_1.T
print(similarity)
