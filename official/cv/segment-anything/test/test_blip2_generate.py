import time

import mindspore
from mindformers import AutoModel, AutoProcessor
from mindformers.tools.image_tools import load_image
from mindspore.ops import operations as P


s = time.time()
# 指定图模式，指定使用训练卡id
from mindspore import ops

mindspore.set_context(mode=0)

# 通过AutoClass创建一阶段预训练任务
model = AutoModel.from_pretrained("blip2_stage1_classification")
# model.set_train(False)

# model = AutoModel.from_pretrained("itt_blip2_stage2_vit_g_llama_7b")
processor = AutoProcessor.from_pretrained("blip2_stage1_classification")
tokenizer = processor.tokenizer

candidate_labels=["sunflower", "tree", "dog", "cat", "toy"]
sentences = ["This is a photo of {}.".format(candidate_label)
                     for candidate_label in candidate_labels]

input_ids = tokenizer(sentences, max_length=77, padding="max_length", return_tensors="ms")["input_ids"]
filepath = "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png"
input_images = processor.image_processor(load_image(filepath))

image_features = model.get_image_feature(input_images)[:, 0]
text_features = model.get_text_feature(input_ids)[:, 0]

logits_per_image = ops.matmul(image_features, text_features.T) / model.temp

probs = P.Softmax()(logits_per_image).asnumpy()

# input_ids2 = tokenizer(["it is a photo of a dog"], padding="max_length", max_length=77, return_tensors="ms")["input_ids"]
# image_features = model.get_image_feature(input_images)[:, 0]
# text_features = model.get_text_feature(input_ids2)[:, 0]
# sims2 = ops.matmul(image_features, text_features.T) / model.temp

print('logits', logits_per_image)
print('prob', probs)
print(image_features[0, :10])
print(text_features[0, :10])
# print("it is a photo of a dog: ", sims2)

e = time.time()

print(f'{e-s:.2f}s')

