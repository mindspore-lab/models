import time

import mindspore
from mindformers.pipeline import pipeline
from mindformers.tools.image_tools import load_image

s = time.time()

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0)

pipeline_task = pipeline(task="zero_shot_image_classification", model="blip2_stage1_classification")

input_data = load_image(
    "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
pipeline_result = pipeline_task(input_data,
                                candidate_labels=["sunflower", "tree", "dog", "cat", "toy"],
                                hypothesis_template="This is a photo of {}.")
print(pipeline_result)

e = time.time()
print(f'{e-s:.2f}s')

# 输出
# [[{'score': 0.99999714, 'label': 'sunflower'},
#   {'score': 1.315181e-06, 'label': 'tree'},
#   {'score': 7.0368844e-07, 'label': 'toy'},
#   {'score': 4.7594781e-07, 'label': 'dog'},
#   {'score': 3.93686e-07, 'label': 'cat'}]]