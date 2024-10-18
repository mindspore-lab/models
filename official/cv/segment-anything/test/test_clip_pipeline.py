from mindformers import pipeline
from mindformers.tools.image_tools import load_image

classifier = pipeline("zero_shot_image_classification",
                      model="clip_vit_l_14@336",
                      candidate_labels=["sunflower", "tree", "dog", "cat", "toy"])
# get_image_features 768 which is not aligned with
img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2."
          "myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
res = classifier(img)
print(res)