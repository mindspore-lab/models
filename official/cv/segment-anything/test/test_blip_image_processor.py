import cv2
from mindformers import AutoProcessor
from mindformers.tools.image_tools import load_image

processor = AutoProcessor.from_pretrained("blip2_stage1_classification")
filepath = "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png"
input_images = processor.image_processor(load_image(filepath))

filepath1 = "./sunflower.png"
image = cv2.cvtColor(cv2.imread(filepath1), cv2.COLOR_BGR2RGB)
input_images1 = processor.image_processor(image)

import numpy as np

input_images2 = processor.image_processor(np.stack([image, image]))
