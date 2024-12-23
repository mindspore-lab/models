import mindspore
from mindformers import CLIPModel, CLIPConfig, AutoModel, AutoProcessor
from mindformers.tools.image_tools import load_image
from mindspore import amp, mint

mindspore.set_context(mode=0, jit_config={'jit_level': 'O1'})

# load model and processor
model = AutoModel.from_pretrained("clip_vit_l_14@336")
model = amp.auto_mixed_precision(model, 'O0')
processor = AutoProcessor.from_pretrained("clip_vit_l_14@336")

# load text
candidate_labels = ["sunflower", "tree", "dog", "cat", "toy"]
sentences = ["This is a photo of {}.".format(candidate_label) for candidate_label in candidate_labels]
input_ids = processor.tokenizer(sentences, max_length=77, padding="max_length", return_tensors="ms")["input_ids"]

# load image
filepath = "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png"
input_images = processor.image_processor(load_image(filepath))

# infer
input_images = input_images.astype(mindspore.float32)
logits_per_image, _ = model(input_images, input_ids)

probs = mint.nn.functional.softmax(logits_per_image, dim=-1).asnumpy()

print('logits', logits_per_image)
print('prob', probs)
