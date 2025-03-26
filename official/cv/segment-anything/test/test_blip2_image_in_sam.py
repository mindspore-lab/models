import mindspore
from mindformers import AutoModel, AutoProcessor
import numpy as np
from mindspore import mint

mindspore.set_context(mode=0, jit_config={'jit_level': 'O1'})

# 通过AutoClass创建一阶段预训练任务
model = AutoModel.from_pretrained("blip2_stage1_classification")
# model = amp.auto_mixed_precision(model, 'O2')._backbone
# model.set_train(False)

# model = AutoModel.from_pretrained("itt_blip2_stage2_vit_g_llama_7b")
processor = AutoProcessor.from_pretrained("blip2_stage1_classification")


images = np.load('./patch.npy')[:20] # (37, 3, 336, 336)
input_images = processor.image_processor(images)

pre_diff = np.load('./blip2_patch.npy') - input_images.asnumpy()
print(f'preprocess diff max: {np.max(np.abs(pre_diff))}')
print(f'preprocess diff : {pre_diff.flatten()[:10]}')

images_np = np.load('./blip2_patch.npy')
input_images  = mindspore.Tensor(images_np)  #
image_features = model.get_image_feature(input_images)[:, 0] # (20, 256)
feat_diff = np.load('./blip2_image_features.npy') - image_features[:20].asnumpy()
print(f'feature diff max: {np.max(np.abs(feat_diff))}')
print(f'feature diff : {feat_diff.flatten()[:10]}')

candidate_labels=["cat", "person", "stone steps", "window", "tree"]
sentences = ["This is a photo of {}.".format(candidate_label)
                     for candidate_label in candidate_labels]

input_ids = processor.tokenizer(sentences, max_length=77, padding="max_length", return_tensors="ms")["input_ids"]
text_features = model.get_text_feature(input_ids)[:, 0]

logits_per_image = mint.matmul(image_features, text_features.T) / model.temp  # (20, 5)

probs = mint.nn.functional.softmax(logits_per_image, dim=-1).asnumpy() # (20, 5)

for i in range(20):
    print(f'\n\n{i}')
    print('logits', logits_per_image[i])

    sorted_res = sorted(zip(probs[i], candidate_labels), key=lambda x: -x[0])
    print('sorted res', sorted_res)

    ind = int(np.argmax(probs[i]))
    print('img feature', image_features[i, :10])
    print('text feature', text_features[ind, :10])
    print('diff', text_features[ind, :10] - image_features[i, :10])

    import matplotlib.pyplot as plt
    plt.imshow(images[i])
    plt.show()
