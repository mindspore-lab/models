import json
import numpy as np
from PIL import Image
from mindspore.dataset import vision
from mindspore.dataset import transforms

class ImageCaption:
    def __init__(self, config) -> None:
        super(ImageCaption, self).__init__()

        self.TrainOrVal = config.TrainOrVal
        self.max_length = config.max_length
        self.sentence_nums = config.sentence_nums

        self.img_dir = config.image_dir

        # 从图片名找到图片id
        self.filename_to_id = {}
        # 从图片id找到句子
        self.imgid_to_sentences = {}

        info_json = open(config.info_json, encoding = 'utf-8')
        content = json.load(info_json)
        for item in content['images']:
            file_name = item['filename']
            img_id = int(item['imgid'])
            self.filename_to_id[file_name] = img_id

            self.imgid_to_sentences[img_id] = []
            for sentence in item['sentences']:
                tokens = ' '.join(sentence['tokens'])
                self.imgid_to_sentences[img_id].append(tokens)

        img_name_file = open(config.image_name, encoding = 'utf-8')
        img_name = img_name_file.readlines()

        self.dataset = []

        # 将一张图片和多个对应描述当作一个样本
        for file_name in img_name:
            file_name = file_name.split('\n')[0]
            temp = {}
            temp['img_id'] = self.filename_to_id[file_name]
            temp['file_name'] = file_name
            self.dataset.append(temp)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transforms = transforms.Compose([
            vision.Normalize(mean=mean, std=std),
            vision.Resize(size=[224, 224]),
            vision.HWC2CHW()
        ])

        self.tokenizer = config.tokenizer
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id

    def __getitem__(self, index):

        item = self.dataset[index]
        image_path = self.img_dir + item['file_name']
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert("RGB")
        image = np.array(image)
        image = self.transforms(image)

        # 将一张图片和多个对应描述当作一个样本
        img_id = item['img_id']
        sentences = self.imgid_to_sentences[img_id][:self.sentence_nums]

        all_caption = []
        all_label = []

        for caption_sentence in sentences:
            caption_encoded = self.tokenizer.encode_plus(
            caption_sentence, max_length=self.max_length, padding=False, return_attention_mask=False, return_token_type_ids=False, truncation=True)

            caption_token_id = caption_encoded['input_ids'][1:-1]
            caption = [self.bos_token_id] + caption_token_id + [self.pad_token_id] * (self.max_length - 1 - len(caption_token_id))
            label = caption_token_id + [self.eos_token_id] + [self.pad_token_id] * (self.max_length  - 1 - len(caption_token_id))

            assert len(caption) == self.max_length and len(label) == self.max_length

            all_caption.append(caption)
            all_label.append(label)

        caption = np.array(all_caption, dtype = np.int32)
        label = np.array(all_label, dtype = np.int32)

        return image, caption, label, img_id

    def __len__(self):
        return len(self.dataset)

def build_data(config):
    data = ImageCaption(config)
    return data