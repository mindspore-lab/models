import os
import random
random.seed(1)
import datasets
from dataclasses import dataclass
from .processor import KGProcessor, get_dataset
import mindspore
from mindspore import ops, Tensor
import mindspore.dataset.transforms as C
import mindspore.common.dtype as mstype
from mindspore.dataset import GeneratorDataset
from mindnlp.transformers import BertTokenizer
from mindspore.dataset import transforms, vision
from PIL import Image

def get_entity_images(entity_img_files, transform, head_ent, tail_ent):
    pixel_images = []
    for head, tail in zip(head_ent, tail_ent):
        pixel_image = []
        if head and tail:   # select one head and one tail
            assert head in entity_img_files and tail in entity_img_files, (head, tail)
            head_file, tail_file = entity_img_files[head], entity_img_files[tail]
            head_imgs = [os.path.join(head_file, file) for file in os.listdir(head_file)]
            tail_imgs = [os.path.join(tail_file, file) for file in os.listdir(tail_file)]
            # select images
            select_head_img = random.sample(head_imgs, k=1) if len(head_imgs) >= 1 else []
            select_tail_img = random.sample(tail_imgs, k=1) if len(tail_imgs) >= 1 else []
            
            # process images
            if len(select_head_img) > 0:
                head_image = Image.open(select_head_img[0]).convert('RGB')
                head_image = transform(head_image)[0]
                pixel_image.append(Tensor(head_image))
            else:
                # pad zero
                pixel_image.append(ops.zeros((3, 224, 224), dtype=mstype.float32))

            if len(select_tail_img) > 0:
                tail_image = Image.open(select_tail_img[0]).convert('RGB')
                tail_image = transform(tail_image)[0]
                pixel_image.append(Tensor(tail_image))
            else:
                # pad zero
                pixel_image.append(ops.zeros((3, 224, 224), dtype=mstype.float32))
        elif head or tail:               # select two head or two tail
            entity = head if head is not None else tail
            assert entity in entity_img_files, (entity)
            entity_file = entity_img_files[entity]
            entity_imgs = [os.path.join(entity_file, file) for file in os.listdir(entity_file)]

            # select
            select_img = random.sample(entity_imgs, k=min(len(entity_imgs), 2)) if len(entity_imgs) >= 1 else []
            for img in select_img:
                entity_image = Image.open(img).convert('RGB')
                entity_image = transform(entity_image)[0]
                pixel_image.append(Tensor(entity_image))
            # pad zero
            for _ in range(2-len(select_img)):
                pixel_image.append(ops.zeros((3, 224, 224), dtype=mstype.float32))
        
        else:
            pixel_image.append(ops.zeros((3, 224, 224), dtype=mstype.float32))
            pixel_image.append(ops.zeros((3, 224, 224), dtype=mstype.float32))
        
        pixel_images.append(ops.stack(pixel_image))

    return pixel_images

class TransferDataset:
    
    """TransferDataset for Huggingface Dataset."""
    def __init__(self, arrow_ds, column_names):
        self.ds = arrow_ds
        self.column_names = column_names

    def __getitem__(self, index):
        return tuple(self.ds[int(index)][name] for name in self.column_names)

    def __len__(self):
        return len(self.ds)

class KGC():
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=False)
        self.processor = KGProcessor(self.tokenizer, args)
        self.label_list = self.processor.get_labels(args.data_dir)
        entity_list = self.processor.get_entities(args.data_dir)
        self.entity_img_files = {entity: os.path.join(args.entity_img_path, entity) for entity in os.listdir(args.entity_img_path)}
        self.transform = transforms.Compose([
                        vision.Resize((224, 224)),
                        vision.CenterCrop(224),
                        vision.ToTensor(),
                        vision.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                            std=(0.26862954, 0.26130258, 0.27577711), is_hwc=False)
                    ])
        print(len(entity_list))
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': entity_list})

        global entities
        with open(self.processor.entity_path, 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip().split("\t")[0])

        relations_tokens = self.processor.get_relations(args.data_dir)
        self.num_relations = len(relations_tokens)
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': relations_tokens})

        self.relation_id_st = self.tokenizer.token_to_id(relations_tokens[0])
        self.relation_id_ed = self.tokenizer.token_to_id(relations_tokens[-1]) + 1
        self.entity_id_st = self.tokenizer.token_to_id(entity_list[0])
        self.entity_id_ed = self.tokenizer.token_to_id(entity_list[-1]) + 1

        # analogy entities and relations
        analogy_entities = self.processor.get_analogy_entities(args.data_dir)
        analogy_relations = self.processor.get_analogy_relations(args.data_dir)
        self.analogy_entity_ids = [self.tokenizer.token_to_id(ent) for ent in analogy_entities]
        self.analogy_relation_ids = [self.tokenizer.token_to_id(rel) for rel in analogy_relations]

        self.setup()

    def create_dataloader(self, dataset, shuffle=False):
        column_names = list(dataset[0].keys())
        dataset = TransferIterableDataset(dataset, column_names=column_names)
        dataloader = GeneratorDataset(source=dataset,
                                        column_names=column_names,
                                        shuffle=shuffle,
                                        num_parallel_workers=1)
        return dataloader

    def setup(self):
        def preprocess_function(features):
            head_ent, tail_ent = features.pop('head_ent'), features.pop('tail_ent')
            pixel_images = get_entity_images(self.entity_img_files, self.transform, head_ent, tail_ent)
            features['pixel_values'] = ops.stack(pixel_images, 0).tolist()
            pre_type = features.pop('pre_type') 
            features['pre_type'] = pre_type
            if 'sep_idx' in features:
                sep_idx, q_head_idx, a_head_idx, rel_idx = features.pop('sep_idx'), features.pop('q_head_idx'), features.pop('a_head_idx'), features.pop('rel_idx')
                features['sep_idx'], features['q_head_idx'], features['a_head_idx'], features['rel_idx'] = sep_idx, q_head_idx, a_head_idx, rel_idx
                
            return features

        self.data_train = get_dataset(self.args, self.processor, "train")
        self.data_val = get_dataset(self.args, self.processor, "dev")
        self.data_test = get_dataset(self.args, self.processor, "test")
        
        raw_dataset = datasets.DatasetDict({'train': datasets.Dataset.from_list(self.data_train),
                                        'val': datasets.Dataset.from_list(self.data_val),
                                        'test': datasets.Dataset.from_list(self.data_test)})
        
        datasets_dict = {}
        for key, raw_ds in raw_dataset.items():
            raw_ds = raw_ds.map(
                preprocess_function,
                batched=True,
                num_proc=1,
            )

            if 'sep_idx' not in raw_ds.features.keys():
                column_names = ['input_ids', 'attention_mask', 'label', 'token_type_ids', 'pixel_values', 'pre_type']
            else:
                column_names =  ['input_ids', 'attention_mask', 'label',
                                 'token_type_ids', 'pixel_values', 'pre_type',
                                  'sep_idx', 'q_head_idx', 'a_head_idx', 'rel_idx']

            source = TransferDataset(raw_ds, column_names)
            ms_ds = GeneratorDataset(
                source=source,
                column_names=column_names,
                shuffle=False,
                num_parallel_workers=1)
            type_cast_op = C.TypeCast(mstype.int32)
            
            print(column_names)
            for column in column_names:
                if column != 'pixel_values':
                    ms_ds = ms_ds.map(input_columns=column, operations=type_cast_op)
                else:
                    ms_ds = ms_ds.map(input_columns=column, operations=C.TypeCast(mstype.float32))
            
            ms_ds = ms_ds.batch(self.args.batch_size, drop_remainder=False)
            datasets_dict[key] = ms_ds
        return datasets_dict

    def get_config(self):
        d = {}
        for k, v in self.__dict__.items():
            if "st" in k or "ed" in k or 'analogy' in k:
                d.update({k:v})
        return d

    def get_tokenizer(self):
        return self.tokenizer
