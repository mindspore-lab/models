import os.path
import os
import random
import pdb

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import mindspore.dataset.vision.utils as utils

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))
    
class ImageLists_VISDA:
    def __init__(self, image_list, root='../../data/multi', transform=None,
                target_transform=None, test=True):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test
        self._index = 0

    def __next__(self):
        if self._index >= len(self.imgs):
            self._index = 0
            #raise StopIteration
        path = os.path.join(self.root, self.imgs[self._index])
        target = self.labels[self._index]
        img = self.loader(path)
        if self.transform is not None:
            img =  self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            item = (img, target)
        else:
            #第三项是图片的path
            item = (img, target, self.imgs[self._index])

        self._index += 1
        return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self.imgs)
    
def return_dataset(args):
#     assert mode == 'train' or mode == 'test', 'mode must chose from [train, test]'
    txt_path = args['txt_path']
    img_root = args['img_root']
    source =args['source']
    target = args['target']
    num = args['num']
    #源域数据 txt
    txt_file_s = os.path.join(txt_path, 'labeled_source_images_' + source + '.txt')
    
    #目标域数据 txt
    txt_file_t = os.path.join(txt_path, 'labeled_target_images_' + target + '_%d.txt' % (num))
    
    # 目标域验证数据 txt
    txt_file_t_val = os.path.join(txt_path, 'validation_target_images_' + target + '_%d.txt' % (num))
    
    # 目标域无标记数据 txt
    txt_file_unl = os.path.join(txt_path, 'unlabeled_target_images_' + target + '_%d.txt' % (num))
    
    if args['net'] == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
        
    data_transforms = {
        'train': transforms.Compose([
            vision.RandomResize(256),
            vision.RandomHorizontalFlip(),
            vision.RandomCrop(crop_size),
            vision.ToTensor(),
            vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], is_hwc=False)
        ]),
        'val': transforms.Compose([
            vision.RandomResize(256),
            vision.RandomHorizontalFlip(),
            vision.RandomCrop(crop_size),
            vision.ToTensor(),
            vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], is_hwc=False)
        ]),
        'test': transforms.Compose([
            vision.RandomResize(256),
            vision.CenterCrop(crop_size),
            vision.ToTensor(),
            vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], is_hwc=False)
        ]),
    }
    
    source_dataset = ImageLists_VISDA(txt_file_s, root=img_root,
                                      transform=data_transforms['train'])
    
    target_dataset = ImageLists_VISDA(txt_file_t, root=img_root,
                                      transform=data_transforms['val'])

    target_dataset_val = ImageLists_VISDA(txt_file_t_val, root=img_root,
                                          transform=data_transforms['val'])

    target_dataset_unl = ImageLists_VISDA(txt_file_unl, root=img_root,
                                          transform=data_transforms['val'])

    target_dataset_test = ImageLists_VISDA(txt_file_unl, root=img_root,
                                           transform=data_transforms['test'])

    class_list = return_classlist(txt_file_s)
    print("%d classes in this dataset" % len(class_list))
    
    return source_dataset, target_dataset, target_dataset_val, target_dataset_unl, target_dataset_test, class_list


def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = []
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            img_path = x.split(' ')[0]
            image_index.append(img_path)
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list, dtype=np.int32)
#     print(selected_list)
    random.shuffle(selected_list)
    image_index = image_index[selected_list]
    label_list = label_list[selected_list]

    return image_index, label_list


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list

if __name__ == '__main__':
    dataset = 'mulit'
    source = 'real'
    target = 'sketch'
    num = 3
    #存储图片名称的txt文件路径
    txt_lis = '../../data/txt/{}/'.format(dataset)
#     存储图片的路径
    img_root = '../../data/{}'.format(dataset)

    source_dataset, target_dataset, target_dataset_val, target_dataset_unl, target_dataset_test = return_dataset(txt_lis,img_root,source,target,num)
    for idx, (image, label) in enumerate(source_dataset.create_tuple_iterator()):
        print(type(image),type(label))
    
