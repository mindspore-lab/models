# Copyright 2024 Xidian University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import mindspore
from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import ToTensor,Normalize
from mindspore import Tensor

import random
import os
from PIL import Image

import numpy as np


class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x
        
def omniglot_character_folders():
    data_folder = 'data/omniglot_resized/'
    character_folders = [os.path.join(data_folder, family, character).replace('\\','/') \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
    random.seed(1)
    random.shuffle(character_folders)

    num_train = 1200
    metatrain_character_folders = character_folders[:num_train]
    metaval_character_folders = character_folders[num_train:]

    return metatrain_character_folders,metaval_character_folders

class OmniglotTask(object):
    # This class is for task generation for both meta training and meta testing.
    # For meta training, we use all 20 samples without valid set (empty here).
    # For meta testing, we use 1 or 5 shot samples for training, while using the same number of samples for validation.
    # If set num_samples = 20 and chracter_folders = metatrain_character_folders, we generate tasks for meta training
    # If set num_samples = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing
    def __init__(self, character_folders, num_classes, train_num,test_num):

        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(self.character_folders,self.num_classes)
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:
            for x in os.listdir(c):
                os.path.join(c, x)  
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num+test_num]

        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        path = os.path.join(*sample.replace('\\','/').split('/')[:-1]).replace('\\','/')
        # return path[:2] + '/'  + path[2:]
        return '/' +path


class FewShotDataset():
    def __init__(self, task, split='train',transform=None, target_transform=None,rotation=0):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels
        self.angle = rotation
    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class Omniglot(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('L')
        image = image.resize((28,28), resample=Image.LANCZOS) # per Chelsea's implementation
        # image = image.rotate(self.angle)
        #image = np.array(image, dtype=np.float32)
        if self.transform != None:
            image = self.transform(image)[0]
        label = self.labels[idx]
        if self.target_transform != None:
            label = self.target_transform(label)
        one_hot = np.zeros(5,dtype='long')   
        one_hot[label] = 1       
        return image, one_hot


class ClassBalancedSampler():
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, dataset, num_per_class, num_cl, num_inst, shuffle=True):      #num_per_class=1,num_cl=5,num_inst = 1
        self._index = 0
        self.dataset = dataset
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle
        self.batch_size = self.num_cl*self.num_inst

    def __iter__(self):
        return self

    def __next__(self):
        list1 = [i for i in range(self.num_inst)]
        random.seed(None)
        random.shuffle(list1)

        if self.shuffle:
            self.batch = [[i + j * self.num_inst for i in list1[:self.num_per_class]] for j in
                          range(self.num_cl)]
        else:
            self.batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in
                          range(self.num_cl)]
        self.batch = [item for sublist in self.batch for item in sublist]           #[]

        if self.shuffle:
            random.shuffle(self.batch)

        if self._index >= len(self.dataset):
            raise StopIteration

        item_images = []
        item_labels = []
        for i in self.batch:
            item_image, item_label = self.dataset[i]
            item_images.append(item_image)
            item_labels.append(item_label)
            self._index += 1
        return (Tensor(item_images), Tensor(item_labels,dtype=mindspore.int32))
    

    def __len__(self):
        return 1



def get_data_loader(task, num_per_class=1, split='train',shuffle=True,rotation=0):
    # NOTE: batch size here is # instances PER CLASS
    normalize = Normalize(mean=[0.92206], std=[0.08426])
    # ,rotation=rotation
    dataset = Omniglot(task,split=split,transform=Compose([Rotate(rotation),ToTensor(),normalize])) 
    if split == 'train':
        loader = ClassBalancedSampler(dataset, num_per_class=num_per_class,
                                      num_cl=task.num_classes, num_inst=task.train_num, shuffle=shuffle)
    else:
        loader = ClassBalancedSampler(dataset, num_per_class=num_per_class,
                                      num_cl=task.num_classes, num_inst=task.test_num, shuffle=shuffle)
    # loader = loader.batch(num_per_class*task.num_classes, drop_remainder=True)
    return loader




if __name__ == '__main__':
    SAMPLE_NUM_PER_CLASS = 1
    CLASS_NUM = 5
    BATCH_NUM_PER_CLASS = 19
    metatrain_character_folders,metatest_character_folders = omniglot_character_folders()
    degrees = random.choice([0,90,180,270])
    task = OmniglotTask(metatrain_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
    sample_dataloader = get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False,rotation=degrees)
    batch_dataloader = get_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True,rotation=degrees)

    samples,sample_labels = sample_dataloader.__iter__().__next__()
    batches,batch_labels = batch_dataloader.__iter__().__next__()
    print(sample_labels)
    print(batch_labels)

