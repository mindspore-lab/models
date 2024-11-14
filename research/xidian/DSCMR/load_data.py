import mindspore.dataset as ds
from scipy.io import loadmat, savemat
import numpy as np
from pdb import set_trace




class CustomDataSet():
    def __init__(
            self,
            images,
            texts,
            labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        return img, text, label

    def __len__(self):
        count = len(self.images)
        assert len(
            self.images) == len(self.labels)
        return count


def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return np.arange(N) == np.repeat(ind, N, axis=1)

def get_loader(path, batch_size):
    img_train = loadmat(path+"train_img.mat")['train_img']
    img_test = loadmat(path + "test_img.mat")['test_img']
    text_train = loadmat(path+"train_txt.mat")['train_txt']
    text_test = loadmat(path + "test_txt.mat")['test_txt']
    label_train = loadmat(path+"train_img_lab.mat")['train_img_lab']
    label_test = loadmat(path + "test_img_lab.mat")['test_img_lab']


    label_train = ind2vec(label_train).astype(int)
    label_test = ind2vec(label_test).astype(int)

    imgs = {'train': img_train, 'test': img_test}
    texts = {'train': text_train, 'test': text_test}
    labels = {'train': label_train, 'test': label_test}
    #set_trace()
    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['train', 'test']}

    #shuffle = {'train': False, 'test': False}
    #dataloader = {x:ds.GeneratorDataset(list(dataset), column_names=['images', 'texts', 'labels']).batch(batch_size) for x in ['train', 'test']}

    traindataset = CustomDataSet(images=img_train, texts=text_train, labels=label_train)
    traindataloader=ds.GeneratorDataset(list(traindataset), column_names=['images', 'texts', 'labels'],shuffle=False).batch(batch_size)

    testdataset = CustomDataSet(images=img_test, texts=text_test, labels=label_test)
    testdataloader = ds.GeneratorDataset(list(testdataset), column_names=['images', 'texts', 'labels'],shuffle=True).batch(
        batch_size)
    #set_trace()
    #input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    #dataloader = input_data.batch(batch_size, drop_remainder=True)

   # dataloader = {x: input_data.batch(batch_size, drop_remainder=True) for x in ['train', 'test']}


    #dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
    #                            shuffle=shuffle[x], num_workers=0) for x in ['train', 'test']}

    img_dim = img_train.shape[1]
    text_dim = text_train.shape[1]
    num_class = label_train.shape[1]

    input_data_par = {}
    input_data_par['img_test'] = img_test
    input_data_par['text_test'] = text_test
    input_data_par['label_test'] = label_test
    input_data_par['img_train'] = img_train
    input_data_par['text_train'] = text_train
    input_data_par['label_train'] = label_train
    input_data_par['img_dim'] = img_dim
    input_data_par['text_dim'] = text_dim
    input_data_par['num_class'] = num_class
    return traindataloader, testdataloader,input_data_par
