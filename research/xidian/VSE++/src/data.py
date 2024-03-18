import json as jsonmod
import os
import mindspore.common.dtype as mstype
import mindspore
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import nltk
import numpy as np
from PIL import Image
from mindspore import Tensor, ops


def get_paths(path, name='coco', use_restval=False):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    The indices are extracted from the Karpathy et al. splits using this
    snippet:

    :param name: Dataset names
    :param use_restval: If True, the the `restval` data is included in train.
    """
    roots = {}
    ids = {}
    if 'coco' == name:
        imgdir = os.path.join(path, 'images')
        capdir = os.path.join(path, 'annotations')
        roots['train'] = {
            'img': os.path.join(imgdir, 'train2014'),
            'cap': os.path.join(capdir, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }
        ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
        ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
        ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
        ids['trainrestval'] = (
            ids['train'],
            np.load(os.path.join(capdir, 'coco_restval_ids.npy')))
        if use_restval:
            roots['train'] = roots['trainrestval']
            ids['train'] = ids['trainrestval']
    elif 'f8k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr8k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}
    elif 'f30k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr30k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}

    return roots, ids


class FlickrDataset:
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, json, split, vocab, transform=None):
        self.root = root
        self.vocab = vocab
        self.split = split
        self.transform = transform
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return len(self.ids)


def collate_fn(image, target, index, img_id, batch_info):
    # Sort a data list by caption length
    # image = Tensor(image, dtype=mindspore.float32)
    data = [image, target, index, img_id]
    data = list(tuple(zip(*data)))
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images=list(images)
    images = [np.squeeze(x) for x in images]
    # images = np.concatenate(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    # targets = np.zeros((len(captions), max(lengths)))
    targets = np.zeros((len(captions), 100))

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end].tolist()

    targets = np.split(targets, targets.shape[0])
    targets = [np.squeeze(x,) for x in targets]
    lengths=[np.array(x) for x in lengths]

    return images, targets, lengths, list(ids)


def get_loader_single(data_name, split, root, json, vocab, transform,
                      batch_size=100, shuffle=True,
                      num_workers=2, ids=None, collate_fn=collate_fn):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dataset = FlickrDataset(root=root,
                            split=split,
                            json=json,
                            vocab=vocab,
                            transform=transform)

    # Data loader
    data_loader = mindspore.dataset.GeneratorDataset(source=dataset,
                                                     shuffle=shuffle,
                                                     column_names=["image", "target", "index", "img_id"],
                                                     num_parallel_workers=num_workers)
    data_loader = data_loader.batch(batch_size, drop_remainder=True, per_batch_map=collate_fn)
    print(split + "=======================")
    return data_loader


def get_transform(data_name, split_name, opt):
    normalizer = vision.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225], is_hwc=False)
    t_list = []
    if split_name == 'train':
        t_list = [vision.RandomResizedCrop(opt.crop_size),
                  vision.RandomHorizontalFlip()]
    elif split_name == 'val':
        t_list = [vision.Resize(256), vision.CenterCrop(224)]
    elif split_name == 'test':
        t_list = [vision.Resize(256), vision.CenterCrop(224)]

    t_end = [vision.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform
class PrecompDataset():
    """
    Load precomputed captions and image features
    Possible options: f8k, f30k, coco, 10crop
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'

        # Captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'r') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index/self.im_div
        image = mindspore.Tensor(self.images[int(img_id)])
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = mindspore.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return self.length
def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab)

    data_loader = mindspore.dataset.GeneratorDataset(source=dset,
                                                     shuffle=shuffle,
                                                     column_names=["image", "target", "index", "img_id"],
                                                     num_parallel_workers=num_workers)
    data_loader = data_loader.batch(batch_size, drop_remainder=True, per_batch_map=collate_fn)
    return data_loader
def get_loaders(data_name, vocab, crop_size, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)

    if opt.data_name.endswith('_precomp'):
        train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                          batch_size, True, workers)
        val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                        batch_size, False, workers)
    else:

        roots, ids = get_paths(dpath, data_name, opt.use_restval)

        transform = get_transform(data_name, 'train', opt)
        train_loader = get_loader_single(opt.data_name, 'train',
                                         roots['train']['img'],
                                         roots['train']['cap'],
                                         vocab, transform, ids=ids['train'],
                                         batch_size=batch_size, shuffle=True,
                                         num_workers=workers,
                                         collate_fn=collate_fn)
        transform = get_transform(data_name, 'test', opt)
        val_loader = get_loader_single(opt.data_name, 'test',
                                       roots['test']['img'],
                                       roots['test']['cap'],
                                       vocab, transform, ids=ids['test'],
                                       batch_size=batch_size, shuffle=False,
                                       num_workers=workers,
                                       collate_fn=collate_fn)

    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, crop_size, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    if opt.data_name.endswith('_precomp'):
        test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                         batch_size, False, workers)
    else:
        roots, ids = get_paths(dpath, data_name, opt.use_restval)

        transform = get_transform(data_name, split_name, opt)
        test_loader = get_loader_single(opt.data_name, split_name,
                                        roots[split_name]['img'],
                                        roots[split_name]['cap'],
                                        vocab, transform, ids=ids[split_name],
                                        batch_size=batch_size, shuffle=False,
                                        num_workers=workers,
                                        collate_fn=collate_fn)

    return test_loader
