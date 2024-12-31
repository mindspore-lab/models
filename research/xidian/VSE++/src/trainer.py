import datetime

import mindspore as ms
import numpy
import numpy as np
from mindspore import nn

from models.model import CustomTrainOneStepCell


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).broadcast_to(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).broadcast_to(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
        d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


class Trainer:
    """一个有两个loss的训练示例"""

    def __init__(self, opts, net, loss, train_dataset, loss_scale=1.0, eval_dataset=None):
        self.opts = opts
        self.loss = loss
        self.train_dataset = train_dataset
        self.train_data_size = self.train_dataset.get_dataset_size()  # 获取训练集batch数
        self.net = net

        milestone = [int(self.opts.lr_update) * self.train_data_size,30*self.train_data_size, self.opts.num_epochs * self.train_data_size]
        learning_rates = [self.opts.learning_rate, self.opts.learning_rate / 10.0,self.opts.learning_rate / 20.0]
        self.lr = nn.piecewise_constant_lr(milestone=milestone,
                                           learning_rates=learning_rates)
        self.opt = nn.Adam(self.net.trainable_params(), learning_rate=self.lr)

        self.net_with_loss = CustomTrainOneStepCell(self.net, self.opt)

        self.run_eval = eval_dataset is not None
        if self.run_eval:
            self.eval_dataset = eval_dataset
            self.best_currscore = 0

    def encode_data(self, data_loader):
        """Encode all images and captions loadable by `data_loader`
        """
        self.net_with_loss.set_train(False)
        ds=data_loader.dataset.get_dataset_size()
        img_embs = None
        cap_embs = None
        for i, val_data in enumerate(data_loader):
            image = val_data['image'].astype(ms.float32)
            target = val_data['target'].astype(ms.int64)
            index = val_data['index']
            img_id = val_data['img_id']
            if i % 10 == 0:
                print(f"{i}/{ds}")
            img_emb, cap_emb = self.net.backbone_network(image, target, index)
            if img_embs is None:
                img_embs = np.zeros((ds * data_loader.dataset.batch_size, img_emb.shape[1]))
                cap_embs = np.zeros((ds * data_loader.dataset.batch_size, cap_emb.shape[1]))

            for i, id in enumerate(img_id):
                img_embs[id] = img_emb.asnumpy().copy()[i]
                cap_embs[id] = cap_emb.asnumpy().copy()[i]

        return img_embs, cap_embs

    def train(self, epochs):
        def validata(img_embs, cap_embs):
            (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, measure=self.opts.measure)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                  (r1, r5, r10, medr, meanr))
            # image retrieval
            (r1i, r5i, r10i, medri, meanri) = t2i(
                img_embs, cap_embs, measure=self.opts.measure)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                  (r1i, r5i, r10i, medri, meanri))
            # sum of recalls to be used for early stopping
            currscore = r1 + r5 + r10 + r1i + r5i + r10i
            print("rsum: ", currscore)
            return currscore

        train_dataset = self.train_dataset.create_dict_iterator()
        self.net_with_loss.set_train(True)
        for epoch in range(epochs):
            print(f"================epoch :{epoch + 1}================", flush=True)
            start_time = datetime.datetime.now()  # 记录每个epoch的开始时间
            # 训练一个epoch
            for batch, train_data in enumerate(train_dataset):
                image = train_data['image'].astype(ms.float32)
                target = train_data['target'].astype(ms.int64)

                index = train_data['index']
                img_id = train_data['img_id']
                loss = self.net_with_loss(image, target, index, img_id)
                if batch % self.opts.log_step == 0:
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{current_time}] step: [{batch + 1}/{self.train_data_size}] "
                          f"loss: {loss} lr: {float(self.lr[epoch*self.train_data_size+batch]):.5f}", flush=True)
            end_time = datetime.datetime.now()  # 记录每个epoch的结束时间
            ds=train_dataset.dataset.get_dataset_size()
            print("Per step costs time(ms):",(end_time - start_time)/ds)  # 将每个step的时间段记录下来
            # 推理并保存最好的那个checkpoint
            if self.run_eval:
                eval_dataset = self.eval_dataset.create_dict_iterator()
                img_embs, cap_embs = self.encode_data(eval_dataset)
                currscore = validata(img_embs, cap_embs)
                if currscore >= self.best_currscore:
                    # 保存最好的那个checkpoint
                    self.best_currscore = currscore
                    ms.save_checkpoint(self.net_with_loss, "best.ckpt")
                    print(f"Updata best score: {currscore}")
                self.net_with_loss.set_train(True)
                print(f"Best score: {self.best_currscore}\n")
