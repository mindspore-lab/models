from os.path import join

import argparse
import mindspore
import pickle
import codecs
import os
from mindspore.dataset import GeneratorDataset
from functools import partial

class GigaDataset():
    def __init__(self, path, split,batch_size,src_word,tgt_word):
        """
        args:
        path: path to dataset
        split: train/val/test
        """
        self.batch_size=batch_size
        self.src_word=src_word
        self.tgt_word=tgt_word

        assert split in ['train', 'val', 'test']
        if split=='train':
            with codecs.open(os.path.join(path, 'raw','src_train.txt'), "r", encoding="utf-8") as f:
                source_data = f.readlines()[:]
            #target_data是输出序列 空格分隔
            with codecs.open(os.path.join(path, 'raw','tgt_train.txt'), "r", encoding="utf-8") as f:
                target_data = f.readlines()[:]
            source_lengths = [len(s) for s in source_data]  # 获得每个元素进行bpe划分之后的长度
            target_lengths = [len(t) + 2 for t in target_data]  # target language sequences have <BOS> and <EOS> tokens
            data = zip(source_data, target_data, source_lengths,target_lengths)
            data=sorted(data,key=lambda x: x[2])
            source_data=[x[0] for x in data]
            target_data=[x[1] for x in data]
        elif split=='val':
            with codecs.open(os.path.join(path, 'raw','src_val.txt'), "r", encoding="utf-8") as f:
                source_data = f.readlines()[:]
            #target_data是输出序列 空格分隔
            with codecs.open(os.path.join(path, 'raw','tgt_val.txt'), "r", encoding="utf-8") as f:
                target_data = f.readlines()[:]
        elif split=='test':
            with codecs.open(os.path.join(path, 'raw','src_test.txt'), "r", encoding="utf-8") as f:
                source_data = f.readlines()[:]
            #target_data是输出序列 空格分隔
            with codecs.open(os.path.join(path, 'raw','tgt_test.txt'), "r", encoding="utf-8") as f:
                target_data = f.readlines()[:]

        source_data=[list(x.rstrip('\r\n').split(' ')) for x in source_data]
        target_data=[list(x.rstrip('\r\n').split(' ')) for x in target_data]

        self.path = path
        self.src = source_data
        self.tgt = target_data
        assert len(self.src) == len(self.tgt)
        self.cur_ind=0
        self.tot_batch=len(self.src)//self.batch_size

    def __len__(self):
        return len(self.src)

    def next_batch(self):
        src,tgt=[],[]
        if self.cur_ind+self.batch_size>=len(self.src):
            self.cur_ind=0
        upper=min(self.cur_ind+self.batch_size,len(self.src))
        for i in range(self.cur_ind,upper):
            src.append([int(x) for x in self.src[i]])
            tgt.append([int(x) for x in self.tgt[i]])
        if self.cur_ind+self.batch_size>=len(self.src):
            self.cur_ind=0
        else:
            self.cur_ind+=self.batch_size
        src,tgt=prepro_batch(self.src_word,self.tgt_word,[src,tgt])
        return src, tgt

def prepro_batch( src_word,tgt_word, batch,):

    # def sort_key(src_tgt):
    #     return (len(src_tgt[0]), len(src_tgt[1]))
    # batch.sort(key=sort_key, reverse=True)

    sources, abstract = batch

    inp_lengths = mindspore.Tensor([len(s) for s in sources],dtype=mindspore.int64)

    tgt = [[tgt_word["<s>"]] + t for t in abstract]
    target = [t + [tgt_word["</s>"]] for t in abstract]

    #tensorize
    sources = tensorized(sources, src_word)
    tgt = tensorized(tgt, tgt_word)
    target = tensorized(target, tgt_word)

    return (sources, inp_lengths, tgt), target

def tensorized(sents_batch, word2id):
    """return [batch_size, max_lengths] tensor"""

    batch_size = len(sents_batch)
    max_lengths = max(len(sent) for sent in sents_batch)
    PAD, UNK = word2id['<pad>'], word2id['<unk>']
    batch = mindspore.ops.ones((batch_size, max_lengths), dtype=mindspore.int64) * PAD

    for sent_i, sent in enumerate(sents_batch):
        for word_i, word in enumerate(sent):
            batch[sent_i, word_i] =mindspore.Tensor(word, dtype=mindspore.int64)

    return batch

if __name__=='__main__':
    # get args
    parser = argparse.ArgumentParser(description="Seq2SeqSum Training Program")
    # model args
    # 注意数据集标明具体名称即可
    parser.add_argument("--data_path", type=str,
                        default="./WMT14/")
    parser.add_argument("--name", type=str,
                        default="WMT14")

    parser.add_argument("--emb_dim", type=int, default=128)  # 128
    parser.add_argument("--n_hidden", type=int, default=256)  # 256
    parser.add_argument("--n_layer", type=int, default=1)

    parser.add_argument("--max_src_len", type=int, default=32)
    parser.add_argument("--max_tgt_len", type=int, default=32)

    # training args
    parser.add_argument("--cuda", action='store_true', default=True)
    parser.add_argument("--batch_size", type=int, default=32)  # 32
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--clip", type=float, default=5.0)

    parser.add_argument("--print_freq", type=int, default=1000)
    parser.add_argument("--ckpt_freq", type=int, default=500)
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()
    with open(os.path.join('WMT14', 'raw', 'src_vocab.pkl'), "rb") as f:
        src_vocab = pickle.load(f)
    with open(os.path.join('WMT14', 'raw', 'tgt_vocab.pkl'), "rb") as f:
        tgt_vocab = pickle.load(f)
    src_vocab['<pad>'] = 0  # 添加填充字符
    tgt_vocab['<pad>'] = 0
    train_loader=GigaDataset(args.data_path, 'train',args.batch_size,src_vocab,tgt_vocab)
    print(args.batch_size)
    for i in range(train_loader.tot_batch):
        srcs, targets=train_loader.next_batch()
        print(srcs[0][0].shape)