import argparse
from functools import partial
import os

from mindspore.nn import Adam
from mindspore.dataset import GeneratorDataset

import pickle
from data import GigaDataset, prepro_batch
from Seq2Seq import Seq2SeqSum
from training import Trainer

#For Training
# 构建vocab
# word2id = {}
# word2id['<pad>'] = 0
# word2id['<unk>'] = 1
# word2id['<start>'] = 2
# word2id['<end>'] = 3
# ind = 4
# for i in range(32, 127):
#     word2id[chr(i)] = ind
#     ind += 1
with open(os.path.join('WMT14', 'raw', 'src_vocab.pkl'), "rb") as f:
    src_vocab = pickle.load(f)
with open(os.path.join('WMT14', 'raw', 'tgt_vocab.pkl'), "rb") as f:
    tgt_vocab = pickle.load(f)

from mindspore.amp import auto_mixed_precision
src_vocab['<pad>']=0#添加填充字符
tgt_vocab['<pad>']=0
def main():
    #get args
    parser = argparse.ArgumentParser(description="Seq2SeqSum Training Program")
    #model args
    #注意数据集标明具体名称即可
    parser.add_argument("--data_path", type=str,
                        default="./WMT14/")
    parser.add_argument("--name", type=str,
                        default="WMT14")


    parser.add_argument("--emb_dim", type=int, default=128)#128
    parser.add_argument("--n_hidden", type=int, default=256)#256
    parser.add_argument("--n_layer", type=int, default=1)



    parser.add_argument("--max_src_len", type=int, default=32)
    parser.add_argument("--max_tgt_len", type=int, default=32)

    #training args
    parser.add_argument("--cuda", action='store_true', default=True)
    parser.add_argument("--batch_size", type=int, default=32)#32
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--clip", type=float, default=5.0)



    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--ckpt_freq", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()

    #保存模型路径
    save_dir='model_'+args.name
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    # make vocab
   # word2id = make_word2id(args.vocab_path, args.vocab_size)
    # id2word = {value:key for key, value in word2id.items()}

    # init data loader
    train_loader=GigaDataset(args.data_path, 'train',args.batch_size,src_vocab,tgt_vocab)

    val_loader=GigaDataset(args.data_path, 'val',args.batch_size,src_vocab,tgt_vocab)

    # train_loader = GeneratorDataset(
    #     GigaDataset(args.data_path, 'train'),
    #     batch_size=args.batch_size,
    #     collate_fn=partial(
    #         prepro_batch,  src_vocab,tgt_vocab
    #         )
    #     )

    # val_loader = DataLoader(
    #     GigaDataset(args.data_path, 'val'),
    #     batch_size=args.batch_size,
    #     collate_fn=partial(
    #         prepro_batch, src_vocab,tgt_vocab
    #         )
    #     )


    # init model


    model = Seq2SeqSum(
        len(src_vocab),len(tgt_vocab), args.emb_dim,
        args.n_hidden, args.n_layer
        )


    model = auto_mixed_precision(model, 'O2')#精度设置
    optimizer = Adam(model.trainable_params(), learning_rate=args.lr)
    trainer = Trainer(optimizer, model, train_loader,
                      val_loader, save_dir,
                      args.clip, args.print_freq, args.ckpt_freq,
                      args.patience,args.epoch )
    trainer.train()

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()
