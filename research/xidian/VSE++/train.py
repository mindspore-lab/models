import argparse
import logging
import os

import pickle

import mindspore
from mindspore import context

import src.data as data
from models.model import ContrastiveLoss
from models.model import VSE, NetWithLoss
from src.trainer import Trainer
from src.vocab import Vocabulary  # NOQA


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data',
                        help='path to datasets')
    parser.add_argument('--data_name', default='f30k_precomp',
                        help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=1, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=200, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', default=True,
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=4096, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune the image encoder.')
    parser.add_argument('--cnn_type', default='vgg19',
                        help="""The CNN used for image encoder
                        (e.g. vgg19, resnet152)""")
    parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--reset_train', action='store_true',
                        help='Ensure the training is always done in '
                             'train mode (Not recommended).')
    parser.add_argument('--seed', default=123)
    parser.add_argument('--device_id', default=5)
    parser.add_argument('--device', default="Ascend")
    opt = parser.parse_args()
    print(opt)
    mindspore.set_seed(int(opt.seed))

    mindspore.set_context(pynative_synchronize=True)
    context.set_context(mode=mindspore.GRAPH_MODE, device_target=opt.device,
                        enable_graph_kernel=False)  # mindspore.PYNATIVE_MODE 动态   mindspore.GRAPH_MODE 静态
    if opt.device=="Ascend":
        context.set_context(device_id=int(opt.device_id))

    # Load Vocabulary Wrapper
    vocab = pickle.load(open(os.path.join(
        opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
    opt.vocab_size = len(vocab)

    # Load data loaders
    train_loader, val_loader = data.get_loaders(
        opt.data_name, vocab, opt.crop_size, opt.batch_size, opt.workers, opt)
    network = VSE(opt)

    criterion = ContrastiveLoss(margin=opt.margin,
                                measure=opt.measure,
                                max_violation=opt.max_violation)
    network_with_loss = NetWithLoss(network, criterion)
    trainer = Trainer(opts=opt, net=network_with_loss, loss=criterion, train_dataset=train_loader, loss_scale=1,
                      eval_dataset=val_loader)
    trainer.train(opt.num_epochs)


if __name__ == '__main__':
    main()
