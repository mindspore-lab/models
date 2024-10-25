import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch', default=256, type=int, help='batch size')
    parser.add_argument('--reg', default=1e-2, type=float, help='weight decay regularizer')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
    parser.add_argument('--save_path', default='temp', help='file name to save model and training record')
    parser.add_argument('--latdim', default=32, type=int, help='embedding size')
    parser.add_argument('--sampNum', default=40, type=int, help='batch size for sampling')
    parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')
    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--shoot', default=10, type=int, help='K of top k')
    parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
    parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
    parser.add_argument('--normalization', default='left', type=str, help='type of normalization')
    parser.add_argument('--encoder', default='lightgcn', type=str, help='type of encoder, selected from lightgcn, gcn, ngcf, gccf')
    parser.add_argument('--num_exps', default=3, type=int, help='number of expert')
    parser.add_argument('--multi_graph', default=True, type=bool,
                        help='whether to use multi graph, True for use, False for not use')
    
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--data', default='beibei', type=str, help='name of dataset')
    parser.add_argument('--mult', default=300, type=float, help='multiplier for the result')
    parser.add_argument('--gpu_id', default=1, type=int, help='gpu id')
    parser.add_argument('--decoder', default='sesg', type=str, help='type of decoder, selected from bilinear, shared_bottom, mmoe, ple, sesg')

    return parser.parse_args()
args = parse_args()
# beibei-gccf tmall-lightgcn

# tianchi
# args.user = 423423
# args.item = 874328
# beibei
args.user = 21716
args.item = 7977
# Tmall
# args.user = 805506#147894
# args.item = 584050#99037
# ML10M
# args.user = 67788
# args.item = 8704
# yelp
# args.user = 19800
# args.item = 22734

args.decay_step = args.trnNum // args.batch
