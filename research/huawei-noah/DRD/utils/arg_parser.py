import argparse
from argparse import ArgumentTypeError

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def config_param_parser():
    parser = argparse.ArgumentParser(description="Experiment Configures and Model Parameters")
    
    # Experiment Configures
    parser.add_argument('--fin', required=True)
    parser.add_argument('--fout',required=True)
    parser.add_argument('--use_cuda', type=str2bool, nargs='?', default=True)
    parser.add_argument('--train_alg', type=str, choices=['drd','drd2','drd_ideal'], required=True)
    parser.add_argument('--pairwise', type=str2bool, nargs='?', default=True)
    parser.add_argument('--eval_positions', type=int, nargs='*',default=[1,2,3,5,10,20])
    parser.add_argument('--topK', type=int , default= 10)
    parser.add_argument('--randseed', type=int , default=41)
    parser.add_argument('--continue_tag', type=str2bool, nargs='?', default=False)
    parser.add_argument('--eval_tag', type=str2bool, nargs='?', default=True)
    
    # Model Parameters
    # parser.add_argument('--alpha', type=float, default=1.0, help="contrall correlation")
    parser.add_argument('--min_alpha', type=float, default=1.0, help="control adjustment")
    parser.add_argument('--alpha', type=float, default=1.0, help="control adjustment")
    parser.add_argument('--beta', type=float, default=0.5, help="control regularize")
    parser.add_argument('--init_prob', type=float, default=0.638, help='control the severity of trust bias')
    parser.add_argument('--eta', type=float, default=1.0, help='control the severity of examination bias')
    parser.add_argument('--clip', type=float, default=0.1, help='the clipping threshold of propensity score')
    # parser.add_argument('--ranker_type', type=str, default='nn', choices=['nn', 'linear', 'setrank'])
    # parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'mae', 'bce'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adadelta'])
    parser.add_argument('--schedul_mode', type=str, default='min', choices=['min', 'max'])
    parser.add_argument('--session_num', type=float, default=1e6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--drop_out', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=10, help="waiting patience for early stop")
    parser.add_argument('--hidden_size', type=int, default=32, help="hidden size of latent layer in NN")
    return parser


if __name__=="__main__":
    parser = config_param_parser()
    args = parser.parse_args()
    print(args)
    