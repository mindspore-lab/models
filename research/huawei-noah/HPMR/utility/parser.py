import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run HPMR.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='./Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='Beibei',
                        help='Choose a dataset from {Beibei,Taobao}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--is_norm', type=int, default=1,
                    help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,   #common parameter
                        help='Embedding size.')
    
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    
    parser.add_argument('--lr', type=float, default=0.001,   #common parameter
                        help='Learning rate.')

    parser.add_argument('--model_type', nargs='?', default='HPMR',
                        help='Specify the name of model (lightgcn,ghcf).') 
    
    parser.add_argument('--adj_type', nargs='?', default='pre',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')

    parser.add_argument('--gpu_id', type=int, default=1,
                        help='Gpu id')

    parser.add_argument('--Ks', nargs='?', default='[10]',
                        help='K for Top-K list')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    parser.add_argument('--memosize', default=2, type=int, help='memory size')

    # HPMR parameters

    parser.add_argument('--wid', nargs='?', default='[0.1,0.1,0.1]',
                        help='negative weight, [0.1,0.1,0.1] for beibei, [0.01,0.01,0.01] for taobao')

    parser.add_argument('--decay', type=float, default=10,
                        help='Regularization, 10 for beibei, 0.01 for taobao')

    parser.add_argument('--coefficient', nargs='?', default='[0.0/6, 5.0/6, 1.0/6]',
                        help='Regularization, [0.0/6, 5.0/6, 1.0/6] for beibei, [1.0/6, 4.0/6, 1.0/6] for taobao')

    parser.add_argument('--mess_dropout', nargs='?', default='[0.2]',
                        help='Keep probability w.r.t. message dropout, 0.2 for beibei and taobao')

    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
   
    parser.add_argument('--encoder', default='gccf', type=str, 
                        help='type of encoder, selected from lightgcn, gcn, ngcf, gccf')
    
    parser.add_argument('--alpha', nargs='?', default='[1, 0, 3]',
                        help='Number of alpha.')

    parser.add_argument('--gnn_layer', default=3, type=int, help='number of gnn layers')

    parser.add_argument('--transfer_gnn_layer', default=1, type=int, 
                        help='number of transfer_gnn layers')

    parser.add_argument('--trans_user', type=int, default=1,
                        help='Whether to transfer user.')
    
    parser.add_argument('--re_mult', type=float, default=2,
                        help='The number of mult of re-enhance.')    

    # parser.add_argument('--rank', default=4, type=int, help='ranks')   
    
    # parser.add_argument('--att_head', default=2, type=int, help='number of attention heads')    

    # parser.add_argument('--sampNum', default=40, type=int, help='batch size for sampling')

    # parser.add_argument('--keepRate', default=0.7, type=float, help='rate for dropout')

    parser.add_argument('--save_emb', type=int, default=0, help='whether to save the embedding')
    
    # parser.add_argument('--atten', type=str, default='', help='attention type')
    return parser.parse_args()

args = parse_args()


if args.dataset == 'Beibei':
    args.wid = '[0.1,0.1,0.1]'
    # args.coefficient = '[0.0/6, 5.0/6, 1.0/6]'
    print('setting for Beibei')
    
    
elif args.dataset == 'Taobao':
    args.wid = '[0.01,0.01,0.01]'
    args.decay = 0.01
    # args.coefficient = '[1.0/6, 4.0/6, 1.0/6]'
    print('setting for Taobao')
    
