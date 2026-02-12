""" config """
import os
import argparse
import yaml

def create_parser():
    """create parser"""
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    parser_config = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser_config.add_argument('-c', '--config', type=str, default='',
                               help='YAML config file specifying default arguments (default="")')

    # The main parser. It inherits the --config argument for better help information.
    parser = argparse.ArgumentParser(description='MindSeq Training', parents=[parser_config])

    parser.add_argument('--model', type=str, required=True, default='informer',
                        help='model name, options: [Autoformer, Informer, Transformer]')
    parser.add_argument('--distribute', action='store_true', default=False)
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
    parser.add_argument('--root_path', type=str, default='./data/ETT/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; \
                        M:multivariate predict multivariate, \
                        S:univariate predict univariate, \
                        MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, \
                        options:[s:secondly, t:minutely, h:hourly, d:daily, \
                        b:business days, w:weekly, m:monthly], \
                        you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--detail_freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, \
                        h:hourly, d:daily, b:business days, w:weekly, m:monthly], \
                        you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='location of model checkpoints')
    parser.add_argument('--seq_len', type=int, default=96,
                        help='input sequence length of Informer encoder')
    parser.add_argument('--label_len', type=int, default=48,
                        help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--ceof', type=int, default=0.5)
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--padding', type=int, default=0, help='padding type')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, \
                        using this argument means not using distilling', default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--attn', type=str, default='prob',
                        help='attention used in encoder, options:[prob, full]')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu',help='activation')
    parser.add_argument('--output_attention', action='store_true',
                        help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true',
                        help='whether to predict unseen future data')
    parser.add_argument('--mix', action='store_false',
                        help='use mix attention in generative decoder', default=True)
    parser.add_argument('--cols', type=str, nargs='+',
                        help='certain cols from the data files as the input features')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test',help='exp description')
    parser.add_argument('--loss', type=str, default='mse',help='loss function')
    parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true',
                        help='use automatic mixed precision training', default=False)
    parser.add_argument('--inverse', action='store_true', help='inverse output data',default=False)
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='use multiple gpus', default=False)
    parser.add_argument('--device', type=str,default='CPU', help='device')
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='device ids of multile gpus')
    parser.add_argument('--device_num',type=int,default=None,help='distribute device num')
    parser.add_argument('--rank_id',type=int,default=None,help='distribute device rank id')
    parser.add_argument('--seed',type=int,default=42,help='seed')
    parser.add_argument('--do_train',action='store_true',default=True,help='pretrained or not')
    parser.add_argument('--ckpt_path',type=str,default='',help='path of pretrained checkpoints')
    parser.add_argument('--longforecast', type=int, default=0, help='super long forecasting')

    return parser_config, parser

def _check_cfgs_in_parser(cfgs: dict, parser: argparse.ArgumentParser):
    actions_dest = [action.dest for action in parser._actions]
    defaults_key = parser._defaults.keys()
    for k in cfgs.keys():
        if k not in actions_dest and k not in defaults_key:
            raise KeyError(f"{k} does not exist in ArgumentParser!")


def parse_args(args=None):
    '''parse args'''
    parser_config, parser = create_parser()
    # Do we have a config file to parse?
    args_config, remaining = parser_config.parse_known_args(args)
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            #_check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
            parser.set_defaults(config=args_config.config)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    return args

def save_args(args: argparse.Namespace, filepath: str, rank: int = 0) -> None:
    """If in master process, save ``args`` to a YAML file. Otherwise, do nothing.
    Args:
        args (Namespace): The parsed arguments to be saved.
        filepath (str): A filepath ends with ``.yaml``.
        rank (int): Process rank in the distributed training. Defaults to 0.
    """
    assert isinstance(args, argparse.Namespace)
    assert filepath.endswith(".yaml")
    if rank != 0:
        return
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as f:
        yaml.safe_dump(args.__dict__, f)
