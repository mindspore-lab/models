import argparse
import yaml
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.communication import get_group_size, get_rank, init

from .Exp_Informer import Exp_Informer
from .Exp_Autoformer import Exp_Autoformer
from .Exp_FEDformer import Exp_FEDformer
from .Exp_TFT import Exp_TFT
from .Exp_Nbeats import Exp_Nbeats
from .Exp_Nbeatsx import Exp_Nbeatsx
from .Exp_JAT import Exp_JAT
from .Exp_ALLOT import Exp_ALLOT
from .Exp_DTRD import Exp_DTRD
from ..utils import set_seed

model_entrypoints = {
    'Informer':Exp_Informer,
    'Autoformer':Exp_Autoformer,
    'FEDformer':Exp_FEDformer,
    'TFT':Exp_TFT,
    'Nbeats': Exp_Nbeats,
    'Nbeatsx': Exp_Nbeatsx,
    'JAT': Exp_JAT,
    'ALLOT': Exp_ALLOT,
    'DTRD': Exp_DTRD,
}

class MindSeqModel():
    def __init__(self,exp,args):
        self.exp = exp
        # self.model = None
        self.args = args
        self.setting = "Setting not set"
        if self.args.model in ['Informer','Autoformer','FEDformer','JAT']:
            # self.model = exp._get_model()
            self.setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
                    self.args.model, self.args.data, self.args.features, 
                    self.args.seq_len, self.args.label_len, self.args.pred_len,
                    self.args.d_model, self.args.n_heads, self.args.e_layers, self.args.d_layers, self.args.d_ff, self.args.attn, self.args.factor, 
                    self.args.embed, self.args.distil, self.args.mix, self.args.des, 0)

    def train(self,itr=0):
        if self.args.model in ['Informer','Autoformer','FEDformer','JAT']:
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(self.setting))
            self.setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
                    self.args.model, self.args.data, self.args.features, 
                    self.args.seq_len, self.args.label_len, self.args.pred_len,
                    self.args.d_model, self.args.n_heads, self.args.e_layers, self.args.d_layers, self.args.d_ff, self.args.attn, self.args.factor, 
                    self.args.embed, self.args.distil, self.args.mix, self.args.des, itr)
            self.exp.train(self.setting)
        elif self.args.model in ['TFT', 'Nbeats', 'Nbeatsx', 'ALLOT', 'DTRD']:
            self.exp.train()
    
    def test(self,itr=0):
        if self.args.model in ['Informer','Autoformer','FEDformer','JAT']:
            print('>>>>>>>start testing : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(self.setting))
            self.setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
                        self.args.model, self.args.data, self.args.features, 
                        self.args.seq_len, self.args.label_len, self.args.pred_len,
                        self.args.d_model, self.args.n_heads, self.args.e_layers, self.args.d_layers, self.args.d_ff, self.args.attn, self.args.factor, 
                        self.args.embed, self.args.distil, self.args.mix, self.args.des, itr)
            self.exp.test(self.setting)
        elif self.args.model in ['TFT', 'Nbeats', 'Nbeatsx', 'ALLOT', 'DTRD']:
            self.exp.test()
    
    def load(self):
        if self.args.ckpt_path:
            if self.args.model not in ['Nbeatsx', 'DTRD']:
                print('>>>Loading from : {}>>'.format(self.args.ckpt_path))
                paras = ms.load_checkpoint(self.args.ckpt_path)
                ms.load_param_into_net(self.exp.model, ms.load_checkpoint(self.args.ckpt_path))
                print('>>>>>>>Load Succeed!>>>>>>>>>>>>>>>>>>>>>>>>>>')
                # ms.save_checkpoint(self.exp.model, "./checkpoints/test_ckpt/ALLOT_PEMS08.ckpt")
        else:
            raise RuntimeError('Need ckpt_path but get None')


def is_model(model_name):
    return model_name in model_entrypoints.keys()

def create_model(
        model_name: str,
        data_name: str,
        pretrained: bool = False,
        checkpoint_path: str='',
        config_file: str='',
        **kwargs,
):
    if checkpoint_path == "" and pretrained:
        raise ValueError("checkpoint_path is mutually exclusive with pretrained")
    
    # 创建一个命名空间对象
    args = argparse.Namespace(
        model_name=model_name,
        data_name=data_name,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        config_file = config_file,
        **kwargs  # 添加 **kwargs 中的参数
    )
    if args.config_file:
        with open(config_file,'r') as f:
            config_data = yaml.safe_load(f)
            args.__dict__.update(config_data)
    if hasattr(args, 'model'):
        args.model = model_name
    if hasattr(args, 'data'):
        args.data = data_name
    data_parser = {
        'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'weather':{'data':'weather.csv','T':'OT','M':[21,21,21],'S':[1,1,1],'MS':[21,21,1]},
    }
    if hasattr(args, 'data'):
        if args.data in data_parser.keys():
            data_info = None
            if hasattr(args, 'data'):
                data_info = data_parser[args.data]
            if hasattr(args, 'data_path') and hasattr(args, 'target'):
                args.data_path = data_info['data']
                args.target = data_info['T']
            if hasattr(args, 'enc_in') and hasattr(args, 'dec_in') and hasattr(args, 'c_out') and hasattr(args, 'features'):
                args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    print('Args in experiment:')
    print(args)

    # set mode
    if hasattr(args, 'device'):
        ms.set_context(device_target=args.device)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    if hasattr(args, 'distribute'):
        if args.distribute:
            init()
            if hasattr(args, 'device_num') and hasattr(args, 'rank_id'):
                args.device_num = get_group_size()
                args.rank_id = get_rank()
            if hasattr(args, 'device_num'):
                ms.set_auto_parallel_context(
                    device_num=args.device_num,
                    parallel_mode="data_parallel",
                    gradients_mean=True,
                    # we should but cannot set parameter_broadcast=True, which will cause error on gpu.
                )
        elif hasattr(args, 'device_num') and hasattr(args, 'rank_id'):
            args.device_num = None
            args.rank_id = None
    if hasattr(args, "seed"):
        set_seed(args.seed)

    # Check model_name
    if not is_model(args.model_name):
        raise RuntimeError(f'Unknow model {args.model_name}, options:{model_entrypoints.keys()}')
    
    exp = model_entrypoints[args.model_name](args)
    Exp = MindSeqModel(exp,args)

    if args.pretrained==True and args.checkpoint_path!='':
        Exp.load()
    
    return Exp

    # setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
    #             args.model, args.data, args.features, 
    #             args.seq_len, args.label_len, args.pred_len,
    #             args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
    #             args.embed, args.distil, args.mix, args.des, 0)
    # print("Setting:", setting)
    # exp = model_entrypoints[args.model_name](args)
    # model = exp._get_model()
    # exp.train(model, setting)

    # if pretrained and checkpoint_path!="":
    #     pass
    # return args

if __name__ == '__main__':
    print(create_model('informer','ETTh1',pretrained = False,config_file = '../../configs/informer/informer_GPU.yaml'))