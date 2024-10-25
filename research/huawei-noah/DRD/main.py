# import imp
from numpy.core.fromnumeric import argsort
# from train_alg.naive_point import Naive_point
# from train_alg.ips_point import IPS_point
# from train_alg.ips_point_beyas import IPS_point_beyas
# from train_alg.ips_point_affine import IPS_point_affine
# from train_alg.em_ips_affine2 import EM_ips_affine2
# from train_alg.mbc import MBC
# from train_alg.dla_list import DLA_list
from train_alg.DRD import DRD
from train_alg.DRD_ideal import DRD_ideal
# from train_alg.oracle import Oracle
from utils.set_seed import setup_seed
from utils.arg_parser import config_param_parser
import mindspore
import warnings
import os

warnings.filterwarnings("ignore")

def main():
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    #torch.cuda.set_device(1)
    #os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    mindspore.set_context(device_target="CPU")
    parser = config_param_parser()
    args = parser.parse_args()

    if args.train_alg == 'drd':
        learner = DRD(args)
    elif args.train_alg == 'drd_ideal':
        learner = DRD_ideal(args)

    setup_seed(args.randseed)
    learner.train()


if __name__=="__main__":
    print('Start ...')
    main()
    print('End ...')