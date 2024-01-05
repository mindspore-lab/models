"""train scripts"""
import os
import mindspore as ms
from config import parse_args
from mindseq.models import create_model

def train(args):
    """main train function"""
    # set mode
    ms.set_context(device_target=args.device)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    print("Device:", args.device)

    # create Exp
    exp = create_model(model_name=args.model, data_name=args.data, \
                       pretrained=not args.do_train, checkpoint_path=args.ckpt_path,**vars(args))
    if args.do_train or args.ckpt_path=='':
        if not os.path.exists("./checkpoints/train_ckpt"):
            os.mkdir("./checkpoints/train_ckpt")
        for i in range(args.itr):
            exp.train(i)
            if args.model not in ['DTRD']:
                exp.test(i)
    else:
        exp.test(0)

if __name__ == "__main__":
    my_args = parse_args()
    # 命令行和配置文件可同时输入，以命令行为准

    # core train
    train(my_args)
