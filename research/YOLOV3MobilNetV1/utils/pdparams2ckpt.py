import paddle
import mindspore as ms
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdparams_path', default='', type=str, help='path of pdparams file')
    parser.add_argument('--ckpt_path', default='', type=str, help='path of ckpt file')
    opt = parser.parse_args()
    ckpt = []
    param_dict = paddle.load(opt.pdparams_path)
    for k, v in param_dict.items():
        if '_mean' in k:
            k = k.replace('_mean', 'moving_mean')
        if '_variance' in k:
            k = k.replace('_variance', 'moving_variance')
        if 'batch_norm' in k:
            if 'weight' in k:
                k = k.replace('weight', 'gamma')
            if 'bias' in k:
                k = k.replace('bias', 'beta')
        ckpt.append({"name": k, "data": ms.Tensor(v.numpy())})
    ms.save_checkpoint(ckpt, opt.ckpt_path)
    print('completed')
