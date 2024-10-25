import sys
sys.path.append('../')
from mindspore import nn
from mindspore.train import Model
from HAMUR.utils.data import Dataset_pre, create_dataset
from HAMUR.basic.features import DenseFeature, SparseFeature
from HAMUR.models.adapter import Mlp7Layer, MlpAdap7Layer2Adp
from HAMUR.models.adapter_dcn import DcnMd, DcnMdAdp
from HAMUR.models.adapter_wd import WideDeepMd, WideDeepMdAdp
import pandas as pd


def get_ali_ccp_data_dict(data_path='./data/ali-ccp'):
    df_train = pd.read_csv(data_path + '/ali_ccp_train_sample.csv')
    df_val = pd.read_csv(data_path + '/ali_ccp_val_sample.csv')
    df_test = pd.read_csv(data_path + '/ali_ccp_test_sample.csv')
    print("train : val : test = %d %d %d" % (len(df_train), len(df_val), len(df_test)))
    train_idx, val_idx = df_train.shape[0], df_train.shape[0] + df_val.shape[0]
    data = pd.concat([df_train, df_val, df_test], axis=0)
    domain_map = {1: 0, 2: 1, 3: 2}
    domain_num = 3
    data["domain_indicator"] = data["301"].apply(lambda fea: domain_map[fea])

    col_names = data.columns.values.tolist()
    dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
    sparse_cols = [col for col in col_names if col not in dense_cols and col not in ['click', 'purchase']]
    print("sparse cols:%d dense cols:%d" % (len(sparse_cols), len(dense_cols)))

    dense_feas = [DenseFeature(col) for col in dense_cols]
    sparse_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in sparse_cols]

    y = data["click"]
    del data["click"]
    x = data
    x_train, y_train = x[:train_idx], y[:train_idx]
    x_val, y_val = x[train_idx:val_idx], y[train_idx:val_idx]
    x_test, y_test = x[val_idx:], y[val_idx:]
    return dense_feas, sparse_feas, x_train, y_train, x_val, y_val, x_test, y_test, domain_num


def main(dataset_path, model_name, epoch, learning_rate, batch_size):
    (dense_feas, sparse_feas, x_train, y_train, x_val, y_val,
     x_test, y_test, domain_num) = get_ali_ccp_data_dict(dataset_path)
    data_train_pre = Dataset_pre(x_train, y_train)
    # data_val_pre = Dataset_pre(x_val, y_val)
    data_test_pre = Dataset_pre(x_test, y_test)

    dataset_train = create_dataset(data_train_pre, batch_size=batch_size, drop_remainder=True)
    # dataset_val = create_dataset(data_val_pre, batch_size=batch_size, drop_remainder=True)
    dataset_test = create_dataset(data_test_pre, batch_size=batch_size, drop_remainder=True)

    if model_name == "mlp":
        net = Mlp7Layer(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[1024, 512, 512, 256, 256, 64, 64])
    elif model_name == "mlp_adp":
        net = MlpAdap7Layer2Adp(dense_feas + sparse_feas, domain_num=domain_num,
                                fcn_dims=[1024, 512, 512, 256, 256, 64, 64], hyper_dims=[64], k=65)
    elif model_name == "dcn_md":
        net = DcnMd(features=dense_feas + sparse_feas, num_domains=domain_num, n_cross_layers=7,
                    mlp_params={"dims": [512, 512, 256, 256, 64, 64]})
    elif model_name == "dcn_md_adp":
        net = DcnMdAdp(features=dense_feas + sparse_feas, num_domains=domain_num, n_cross_layers=7,
                       k=25, mlp_params={"dims": [512, 512, 256, 256, 64, 64]}, hyper_dims=[64])
    elif model_name == "wd_md":
        net = WideDeepMd(wide_features=dense_feas, deep_features=sparse_feas, num_domains=domain_num,
                         mlp_params={"dims": [512, 512, 256, 256, 64, 64], "dropout": 0.2, "activation": "relu"})
    elif model_name == "wd_md_adp":
        net = WideDeepMdAdp(wide_features=dense_feas, deep_features=sparse_feas, num_domains=domain_num, k=25,
                            mlp_params={"dims": [512, 512, 256, 256, 64, 64], "dropout": 0.2, "activation": "relu"},
                            hyper_dims=[64])
    else:
        raise Exception('recheck the mode name')
    loss = nn.BCELoss()
    optim = nn.Adam(params=net.trainable_params(), learning_rate=learning_rate, weight_decay=0.0)

    model = Model(net, loss_fn=loss, optimizer=optim, metrics={'loss'})

    model.train(epoch, dataset_train)  # num_epochs是训练的轮数，往往训练多轮才能使模型收敛

    result = model.eval(dataset_test)
    print(result)


if __name__ == '__main__':
    import argparse
    import warnings
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="data/ali-ccp")
    parser.add_argument('--model_name', default='wd_md_adp')
    parser.add_argument('--epoch', type=int, default=1)  # 200
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=50)  # 4096*10

    args = parser.parse_args()
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate,
         args.batch_size)
"""
python run_ali_ccp_ctr_ranking_multi_domain.py --model_name widedeep
python run_ali_ccp_ctr_ranking_multi_domain.py --model_name deepfm
python run_ali_ccp_ctr_ranking_multi_domain.py --model_name dcn
"""
