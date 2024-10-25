import sys
sys.path.append('../')
from HAMUR.utils.data import Dataset_pre, create_dataset
from HAMUR.basic.features import DenseFeature, SparseFeature
from HAMUR.models.adapter import Mlp2Layer, MlpAdap2Layer1Adp
from HAMUR.models.adapter_dcn import DcnMd, DcnMdAdp
from HAMUR.models.adapter_wd import WideDeepMd, WideDeepMdAdp
from tqdm import tqdm
from mindspore import nn
from mindspore.train import Model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd


def get_movielens_data_rank_multi_domain(data_path="examples/ranking/data/ml-1m"):
    data = pd.read_csv(data_path + "/ml-1m-sample.csv")
    data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])
    del data["genres"]

    group1 = {1, 18}
    group2 = {25}
    group3 = {35, 45, 50, 56}

    domain_num = 3

    data["domain_indicator"] = data["age"].apply(lambda x: map_group_indicator(x, [group1, group2, group3]))

    useless_features = ['title', 'timestamp']

    dense_features = ['age']
    sparse_features = ['user_id', 'movie_id', 'gender', 'occupation', 'zip', "cate_id", "domain_indicator"]
    target = "rating"

    for feature in dense_features:
        data[feature] = data[feature].apply(lambda x: convert_numeric(x))
    if dense_features:
        sca = MinMaxScaler()  # scaler dense feature
        data[dense_features] = sca.fit_transform(data[dense_features])

    for feature in useless_features:
        del data[feature]
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1

    data[target] = data[target].apply(lambda x: convert_target(x))

    for feat in tqdm(sparse_features):  # encode sparse feature
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    dense_feas = [DenseFeature(feature_name) for feature_name in dense_features]
    sparse_feas = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) for feature_name
                   in sparse_features]

    y = data[target]
    ll = len(data)
    train_x = data[:int(0.8 * ll)]
    test_x = data[int(0.8 * ll):]
    test_x.reset_index(inplace=True, drop=True)
    train_y = y[:int(0.8 * ll)]
    test_y = y[int(0.8 * ll):]
    test_y.reset_index(inplace=True, drop=True)

    return dense_feas, sparse_feas, train_x, train_y, test_x, test_y, domain_num


def map_group_indicator(age, list_group):
    l1 = len(list(list_group))
    for i in range(l1):
        if age in list_group[i]:
            return i


def convert_target(val):
    v = int(val)
    if v > 3:
        return int(1)
    else:
        return int(0)


def convert_numeric(val):
    """
    Forced conversion
    """
    return int(val)


def df_to_dict_multi_domain(data, columns):
    """
    Convert the array to a dict type input that the network can accept
    Args:
        data (array): 3D datasets of type DataFrame (Length * Domain_num * feature_num)
        columns (list): feature name list
    Returns:
        The converted dict, which can be used directly into the input network
    """

    data_dict = {}
    for i in range(len(columns)):
        data_dict[columns[i]] = data[:, :, i]
    return data_dict


def main(dataset_path, model_name, epoch, learning_rate, batch_size):
    (dense_feas, sparse_feas, train_x, train_y, test_x, test_y,
     domain_num) = get_movielens_data_rank_multi_domain(dataset_path)
    data_train_pre = Dataset_pre(train_x, train_y)
    data_test_pre = Dataset_pre(test_x, test_y)

    dataset_train = create_dataset(data_train_pre, batch_size=batch_size, drop_remainder=True)
    dataset_test = create_dataset(data_test_pre, batch_size=batch_size, drop_remainder=True)

    if model_name == "mlp":
        net = Mlp2Layer(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
    elif model_name == "mlp_adp":
        net = MlpAdap2Layer1Adp(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128],
                                hyper_dims=[128], k=35)
    elif model_name == "dcn_md":
        net = DcnMd(features=dense_feas + sparse_feas, num_domains=domain_num, n_cross_layers=2,
                    mlp_params={"dims": [256, 128]})
    elif model_name == "dcn_md_adp":
        net = DcnMdAdp(features=dense_feas + sparse_feas, num_domains=domain_num, n_cross_layers=2,
                       k=30, mlp_params={"dims": [256, 128]}, hyper_dims=[128])
    elif model_name == "wd_md":
        net = WideDeepMd(wide_features=dense_feas + sparse_feas, num_domains=domain_num, deep_features=sparse_feas,
                         mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
    elif model_name == "wd_md_adp":
        net = WideDeepMdAdp(wide_features=dense_feas, num_domains=domain_num, deep_features=sparse_feas, k=45,
                            mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"}, hyper_dims=[128])
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="./data/ml-1m")
    parser.add_argument('--model_name', default='wd_md')
    parser.add_argument('--epoch', type=int, default=1)  # 100
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=50)  # 4096
    args = parser.parse_args()
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size)
