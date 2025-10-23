import numpy as np
import pandas as pd
import mindspore
import gc
from tqdm import tqdm
from model.D3 import D3
import mindspore.dataset as ds
from mindspore import dtype as mstype
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from mindspore import nn, Model, context

def get_ali_ccp_data_dict_pd(args, data_path='./data/'):
    data_type = {'click':np.int8, 'purchase': np.int8, '101':np.int32, '121':np.uint8, '122':np.uint8, '124':np.uint8, '125':np.uint8, '126':np.uint8, '127':np.uint8, '128':np.uint8, '129':np.uint8, '205':np.int32, '206':np.int16, '207':np.int32, '210':np.int32, '216':np.int32, '508':np.int16, '509':np.int32, '702':np.int32, '853':np.int32, '301':np.int8, '109_14':np.int16, '110_14':np.int32, '127_14':np.int32, '150_14':np.int32, 'D109_14': np.float16, 'D110_14': np.float16, 'D127_14': np.float16, 'D150_14': np.float16, 'D508': np.float16, 'D509': np.float16, 'D702': np.float16, 'D853': np.float16}

    df_train = pd.read_csv(data_path + '/ali_ccp_train.csv', dtype=data_type)
    df_val = pd.read_csv(data_path + '/ali_ccp_val.csv', dtype=data_type)
    df_test = pd.read_csv(data_path + '/ali_ccp_test.csv', dtype=data_type)

    print("train : val : test = %d %d %d" % (df_train.shape[0], df_val.shape[0], df_test.shape[0]))
    lengths = [df_train.shape[0], df_val.shape[0], df_test.shape[0]]
    train_idx = lengths[0]
    data = pd.concat([df_train, df_val, df_test], axis=0)
    print(data.head(5))
    del df_train, df_val, df_test
    col_names = data.columns
    dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
    sparse_cols = [col for col in col_names if col not in dense_cols and col not in ['click', 'purchase']]
    print('dense cols:', dense_cols, 'sparse cols:', sparse_cols)
    print("sparse cols:%d dense cols:%d" % (len(sparse_cols), len(dense_cols)))
    y = data.loc[:,"click"]
    sparse_x = data[sparse_cols]
    sparse_x_unique = [238635, 98, 14, 3, 8, 4, 4, 3, 5, 467298, 6929, 263942, 80232, 106399, 5888, 104830, 51878, 37148, 4, 5853, 105622, 53843, 31858]
    x_sparse_train, x_sparse_test = sparse_x[:train_idx], sparse_x[train_idx:]
    y_train, y_test = y[:train_idx], y[train_idx:]
    x_sparse_train, y_train = mindspore.Tensor(x_sparse_train.values), mindspore.Tensor(y_train.values, dtype=mstype.float32).reshape(-1,1)
    x_sparse_test, y_test = mindspore.Tensor(x_sparse_test.values), mindspore.Tensor(y_test.values, dtype=mstype.float32).reshape(-1,1)
    sampler = ds.SequentialSampler()
    train_dataset = ds.NumpySlicesDataset({'features': x_sparse_train, 'labels': y_train}, sampler=sampler)
    test_dataset = ds.NumpySlicesDataset({'features': x_sparse_test, 'labels': y_test}, sampler=sampler)
    train_dataset = train_dataset.batch(batch_size=args.batch_size)
    train_dataloader = train_dataset.create_tuple_iterator()
    test_dataset = test_dataset.batch(batch_size=args.batch_size)
    test_dataloader = test_dataset.create_tuple_iterator()
    return train_dataloader, None, test_dataloader, sparse_x_unique, dense_cols, sparse_cols, lengths

def main(args):
    context.set_context(device_target=args.device)
    train_dataloader, valid_dataloader, test_dataloader, feature_dims, dense_cols, sparse_cols, lengths = get_ali_ccp_data_dict_pd(args, args.data_path)
    model = D3(feature_dims, [], sparse_cols, args.embed_size, selected_ID_features = list(range(len(sparse_cols))), hid_dim1=args.d3_hid_dim1, hid_dim2=args.d3_hid_dim2, mlp_dims=args.d3_mlp_dims)
    optimizer = mindspore.nn.AdamWeightDecay(model.parameters(), learning_rate=args.lr, weight_decay=1e-5)
    criterion = mindspore.nn.BCELoss(reduction='none')
    for i in range(args.epochs):
        model.train_(args, train_dataloader, valid_dataloader, optimizer, criterion, i)
    model.test_(args, test_dataloader, roc_auc_score, log_loss)
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embed_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--d3_hid_dim1', type=int, default=64)
    parser.add_argument('--d3_hid_dim2', type=int, default=128)
    parser.add_argument('--d3_mlp_dims',type=list, default=[16,16])
    parser.add_argument('--selected_ID_features', type=list, default=list(range(23)))
    parser.add_argument('--start_weight_loss_step', type=int, default=15000)
    args = parser.parse_args()

    main(args)