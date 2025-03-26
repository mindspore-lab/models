import mindspore
import mindspore.nn as nn
from mindspore import ops
from mindspore import dtype as mstype
import numpy as np
from utils import MLP, DANet
from tqdm import tqdm

class DSFS(nn.Cell):
    def __init__(self, feature_num, embed_size, hid_dim, output_dims):
        super(DSFS, self).__init__(auto_prefix=True)
        self.output_dims = output_dims
        self.linear_layers = nn.CellList([MLP(embed_size, False, [hid_dim]) for i in range(feature_num)])
        self.shared_weight = mindspore.Parameter(mindspore.Tensor(shape=(output_dims[0], output_dims[1]), dtype=mstype.float32))
        self.shared_bias = mindspore.Parameter(mindspore.ops.zeros(output_dims[1]))
        self.trans = MLP(feature_num * hid_dim, False, [feature_num * hid_dim], 0)
        self.trans_weight = MLP(feature_num * hid_dim, False, [output_dims[0]*output_dims[1]], 0)
        self.trans_bias = MLP(feature_num * hid_dim, False, [output_dims[1]], 0)

    def construct(self, x):
        b, f, e = x.shape
        trans_features = []
        for i in range(f):
            feature = x[:, i, :].clone()
            feature = self.linear_layers[i](feature)
            trans_features.append(feature)
        trans_features = mindspore.ops.stack(trans_features, dim=1)
        residual_output = self.trans(trans_features.view(b,-1)) + trans_features.reshape(b, -1)
        weight = self.trans_weight(residual_output).reshape(b, self.output_dims[0], self.output_dims[1])
        bias = self.trans_bias(residual_output).reshape(b, self.output_dims[1])
        return weight, bias

class D3(nn.Cell):
    def __init__(self, feature_dims, dense_cols, sparse_cols, embed_dim=8, selected_ID_features=list(range(23)), hid_dim1=64, hid_dim2=32, mlp_dims=[32,32]):
        super(D3, self).__init__(auto_prefix=True)
        self.embedding = nn.Embedding(sum(feature_dims), embedding_dim=embed_dim)
        self.embed_dim = embed_dim
        self.dense_cols = dense_cols
        self.sparse_cols = sparse_cols
        self.offsets = np.array((0, *np.cumsum(feature_dims)[:-1]))
        self.offsets = mindspore.Tensor(self.offsets, dtype=mstype.int32)
        self.selected_ID_features = selected_ID_features
        self.total_fea_num = len(dense_cols) + len(sparse_cols)
        self.DSFS_tr = DSFS(len(self.selected_ID_features), embed_dim, hid_dim1, [self.total_fea_num * self.embed_dim, hid_dim2])
        self.output_mlp = MLP(hid_dim2, True, mlp_dims, 0)
        self.add_attention = DANet(self.embed_dim)
        self.gate = MLP(self.total_fea_num * self.embed_dim, False, [64,16,2], 0)
        self.desired_std = 0.5
        self.desired_mean = 0.5
        self.alpha_1 = 1.0
        self.alpha_2 = 1.0
        self.lambda_1 = 0.5
        self.lambda_2 = 0.5

    def construct(self, sparse):
        b,f = sparse.shape
        slot_id = sparse[:, 18].clone()
        sparse = sparse + self.offsets.unsqueeze(0)
        sparse = self.embedding(sparse)
        DSFS_input = sparse.clone()
        DSFS_input, attn_weights = self.add_attention(DSFS_input)
        attn_weights = attn_weights
        attn_weights = mindspore.ops.mean(attn_weights, axis=1)
        log_attn_weights = mindspore.ops.log(attn_weights + 1e-9) # b,f*f
        entropy = -mindspore.ops.sum(attn_weights * log_attn_weights, dim=-1).reshape(b)
        np_lis = entropy.asnumpy()
        mean = np_lis.mean()
        std = np_lis.std()
        arr = (np_lis - mean) / std
        desired_std = self.desired_std
        desired_mean = self.desired_mean
        arr = arr * desired_std + desired_mean
        arr = np.clip(arr, 0.1, 1)
        loss_weight = mindspore.Tensor(arr, dtype=mstype.float32).reshape(b, 1)
        gate = mindspore.ops.Softmax(self.gate(sparse.reshape(b, -1)), dim=1) # b, 2
        return_gate = gate
        se_output = sparse
        tr_weight, tr_bias = self.DSFS_tr(DSFS_input) # b, f*e, hid[1] ; b, hid[1]
        tr_weight = mindspore.ops.multiply(mindspore.ops.multiply(tr_weight, gate[:,0].reshape(b,1,1)) , mindspore.ops.multiply(self.DSFS_tr.shared_weight.unsqueeze(0), gate[:,1].reshape(b,1,1)))
        tr_output = mindspore.ops.matmul(se_output.reshape(b, 1, -1), tr_weight).reshape(b, -1) + tr_bias # b,1,f*e * b,f*e,hid[1] => b,hid[1]
        output = mindspore.ops.sigmoid(self.output_mlp(tr_output)) # b, 1
        return output, self.alpha_1 * loss_weight + self.lambda_1, self.alpha_2 * return_gate[:,0].reshape(b,1) + self.lambda_2

    def train_(self, args, train_dataloader, valid_dataloader, optimizer, criterion, epoch=0):
        def forward_fn(data, labels):
            output, loss_weight, gate_weight = self(data)
            loss = criterion(output, labels.float()) # Assuming target needs to be float
            if args.start_weight_loss_step and batch_idx > args.start_weight_loss_step:
                loss = ops.multiply(loss, loss_weight)
                loss = ops.multiply(loss, gate_weight)
            loss = ops.mean(loss)
            return loss, output
        grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        self.set_train(True)
        total_loss = 0.0
        total_length = 0
        tk0 = tqdm(train_dataloader, desc=f"train epoch {epoch}", leave=True, colour='blue')
        for batch_idx, (data) in enumerate(tk0):
            sparse, target = data['features'], data['labels']
            (loss, _), grads = grad_fn(sparse, target)
            optimizer(grads)
            total_loss += loss.asnumpy() * len(target)
            total_length += len(target)
            if (batch_idx + 1) % 100 == 0:  # Example log interval of 100
                tk0.set_postfix(train_loss=total_loss / total_length)
                total_loss = 0.0  # Reset total loss after logging

    def test_(self, args, test_dataloader, auc, log_loss):
        self.set_train(False)
        y_true, y_pred, slot_id = [], [], []
        tk0 = tqdm(test_dataloader, desc=f"test ", leave=True, colour='blue')
        for batch_idx, (data) in enumerate(tk0):
            sparse, target = data['features'], data['labels']
            scene_id = sparse[:, 18]
            output, _, _ = self(sparse)
            y_true.extend(target.tolist())
            y_pred.extend(output.tolist())
            slot_id.extend(scene_id.tolist())
        y_true, y_pred, slot_id = np.array(y_true).flatten(), np.array(y_pred), np.array(slot_id)
        for i in range(1,4):
            tqdm.write(f"Slot {i} AUC: {auc(y_true[slot_id==i], y_pred[slot_id==i])}")
            tqdm.write(f"Slot {i} Log Loss: {log_loss(y_true[slot_id==i], y_pred[slot_id==i])}")