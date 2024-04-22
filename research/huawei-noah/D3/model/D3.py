import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from utils import MLP, DANet
from tqdm import tqdm
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, log_loss

class DSFS(nn.Module):
    def __init__(self, feature_num, embed_size, hid_dim, output_dims):
        super().__init__()
        self.output_dims = output_dims
        self.linear_layers = nn.ModuleList([MLP(embed_size, False, [hid_dim]) for i in range(feature_num)])
        self.shared_weight = nn.Parameter(torch.empty(output_dims[0], output_dims[1]))
        self.shared_bias = nn.Parameter(torch.zeros(output_dims[1]))
        torch.nn.init.xavier_uniform_(self.shared_weight.data)
        self.trans = MLP(feature_num * hid_dim, False, [feature_num * hid_dim], 0)
        self.trans_weight = MLP(feature_num * hid_dim, False, [output_dims[0]*output_dims[1]], 0)
        self.trans_bias = MLP(feature_num * hid_dim, False, [output_dims[1]], 0)

    def forward(self, x):
        b, f, e = x.shape
        trans_features = []
        for i in range(f):
            feature = x[:, i, :].clone().detach()
            feature = self.linear_layers[i](feature)
            trans_features.append(feature)
        trans_features = torch.stack(trans_features, dim=1)
        residual_output = self.trans(trans_features.view(b,-1)) + trans_features.reshape(b, -1)
        weight = self.trans_weight(residual_output).reshape(b, self.output_dims[0], self.output_dims[1])
        bias = self.trans_bias(residual_output).reshape(b, self.output_dims[1])
        return weight, bias

class D3(nn.Module):
    def __init__(self, feature_dims, dense_cols, sparse_cols, embed_dim=8, selected_ID_features=list(range(23)), hid_dim1=64, hid_dim2=32, mlp_dims=[32,32]):
        super().__init__()
        self.embedding = nn.Embedding(sum(feature_dims), embedding_dim=embed_dim)
        self.embed_dim = embed_dim
        self.dense_cols = dense_cols
        self.sparse_cols = sparse_cols
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        self.offsets = np.array((0, *np.cumsum(feature_dims)[:-1]))
        self.selected_ID_features = selected_ID_features
        self.total_fea_num = len(dense_cols) + len(sparse_cols)
        self.DSFS_tr = DSFS(len(self.selected_ID_features), embed_dim, hid_dim1, [self.total_fea_num * self.embed_dim, hid_dim2])
        self.output_mlp = MLP(hid_dim2, True, mlp_dims, 0)

        self.add_attention = DANet(self.embed_dim)
        self.gate = MLP(self.total_fea_num * self.embed_dim, False, [64,16,2], 0)

        # hyperparameters
        self.desired_std = 0.5
        self.desired_mean = 0.5
        self.alpha_1 = 1.0
        self.alpha_2 = 1.0
        self.lambda_1 = 0.5
        self.lambda_2 = 0.5

    def forward(self, sparse):
        b,f = sparse.shape
        slot_id = sparse[:, 18].clone().detach()
        sparse = sparse + sparse.new_tensor(self.offsets).unsqueeze(0)
        sparse = self.embedding(sparse)
        DSFS_input = sparse.clone()
        DSFS_input, attn_weights = self.add_attention(DSFS_input)

        attn_weights = attn_weights.detach()
        attn_weights = torch.mean(attn_weights, dim=1)
        log_attn_weights = torch.log(attn_weights + 1e-9) # b,f*f
        entropy = -torch.sum(attn_weights * log_attn_weights, dim=-1).reshape(b)
        np_lis = entropy.cpu().numpy()
        mean = np_lis.mean()
        std = np_lis.std()
        arr = (np_lis - mean) / std
        desired_std = self.desired_std
        desired_mean = self.desired_mean
        arr = arr * desired_std + desired_mean
        arr = np.clip(arr, 0.1, 1)
        loss_weight = torch.tensor(arr, dtype=torch.float32).reshape(b, 1).to(sparse.device)

        gate = torch.softmax(self.gate(sparse.reshape(b, -1)), dim=1) # b, 2
        return_gate = gate.detach()

        se_output = sparse
        # feature transformation
        tr_weight, tr_bias = self.DSFS_tr(DSFS_input) # b, f*e, hid[1] ; b, hid[1]
        tr_weight = torch.multiply(torch.multiply(tr_weight, gate[:,0].reshape(b,1,1)) , torch.multiply(self.DSFS_tr.shared_weight.unsqueeze(0), gate[:,1].reshape(b,1,1)))
        tr_output = torch.matmul(se_output.reshape(b, 1, -1), tr_weight).reshape(b, -1) + tr_bias # b,1,f*e * b,f*e,hid[1] => b,hid[1]

        output = torch.sigmoid(self.output_mlp(tr_output)) # b, 1

        return output, self.alpha_1 * loss_weight + self.lambda_1, self.alpha_2 * return_gate[:,0].reshape(b,1) + self.lambda_2

    def train_(self, args, train_dataloader, valid_dataloader, optimizer, criterion, device, epoch=0):
        self.train()
        total_loss = 0.0
        total_length = 0
        tk0 = tqdm(train_dataloader, desc=f"train epoch {epoch}", leave=True, colour='blue')
        for batch_idx, (sparse, target) in enumerate(tk0):
            sparse, target = sparse.to(device), target.to(device)
            optimizer.zero_grad()
            output, loss_weight, gate_weight = self(sparse)
            loss = criterion(output, target.float()) # b, 1
            if args.start_weight_loss_step and batch_idx > args.start_weight_loss_step:
                loss = torch.multiply(loss, loss_weight)
                loss = torch.multiply(loss, gate_weight)
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*len(target)
            total_length += len(target)
            tk0.set_postfix(ordered_dict={'train_loss': total_loss/total_length})

    def test_(self, args, test_dataloader, auc, log_loss, device):
        self.eval()
        y_true, y_pred, slot_id = [], [], []
        tk0 = tqdm(test_dataloader, desc=f"test ", leave=True, colour='blue')
        with torch.no_grad():
            for batch_idx, (sparse, target) in enumerate(tk0):
                sparse, target = sparse.to(device), target.to(device)
                scene_id = sparse[:, 18]
                output, _, _ = self(sparse)
                y_true.extend(target.tolist())
                y_pred.extend(output.tolist())
                slot_id.extend(scene_id.tolist())
        y_true, y_pred, slot_id = np.array(y_true).flatten(), np.array(y_pred), np.array(slot_id)
        for i in range(1,4):
            tqdm.write(f"Slot {i} AUC: {auc(y_true[slot_id==i], y_pred[slot_id==i])}")
            tqdm.write(f"Slot {i} Log Loss: {log_loss(y_true[slot_id==i], y_pred[slot_id==i])}")