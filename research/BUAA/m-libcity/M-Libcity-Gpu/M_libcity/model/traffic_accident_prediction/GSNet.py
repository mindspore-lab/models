import mindspore
import mindspore.nn as nn
import numpy as np
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import One
from model.abstract_traffic_state_model import AbstractTrafficStateModel
import pickle as pkl
import os

class Residual(nn.Cell):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def construct(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Cell):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm([dim])
        self.fn = fn

    def construct(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Cell):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(dim, hidden_dim),
            nn.GELU(),
            nn.Dense(hidden_dim, dim)
        )

    def construct(self, x):
        return self.net(x)

class Attention(nn.Cell):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Dense(dim, dim * 3)
        self.to_out = nn.Dense(dim, dim)

    def construct(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        qkv = qkv.reshape([b,n,3,h,-1])
        qkv = qkv.transpose(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        dots = mindspore.ops.bmm(q, k.transpose(0,1,3,2)) * self.scale

        if mask is not None:
            mask = mindspore.numpy.pad(mask.flatten(start_dim=1), (1, 0), mode="constant",constant_values=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots=mindspore.ops.masked_fill(dots,~mask, float('-inf'))
            del mask

        attn = mindspore.ops.softmax(dots)
        out = mindspore.ops.bmm(attn,v)
        out = out.reshape(b,n,-1)
        out = self.to_out(out)
        return out

class Transformer(nn.Cell):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.SequentialCell([])
        for _ in range(depth):
            self.layers.append(nn.SequentialCell([
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def construct(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class _ViT(nn.Cell):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, data_type, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = mindspore.Parameter(mindspore.numpy.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Dense(patch_dim, dim)
        self.cls_token = mindspore.Parameter(mindspore.numpy.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()
        if 'NYC' in data_type:
            self.past_region_features_fc = nn.Dense(329, dim) #nyc -> 47*7
        else:
            self.past_region_features_fc = nn.Dense(280, dim) #chicago -> 40*7
        
        self.mlp_head = nn.SequentialCell(
            nn.Dense(dim, mlp_dim),
            nn.GELU(),
            nn.Dense(mlp_dim, num_classes)
        )
        self.reshape = ops.Reshape()

    def construct(self, img, non_risk_features, mask=None):

        p = self.patch_size
        b,c,h,w=img.shape
        h=int(h/p)
        img = img.reshape(b,c,h,p,w)
        w=int(w/p)
        img = img.reshape(b,c,h,p,w,p)
        img = img.transpose(0,2,4,3,5,1)
        img = img.reshape(b,h*w,p,p,c)
        x = img.reshape(b,h*w,-1)
        # x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        cls_tokens = self.cls_token.repeat_interleave(img.shape[0], 0)
        x = mindspore.ops.cat((cls_tokens, x), 1)
        x += self.pos_embedding
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])
        
        return self.mlp_head((x+self.past_region_features_fc(non_risk_features))).reshape(img.shape[0],-1,20,20)

def mask_loss(predicts,labels,region_mask,data_type="nyc"):
    """
    
    Arguments:
        predicts {Tensor} -- predict，(batch_size,pre_len,W,H)
        labels {Tensor} -- label，(batch_size,pre_len,W,H)
        region_mask {np.array} -- mask matrix，(W,H)
        data_type {str} -- nyc/chicago
    
    Returns:
        {Tensor} -- MSELoss,(1,)
    """

    batch_size,pre_len,_,_,_ = predicts.shape
    region_mask /= region_mask.mean()
    loss = ((labels-predicts)*region_mask)**2
    if data_type=='nyc':
        ratio_mask = mindspore.numpy.zeros(labels.shape)
        index_1 = labels <=0
        index_2 = (labels > 0) & (labels <= 0.04)
        index_3 = (labels > 0.04) & (labels <= 0.08)
        index_4 = labels > 0.08
        ratio_mask[index_1] = 0.05
        ratio_mask[index_2] = 0.2
        ratio_mask[index_3] = 0.25
        ratio_mask[index_4] = 0.5
        loss *= ratio_mask
    elif data_type=='chicago':
        ratio_mask = mindspore.numpy.zeros(labels.shape)
        index_1 = labels <=0
        index_2 = (labels > 0) & (labels <= 1/17)
        index_3 = (labels > 1/17) & (labels <= 2/17)
        index_4 = labels > 2/17
        ratio_mask[index_1] = 0.05
        ratio_mask[index_2] = 0.2
        ratio_mask[index_3] = 0.25
        ratio_mask[index_4] = 0.5
        loss *= ratio_mask
    return loss.mean()

class GSNet(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super(GSNet, self).__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')
        
        self.dataset=config['dataset']
        self.dataset = './raw_data/' + self.dataset + '/'
        self.mask_filename=os.path.join(self.dataset,"risk_mask.pkl")
        
        self.risk_mask = pkl.load(open(self.mask_filename,'rb')).astype(np.float32)
        self.risk_mask = mindspore.Tensor.from_numpy(self.risk_mask)

        self.gsnet = _ViT(image_size=20, patch_size=5, num_classes=400, channels=7,
            dim=64, depth=6, heads=8, mlp_dim=128, data_type = config['dataset'])

    
    def forward(self, batch):
        b,t,d,w,h=batch['X'].shape
        x=batch['X'][:,:,0,:,:]

        x_ext=batch['X'][:,:,1:,0,0].flatten(start_dim=1)

        result=self.gsnet(x,x_ext)
        result=result.unsqueeze(-1)
        return result

    def calculate_loss(self, batch):
        # [batch_size, output_dim, num_cols, num_rows, output_window]
        y_pred = self.forward(batch)
        y_true = batch['y']
        loss=mask_loss(y_pred,y_true,self.risk_mask)
        return loss

    def predict(self, batch):
        return self.forward(batch)

    def construct(self, X, y, X_ext, y_ext):
        batch = {}
        batch['X'] = X
        batch['y'] = y
        batch['X_ext'] = X_ext
        batch['y_ext'] = y_ext
        # 适用于不同ms版本的代码备份
        # batch['X'] = mindspore.Tensor(input_data=X, dtype=mindspore.float32)
        # batch['y'] = mindspore.Tensor(input_data=y, dtype=mindspore.float32)
        # batch['X_ext'] = mindspore.Tensor(input_data=X_ext, dtype=mindspore.float32)
        # batch['y_ext'] = mindspore.Tensor(input_data=y_ext, dtype=mindspore.float32)

        if self.mode == "train":
            return self.calculate_loss(batch)
        elif self.mode == "eval":
            y_true = ops.transpose(batch['y'][..., :1], (0, 4, 2, 3, 1))
            y_pred = self.predict(batch)
            return y_pred, y_true
    
    def set_loss(self, loss_fn):
        pass

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"