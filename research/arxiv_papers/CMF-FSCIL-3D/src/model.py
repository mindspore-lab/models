from collections import OrderedDict
import timm
import mindspore
from easydict import EasyDict
from mindspore import nn, ops, Tensor, Parameter
from mindspore.common.initializer import initializer, Normal
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from models.pointnet2.pointnet2 import Pointnet2_Ssg
import numpy as np
from easydict import EasyDict
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import mahalanobis
import heapq
import mindspore.numpy as ms_np

class QuickGELU(nn.Cell):
    def construct(self, x):
        return x * ops.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Cell):
    def __init__(self, d_model: int, n_head: int, attn_mask: Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm((d_model,))
        self.mlp = nn.SequentialCell(
            OrderedDict(
                [
                    ("c_fc", nn.Dense(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Dense(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm((d_model,))
        self.attn_mask = attn_mask

    def attention(self, x: Tensor):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.astype(x.dtype)
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def construct(self, x: Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Cell):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.CellList(
            [ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def construct(self, x: Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class Model_with_Image(nn.Cell):

    def __init__(self, point_encoder, **kwargs):
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.context_length = kwargs.context_length
        self.vision_width = kwargs.vision_width
        self.visual = kwargs.vision_model

        self.transformer = Transformer(
            width=kwargs.transformer_width,
            layers=kwargs.transformer_layers,
            heads=kwargs.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )
        self.vocab_size = kwargs.vocab_size
        self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        self.positional_embedding = Parameter(
            Tensor(
                ops.zeros(
                    (self.context_length, kwargs.transformer_width),
                    dtype=mindspore.float32,
                )
            )
        )
        self.ln_final = nn.LayerNorm((kwargs.transformer_width,))

        self.image_projection = Parameter(
            Tensor(
                ops.zeros(
                    (kwargs.vision_width, kwargs.embed_dim), dtype=mindspore.float32
                )
            )
        )
        self.text_projection = Parameter(
            Tensor(
                ops.zeros(
                    (kwargs.transformer_width, kwargs.embed_dim),
                    dtype=mindspore.float32,
                )
            )
        )
        self.logit_scale = Parameter(
            Tensor(ops.ones([]) * np.log(1 / 0.07), dtype=mindspore.float32)
        )

        self.initialize_parameters()

        self.point_encoder = point_encoder

        self.pc_projection = Parameter(
            initializer(Normal(sigma=512**-0.5), [kwargs.pc_feat_dims, 512], mindspore.float32),
            name='pc_projection'
        )

    def encode_image(self, image):
        x = self.visual(image)
        x = ops.matmul(x, self.image_projection)
        return x

    def encode_text(self, text):
        text = text.astype(mindspore.int32)  
        x = self.token_embedding(text)  
        x = x + self.positional_embedding
        x = ops.transpose(x, (1, 0, 2))  
        x = self.transformer(x)
        x = ops.transpose(x, (1, 0, 2))  
        x = self.ln_final(x)
        x = ops.matmul(
            x[ops.arange(x.shape[0]), text.argmax(axis=-1)], self.text_projection
        )
        return x

    def build_attention_mask(self):
        mask = Tensor(
            ops.fill(
                mindspore.float32,
                (self.context_length, self.context_length),
                float("-inf"),
            )
        )
        mask = ops.triu(mask, 1)  
        return mask

    def initialize_parameters(self):
        self.token_embedding.embedding_table = Tensor(
            initializer(
                Normal(sigma=0.02),
                self.token_embedding.embedding_table.shape,
                self.token_embedding.embedding_table.dtype,
            )
        )
        self.positional_embedding = Tensor(
            initializer(
                Normal(sigma=0.01),
                self.positional_embedding.shape,
                self.positional_embedding.dtype,
            )
        )

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5

        for block in self.transformer.resblocks:
            block.attn.in_proj_weight = Tensor(
                initializer(
                    Normal(sigma=attn_std),
                    block.attn.in_proj_weight.shape,
                    block.attn.in_proj_weight.dtype,
                )
            )
            block.attn.out_proj.weight = Tensor(
                initializer(
                    Normal(sigma=proj_std),
                    block.attn.out_proj.weight.shape,
                    block.attn.out_proj.weight.dtype,
                )
            )
            block.mlp[0].weight = Tensor(
                initializer(
                    Normal(sigma=fc_std),
                    block.mlp[0].weight.shape,
                    block.mlp[0].weight.dtype,
                )
            )
            block.mlp[2].weight = Tensor(
                initializer(
                    Normal(sigma=proj_std),
                    block.mlp[2].weight.shape,
                    block.mlp[2].weight.dtype,
                )
            )

        self.image_projection = Tensor(
            initializer(
                Normal(sigma=self.vision_width**-0.5),
                self.image_projection.shape,
                self.image_projection.dtype,
            )
        )
        self.text_projection = Tensor(
            initializer(
                Normal(sigma=self.transformer.width**-0.5),
                self.text_projection.shape,
                self.text_projection.dtype,
            )
        )

    def encode_pc(self, pc):
        pc_feat, pc_all, pc_main = self.point_encoder(pc)
        pc_embed = ops.matmul(pc_feat, self.pc_projection)
        return pc_embed, pc_all, pc_main

    def construct(self, pc, text, image=None):
        text_embed_all = []
        for i in range(text.shape[0]):
            text_for_one_sample = text[i]
            text_embed = self.encode_text(text_for_one_sample)
            text_embed = text_embed / ops.norm(text_embed, dim=-1, keepdim=True)
            text_embed = ops.mean(text_embed, axis=0)
            text_embed = text_embed / ops.norm(text_embed, dim=-1, keepdim=True)
            text_embed_all.append(text_embed)

        text_embed_all = ops.stack(text_embed_all)
        pc_embed, pc_all, pc_main = self.encode_pc(pc)
        if image is not None:
            image_embed = self.encode_image(image)
            return {
                "text_embed": text_embed_all,
                "pc_embed": pc_embed,
                "pc_all": pc_all,
                "pc_main": pc_main,
                "image_embed": image_embed,
                "logit_scale": ops.exp(self.logit_scale),
            }

        else:
            return {
                "text_embed": text_embed_all,
                "pc_embed": pc_embed,
                "pc_all": pc_all,
                "pc_main": pc_main,
                "logit_scale": ops.exp(self.logit_scale),
            }


class IncrementalMedianCalculator:
    def __init__(self, num_dimensions=256):
        self.min_heaps = [[] for _ in range(num_dimensions)]
        self.max_heaps = [[] for _ in range(num_dimensions)]

    def add_number(self, num):
        for i in range(len(num)):
            if len(self.max_heaps[i]) == 0 or num[i] <= -self.max_heaps[i][0]:
                heapq.heappush(self.max_heaps[i], -num[i])
            else:
                heapq.heappush(self.min_heaps[i], num[i])

            if len(self.max_heaps[i]) > len(self.min_heaps[i]) + 1:
                heapq.heappush(self.min_heaps[i], -heapq.heappop(self.max_heaps[i]))
            elif len(self.min_heaps[i]) > len(self.max_heaps[i]):
                heapq.heappush(self.max_heaps[i], -heapq.heappop(self.min_heaps[i]))

    def get_median(self):
        medians = []
        for i in range(len(self.min_heaps)):
            if len(self.max_heaps[i]) == len(self.min_heaps[i]):
                median = (-self.max_heaps[i][0] + self.min_heaps[i][0]) / 2.0
            else:
                median = -self.max_heaps[i][0]
            medians.append(median)
        return np.array(medians)


class NCMClassifier(nn.Cell):
    def __init__(self, pc_encoder):
        super(NCMClassifier, self).__init__()
        self.encoder = pc_encoder
        self.feature_center = {}
        self.cate_num = {}
        self.total_weight = defaultdict(float)
        self.device = mindspore.context.get_context("device_target")  
        self.median_calculators = defaultdict(IncrementalMedianCalculator)
        self.method = "mean"
        
        self.encoder.set_train(False)  
        for param in self.encoder.get_parameters():
            param.requires_grad = False  

    def construct(self, x):
        
        pass

    def to(self, *args, **kwargs):
        device = kwargs.get("device", None)
        if device:
            mindspore.context.set_context(device_target=device)
        return self

    def train(self, category, pc_datas, weights=None, method="mean"):
        self.method = method
        pc_datas = [Tensor(pc) if isinstance(pc, np.ndarray) else pc for pc in pc_datas]  
        if len(pc_datas) == 1:
            return
        pc_embed, _, _ = self.encoder(ms_np.stack(pc_datas, axis=0))
        for i, (feature, cat) in enumerate(zip(pc_embed, category)):
            if method == "mean":
                self._update_mean(cat, feature)
            elif method == "weighted_mean" and weights is not None:
                self._update_weighted_mean(cat, feature, weights[i: i + 1])
            elif method == "median":
                self.median_calculators[cat].add_number(feature.asnumpy())  
        
    def _update_mean(self, label, features):
        if label not in self.feature_center:
            self.feature_center[label] = Tensor(np.zeros(features.shape), dtype=mindspore.float32)
            self.cate_num[label] = 0
        self.feature_center[label] += features
        self.cate_num[label] += 1

    def _update_weighted_mean(self, label, features, weights):
        if label not in self.feature_center:
            self.feature_center[label] = Tensor(np.zeros(features.shape), dtype=mindspore.float32)
            self.total_weight[label] = 0
        self.feature_center[label] += features * weights[0]
        self.total_weight[label] += weights[0]

    def train_last(self, method="mean"):
        if method == "mean":
            for cate in self.feature_center:
                self.feature_center[cate] /= self.cate_num[cate]
        elif method == "weighted_mean":
            for cate in self.feature_center:
                self.feature_center[cate] /= self.total_weight[cate]
        elif method == "median":
            for label in self.median_calculators:
                self.feature_center[label] = Tensor(
                    np.array([self.median_calculators[label].get_median()]),
                    dtype=mindspore.float32
                )

    def predict(self, pc_data):
        pc_emb, _, _ = self.encoder(pc_data)
        batch_size = pc_emb.shape[0]

        results = {"cos": [], "ed": [], "dp": []}
        for i in range(batch_size):
            max_cos = {"val": -1, "cate": -1}
            min_ed = {"val": float("inf"), "cate": -1}
            max_dp = {"val": float("-inf"), "cate": -1}
            emb = pc_emb[i]
            for cate, center in self.feature_center.items():
                if center.ndim == 2:
                    center = center.squeeze(0)
                cos = self.cosine_similarity(emb, center)
                if cos > max_cos["val"]:
                    max_cos["val"] = cos
                    max_cos["cate"] = cate

                ed = ops.norm(emb - center)
                if ed < min_ed["val"]:
                    min_ed["val"] = ed
                    min_ed["cate"] = cate

                dp = ops.matmul(emb, center)
                if dp > max_dp["val"]:
                    max_dp["val"] = dp
                    max_dp["cate"] = cate

            results["cos"].append(max_cos)
            results["ed"].append(min_ed)
            results["dp"].append(max_dp)

        return results

    @staticmethod
    def cosine_similarity(a, b):
        return ops.cosine_similarity(a.expand_dims(0), b.expand_dims(0)).asnumpy()

def get_metric_names():
    return ["loss", "pc_image_acc", "pc_text_acc", "pc_all_acc", "pc_main_acc"]

class PatchEmbedding(nn.Cell):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=True)

    def construct(self, x):
        x = self.proj(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = ops.transpose(x, (0, 2, 1))
        return x

class Attention(nn.Cell):
    def __init__(self, dim, num_heads=12, qkv_bias=False):
        super(Attention, self).__init__()
        qkv_bias = True
        self.num_heads = num_heads
        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.scale = Tensor((dim // num_heads) ** -0.5, mindspore.float32)
        self.proj = nn.Dense(dim, dim)

    def construct(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = ops.softmax(attn, axis=-1)
        x = ops.matmul(attn, v)
        x = ops.transpose(x, (0, 2, 1, 3)).view(B, N, C)
        x = self.proj(x)
        return x

class MLP(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Cell):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm((dim,))
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm((dim,))
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def construct(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Cell):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, drop_rate=0.0):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = mindspore.Parameter(ops.zeros((1, 1, embed_dim), mindspore.float32))
        self.pos_embed = mindspore.Parameter(ops.zeros((1, num_patches + 1, embed_dim), mindspore.float32))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.CellList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm((embed_dim,))

        self.head = nn.Dense(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def construct(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = ops.tile(self.cls_token, (B, 1, 1))
        x = ops.concat((cls_tokens, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        return x[:, 0]

def PN_SSG():
    vision_model = VisionTransformer(img_size=224, patch_size=16, in_channels=3, num_classes=0, embed_dim=768, depth=12, num_heads=12)

    point_encoder = Pointnet2_Ssg()
    pc_feat_dims = 256

    model = Model_with_Image(
        embed_dim=512,
        vision_width=768,
        point_encoder=point_encoder,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        pc_feat_dims=pc_feat_dims,
    )
    
    # pretrain_slip_model_params = load_checkpoint(
    #     "data/initialize_models/slip_base_100ep.ckpt"
    # )
    # load_param_into_net(model, pretrain_slip_model_params)

    # for name, param in model.parameters_and_names():
    #     if name in pretrain_slip_model_params:
    #         param.set_data(pretrain_slip_model_params[name].data)
    #         param.requires_grad = False
    #         print(f"load {name} and freeze")


    return model
