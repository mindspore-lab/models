import mindspore.common.dtype as mstype
import numpy as np
from mindspore import ParameterTuple, Tensor, nn
from mindspore.common.initializer import Uniform, initializer
from mindspore.common.parameter import Parameter
from mindspore.nn import Dropout
from mindspore.nn.optim import Adam
from mindspore.ops import Squeeze
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

ms_type = mstype.float32
ds_type = mstype.int32


def init_method(method, shape, name, max_val=1.0):
    """
    parameter init method
    """
    if method in ["uniform"]:
        params = Parameter(initializer(Uniform(max_val), shape, ms_type), name=name)
    elif method == "one":
        params = Parameter(initializer("ones", shape, ms_type), name=name)
    elif method == "zero":
        params = Parameter(initializer("zeros", shape, ms_type), name=name)
    elif method == "normal":
        params = Parameter(initializer("normal", shape, ms_type), name=name)
    return params


def normal_weight(shape, num_units):
    norm = np.random.normal(0.0, num_units**-0.5, shape).astype(np.float32)
    return Tensor(norm)


class DenseLayer(nn.Cell):
    """
    Dense Layer for DCN Model;
    Containing: activation, matmul, bias_add;
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        weight_bias_init,
        act_str,
        keep_prob=0.5,
        use_activation=True,
        convert_dtype=True,
        drop_out=False,
    ):
        super().__init__()
        weight_init, bias_init = weight_bias_init
        self.weight = init_method(weight_init, [input_dim, output_dim], name="weight")
        self.bias = init_method(bias_init, [output_dim], name="bias")
        self.act_func = self._init_activation(act_str)
        self.matmul = P.MatMul(transpose_b=False)
        self.bias_add = P.BiasAdd()
        self.cast = P.Cast()
        self.dropout = Dropout(p=1 - keep_prob)
        self.use_activation = use_activation
        self.convert_dtype = convert_dtype
        self.drop_out = drop_out

    def _init_activation(self, act_str):
        act_str = act_str.lower()
        if act_str == "relu":
            act_func = P.ReLU()
        elif act_str == "sigmoid":
            act_func = P.Sigmoid()
        elif act_str == "tanh":
            act_func = P.Tanh()
        return act_func

    def construct(self, x):
        """
        Construct Dense layer
        """
        if self.training and self.drop_out:
            x = self.dropout(x)
        if self.convert_dtype:
            x = self.cast(x, mstype.float16)
            weight = self.cast(self.weight, mstype.float16)
            bias = self.cast(self.bias, mstype.float16)
            wx = self.matmul(x, weight)
            wx = self.bias_add(wx, bias)
            if self.use_activation:
                wx = self.act_func(wx)
            wx = self.cast(wx, mstype.float32)
        else:
            wx = self.matmul(x, self.weight)
            wx = self.bias_add(wx, self.bias)
            if self.use_activation:
                wx = self.act_func(wx)
        return wx


class CrossLayer(nn.Cell):
    """
    Cross Layer for  DCN Model;
    """

    # pylint: disable=W0613
    def __init__(
        self, cross_raw_dim, cross_col_dim, weight_bias_init, convert_dtype=True
    ):
        super().__init__()
        weight_init, bias_init = weight_bias_init
        self.cross_weight = init_method(weight_init, [cross_col_dim, 1], name="weight")
        self.cross_bias = init_method(bias_init, [cross_col_dim, 1], name="bias")
        self.matmul = nn.MatMul()
        self.bias_add = P.BiasAdd()
        self.tensor_add = P.TensorAdd()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.expand_dims = P.ExpandDims()
        self.convert_dtype = convert_dtype
        self.squeeze = Squeeze(2)

    def construct(self, inputs, x_0):
        """
        Construct Cross layer
        """
        x_0 = self.expand_dims(x_0, 2)
        x_l = self.expand_dims(inputs, 2)
        x_lw = C.tensor_dot(x_l, self.cross_weight, ((1,), (0,)))
        dot = self.matmul(x_0, x_lw)
        y_l = dot + self.cross_bias + x_l
        y_l = self.squeeze(y_l)
        return y_l


class EmbeddingLookup(nn.Cell):
    """
    A embeddings lookup table with a fixed dictionary and size.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
    """

    # pylint: disable=W0613
    def __init__(
        self,
        vocab_size,
        embedding_size,
        use_one_hot_embeddings=False,
        initializer_range=0.02,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.embedding_table = Parameter(
            normal_weight([vocab_size, embedding_size], embedding_size)
        )
        self.expand = P.ExpandDims()
        self.shape_flat = (-1,)
        self.gather = P.Gather()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, input_ids):
        """Get a embeddings lookup table with a fixed dictionary and size."""
        input_shape = self.shape(input_ids)

        flat_ids = self.reshape(input_ids, self.shape_flat)
        if self.use_one_hot_embeddings:
            one_hot_ids = self.one_hot(
                flat_ids, self.vocab_size, self.on_value, self.off_value
            )
            output_for_reshape = self.array_mul(one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)

        out_shape = input_shape + (self.embedding_size,)
        output = self.reshape(output_for_reshape, out_shape)
        return output, self.embedding_table


class DeepCrossModel(nn.Cell):
    """
    Deep and Cross Model
    """

    def __init__(self):
        super().__init__()
        self.concat = P.Concat(axis=1)
        self.reshape = P.Reshape()
        self.deep_reshape = P.Reshape()
        self.deep_mul = P.Mul()
        self.vocab_size = 20000
        self.emb_dim = 8
        self.batch_size = 128
        self.field_size = 8
        self.input_size = self.field_size * self.emb_dim
        self.deep_layer_dim = 64
        self.deep_embeddinglookup = EmbeddingLookup(self.vocab_size, self.emb_dim)
        self.keep_prob = 1.0

        self.cross_layer_1 = CrossLayer(
            self.batch_size,
            self.input_size,
            weight_bias_init=["normal", "normal"],
            convert_dtype=False,
        )
        self.cross_layer_2 = CrossLayer(
            self.batch_size,
            self.input_size,
            weight_bias_init=["normal", "normal"],
            convert_dtype=False,
        )
        self.cross_layer_3 = CrossLayer(
            self.batch_size,
            self.input_size,
            weight_bias_init=["normal", "normal"],
            convert_dtype=False,
        )
        self.cross_layer_4 = CrossLayer(
            self.batch_size,
            self.input_size,
            weight_bias_init=["normal", "normal"],
            convert_dtype=False,
        )
        self.cross_layer_5 = CrossLayer(
            self.batch_size,
            self.input_size,
            weight_bias_init=["normal", "normal"],
            convert_dtype=False,
        )
        self.cross_layer_6 = CrossLayer(
            self.batch_size,
            self.input_size,
            weight_bias_init=["normal", "normal"],
            convert_dtype=False,
        )

        self.dense_layer_1 = DenseLayer(
            self.input_size,
            self.deep_layer_dim,
            weight_bias_init=["normal", "normal"],
            act_str="relu",
            keep_prob=self.keep_prob,
            convert_dtype=False,
            drop_out=False,
        )
        self.dense_layer_2 = DenseLayer(
            self.deep_layer_dim,
            self.deep_layer_dim,
            weight_bias_init=["normal", "normal"],
            act_str="relu",
            keep_prob=self.keep_prob,
            convert_dtype=False,
            drop_out=False,
        )

    def construct(self, id_hldr):
        """dcn construct"""
        # mask = self.reshape(wt_hldr, (self.batch_size, self.field_size, 1))
        deep_id_embs, _ = self.deep_embeddinglookup(id_hldr)
        # vx = self.deep_mul(deep_id_embs, mask)
        vx = deep_id_embs
        input_x = self.deep_reshape(vx, (-1, self.field_size * self.emb_dim))
        d_1 = self.dense_layer_1(input_x)
        d_2 = self.dense_layer_2(d_1)
        c_1 = self.cross_layer_1(input_x, input_x)
        c_2 = self.cross_layer_2(c_1, input_x)
        c_3 = self.cross_layer_3(c_2, input_x)
        c_4 = self.cross_layer_4(c_3, input_x)
        c_5 = self.cross_layer_5(c_4, input_x)
        c_6 = self.cross_layer_6(c_5, input_x)
        out = self.concat((d_2, c_6))
        # out = self.dense_layer_3(out)
        
        return out
    

class DCN_LLaMA(nn.Cell):
    def __init__(self, ctr_model, llama):
        super().__init__()
        self.ctr_model = ctr_model
        self.llama = llama
        self.num_prefix_prompt = 3
        self.num_hidden_layers = self.llama.config.num_layers
        self.hidden_size = self.llama.config.hidden_size
        
        self.prefix_encoder = DenseLayer(
            self.ctr_model.input_size + self.ctr_model.deep_layer_dim,
            self.num_hidden_layers * 2 * self.hidden_size * self.num_prefix_prompt,
            weight_bias_init=["normal", "normal"],
            act_str="relu",
            use_activation=False,
            convert_dtype=False,
            drop_out=False,
        )
        self.output_layer = DenseLayer(
            self.hidden_size,
            1,
            weight_bias_init=["normal", "normal"],
            act_str="sigmoid",
            keep_prob=1.0,
            use_activation=False,
            convert_dtype=False,
            drop_out=False,
        )
        
    def construct(self, id_hldr, token, attention_mask):
        out = self.ctr_model(id_hldr)
        
        """llama with prefix"""
        bsz = token.shape[0]
        num_head = self.llama.model.layers[0].attention.n_head
        past_key_values = self.prefix_encoder(out) # [bsz, num_hidden_layers * 2 * hidden_size]
        past_key_values = past_key_values.view(bsz, self.num_prefix_prompt, self.num_hidden_layers, 2, num_head, -1) # [bsz, num_prefix_prompt, num_hidden_layers, 2, num_head, head_dim]
        past_key_values = past_key_values.permute(2, 3, 0, 4, 1, 5) # [num_hidden_layers, 2, bsz, num_head, num_prefix_prompt, head_dim]
        
        outputs = self.llama(
            input_ids=token,
            output_hidden_states=True,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        encoder_hidden_states = ms.ops.ReduceSum(keep_dims=False)(outputs.last_hidden_state * attention_mask.unsequeeze(-1), axis=1) / ms.ops.ReduceSum(keep_dims=True)(attention_mask, axis=-1)
        out = self.output_layer(encoder_hidden_states)


class NetWithLossClass(nn.Cell):
    """
    NetWithLossClass definition.
    """

    def __init__(self, network):
        super().__init__(auto_prefix=False)
        self.loss = P.SigmoidCrossEntropyWithLogits()
        self.network = network
        self.reduce_mean = P.ReduceMean(keep_dims=False)

    def construct(self, batch_ids, label, token, attention_mask):
        predict = self.network(batch_ids, token, attention_mask)
        log_loss = self.loss(predict, label)
        mean_log_loss = self.reduce_mean(log_loss)
        loss = mean_log_loss
        return loss


class TrainStepWrap(nn.Cell):
    """
    TrainStepWrap definition
    """

    def __init__(self, network, lr=0.0001, eps=1e-8, loss_scale=1000.0):
        super().__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.set_train()
        self.weights = ParameterTuple(network.network.ctr_model.trainable_params())
        self.optimizer = Adam(
            self.weights, learning_rate=lr, eps=eps, loss_scale=loss_scale
        )
        self.hyper_map = C.HyperMap()
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = loss_scale

    def construct(self, batch_ids, label, token, attention_mask):
        weights = self.weights
        loss = self.network(batch_ids, label, token, attention_mask)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(batch_ids, batch_wts, label, sens)
        return F.depend(loss, self.optimizer(grads))


class PredictWithSigmoid(nn.Cell):
    """
    Eval model with sigmoid.
    """

    def __init__(self, network):
        super().__init__(auto_prefix=False)
        self.network = network
        self.sigmoid = P.Sigmoid()

    def construct(self, batch_ids, batch_wts, labels):
        logits = self.network(batch_ids, batch_wts)
        pred_probs = self.sigmoid(logits)
        return logits, pred_probs, labels