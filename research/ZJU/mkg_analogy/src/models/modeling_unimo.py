from typing import Any, Optional, Tuple
import math
import numpy as np
import mindspore
from mindspore import nn, Tensor, ops, ParameterTuple, Parameter, dtype_to_nptype
from mindspore.common.initializer import initializer, Uniform, Normal


from mindnlp.models import PretrainedConfig
from mindnlp.transformers.modeling_utils import PreTrainedModel
from mindnlp.transformers.activations import ACT2FN
from mindnlp.transformers.ms_utils import apply_chunking_to_forward
from mindnlp.transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput
)

# some function
def get_extended_attention_mask(attention_mask, input_shape: Tuple[int]):
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) \
                    or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=mindspore.int32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


def get_head_mask(
        head_mask, num_hidden_layers: int, is_attention_chunked: bool = False
    ):
        head_mask = [None] * num_hidden_layers

        return head_mask


# models
class UnimoConfig(PretrainedConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class UnimoPreTrainedModel(PreTrainedModel):
    config_class = UnimoConfig
    base_model_prefix = "clip"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init_weights(self, module):
        pass


class CLIPVisionEmbeddings(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = Parameter(ops.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            pad_mode='valid',
            has_bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.position_ids = ops.broadcast_to(ops.arange(self.num_positions), (1, -1))

    def construct(self, pixel_values):
        # pixel_values: (bsz, 2, 3, 224, 224)
        batch_size = pixel_values.shape[0]
        patch_embeds = ops.cat([
                            self.patch_embedding(pixel_values[:, 0]).flatten(start_dim=2).swapaxes(1, 2),
                            self.patch_embedding(pixel_values[:, 1]).flatten(start_dim=2).swapaxes(1, 2)],
                            axis=1
                        )   # bsz, 98, 768

        class_embeds = self.class_embedding.broadcast_to((batch_size, 1, -1))

        embeddings = ops.cat([class_embeds, patch_embeds], axis=1)
        embeddings = embeddings + ops.cat([self.position_embedding(self.position_ids),
                                           self.position_embedding(self.position_ids)[:, 1:]],
                                           axis=1)

        return embeddings


class BertEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size,
                                            padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.position_ids = ops.arange(config.max_position_embeddings).reshape((1, -1))
        self.token_type_ids = ops.zeros(self.position_ids.shape, dtype=mindspore.int64)

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        past_key_values_length: int = 0,
    ):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + \
                                             past_key_values_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0],
                                                                                  seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CLIPAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} \
            and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        return ops.swapaxes(tensor.view(bsz, seq_len, self.num_heads, self.head_dim), 1, 2)

    def construct(
        self,
        hidden_states,
        output_attentions,
        past_key_values,
    ) -> Tuple[ops.Tensor, Optional[ops.Tensor], Optional[Tuple[ops.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if past_key_values is not None:
            key_states = ops.cat([past_key_values[0], key_states], axis=2)
            value_states = ops.cat([past_key_values[1], value_states], axis=2)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)

        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.shape[1]
        attn_weights = ops.bmm(query_states, key_states.swapaxes(1, 2))

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attentionze {(bsz * self.num_heads, tgt_len, src_len)}, \
                    but is {attn_weights.shape}"
            )
        attn_weights = ops.softmax(attn_weights, axis=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = ops.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = ops.bmm(attn_probs, value_states)

        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_oe {(bsz, self.num_heads, tgt_len, self.head_dim)},\
                      but is {attn_output.shape}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class CLIPMLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Dense(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Dense(config.intermediate_size, config.hidden_size)

    def construct(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class BertSelfAttention(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads   # 12
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size    # 768

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.fusion = BertFusion(config)    #

        # adaptive analogy mask
        self.adaptive_weight = ParameterTuple([
                Parameter(initializer(Uniform(0.5), [1,], mindspore.float32), name="param1"),  # example to query
                Parameter(initializer(Uniform(0.5), [1,], mindspore.float32), name="param2")   # query to example
        ])

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.transpose(0, 2, 1, 3)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        visual_hidden_state=None,
        output_qks=None,
        sep_idx=None
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        qks = (key_layer, value_layer) if output_qks else None

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if sep_idx is not None:
            for i, idx in enumerate(sep_idx):
                # example to answer
                attention_scores[i, :, :idx[2], idx[2]:] = ops.clamp(self.adaptive_weight[0],
                                                                     0, 0.5) \
                                        * attention_scores[i, :, :idx[2], idx[2]:]
                # answer to example
                attention_scores[i, :, idx[2]:, idx[2]:] = ops.clamp(self.adaptive_weight[1],
                                                                     0.5, 1) \
                                        * attention_scores[i, :, idx[2]:, idx[2]:]

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(axis=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = ops.matmul(attention_probs, value_layer)

        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)    # bsz, 128, 768

        fusion_output = self.fusion(context_layer, visual_hidden_state) \
            if visual_hidden_state is not None else None # add

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs, fusion_output, qks


class BertSelfOutput(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertFusion(nn.Cell):
    def __init__(self, config):
        super().__init__()
        # self.fusion_function = config.fusion_function
        self.fusion_function = 'softmax'

    def construct(
        self,
        hidden_states,
        visual_hidden_state=None,
    ):
        fusion_scores = ops.matmul(hidden_states, visual_hidden_state.swapaxes(-1, -2))
        # if attention_mask is not None:
        #     # attention_mask: bsz, 1, 1, 128; fusion_scores: bsz, 128, 49
        #     fusion_scores = fusion_scores + attention_mask.squeeze(1).swapaxes(1, 2)
        if self.fusion_function == 'softmax':
            fusion_probs = nn.Softmax(axis=-1)(fusion_scores)
            fusion_output = ops.matmul(fusion_probs, visual_hidden_state)
        elif self.fusion_function == 'max':
            fusion_probs = fusion_scores.max(axis=-1)
        return fusion_output


class BertAttention(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        visual_hidden_state=None,
        output_qks=None,
        sep_idx=None,
    ):
        self_outputs, fusion_output, qks = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            visual_hidden_state,
            output_qks,
            sep_idx
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs, fusion_output, qks


class BertIntermediate(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        self.fusion_dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states, fusion_output=None):
        hidden_states = self.dense(hidden_states)
        if fusion_output is not None:
            fusion_states = self.fusion_dense(fusion_output)
            hidden_states = hidden_states + fusion_states
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CLIPEncoderLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm((self.embed_dim,), epsilon=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm((self.embed_dim,), epsilon=config.layer_norm_eps)

    def construct(
        self,
        hidden_states,
        output_attentions,
        past_key_values,
    ):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BertLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        visual_hidden_state=None,
        output_qks=None,
        sep_idx=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs, fusion_output, qks = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            visual_hidden_state=visual_hidden_state,
            output_qks=output_qks,
            sep_idx=sep_idx,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim, attention_output,
            fusion_output
        )
        outputs = (layer_output,) + outputs
        if output_qks:
            outputs += (qks,)

        return outputs

    def feed_forward_chunk(self, attention_output, fusion_output):
        intermediate_output = self.intermediate(attention_output, fusion_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class UnimoEncoder(nn.Cell):
    def __init__(self, vision_config, text_config):
        super().__init__()
        self.vision_config = vision_config
        self.text_config = text_config

        self.vision_layers = nn.CellList([CLIPEncoderLayer(vision_config) \
                                          for _ in range(vision_config.num_hidden_layers)])
        self.text_layer = nn.CellList([BertLayer(text_config) \
                                       for _ in range(text_config.num_hidden_layers)])

    def construct(
        self,
        vision_embeds=None,
        text_embeds=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sep_idx=None,
    ):
        assert self.vision_config.num_hidden_layers == self.text_config.num_hidden_layers

        all_vision_hidden_states = () if output_hidden_states else None
        all_text_hidden_states = () if output_hidden_states else None
        all_vision_attentions = () if output_attentions else None
        all_text_attentions = () if output_attentions else None

        vision_hidden_states = vision_embeds
        text_hidden_states = text_embeds
        for idx in range(self.vision_config.num_hidden_layers):
            if output_hidden_states:
                all_vision_hidden_states = all_vision_hidden_states + (vision_hidden_states, )
                all_text_hidden_states = all_text_hidden_states + (text_hidden_states, )

            # vision
            # TODO: 9-12 layers past text as pkv to vision
            past_key_values = text_layer_output[-1] if idx >= 8 else None
            vision_layer_module = self.vision_layers[idx]
            vision_layer_output = vision_layer_module(
                    vision_hidden_states,
                    output_attentions=output_attentions,
                    past_key_values=past_key_values,
            )
            vision_hidden_states = vision_layer_output[0]

            # text
            # TODO: 9-12 layers past vison qks to text
            last_hidden_state = vision_hidden_states if idx >= 8 else None
            output_qks = True if idx >= 7 else None
            layer_head_mask = head_mask[idx] if head_mask is not None else None
            text_layer_module = self.text_layer[idx]
            text_layer_output = text_layer_module(
                    text_hidden_states,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    visual_hidden_state=last_hidden_state,
                    output_attentions=output_attentions,
                    output_qks=output_qks,
                    sep_idx=sep_idx,
            )
            text_hidden_states = text_layer_output[0]
            if output_attentions:
                all_vision_attentions = all_vision_attentions + (vision_layer_output[1], )
                all_text_attentions = all_text_attentions + (text_layer_output[1], )

        if output_hidden_states:
                all_vision_hidden_states = all_vision_hidden_states + (vision_hidden_states, )
                all_text_hidden_states = all_text_hidden_states + (text_hidden_states, )

        if not return_dict:
            return tuple(
                v for v in [
                    text_hidden_states,
                    all_text_hidden_states,
                    all_text_attentions,
                ] if v is not None)
        return BaseModelOutput(
            last_hidden_state=text_hidden_states,
            hidden_states=all_text_hidden_states,
            attentions=all_text_attentions
        )


class BertPooler(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size, activation='tanh')

    def construct(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return pooled_output


class UnimoModel(nn.Cell):
    def __init__(self, vision_config, text_config, add_pooling_layer=True):
        super(UnimoModel, self).__init__()
        # vision model
        self.vision_config = vision_config
        self.vision_embeddings = CLIPVisionEmbeddings(vision_config)
        self.vision_pre_layrnorm = nn.LayerNorm((vision_config.hidden_size,))
        self.vision_post_layernorm = nn.LayerNorm((vision_config.hidden_size,))

        # text model
        self.text_config = text_config
        self.text_embeddings = BertEmbeddings(text_config)
        self.text_pooler = BertPooler(text_config) if add_pooling_layer else None

        # all
        self.encoder = UnimoEncoder(vision_config, text_config)

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        sep_idx=None,

        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # pre vision
        vision_embedding_output = self.vision_embeddings(pixel_values)
        vision_embedding_output = self.vision_pre_layrnorm(vision_embedding_output)

        # pre text
        input_shape = input_ids.shape
        batch_size, seq_length = input_shape
        if attention_mask is None:
            attention_mask = ops.ones(((batch_size, seq_length)))
        if token_type_ids is None:
            if hasattr(self.text_embeddings, "token_type_ids"):
                buffered_token_type_ids = self.text_embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = \
                    buffered_token_type_ids.broadcast_to((batch_size, seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=mindspore.int32)


        extended_attention_mask: ops.Tensor = get_extended_attention_mask(attention_mask,
                                                                          input_shape)
        head_mask = get_head_mask(head_mask, self.text_config.num_hidden_layers)    # [None]*12

        text_embedding_output = self.text_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        # all encoder
        encoder_outputs = self.encoder(
            vision_embeds=vision_embedding_output,
            text_embeds=text_embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            sep_idx=sep_idx,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.text_pooler(sequence_output) if self.text_pooler is not None \
                                                else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _init_text_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Dense):
            # cf https://github.com/pyops/pyops/pull/5617
            module.weight.set_data(initializer(Normal(self.text_config.initializer_range),
                                                    module.weight.shape, module.weight.dtype))
            if module.has_bias:
                module.bias.set_data(initializer('zeros', module.bias.shape, module.bias.dtype))

        elif isinstance(module, nn.Embedding):
            embedding_table = initializer(Normal(self.text_config.initializer_range),
                                                 module.embedding_table.shape,
                                                 module.embedding_table.dtype)
            if module.padding_idx:
                embedding_table[module.padding_idx] = 0
            module.embedding_table.set_data(embedding_table)
        elif isinstance(module, nn.LayerNorm):
            module.gamma.set_data(initializer('ones', module.gamma.shape, module.gamma.dtype))
            module.beta.set_data(initializer('zeros', module.beta.shape, module.beta.dtype))

    def get_input_embeddings(self):
        return self.text_embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.text_embeddings.word_embeddings = value

    def resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.embedding_table.shape
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)

        # initialize all new embeddings (in particular added tokens)
        self._init_text_weights(new_embeddings)

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.embedding_table.data[:num_tokens_to_copy, :] = \
            old_embeddings.embedding_table.data[:num_tokens_to_copy, :]

        return new_embeddings


class UnimoForMaskedLM(nn.Cell):
    def __init__(self, vision_config, text_config):
        super().__init__()
        self.unimo = UnimoModel(vision_config, text_config)
        self.cls = UnimoOnlyMLMHead(text_config)
        self.config = text_config

        self.tie_weights()

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        sep_idx=None,

        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        outputs = self.unimo(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            sep_idx=sep_idx,
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores, trans_hidden_states = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size),
                                       labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), trans_hidden_states

    def get_input_embeddings(self):
        return self.unimo.text_embeddings.word_embeddings

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def tie_weights(self):
        output_embeddings = self.get_output_embeddings()
        self._tie_or_clone_weights(output_embeddings, self.unimo.get_input_embeddings())

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        output_embeddings.weight = input_embeddings.embedding_table
        output_embeddings._params['weight'] = input_embeddings.embedding_table
        if output_embeddings.has_bias:
            output_embeddings.bias.set_data(ops.pad(
                output_embeddings.bias.data,
                (0, output_embeddings.weight.shape[0] -
                 output_embeddings.bias.shape[0]),
                "constant",
                0,
            ))
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_channels = input_embeddings.vocab_size

    def resize_token_embeddings(self, new_num_tokens):
        self.unimo.resize_token_embeddings(new_num_tokens)
        self.tie_weights()

class UnimoOnlyMLMHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.predictions = UnimoLMPredictionHead(config)

    def construct(self, sequence_output):
        prediction_scores, trans_hidden_states = self.predictions(sequence_output)
        return prediction_scores, trans_hidden_states


class UnimoLMPredictionHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        self.bias = Parameter(initializer('zeros', config.vocab_size), 'decoder.bias')

        self.decoder.bias = self.bias

    def construct(self, hidden_states):
        trans_hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(trans_hidden_states)
        return hidden_states, trans_hidden_states


class BertPredictionHeadTransform(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


if __name__ == '__main__':
    from mindnlp.models import BertConfig, CLIPVisionConfig, BertModel, CLIPModel

    bert_config = BertConfig.from_json_file('.mindnlp/models/bert-base-uncased/config.json')
    # bert = BertModel.from_pretrained('bert-base-uncased')
    clip_config = CLIPVisionConfig.from_json_file('/data/lilei/models/research/mm/mkg_analogy/mind_models/clip-vit-base-patch32/vision_config.json')

    model = UnimoForMaskedLM(clip_config, bert_config)

    input_ids = ops.randint(0, 100, (2, 128))
    attention_mask = ops.ones((2, 128))
    pixel_values = ops.randint(0, 255, (2, 2, 3, 224, 224)).astype(mindspore.float32)

    output = model(input_ids=input_ids,
                   attention_mask=attention_mask,
                   pixel_values=pixel_values)
