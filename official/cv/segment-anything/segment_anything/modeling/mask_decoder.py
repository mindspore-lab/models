import mindspore as ms
from mindspore import nn, ops, mint
import mindspore.mint.nn as mnn
import mindspore.mint.nn.functional as F
from typing import Tuple, Type

from .common import LayerNorm2d, GELU


class MaskDecoder(mnn.Cell):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: mnn.Cell,
        num_multimask_outputs: int = 3,
        activation: Type[mnn.Cell] = GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.SequentialCell(
            nn.Conv2dTranspose(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2,
                               pad_mode='pad', has_bias=True),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.Conv2dTranspose(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2,
                               pad_mode='pad', has_bias=True),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.CellList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def construct(
        self,
        image_embeddings: ms.Tensor,
        image_pe: ms.Tensor,
        sparse_prompt_embeddings: ms.Tensor,
        dense_prompt_embeddings: ms.Tensor,
        multimask_output: bool,
        output_best_mask: bool = False,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (ms.Tensor): the embeddings from the image encoder
          image_pe (ms.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (ms.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (ms.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          ms.Tensor: batched predicted masks
          ms.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # remove dynamic shape, for training
        if output_best_mask:
            ind = ops.stop_gradient(mint.argmax(iou_pred[:, 1:], dim=1, keepdim=True)) + 1  # (bs, 1)
            best_iou = ops.gather(iou_pred, input_indices=ind, axis=1, batch_dims=1)  # (bs, 1)
            best_mask = ops.gather(masks, input_indices=ind, axis=1, batch_dims=1) # (bs, 1, h, w)
            best_mask = int(multimask_output) * best_mask + (1 - int(multimask_output)) * masks[:, :1]  # (bs, 1, h, w)
            best_iou = int(multimask_output) * best_iou + (1 - int(multimask_output)) * iou_pred[:, :1]  # (bs, 1)
            return best_mask, best_iou

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: ms.Tensor,
        image_pe: ms.Tensor,
        sparse_prompt_embeddings: ms.Tensor,
        dense_prompt_embeddings: ms.Tensor,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = mint.cat([self.iou_token.embedding_table, self.mask_tokens.embedding_table], dim=0)
        output_tokens = output_tokens.unsqueeze(0).repeat(sparse_prompt_embeddings.shape[0], axis=0)
        tokens = mint.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = mint.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = mint.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.swapaxes(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = mint.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(mnn.Cell):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.CellList(list(
            mnn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        ))
        self.sigmoid_output = sigmoid_output

    def construct(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = mint.sigmoid(x)
        return x
