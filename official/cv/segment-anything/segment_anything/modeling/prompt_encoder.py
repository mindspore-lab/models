import numpy as np
import mindspore as ms
from mindformers import Blip2Classifier
from mindspore import nn, ops

from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d


class PromptEncoder(nn.Cell):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Cell] = nn.GELU,
        text_encoder=None,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.CellList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.SequentialCell(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2, has_bias=True),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2, has_bias=True),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1, has_bias=True),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

        self.text_embeddings = nn.Embedding(1, embed_dim)
        self.text_encoder: Blip2Classifier = text_encoder

    def get_dense_pe(self) -> ms.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          ms.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: ms.Tensor,
        labels: ms.Tensor,
        pad: bool,
    ) -> ms.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = ops.zeros((points.shape[0], 1, 2))
            padding_label = -ops.ones((labels.shape[0], 1), dtype=ms.int32)
            points = ops.cat([points, padding_point], axis=1)
            labels = ops.cat([labels, padding_label], axis=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        # point_embedding[labels == -1] = 0.0
        # point_embedding[labels == -1] += self.not_a_point_embed.embedding_table
        # point_embedding[labels == 0] += self.point_embeddings[0].embedding_table
        # point_embedding[labels == 1] += self.point_embeddings[1].embedding_table

        mask = ops.broadcast_to((labels==-1).unsqueeze(-1), point_embedding.shape).astype(ms.float32)
        point_embedding =  point_embedding * ops.logical_not(mask).astype(ms.float32) \
                           +  mask * ops.expand_dims(self.not_a_point_embed.embedding_table, axis=1)

        mask = ops.broadcast_to((labels==0).unsqueeze(-1), point_embedding.shape).astype(ms.float32)
        point_embedding += mask * ops.expand_dims(self.point_embeddings[0].embedding_table, axis=1)

        mask = ops.broadcast_to((labels==1).unsqueeze(-1), point_embedding.shape).astype(ms.float32)
        point_embedding += mask * ops.expand_dims(self.point_embeddings[1].embedding_table, axis=1)

        return point_embedding

    def _embed_boxes(self, boxes: ms.Tensor) -> ms.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].embedding_table
        corner_embedding[:, 1, :] += self.point_embeddings[3].embedding_table
        return corner_embedding

    def _embed_masks(self, masks: ms.Tensor) -> ms.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _embed_texts(self, texts: ms.Tensor) -> ms.Tensor:
        """Embeds text inputs."""
        text_embedding = texts + self.text_embeddings.embedding_table
        return text_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[ms.Tensor, ms.Tensor]],
        boxes: Optional[ms.Tensor],
        masks: Optional[ms.Tensor],
        texts: Optional[ms.Tensor] = None,
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        elif texts is not None:
            return texts.shape[0]
        else:
            return 1

    def construct(
        self,
        points: Optional[Tuple[ms.Tensor, ms.Tensor]],
        boxes: Optional[ms.Tensor],
        masks: Optional[ms.Tensor],
        texts: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(ms.Tensor, ms.Tensor) or none): point coordinates
            and labels to embed.
          boxes (ms.Tensor or none): boxes to embed
          masks (ms.Tensor or none): masks to embed
          texts (ms.Tensor or none): blip2 outputted text(inference) or image(training) features  to embed

        Returns:
          ms.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          ms.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks, texts)
        # sparse_embeddings = ms.numpy.empty((bs_prompt, 0, self.embed_dim))
        sparse_embeddings = None  # (bs_prompt, sum_of_prompts, embed_dim)
        if points is not None:
            coords, labels = points  # (bs_prompt, num_point, 2), multiple point a time
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None)) # (bs_prompt, num_point, embed_dim)
            sparse_embeddings = point_embeddings if sparse_embeddings is None \
                else ops.cat([sparse_embeddings, point_embeddings], axis=1)
        if boxes is not None:  # (bs_prompt, 4)  # one box a time
            box_embeddings = self._embed_boxes(boxes)  # (bs_prompt, 2, embed_dim), 1 box worth 2 points
            sparse_embeddings = box_embeddings if sparse_embeddings is None \
                else ops.cat([sparse_embeddings, box_embeddings], axis=1)

        if texts is not None:
            assert len(texts.shape) == 2
            texts = texts.unsqueeze(1)  # (bs_prompt, 1, embed_dim)  # one text a time
            text_embeddings = self._embed_texts(texts)  # (bs_prompt, 1, 256)
            sparse_embeddings = text_embeddings if sparse_embeddings is None \
                else ops.cat([sparse_embeddings, text_embeddings], axis=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.embedding_table.reshape(1, -1, 1, 1).broadcast_to(
                (bs, self.no_mask_embed.embedding_table.shape[1],
                 self.image_embedding_size[0], self.image_embedding_size[1])
            )

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Cell):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.positional_encoding_gaussian_matrix = \
            ms.Parameter(scale * ops.randn(2, num_pos_feats), requires_grad=False)

    def _pe_encoding(self, coords: ms.Tensor) -> ms.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        # aa = coords @ self.positional_encoding_gaussian_matrix
        dtype = coords.dtype
        coords = ops.matmul(coords, self.positional_encoding_gaussian_matrix.astype(dtype))
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return ops.cat([ops.sin(coords), ops.cos(coords)], axis=-1)

    def construct(self, size: Tuple[int, int]) -> ms.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        grid = ops.ones((h, w), dtype=ms.float32)
        y_embed = grid.cumsum(axis=0) - 0.5
        x_embed = grid.cumsum(axis=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(ops.stack([x_embed, y_embed], axis=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: ms.Tensor, image_size: Tuple[int, int]
    ) -> ms.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.copy()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(ms.float32))  # B x N x C
