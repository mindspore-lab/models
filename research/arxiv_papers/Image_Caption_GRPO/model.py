import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from module.utils import Embeddings
from module.resnet import resnet50

class CaptionModel(nn.Cell):
    def __init__(self, config) -> None:
        super(CaptionModel, self).__init__()

        self.head_nums = config.head_nums
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id

        self.generation_length = config.generation_length
        self.img_length = config.img_length
        self.max_length = config.max_length
        self.sentence_nums = config.sentence_nums
        self.decoder_layer_nums = config.decoder_layer_nums

        self.mask = ops.triu(ops.full((self.max_length, self.max_length), fill_value = float(-10000), dtype = ms.float32), diagonal = 1)

        self.image_encoder = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1, pretrained_ckpt=config.resnet_model)
        self.map = nn.Dense(config.image_embedding_dim, config.hidden_dim)
        self.caption_encoder = Embeddings(config)

        decoder_layer = nn.TransformerDecoderLayer(d_model = config.hidden_dim, nhead = config.head_nums, dropout = config.dropout, batch_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = config.decoder_layer_nums)

        self.classify = nn.Dense(config.hidden_dim, config.vocab_size)
        self.loss_fun = nn.CrossEntropyLoss(reduction = 'sum', ignore_index = config.pad_token_id)

    def construct(self, img, caption_index = None, label = None, img_id = None, img_embed = None):
        if img_embed is None:
            img_embedding = self.image_encoder(img)
            img_embedding = img_embedding.reshape(img_embedding.shape[0], img_embedding.shape[1], -1).permute(0, 2, 1)
            img_embedding = self.map(img_embedding)
        else:
            img_embedding = img_embed

        if caption_index is None:
            return img_embedding

        if caption_index.dim() == 3:
            img_embedding = img_embedding.unsqueeze(1).repeat_interleave(caption_index.shape[1], 1).reshape(-1, img_embedding.shape[-2], img_embedding.shape[-1])
            caption_index = caption_index.reshape(-1, caption_index.shape[-1])

        padding_mask = ((caption_index != self.pad_token_id) & (caption_index != self.eos_token_id)).to(ms.float32)
        caption_embedding = self.caption_encoder(caption_index)

        out = caption_embedding
        caption_mask = self.mask[:caption_embedding.shape[1], :caption_embedding.shape[1]]
        out = self.decoder(tgt = out, memory = img_embedding, tgt_mask = caption_mask, tgt_key_padding_mask = ~(padding_mask > 0))

        pred = self.classify(out)

        if label is None:
            return pred
        else:
            pred = pred.reshape(-1, self.vocab_size)
            label = label.reshape(-1)
            loss = self.loss_fun(pred, label)
            loss = loss / ops.sum(padding_mask)
            return loss