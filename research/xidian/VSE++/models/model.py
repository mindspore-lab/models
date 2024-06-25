import mindspore
import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor
from mindspore import ops, load_param_into_net
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from models import model_vgg19


def l2norm(X):
    norm = ops.pow(X, 2)
    norm = norm.sum(axis=1).sqrt()

    X = ops.div(X, ops.expand_dims(norm, 1))

    return X


def EncoderImage(data_name, img_dim, embed_size, finetune=False,
                 cnn_type='vgg19', use_abs=False, no_imgnorm=False):

    #img_enc = EncoderImageFull(
    #    embed_size, finetune, cnn_type, use_abs, no_imgnorm)
    img_enc = EncoderImagePrecomp(
            img_dim, embed_size, use_abs, no_imgnorm)	
    return img_enc

class EncoderImagePrecomp(nn.Cell):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.fc = nn.Dense(img_dim, embed_size)

        self.init_weights()



    def init_weights(self):
        r = np.sqrt(6.) / np.sqrt(self.fc.in_channels +
                                  self.fc.out_channels)
        self.fc.weight.data.set_data(ops.uniform(self.fc.weight.data.shape, Tensor(-r, mindspore.float32),
                                                 Tensor(r, mindspore.float32), dtype=mindspore.float32))

        self.fc.bias.data.set_data(mindspore.numpy.full(self.fc.bias.data.shape, Tensor(0, mindspore.float32)))

    def construct(self, images):
        features = self.fc(images)

        if not self.no_imgnorm:
            features = l2norm(features)

        if self.use_abs:
            features = ops.abs(features)

        return features

   
class EncoderImageFull(nn.Cell):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True)

        # For efficient memory usage.
        for param in self.cnn.get_parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            # self.cnn.classifier
            self.fc = nn.Dense(4096,
                               embed_size)
            self.cnn.classifier = nn.SequentialCell(
                *list(self.cnn.classifier)[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Dense(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.SequentialCell()

        self.init_weights()

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = model_vgg19.vgg19(pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = model_vgg19.vgg19(pretrained=False)

        # if arch.startswith('alexnet') or arch.startswith('vgg'):
        #     model.features = nn.DataParallel(model.features)
        # else:
        #     model = nn.DataParallel(model)

        # if torch.cuda.is_available():
        #     model.cuda()
        return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        # super(EncoderImageFull, self).load_state_dict(state_dict)

        load_param_into_net(super(EncoderImageFull, self), state_dict[0])

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_channels +
                                  self.fc.out_channels)
        # fc_weight = np.load("models/fc_weights.npy", allow_pickle=True)
        # self.fc.weight.data.set_data(mindspore.Tensor(fc_weight))
        self.fc.weight.data.set_data(ops.uniform(self.fc.weight.data.shape, Tensor(-r, mindspore.float32),
                                                 Tensor(r, mindspore.float32), dtype=mindspore.float32))
        # self.fc.weight.data.uniform_(-r, r)
        # self.fc.bias.data.fill_(0)
        self.fc.bias.data.set_data(mindspore.numpy.full(self.fc.bias.data.shape, Tensor(0, mindspore.float32)))

    def construct(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = ops.abs(features)

        return features


class EncoderText(nn.Cell):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers=num_layers, batch_first=True)

        self.init_weights()

    def init_weights(self):
        self.embed.init_tensor = ops.uniform(self.embed.init_tensor.shape, Tensor(-0.1, mindspore.float32),
                                             Tensor(0.1, mindspore.float32), dtype=mindspore.float32)

    def construct(self, captions, lengths):

        out = self.embed(captions.astype(mindspore.int64))
        out, _ = self.rnn(out)
        I = lengths.view(-1, 1, 1)
        I = I.broadcast_to((captions.shape[0], 1, self.embed_size)) - 1
        out = ops.gather_elements(out, 1, I).squeeze(1).astype(ms.float32)

        out = l2norm(out)
        if self.use_abs:
            out = ops.abs(out)
        return out


def cosine_sim(im, s):
    return ops.MatMul()(im, s.T)


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).broadcast_to(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).broadcast_to(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Cell):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim
        self.max_violation = max_violation

    def construct(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        # scores = ms.ops.randn(im.shape[0], im.shape[0])
        diagonal = ms.numpy.diag(scores).view(im.shape[0], 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.T.expand_as(scores)

        cost_s = ops.clip_by_value(self.margin + scores - d1, clip_value_min=Tensor(0, mindspore.float32))
        cost_im = ops.clip_by_value(self.margin + scores - d2, clip_value_min=Tensor(0, mindspore.float32))
        # clear diagonals
        mask = ops.eye(scores.shape[0], scores.shape[0], mindspore.float32) > .5
        cost_s = cost_s.masked_fill(mask, 0)
        cost_im = cost_im.masked_fill(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)
            cost_im = cost_im.max(0)

        return cost_s.sum() + cost_im.sum()


class VSE(nn.Cell):
    def __init__(self, opt):
        super(VSE, self).__init__()
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    opt.finetune, opt.cnn_type,
                                    use_abs=opt.use_abs,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_abs=opt.use_abs)

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        load_param_into_net(self.txt_enc, state_dict[1])

    

    def construct(self, images, captions, lengths, ids=None):
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions=captions, lengths=lengths).astype(ms.float32)
        # measure accuracy and record loss
        return img_emb, cap_emb


class ClipGradients(nn.Cell):

    def __init__(self):
        super(ClipGradients, self).__init__()
        self.clip_by_norm = nn.ClipByNorm()
        self.cast = mindspore.ops.operations.Cast()
        self.dtype = mindspore.ops.operations.DType()

    def construct(self,
                  grads,
                  clip_type=1,
                  clip_value=1.0):
        """Defines the gradients clip."""
        if clip_type not in (0, 1):
            return grads
        new_grads = ()
        for grad in grads:
            dt = self.dtype(grad)
            if clip_type == 0:
                t = ms.ops.clip_by_value(grad, self.cast(F.tuple_to_array((-clip_value,)), dt),
                                         self.cast(F.tuple_to_array((clip_value,)), dt))
            else:
                t = self.clip_by_norm(grad, self.cast(F.tuple_to_array((clip_value,)), dt))
            t = self.cast(t, dt)
            new_grads = new_grads + (t,)
        return new_grads


class NetWithLoss(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(NetWithLoss, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, images, captions, lengths, ids):
        im, s = self._backbone(images, captions, lengths, ids)
        loss = self._loss_fn(im, s)
        return loss

    @property
    def backbone_network(self):
        return self._backbone


clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class CustomTrainOneStepCell(nn.TrainOneStepCell):
    def __init__(self, network, optimizer):
        super(CustomTrainOneStepCell, self).__init__(network, optimizer, sens=1)
        self.cast = P.Cast()
        self.hyper_map = C.HyperMap()

    def construct(self, images, captions, index, img_id):
        loss = self.network(images, captions, index, img_id)  # 计算当前输入的损失函数值
        sens_g = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)

        grads = self.grad(self.network, self.weights)(images, captions, index, img_id, sens_g)  # 进行反向传播，计算梯度
        grads = self.hyper_map(F.partial(clip_grad, 0, 2), grads)
        grads = self.grad_reducer(grads)
        return ops.depend(loss, self.optimizer(grads))
