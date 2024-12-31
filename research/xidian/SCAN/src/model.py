# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""SCAN model"""
from src.rnns import GRU
from mindspore.ops import functional as F
from mindspore import load_checkpoint, load_param_into_net
import mindspore.numpy
import mindspore.ops as ops
from mindspore import nn
import mindspore as ms
import mindspore.common.initializer as init
from ipdb import set_trace
import numpy as np
from collections import OrderedDict
from time import *
import mindspore.common.dtype as mstype

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    # X:  (128, 36, 1024)
    # dim: -1
    norm = ops.Sqrt()(ops.Pow()(X, 2).sum(axis=dim, keepdims=True)) + eps   #(128, 36, 1)
    X = ops.Div()(X, norm)   #(128, 36, 1024)
    return X



class EncoderImage(nn.Cell):

    def __init__(self, img_dim, embed_size,no_imgnorm):
        super(EncoderImage, self).__init__()
        self.img_dim = img_dim
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Dense(img_dim, embed_size)#.to_float(mstype.float16)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.img_dim + self.embed_size)
        # if isinstance(self.fc, nn.Dense):
        self.fc.weight.set_data(init.initializer(
            init.Uniform(r), self.fc.weight.shape, self.fc.weight.dtype))
        self.fc.bias.set_data(init.initializer(
            init.Constant(0), self.fc.bias.shape, self.fc.bias.dtype))



    def construct(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)

        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)
        return features





# RNN Based Language Model
class EncoderText(nn.Cell):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False,
                batch_size=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        #word_dim    300
        #embed_size  1024
        # word embedding
        w_init = ms.common.initializer.Uniform(scale=0.1)
        self.embed = nn.Embedding(vocab_size, word_dim,embedding_table=w_init)#.to_float(mstype.float16)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        
        self.rnn = GRU(word_dim, 
                          embed_size, 
                          num_layers,
                          has_bias=True,   #Whether the cell has bias b_ih and b_hh.
                          batch_first=True,   #the input and output tensors are provided as (batch, seq, feature)
                          bidirectional=True
                         )


    def construct(self, x, lengths):
        """Handles variable size captions
        x:   (128, n)    float32
        lengths   (128)  int32
        """
        # Embed word ids to vectors
        # print(x.shape)
        # print(lengths.shape)
        #begin_time = time()
        x = self.embed(x)  #(128, n, 300)  float32

        cap_emb, _ = self.rnn(x, seq_length = lengths )   #(128, 37, 2048)

        
        if self.use_bi_gru:
           cap_emb = (cap_emb[:,:,:cap_emb.shape[2]//2] + cap_emb[:,:,cap_emb.shape[2]//2:])/2   #(128, 37, 1024)
        if not self.no_txtnorm:
           cap_emb = l2norm(cap_emb, dim=-1)
        return cap_emb, lengths
    
class Softmax(nn.Cell):
    def __init__(self, axis=-1):
        super(Softmax, self).__init__()
        self.axis = axis
        self.max = ops.ReduceMax(keep_dims=True)
        self.sum = ops.ReduceSum(keep_dims=True)
        self.sub = ops.Sub()
        self.exp = ops.Exp()
        self.div = ops.RealDiv()
        self.cast = ops.Cast()

    def construct(self, x):
        x = self.cast(x, mstype.float32)
        x = self.sub(x, self.max(x, self.axis))
        x = self.div(self.exp(x), self.sum(self.exp(x), self.axis))
        return x

def func_attention(query, context, raw_feature_norm, smooth, eps=1e-8,transpose=None,LeakyReLU=None):
    """
    query: (n_context, queryL, d)   
    context: (n_context, sourceL, d)  
    """
    batch_size_q, queryL = query.shape[0], query.shape[1]      #128   20
    batch_size, sourceL = context.shape[0], context.shape[1]   #128   36

    queryT = transpose(query, (0, 2, 1))
    attn = ops.BatchMatMul()(context, queryT)   #(128, 36, n)
    attn = LeakyReLU(attn)
    attn = l2norm(attn, 2)
    
    
    attn = transpose(attn, (0, 2, 1))   ##(128, 20, 36)
    attn = attn.view(-1, sourceL)   #(128*n, 36)
    attn = Softmax()(attn*smooth)
    attn = attn.view(batch_size, queryL, sourceL)   #(128, n, 36)
    attnT = transpose(attn, (0, 2, 1))#.contiguous()  #(128, 36, n)
    contextT = transpose(context, (0, 2, 1))   #(128, 1024, 36)

    
    weightedContext = ops.BatchMatMul()(contextT, attnT)   #(128, 1024, 12)
    weightedContext = transpose(weightedContext, (0, 2, 1))  #(128, 12, 1024)
    
    return weightedContext, attnT  #(128, 12, 1024)    (128, 36, n)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):    #(128, 15, 1024)
    #x1,x2 (8, 20, 1024)
    """Returns cosine similarity between x1 and x2, computed along dim."""

    l2normalize = ops.L2Normalize(axis=-1, epsilon=1e-4)
    x1 = l2normalize(x1)
    x2 = l2normalize(x2)

    temp = ops.ReduceSum()(x1 * x2,-1)

    return ops.Squeeze()(temp) 


def xattn_score_t2i(images, captions, caption_mask,   ##(128, 87)
                    lambda_softmax,
                    agg_func,
                    lambda_lse,
                    raw_feature_norm):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    masked_select = ops.MaskedSelect()
    transpose = ops.Transpose()
    LeakyReLU = nn.LeakyReLU(0.1)
    
    
    caption_mask = ms.numpy.tile(caption_mask[:, :, None],(1, 1, 1024))
    caption_mask = ops.cast(caption_mask,  ms.bool_)
    similarities = []
    n_image = images.shape[0] #128
    n_caption = captions.shape[0]  #128
    
    for i in range(n_caption):
        cap_i_expand = ms.numpy.tile(captions[i][None,:,:],(n_image, 1, 1))  #(128, 15, 1024)
        
        begin_time3 = time()
        weiContext, attn = func_attention(cap_i_expand, images, raw_feature_norm, smooth=lambda_softmax,transpose=transpose,LeakyReLU=LeakyReLU)   ##(128, 12, 1024)    (128, 36, n)


        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)  #128*n


        if agg_func == 'LogSumExp':
            row_sim = ops.Mul()(row_sim,lambda_lse)
            row_sim = ops.Exp()(row_sim)
            row_sim = row_sim.sum(axis=1, keepdims=True)

            row_sim = ops.Log()(row_sim) / lambda_lse   #128*1
        elif agg_func == 'Mean':
            row_sim = ops.ReduceMean(keep_dims=True)(row_sim,axis=1)
        similarities.append(row_sim)

    similarities = ops.Concat(axis = 1)(similarities) # (128, 128)
    return similarities

#image  (64, 128, 36, 1024)
#caption  (64, 128, 87, 1024)
#each_batch    64
#n_image    37
def func_attention_xin(image,    #(128, 128, 36, 1024)
                       caption,  #(128, 128, 87, 1024)
                       n_image, 
                       lambda_softmax,
                      ):
        transpose = ops.Transpose()
        LeakyReLU = nn.LeakyReLU(0.1)
        softmax = Softmax()
        bmm = ops.BatchMatMul()
        caption = transpose(caption, (0, 1, 3, 2))
        cast = ops.Cast()
#         begin_time = time()
        
        image = cast(image, ms.float16)
        caption = cast(caption, ms.float16)
        attn = bmm(image, caption)   #(128, 128, 36, 87)
        attn = LeakyReLU(attn)

        attn = l2norm(attn, -1)                #(128, 128, 36, 87)
        attn = transpose(attn, (0, 1, 3, 2))    #(128, 128, 87, 36)
        
        
        attn = softmax(attn*lambda_softmax)              ##(128, 128, 87, 36)
        attn = cast(attn, ms.float16)
        weightedContext = bmm(attn, image)   #(64, 128, 87, 1024)

        
        return weightedContext



def xattn_score_t2i_xin(images,   #(128, 36, 1024)
                    captions,  #(128, 87, 1024)
                    caption_masks,  #(128, 87)
                    lambda_softmax,
                    agg_func,
                    lambda_lse,
                    raw_feature_norm):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    
    batch_size = captions.shape[0]
    
    captions = ms.numpy.tile(captions[:,None,:,:],(1, batch_size, 1, 1))   #(128, 128, 87, 1024)
    images = ms.numpy.tile(images[None,:,:,:],(batch_size, 1, 1, 1))     #(128, 128, 36, 1024)
    caption_masks = ms.numpy.tile((caption_masks)[:,None,:],( 1, batch_size, 1))   #(128, 128, 87)


    weightedContext = func_attention_xin(images, captions,       
                                         batch_size,          #128
                                         lambda_softmax,   #6
                                          )   #(64, 128, 87, 1024)


    row_sim = cosine_similarity(captions, weightedContext, dim=-1) * caption_masks  #128*n   (128, 128, 87)

    if agg_func == 'LogSumExp':
        row_sim = ops.Mul()(row_sim,lambda_lse)
        row_sim = ops.Exp()(row_sim)
        row_sim = row_sim.sum(axis=-1)
        row_sim = ops.Log()(row_sim) / lambda_lse   
    elif agg_func == 'Mean':
        row_sim = ops.ReduceMean(keep_dims=True)(row_sim,axis=1)
    return row_sim.T   





class ContrastiveLoss(nn.Cell):
    def __init__(self, lambda_softmax, agg_func, lambda_lse, cross_attn, raw_feature_norm, margin, max_violation):
        super(ContrastiveLoss, self).__init__()
        self.cross_attn = cross_attn
        self.margin = margin
        self.max_violation = max_violation
        self.lambda_softmax = lambda_softmax
        self.agg_func = agg_func
        self.lambda_lse = lambda_lse
        self.raw_feature_norm = raw_feature_norm
        self.eye =  ops.Eye()

        self.length_list = []
        self.cast = ops.Cast()

    def construct(self, im, s, s_l):
        #im   (128, 36, 1024)
        scores=None

        if self.cross_attn == 't2i':   
            scores = xattn_score_t2i_xin(im, s, s_l,
                                     lambda_softmax = self.lambda_softmax,
                                     agg_func = self.agg_func,
                                     lambda_lse = self.lambda_lse,
                                     raw_feature_norm = self.raw_feature_norm)   #128*128

        diagonal = ms.numpy.diag(scores).view(im.shape[0], 1)

        d1 = diagonal.expand_as(scores)
        d2 = diagonal.T.expand_as(scores)
        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = ms.ops.clip_by_value(self.margin + scores - d1 , 0., (self.margin + scores - d1).max())
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = ms.ops.clip_by_value(self.margin + scores - d2 , 0., (self.margin + scores - d2).max())

        # clear diagonals
        eye_size = scores.shape[0]
        I = self.eye(eye_size,eye_size,ms.int32) > 0.5

        cost_s = cost_s.masked_fill(I, 0)
        cost_im = cost_im.masked_fill(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:   #执行
            cost_s = cost_s.max(axis=1)
            cost_im = cost_im.max(axis=0)
        loss_ = cost_s.sum() + cost_im.sum()
        return loss_


#model  loss
class BuildValNetwork(nn.Cell):
    def __init__(self, img_enc, txt_enc, criterion):
        super(BuildValNetwork, self).__init__()
        self.net_image = img_enc
        self.net_caption = txt_enc
        self.criterion = criterion

    def construct(self, images, captions, lengths,lengths_int ):
        #images   (128, 36, 2048)
        #captions       (128,  n)
        #lengths        (128, 1)
        #ids            (128, 1)
        # 
        img_emb = self.net_image(images)
        cap_emb, cap_lens = self.net_caption(captions, lengths)
        
        return img_emb, cap_emb



class BuildTrainNetwork(nn.Cell):
    def __init__(self, img_enc, txt_enc, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.net_image = img_enc
        self.net_caption = txt_enc
        self.criterion = criterion
        
    def construct(self, images, captions, lengths, lengths_int):
        #images   (128, 36, 2048)
        #captions       (128,  n)
        #lengths        (128, 1)
        #ids            (128, 1)
        
        img_emb = self.net_image(images)
        cap_emb, cap_lens = self.net_caption(captions, lengths)
        loss = self.criterion(img_emb, cap_emb, lengths_int)

        return loss



class ClipGradients(nn.Cell):
    """
    Clip gradients.

    Args:
        grads (list): List of gradient tuples.
        clip_type (Tensor): The way to clip, 'value' or 'norm'.
        clip_value (Tensor): Specifies how much to clip.

    Returns:
        List, a list of clipped_grad tuples.
    """
    def __init__(self):
        super(ClipGradients, self).__init__()
        self.clip_by_norm = nn.ClipByNorm()
        self.cast = mindspore.ops.operations.Cast()
        self.dtype = mindspore.ops.operations.DType()
    def construct(self,
                  grads,
                  clip_type = 1,
                  clip_value = 1.0):
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






class CustomTrainOneStepCell(nn.Cell):


    def __init__(self, network, optimizer):
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network                           
        self.network.set_grad()                          
        self.optimizer = optimizer                       
        self.weights = self.optimizer.parameters         
        self.grad = ops.GradOperation(get_by_list=True)
        self.clip_grad = ClipGradients()   

    def construct(self, *inputs):
        loss = self.network(*inputs)                            
        grads = self.grad(self.network, self.weights)(*inputs)  

        loss = F.depend(loss, self.optimizer(grads))                   
        return loss

    
    
def save_state_dict(img_enc, txt_enc,prefix,epoch,is_best=False):
    # state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
    # return state_dict
    if not is_best:
        filename = 'checkpoint_{}.ckpt'.format(epoch)
        image_path = prefix + "image" + filename
        text_path = prefix + "text" + filename
        ms.save_checkpoint(img_enc, image_path)
        ms.save_checkpoint(txt_enc, text_path)
    else:
        image_path = prefix + "image" + 'model_best.ckpt'
        text_path = prefix + "text" + 'model_best.ckpt'
        ms.save_checkpoint(img_enc, image_path)
        ms.save_checkpoint(txt_enc, text_path)

class SCAN(object):
    """
    Stacked Cross Attention Network (SCAN) model
    """
    def __init__(self, opt,train_dataset_len):

        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm,
                                  batch_size=opt.batch_size)


        self.criterion = ContrastiveLoss(lambda_softmax = opt.lambda_softmax, 
                                         agg_func = opt.agg_func, 
                                         lambda_lse = opt.lambda_lse, 
                                         cross_attn = opt.cross_attn,
                                         raw_feature_norm = opt.raw_feature_norm,
                                         margin = opt.margin,
                                         max_violation = opt.max_violation)
        self.trainloss = BuildTrainNetwork(self.img_enc,self.txt_enc, self.criterion)
        self.valnet = BuildValNetwork(self.img_enc,self.txt_enc, self.criterion)



        batch_each_epoch = (train_dataset_len // 128) + 1
        milestone = []
        learning_rates = []
        for i in range(opt.num_epochs):
            milestone.append((i+1)*batch_each_epoch)
            learning_rates.append(opt.learning_rate * (0.1 ** (i // opt.lr_update)))
        output = nn.dynamic_lr.piecewise_constant_lr(milestone, learning_rates)
        params = list(self.txt_enc.trainable_params())
        params += list(self.img_enc.fc.trainable_params())
        self.params = params
        self.optimizer = nn.Adam(self.params, learning_rate=output)
        self.model = CustomTrainOneStepCell(self.trainloss, self.optimizer)
        self.squeeze = ops.Squeeze()

    def save_state_dict(self,prefix,epoch,is_best=False):
        # state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        # return state_dict
        if not is_best:
            filename = 'checkpoint_{}.ckpt'.format(epoch)
            image_path = prefix + "image" + filename
            text_path = prefix + "text" + filename
            ms.save_checkpoint(self.img_enc, image_path)
            ms.save_checkpoint(self.txt_enc, text_path)
        else:
            image_path = prefix + "image" + 'model_best.ckpt'
            text_path = prefix + "text" + 'model_best.ckpt'
            ms.save_checkpoint(self.img_enc, image_path)
            ms.save_checkpoint(self.txt_enc, text_path)




    def load_state_dict(self, image_weight_path,text_weight_path):
        image_param_dict = load_checkpoint(image_weight_path)
        load_param_into_net(self.img_enc, image_param_dict)
        text_param_dict = load_checkpoint(text_weight_path)
        load_param_into_net(self.txt_enc, text_param_dict)

        # set_trace()

    def train_start(self):
        """switch to train mode
        """
        self.model.set_train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.valnet.set_train(False)

    def val_emb(self, images, captions, lengths,volatile):
        lengths = self.squeeze(lengths)
        lengths_int = lengths.asnumpy().tolist()
        img_emb, cap_emb, cap_lens = self.valnet(images, captions, lengths,lengths_int)  #,val_loss
        return img_emb, cap_emb, cap_lens#,val_loss

    def train_emb(self, images, captions, lengths, ids=None, batch_id=None):
        begin_time = time()
        lengths = self.squeeze(lengths)
        lengths_int = lengths.asnumpy().tolist()
        result = self.model(images, captions, lengths, lengths_int)
        end_time = time()
        run_time = end_time-begin_time
        print ('each train time',run_time)