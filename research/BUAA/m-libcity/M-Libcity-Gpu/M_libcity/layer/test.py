import sys
import numpy as npy
import mindspore.numpy as np
from Attention import SpatialAttentionLayer
from AttentionforRNN import RnnAttnOutput
from AVGCN import AVGCN
from LinearInterpolationEmbedding import LinearInterpolationEmbedding
from DCNN import DCNN
from Dense import Dense
from GCGRU import GCGRU
from GCLSTM import GCLSTM
from LocationEmbedding import LocationEmbedding
from PoswiseFeedForward import PositionwiseFeedForward
from SpatialGatedCNN import SpatialGatedCNN
from SpatialViewCNN import SpatialViewCNN
from TCN import TCN
from TemporalViewCNN import TemporalViewCNN
import mindspore
from mindspore import Tensor

def test_Attention():
    #随机矩阵
    """
    ok
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    stdnormal = mindspore.ops.StandardNormal()
    x = stdnormal((32, 307, 3, 24))
    
    net = SpatialAttentionLayer(0, 24, 3, 307)
    output = net(x)
    assert output.shape == (32, 3, 3)
    
def test_AttentionforRNN():
    """
    运算有错误
    eg. net = RnnAttnOutput(24., 24)
    output = net(x) #x.shape = (32, 307, 24)
    output.shape = (32,307,307)
    """
    #随机矩阵
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    stdnormal = mindspore.ops.StandardNormal()
    #(batch, seq_len, hidden_size)
    x = stdnormal((32, 307, 24))
    net = RnnAttnOutput(24., 24)
    output = net(x)
    print(output.shape) #(32,307,307)
    assert output.shape == (32, 24, 24)
    
def test_AVGCN():
    """
    运算错误,第55行x_gconv = x_g_op * weights_op
    x.shape[-2] = [32, 307, 3, 3, 1] and y.shape[-2] = [1, 307, 3, 32, 12]
    """
    #随机矩阵
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    stdnormal = mindspore.ops.StandardNormal()
    x = stdnormal((32, 307, 3))
    node_embeddings = stdnormal((307 ,3))
    
    net = AVGCN(3, 3, 3, 3)
    output = net(x, node_embeddings)
    assert output.shape == (32, 307, 3)
    
def test_DCNN():
    #随机矩阵
    """
    ok
    """
    mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE,device_target='CPU')
    dim=1
    adj=npy.random.randint(0,2,[10,10])
    input = np.randn([32,10,1])
    state = np.randn([32,10,10])
    model=DCNN(adj_mx=adj)
    output=model(input,state)
    print(output.shape)
    assert output.shape == (32,10*10)
    
def test_Dense():
    #随机矩阵
    """
    ok
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    stdnormal = mindspore.ops.StandardNormal()
    #(*,in_channels)
    x = stdnormal((307,32))
    #(in_channels,out_channels)
    net = Dense(32,24)
    output = net(x)
    print(output.shape)
    assert output.shape == (307,24)
    
def test_GCGRU():
    #随机矩阵
    """
    ok
    """
    mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE)
    N=20
    dim=1
    adj=npy.random.randint(0,2,[10,10])
    input = np.randn([1,10*1])
    state = np.randn([1,10*10])
    model=GCGRU(adj_mx=adj)
    output=model(input,state)
    print(output.shape)
    assert output.shape == (1,10*10)
    
def test_GCLSTM():
    #随机矩阵
    """
    ok
    """
    mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE,device_target='CPU')
    dim=1
    adj=np.randn([10,10])
    input = np.randn([32,10,1])
    state = np.randn([2,32,10,10])
    model=GCLSTM(adj=adj)
    output,_=model(input,state)
    print(output.shape)
    assert output.shape == (32,10,10)

def test_LinearInterpolationEmbedding():
    mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE,device_target='CPU')
    dim=1
    loc_len=[4]
    td_lower=mindspore.Tensor([[[0],[0],[0],[0]]])
    td_upper=mindspore.Tensor([[[10],[10],[10],[10]]])
    ld_lower=mindspore.Tensor([[[0],[0],[0],[0]]])
    ld_upper=mindspore.Tensor([[[10],[10],[10],[10]]])
    loc=mindspore.Tensor([[5,5,5,5]])
    model=LinearInterpolationEmbedding()
    output=model(td_upper,td_lower,ld_upper,ld_lower,loc,loc_len)
    print(output.shape)
    assert output.shape == (1,1)
    
def test_LocationEmbedding():
    """
    ok
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    stdnormal = mindspore.ops.StandardNormal()
    #(1, T, 1, C)
    #x = stdnormal((1, 36, 1, 24))
    #(1, 1, N, C)
    x = stdnormal((1, 1, 12, 24))
    #input_length, num_of_vertices, embedding_size
    net = LocationEmbedding(36,12,24,temporal=False)
    output = net(x)
    print(output.shape)
    assert output.shape == (1, 1, 12, 24)

def test_PoswiseFeedForward():
    """
    ok
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    stdnormal = mindspore.ops.StandardNormal()
    #(*,d_in)
    x = stdnormal((36,12))
    #d_in, d_hid
    net = PositionwiseFeedForward(12,24)
    output = net(x)
    print(output.shape)#(36, 12)
    #assert output.shape == (?)
    
def test_SpatialGatedCNN():
    """
    ok
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    stdnormal = mindspore.ops.StandardNormal()
    #(b,c_in,t,N)
    x = stdnormal((36,24,12,64))
    #input_size,output_size
    net = SpatialGatedCNN(24,12)
    output = net(x)
    print(output.shape)
    assert output.shape == (36,12,12,64)
    
def test_SpatialViewCNN():
    """
    ok
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    stdnormal = mindspore.ops.StandardNormal()
    #(b,c_in,t,N)
    x = stdnormal((1, 120, 1024, 640))
    #inp_channel, oup_channel, kernel_size
    net = SpatialViewCNN(120,240,4)
    output = net(x)
    print(output.shape)
    assert output.shape == (1,240,1024,640)
    
def test_TCN():
    """
    ok
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    stdnormal = mindspore.ops.StandardNormal()
    #(b,c_in,t,N)
    x = stdnormal((1, 120, 1024, 640))
    #t_kernel_size, channel_in, channel_out, vertex_num, gate_type
    net = TCN(4,120,240,3,'gtu')
    output = net(x)
    print(output.shape) #(1, 240, 1021, 640)
    
def test_TemporalViewCNN():
    """
    ok
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    stdnormal = mindspore.ops.StandardNormal()
    #(b,t,n,d)
    x = stdnormal((1, 120, 128, 64))
    #input_size, output_size, time_strides
    net = TemporalViewCNN(120,240,4)
    output = net(x)
    print(output.shape) #(1, 240, 8192)

test_Attention()
test_AttentionforRNN()
test_AVGCN()
test_DCNN()
test_Dense()
test_GCGRU()
test_GCLSTM()
test_LinearInterpolationEmbedding()
test_LocationEmbedding()
test_PoswiseFeedForward()
test_SpatialGatedCNN()
test_SpatialViewCNN()
test_TCN()
test_TemporalViewCNN()