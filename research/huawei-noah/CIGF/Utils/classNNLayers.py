import numpy as np
import mindspore
from mindspore.common.initializer import initializer, XavierNormal
from mindspore import Parameter
from mindspore.ops import operations as P
from mindspore.nn import Dropout
import mindspore.ops as ops

paramId = 0
biasDefault = False
params = {}
regParams = {}
ita = 0.2
leaky = 0.1


def getOrDefineParam(name, shape, dtype=mindspore.float32, reg=False, initia='xavier', trainable=True, reuse=False):
	global params
	global regParams
	if name in params:
		assert reuse, 'Reusing Param %s Not Specified' % name
		if reg and name not in regParams:
			regParams[name] = params[name]
		return params[name]
	return defineParam(name, shape, dtype, reg, initia, trainable)

def getParamId():
	global paramId
	paramId += 1
	return paramId

def Bias(data, name=None, reg=False, reuse=False, initia='zeros'):
	inDim =  ops.Shape()(data)[-1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	temBiasName = temName + 'Bias'
	bias = getOrDefineParam(temBiasName, inDim, reg=False, initia=initia, reuse=reuse)
	if reg:
		regParams[temBiasName] = bias
	return data + bias

def Activate(data, method, useBN=False):
	global leaky
	if useBN:
		ret = BN(data)
	else:
		ret = data
	ret = ActivateHelp(ret, method)
	return ret

def ActivateHelp(data, method):
	if method == 'relu':
		ret = P.ReLU()(data)
	elif method == 'sigmoid':
		ret = P.Sigmoid()(data)
	elif method == 'tanh':
		ret = P.Tanh()(data)
	elif method == 'softmax':
		ret = P.Softmax()(data)
	elif method == 'leakyRelu':
		ret = P.Maximum()(leaky*data, data)
	elif method == 'twoWayLeakyRelu6':
		temMask = P.Greater()(data, 6.0).to_float(mindspore.float32)
		ret = temMask * (6 + leaky * (data - 6)) + (1 - temMask) * P.Maximum()(leaky * data, data)
	elif method == '-1relu':
		ret = P.Maximum()(-1.0, data)
	elif method == 'relu6':
		ret = P.Maximum()(0.0, P.Minimum()(6.0, data))
	elif method == 'relu3':
		ret = P.Maximum()(0.0, P.Minimum()(3.0, data))
	else:
		raise Exception('Error Activation Function')
	return ret

class FC(nn.Cell):
    def __init__(self, inp, outDim, name=None, useBias=False, activation=None, reg=False, useBN=False, dropout=None, initia='xavier', reuse=False, biasReg=False, biasInitializer='zeros'):
        self.useBias = useBias
        self.inp = inp
        self.name = name
        self.outDim = outDim
        self.reg = reg
        self.initia = initia
        self.reuse = reuse
        self.drop_out = dropout
        self.useBias = useBias
        global params
        global regParams
        global leaky

    def construct(self,x):
        inDim = ops.Shape()(self.inp)[1]
        temName = self.name if self.name!=None else 'defaultParamName%d'%getParamId()
        W = getOrDefineParam(temName, [inDim, self.outDim], reg=self.reg, initia=self.initia, reuse=self.reuse)
        if self.dropout != None:
            ret = Dropout(p=dropout)(inp) @ W
        else:
            ret = inp @ W
        if useBias:
            ret = Bias(ret, name=name, reuse=reuse, reg=biasReg, initia=biasInitializer)
        if activation != None:
            ret = Activate(ret, activation)
        return ret