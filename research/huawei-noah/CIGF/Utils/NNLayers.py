import numpy as np
import mindspore
from mindspore.common.initializer import initializer, XavierNormal
from mindspore import Parameter
from mindspore.ops import operations as P
from mindspore.nn import Dropout
import mindspore.ops as ops
from mindspore import nn

paramId = 0
biasDefault = False
params = {}
regParams = {}
ita = 0.2
leaky = 0.1

def getParamId():
	global paramId
	paramId += 1
	return paramId

def setIta(ITA):
	ita = ITA

def setBiasDefault(val):
	global biasDefault
	biasDefault = val

def getParam(name):
	return params[name]

def addReg(name, param):
	global regParams
	if name not in regParams:
		regParams[name] = param
	else:
		print('ERROR: Parameter already exists')

def addParam(name, param):
	global params
	if name not in params:
		params[name] = param

def defineRandomNameParam(shape, dtype=mindspore.float32, reg=False, initia='xavier', trainable=True):
	name = 'defaultParamName%d'%getParamId()
	return defineParam(name, shape, dtype, reg, initia, trainable)

def defineParam(name, shape,requires_grad=False, dtype=mindspore.float32, reg=False, initia='xavier', trainable=True):
	global params
	global regParams
	assert name not in params, 'name %s already exists' % name
	if initia == 'xavier':
		ret = Parameter(initializer("xavier_uniform", shape, dtype), name=name, requires_grad=requires_grad)
	elif initia == 'trunc_normal':
		ret = Parameter(initializer("truncatedNormal", shape, dtype), name=name, requires_grad=requires_grad)
	elif initia == 'zeros':
		ret = Parameter(initializer("zeros", shape, dtype), name=name, requires_grad=requires_grad)
	elif initia == 'ones':
		ret = Parameter(initializer("ones", shape, dtype), name=name, requires_grad=requires_grad)
	else:
		print('ERROR: Unrecognized initializer')
		exit()
	params[name] = ret
	if reg:
		regParams[name] = ret
	return ret

def getOrDefineParam(name, shape,requires_grad, dtype=mindspore.float32, reg=False, initia='xavier', trainable=True, reuse=False):
	global params
	global regParams
	if name in params:
		assert reuse, 'Reusing Param %s Not Specified' % name
		if reg and name not in regParams:
			regParams[name] = params[name]
		return params[name]
	return defineParam(name, shape,requires_grad, dtype, reg, initia, trainable)

def BN(inp, name=None):
	global ita
	dim = inp.get_shape()[1]
	name = 'defaultParamName%d'%getParamId()
	scale = tf.Variable(tf.ones([dim]))
	shift = tf.Variable(tf.zeros([dim]))
	fcMean, fcVar = tf.nn.moments(inp, axes=[0])
	ema = tf.train.ExponentialMovingAverage(decay=0.5)
	emaApplyOp = ema.apply([fcMean, fcVar])
	with tf.control_dependencies([emaApplyOp]):
		mean = tf.identity(fcMean)
		var = tf.identity(fcVar)
	ret = tf.nn.batch_normalization(inp, mean, var, shift,
		scale, 1e-8)
	return ret

# def FC(inp, outDim, name=None, useBias=False, activation=None, reg=False, useBN=False, dropout=None, initia='xavier', reuse=False, biasReg=False, biasInitializer='zeros'):
# 	global params
# 	global regParams
# 	global leaky
# 	inDim = ops.Shape()(inp)[1]
# 	temName = name if name!=None else 'defaultParamName%d'%getParamId()
# 	W = getOrDefineParam(temName, [inDim, outDim],requires_grad=True, reg=reg, initia=initia, reuse=reuse)
# 	if dropout != None:
# 		ret = Dropout(p=dropout)(inp) @ W
# 	else:
# 		ret = inp @ W
# 	if useBias:
# 		ret = Bias(ret, name=name, reuse=reuse, reg=biasReg, initia=biasInitializer)
# 	if useBN:
# 		ret = BN(ret)
# 	if activation != None:
# 		ret = Activate(ret, activation)
# 	return ret

class FC(nn.Cell):
    def __init__(self,indim,outDim, name=None, useBias=True, activation=None, reg=False, useBN=False, dropout=None, initia='xavier', reuse=False, biasReg=False, biasInitializer='zeros'):
        super(FC, self).__init__()
		# global params
		# global regParams
		# global leaky
        self.dropout = dropout
        self.inDim = indim
        self.activation = activation
        self.useBias = useBias
        self.temName = name if name!=None else 'defaultParamName%d'%getParamId()
        self.W = getOrDefineParam(self.temName, [self.inDim, outDim],requires_grad=True, reg=reg, initia=initia, reuse=reuse)
        self.Bias = Bias(indim=outDim,name=self.temName, reuse=reuse, reg=biasReg, initia=biasInitializer)
    def construct(self,inp):
        if self.dropout != None:
            ret = Dropout(p=dropout)(inp) @ self.W
        else:
            ret = inp @ self.W
        if self.useBias:
            ret = self.Bias(ret)
        if self.activation != None:
            ret = Activate(ret, self.activation)
        return ret

class Bias(nn.Cell):
    def __init__(self,indim,name=None, reg=False, reuse=False, initia='zeros'):
        super(Bias, self).__init__()

        self.inDim = indim
        self.temName = name if name!=None else 'defaultParamName%d'%getParamId()
        self.temBiasName = self.temName + 'Bias'
        self.bias = getOrDefineParam(self.temBiasName, self.inDim,requires_grad=True, reg=False, initia=initia, reuse=reuse)
    def construct(self,data):
		
        return data + self.bias

# def Bias(data, name=None, reg=False, reuse=False, initia='zeros'):
# 	inDim =  ops.Shape()(data)[-1]
# 	temName = name if name!=None else 'defaultParamName%d'%getParamId()
# 	temBiasName = temName + 'Bias'
# 	bias = getOrDefineParam(temBiasName, inDim,requires_grad=True, reg=False, initia=initia, reuse=reuse)
# 	if reg:
# 		regParams[temBiasName] = bias
# 	return data + bias

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

def Activate(data, method, useBN=False):
	global leaky
	if useBN:
		ret = BN(data)
	else:
		ret = data
	ret = ActivateHelp(ret, method)
	return ret

def Regularize(names=None, method='L2'):
	ret = 0
	if method == 'L1':
		if names != None:
			for name in names:
				ret += ops.ReduceSum(keep_dims=False)(ops.Abs()(getParam(name)))
		else:
			for name in regParams:
				ret += ops.ReduceSum(keep_dims=False)(ops.Abs()(regParams[name]))
	elif method == 'L2':
		if names != None:
			for name in names:
				ret += ops.ReduceSum(keep_dims=False)(ops.Square()(getParam(name)))
		else:
			for name in regParams:
				ret += ops.ReduceSum(keep_dims=False)(ops.Square()(regParams[name]))
	return ret

def Dropout(data, rate):
	if rate == None:
		return data
	else:
		return tf.nn.dropout(data, rate=rate)