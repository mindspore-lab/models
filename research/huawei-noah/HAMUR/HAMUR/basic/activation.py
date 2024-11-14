"""
-*- coding: utf-8 -*-
@Time    : 9/12/2023 3:47 pm
@Author  : Xiaopeng Li
@File    : activation.py

"""
import mindspore.nn as nn


def activation_layer(act_name):
    """Construct activation layers

    Args:
        act_name: str or nn.Cell, name of activation function or Cell class

    Returns:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU()
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
        elif act_name.lower() == "softmax":
            act_layer = nn.Softmax(axis=1)  # Specify the axis if needed
    elif issubclass(act_name, nn.Cell):
        act_layer = act_name()
    else:
        raise Exception("Activation layer name is not supported")
    return act_layer
