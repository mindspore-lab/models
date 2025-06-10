import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import TextCNN as c
import TextRNN as r
import TextGE as e

class ge(nn.Module):
    def __init__(self, cell, vocab_size, embed_size, filter_num, filter_size, hidden_dim, num_layers, class_num,
                 dropout_rate, g, k):
        super(ge, self).__init__()
        self.cnn = c.TextCNN(vocab_size, embed_size, filter_num, filter_size, dropout_rate)
        self.rnn = r.TextRNN(cell, vocab_size, embed_size, hidden_dim, num_layers, dropout_rate)
        self.g_e = e.GroupEnhance(g)
        self.dropout = nn.Dropout(dropout_rate)
        self.k_pool = nn.AdaptiveMaxPool1d(k) # k-max-pooling
        self.max_pool = nn.AdaptiveMaxPool1d(1) # max_pooling

        self.conv1 = nn.Conv1d(in_channels=4 * hidden_dim, out_channels=20, kernel_size=3, padding='same')

        self.output_layer = nn.Linear(filter_num * 4, class_num) 
        self.output_layer1 = nn.Linear(20 * k, class_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        cnn_res = self.cnn(x)  # (B,filter_num * length(filter_size),L)
        rnn_res = self.rnn(x)  # (B, H*2, L） H：hidden_dim，NL：num_layers
        cge = self.g_e(cnn_res)  
        rge = self.g_e(rnn_res) 
        # _cat = torch.cat((cnn_res, rnn_res), dim=1) 
        _cat = torch.cat((self.k_pool(cge), self.k_pool(rge)), dim=1) 
        _ = F.relu(self.k_pool(self.dropout(self.conv1(_cat)))) 
        _ = _.view(_.shape[0], -1)  # (B,out_channels*k) 
        _ = self.output_layer1(_)
        _ = self.softmax(_)
        return _

# if __name__ == "__main__":
#     x = torch.rand((2, 10))
#     ge = ge("bi-gru",20,20,4,[3],2,1,2,0.5,1,1)
#     res = ge(x)
#     # print(x)
#     print(res)
