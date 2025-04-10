import torch
from torch import nn


class TextCNN(nn.Module):
	def __init__(self, vocab_size, embed_size, filter_num, filter_size, dropout_rate):
		super(TextCNN, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embed_size) 
		self.cnn_list = nn.ModuleList()
		for size in filter_size:
			self.cnn_list.append(nn.Conv1d(embed_size, filter_num, size, padding='same'))
		self.relu = nn.ReLU()
		self.max_pool = nn.AdaptiveMaxPool1d(1)
		self.dropout = nn.Dropout(dropout_rate)
		# self.output_layer = nn.Linear(filter_num * len(filter_size), class_num)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		x = x.long()
		_ = self.embedding(x) 
		_ = _.permute(0, 2, 1) 
		result = [] 
		for self.cnn in self.cnn_list:
			__ = self.cnn(_) 
			__ = self.relu(__)
			result.append(__) 

		_ = torch.cat(result, dim=1) 
		_ = self.dropout(_)
		_ = self.softmax(_)
		return _