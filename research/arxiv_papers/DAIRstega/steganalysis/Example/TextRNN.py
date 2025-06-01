import torch
import torch.nn as nn


class TextRNN(nn.Module):
    """RNN"""
    def __init__(self, cell, vocab_size, embed_size, hidden_dim, num_layers, dropout_rate):
        super(TextRNN, self).__init__()
        self._cell = cell

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = None
        if cell == 'rnn':
            self.rnn = nn.RNN(embed_size, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
            out_hidden_dim = hidden_dim * num_layers
        elif cell == 'bi-rnn':
            self.rnn = nn.RNN(embed_size, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
            out_hidden_dim = 2 * hidden_dim * num_layers
        elif cell == 'gru':
            self.rnn = nn.GRU(embed_size, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
            out_hidden_dim = hidden_dim * num_layers
        elif cell == 'bi-gru':
            self.rnn = nn.GRU(embed_size, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
            out_hidden_dim = 2 * hidden_dim * num_layers
        elif cell == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
            out_hidden_dim = hidden_dim * num_layers
        elif cell == 'bi-lstm':
            self.rnn = nn.LSTM(embed_size, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
            out_hidden_dim = 2 * hidden_dim * num_layers
        else:
            raise Exception("no such rnn cell")

        # self.output_layer = nn.Linear(out_hidden_dim, k)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
		:param x:(B，L）B：batch_size, L：sentence_length
		:return: (B, hidden_dim*2, L） bi-gru
		"""
        x = x.long()
        _ = self.embedding(x) # (B, L, E)
        __, h_out = self.rnn(_) # Bi-gru:(B, L, H*2), (2*n_l, B, H); gru:(B, L, H), (1, B, H)
        # h_out = h_out.reshape(1, -1, h_out.shape[0] * h_out.shape[2]) # (1, B, 2*n_l*H)
        __ = __.permute(0, 2, 1)
        _ = self.softmax(__)
        return _
