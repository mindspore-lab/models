import numpy as np
import mindspore as ms
import mindspore.nn as nn

def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index

def bj(a, b, eps=1e-6):
    return (ms.ops.abs(a - b) > eps).sum()

# ########## fourier layer #############
class FourierBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        # get modes on frequency domain
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        print('modes={}, index={}'.format(modes, self.index))

        self.scale = (1 / (in_channels * out_channels))
        # self.weights1 = ms.Parameter(
        #     self.scale * ms.ops.UniformReal()((8, in_channels // 8, out_channels // 8, len(self.index)))
        #     )
        self.weights1_real = ms.Parameter(
            self.scale * ms.ops.UniformReal()((8, in_channels // 8, out_channels // 8, len(self.index)))
            )
        self.weights1_imag = ms.Parameter(
            self.scale * ms.ops.UniformReal()((8, in_channels // 8, out_channels // 8, len(self.index)))
            )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return (input.unsqueeze(-1) * weights.unsqueeze(0)).sum(-2)

    def construct(self, q, k, v, mask):
        self.weights1 = ms.ops.Complex()(self.weights1_real, self.weights1_imag)
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)

        rfft_net = ms.ops.FFTWithSize(signal_ndim=1, inverse=False, real=True)
        # Compute Fourier coefficients
        x_ft = rfft_net(x)
        # Perform Fourier neural operations
        out_ft = ms.ops.Zeros()((B, H, E, L // 2 + 1), ms.float32)
        out_ft = ms.ops.Complex()(out_ft, out_ft)

        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])

        # Return to time domain
        irfft_net = ms.ops.FFTWithSize(signal_ndim=1, inverse=True, real=True, signal_sizes=[x.shape[-1]])
        x = irfft_net(ms.ops.Cast()(out_ft, ms.complex64))
        return (x, None)


# ########## Fourier Cross Former ####################
class FourierCrossAttention(nn.Cell):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random',
                 activation='tanh', policy=0):
        super(FourierCrossAttention, self).__init__()
        print(' fourier enhanced cross attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        # get modes for queries and keys (& values) on frequency domain
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
        print('modes_kv={}, index_kv={}'.format(len(self.index_kv), self.index_kv))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1_real = ms.Parameter(
            self.scale * ms.ops.UniformReal()((8, in_channels // 8, out_channels // 8, len(self.index_q)))
            )
        self.weights1_imag = ms.Parameter(
            self.scale * ms.ops.UniformReal()((8, in_channels // 8, out_channels // 8, len(self.index_q)))
            )
        # self.weights1 = ms.Parameter(
        #     self.scale * ms.ops.UniformReal()((8, in_channels // 8, out_channels // 8, len(self.index_q)))
        #     )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return (input.unsqueeze(-1) * weights.unsqueeze(0)).sum(-2)

    def construct(self, q, k, v, mask):
        # size = [B, L, H, E]
        self.weights1 = ms.ops.Complex()(self.weights1_real, self.weights1_imag)
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)
        # Compute Fourier coefficients
        xq_ft_ = ms.ops.zeros((B, H, E, len(self.index_q)), ms.float32)
        xq_ft_ = ms.ops.Complex()(xq_ft_, xq_ft_)
        rfft_net = ms.ops.FFTWithSize(signal_ndim=1, inverse=False, real=True)
        xq_ft = rfft_net(xq)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = ms.ops.zeros((B, H, E, len(self.index_kv)), ms.float32)
        xk_ft_ = ms.ops.Complex()(xk_ft_, xk_ft_)
        xk_ft = rfft_net(xk)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        # perform attention mechanism on frequency domain
        xqk_ft = xq_ft_.swapaxes(-1, -2).matmul(xk_ft_)

        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = ms.nn.Softmax(axis=-1)(abs(xqk_ft))
            xqk_ft = ms.ops.Complex()(xqk_ft, ms.ops.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = xk_ft_.matmul(xqk_ft.swapaxes(-1, -2))
        xqkvw = (xqkv_ft.unsqueeze(-2) * self.weights1.unsqueeze(0)).sum(2)
        out_ft = ms.ops.zeros((B, H, E, L // 2 + 1), ms.float32)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # Return to time domain
        irfft_net = ms.ops.FFTWithSize(signal_ndim=1, inverse=True, real=True, signal_sizes=[xq.shape[-1]])
        out = irfft_net(ms.ops.Cast()(out_ft / self.in_channels / self.out_channels, ms.complex64))
        return (out, None)
    



