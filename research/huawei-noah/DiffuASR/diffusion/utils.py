# -*- encoding: utf-8 -*-
# here put the import lib
import mindspore
import mindspore.ops as ops
import numpy as np
from PIL import Image
import math


def img_to_tensor(im):
  return mindspore.tensor(np.array(im.convert('RGB'))/255).permute(2, 0, 1).unsqueeze(0) * 2 - 1

def tensor_to_image(t):
  return Image.fromarray(np.array(((t.squeeze().permute(1, 2, 0)+1)/2).clip(0, 1)*255).astype(np.uint8))

def gather(consts: mindspore.Tensor, t: mindspore.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather_elements(-1, t)
    return c.reshape(-1, 1, 1, 1)

def q_xt_x0(x0, t, alpha_bar):
    """get the noised x and noise"""
    # alpha_bar: (step_num)
    mean = gather(alpha_bar, t) ** 0.5 * x0 # (bs, n_channel, max_len, hidden_size)
    var = 1-gather(alpha_bar, t)    # (bs, 1, 1, 1)
    eps = ops.randn_like(x0)
    return mean + (var ** 0.5) * eps, eps # also returns noise

def compute_alpha(beta, t):
    beta = ops.cat([ops.zeros(1), beta], axis=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def p_xt(xt, noise, t, next_t, beta, eta):
    at = compute_alpha(beta, t.long())
    at_next = compute_alpha(beta, next_t.long())
    x0_t = (xt - noise * (1 - at).sqrt()) / at.sqrt()
    c1 = (eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    eps = ops.randn(xt.shape)
    xt_next = at_next.sqrt() * x0_t + c1 * eps + c2 * noise
    return xt_next


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = ops.exp(ops.arange(half_dim, dtype=mindspore.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = ops.cat([ops.sin(emb), ops.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = ops.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*ops.sigmoid(x)


def Normalize(in_channels):
    return mindspore.nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-6, affine=True)


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, mindspore.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, mindspore.Tensor) else mindspore.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    # print(logvar2.shape)
    # temp1 = 0.5 * (-1.0 + logvar2 - logvar1 + th.exp(logvar1 - logvar2))
    # print(f'const = {temp1.mean()}, coef={(th.exp(-logvar2) * 0.5).mean()}, mse={((mean1 - mean2) ** 2).mean().item()}')

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + ops.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * ops.exp(-logvar2)
    )




