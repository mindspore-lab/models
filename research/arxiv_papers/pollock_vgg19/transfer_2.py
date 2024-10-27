import mindspore as ms
from mindspore import nn, ops, Tensor, context
from mindspore.context import PYNATIVE_MODE
from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import Resize, ToTensor
import numpy as np
import cv2
from PIL import Image
import time
from mindspore.ops import value_and_grad
import mindspore.numpy as mnp
from mindspore.amp import StaticLossScaler

context.set_context(mode=PYNATIVE_MODE)

ms.set_seed(0)

t0 = time.time()

# 加载本地VGG19模型
class VGG19(nn.Cell):
    def __init__(self, ckpt_path):
        super(VGG19, self).__init__()
        self.features = nn.SequentialCell([
            nn.Conv2d(3, 64, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        ])
        # 加载预训练权重
        if ckpt_path:
            param_dict = ms.load_checkpoint(ckpt_path)
            ms.load_param_into_net(self, param_dict)

model = VGG19(ckpt_path='vgg19.ckpt')  # 替换为你的ckpt文件路径

batch_size = 1

for param in model.trainable_params():
    param.requires_grad = False

model.set_train(False)

# 修改归一化处理
mu = mnp.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
std = mnp.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
unnormalize = lambda x: x * std + mu
normalize = lambda x: (x - mu) / std

transform_test = Compose([
    Resize((512, 512)),
    ToTensor(),
])

# 修改自定义函数以处理元组
def custom_expand_dims(tensor_tuple, axis=0):
    # 假设元组中只有一个元素，即张量
    tensor = tensor_tuple[0] if isinstance(tensor_tuple, tuple) else tensor_tuple
    shape = list(tensor.shape)
    shape.insert(axis, 1)
    return tensor.reshape(shape)

content_img = Image.open('/root/autodl-fs/test/A.jpg').convert('RGB')
image_size = content_img.size
content_img = transform_test(content_img)
# 使用修改后的自定义函数
content_img = custom_expand_dims(content_img, 0)
# 将 NumPy 数组转换为 MindSpore Tensor
content_img = ms.Tensor(content_img, dtype=ms.float32)

style_img = Image.open('/root/autodl-fs/test/pollock.png').convert('RGB')
style_img = transform_test(style_img)
# 使用修改后的自定义函数
style_img = custom_expand_dims(style_img, 0)
# 将 NumPy 数组转换为 MindSpore Tensor
style_img = ms.Tensor(style_img, dtype=ms.float32)

#  var_img 声明为 Parameter 以支持梯度
var_img = ms.Parameter(content_img, requires_grad=True, name="var_img")

class ShuntModel(nn.Cell):
    def __init__(self, model):
        super(ShuntModel, self).__init__()
        self.module = model.features
        self.con_layers = [22]
        # 调整风格层，选择更低层的特征来捕捉更细致的纹理
        self.sty_layers = [1, 3, 6, 8, 11, 13, 15, 20, 22, 24, 29]
        for i, layer in enumerate(self.module):
            if isinstance(layer, nn.MaxPool2d):
                self.module[i] = nn.AvgPool2d(kernel_size=2, stride=2)

    def construct(self, tensor: ms.Tensor) -> dict:
        sty_feat_maps = []; con_feat_maps = []
        x = normalize(tensor)
        for i, layer in enumerate(self.module):
            x = layer(x)
            if i in self.con_layers: con_feat_maps.append(x)
            if i in self.sty_layers: sty_feat_maps.append(x)
        return {"Con_features": con_feat_maps, "Sty_features": sty_feat_maps}

model = ShuntModel(model)
sty_target = model(style_img)["Sty_features"]
con_target = model(content_img)["Con_features"]
gram_target = []
for i in range(len(sty_target)):
    b, c, h, w = sty_target[i].shape
    tensor_ = sty_target[i].view(b * c, h * w)
    gram_i = ops.matmul(tensor_, tensor_.transpose(1, 0)).div(b*c*h*w)
    gram_target.append(gram_i)

# 添加纹理损失函数
def texture_loss(x):
    b, c, h, w = x.shape
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return (ops.reduce_mean(dx**2) + ops.reduce_mean(dy**2)) / (c * h * w)

# 添加滴画效果损失函数
def drip_effect_loss(x):
    b, c, h, w = x.shape
    # 计算垂直方向的梯度
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    # 鼓励垂直方向的连续性，同时保留一些随机性
    return -ops.reduce_mean(ops.abs(dy)) / (c * h * w) + ops.reduce_mean(ops.abs(dy - ops.mean(dy))) / (c * h * w)

# 调整损失权重
lam1 = 1e0    # 增加内容损失权重
lam2 = 1e7    # 略微降低风格损失权重
lam3 = 1e-2   # 增加总变差损失权重
lam4 = 1e6    # 大幅增加纹理损失权重
lam5 = 1e5    # 增加滴画效果损失权重

# 修改学习率调度器的设置
lr_schedule = nn.cosine_decay_lr(min_lr=0.0, max_lr=0.05, total_step=1000, step_per_epoch=100, decay_epoch=10)
optimizer = nn.Adam(params=[var_img], learning_rate=lr_schedule[0])

# 定义前向函数
mse_loss = nn.MSELoss()

# 定义一个辅助函数来计算梯度
def compute_grad(x):
    def func(y):
        return ops.reduce_sum(y)
    return ops.grad(func)(x)

# 修改前向函数
def forward_fn(var_img):
    output = model(var_img)
    sty_output = output["Sty_features"]
    con_output = output["Con_features"]
    
    con_loss = sum(nn.MSELoss()(co, ct) for co, ct in zip(con_output, con_target))
    
    sty_loss = ms.Tensor(0, dtype=ms.float32)
    for so, gt in zip(sty_output, gram_target):
        b, c, h, w = so.shape
        tensor_ = so.view(b * c, h * w)
        gram_i = ops.matmul(tensor_, tensor_.transpose(1, 0)).div(b*c*h*w)
        sty_loss = sty_loss + nn.MSELoss()(gram_i, gt)
    
    b, c, h, w = var_img.shape
    TV_loss = (ops.reduce_sum(ops.abs(var_img[:, :, :, :-1] - var_img[:, :, :, 1:])) +
               ops.reduce_sum(ops.abs(var_img[:, :, :-1, :] - var_img[:, :, 1:, :]))) / (b*c*h*w)
    
    tex_loss = sum(texture_loss(so) for so in sty_output)
    drip_loss = drip_effect_loss(var_img)
    
    total_loss = con_loss * lam1 + sty_loss * lam2 + TV_loss * lam3 + tex_loss * lam4 + drip_loss * lam5
    return total_loss, con_loss, sty_loss, TV_loss, tex_loss, drip_loss

# 修改梯度函数
grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

# 在环之前添加 gram_matrix 函数的定义
def gram_matrix(x):
    b, c, h, w = x.shape
    features = x.view(b * c, h * w)
    gram = ops.matmul(features, features.transpose(1, 0))
    return gram.div(b * c * h * w)

# 主循环
for itera in range(1001):
    # 更新学习率
    if itera < len(lr_schedule):
        lr = lr_schedule[itera]
        optimizer.learning_rate.assign_value(Tensor(lr, ms.float32))
    
    (loss, con_loss, sty_loss, TV_loss, tex_loss, drip_loss), grads = grad_fn(var_img)
    optimizer(grads)
    var_img.set_data(ops.clamp(var_img, 0, 1))
    
    if itera % 100 == 0:
        print(f'itera: {itera}, total loss: {loss.asnumpy():.4f}')
        print(f'con_loss: {con_loss.asnumpy():.4f}, sty_loss: {sty_loss.asnumpy():.4f}, TV_loss: {TV_loss.asnumpy():.4f}')
        print(f'tex_loss: {tex_loss.asnumpy():.4f}, drip_loss: {drip_loss.asnumpy():.4f}')
        print(f'var_img mean: {var_img.mean().asnumpy():.4f}, std: {var_img.std().asnumpy():.4f}')
        print(f'time: {time.time() - t0:.2f} seconds')

    if itera % 100 == 0:
        save_img = var_img.copy()
        save_img = ops.clamp(save_img, 0, 1)
        save_img = save_img[0].transpose(1, 2, 0).asnumpy() * 255
        save_img = save_img[..., ::-1].astype('uint8')
        save_img = cv2.resize(save_img, image_size)
        cv2.imwrite(f'/root/autodl-fs/test/output/transfer{itera}.jpg', save_img)

    loss_scale = StaticLossScaler(1024)
    loss = loss_scale.scale(loss)
    grads = loss_scale.unscale(grads)
