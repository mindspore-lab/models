import os
import mindspore
from mindspore import nn, Tensor
from mindspore.train import Model
from mindspore.train.callback import LossMonitor
import mindspore.context as context
import mindspore.ops as ops
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.vision import Decode, Normalize, HWC2CHW, Resize

# 设置MindSpore环境
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# 定义简化版的VGG-like网络
class VGG19(nn.Cell):
    def __init__(self):
        super(VGG19, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

    def construct(self, x):
        x = ops.relu(self.conv1_1(x))
        x = ops.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = ops.relu(self.conv2_1(x))
        x = ops.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = ops.relu(self.conv3_1(x))
        x = ops.relu(self.conv3_2(x))
        x = ops.relu(self.conv3_3(x))
        x = self.pool3(x)
        x = ops.relu(self.conv4_1(x))
        x = ops.relu(self.conv4_2(x))
        x = ops.relu(self.conv4_3(x))
        x = self.pool4(x)
        return x

# 定义损失函数：内容损失 + 风格损失
class LossFunctions(nn.Cell):
    def __init__(self, vgg19):
        super(LossFunctions, self).__init__()
        self.vgg19 = vgg19
        self.mse_loss = nn.MSELoss()

    def construct(self, content_image, style_image, generated_image):
        # 计算内容损失
        content_features = self.vgg19(content_image)
        generated_features = self.vgg19(generated_image)
        content_loss = self.mse_loss(content_features, generated_features)

        # 计算风格损失：通过 Gram Matrix 计算
        style_features = self.vgg19(style_image)
        style_generated_features = self.vgg19(generated_image)
        
        style_loss = self.mse_loss(style_features, style_generated_features)

        # 总损失 = 内容损失 + 风格损失
        total_loss = content_loss + style_loss
        return total_loss

# 数据预处理
def load_facescrub_data(image_paths, batch_size=32, image_size=256):
    def generate_images():
        for image_path in image_paths:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size))  # Resize to 256x256
            img = img / 255.0  # Normalize to [0, 1]
            img = img.transpose(2, 0, 1)  # HWC to CHW
            img = Tensor(img, dtype=mindspore.float32)
            yield img, img  # For style and content, we'll use the same image

    dataset = GeneratorDataset(generator=generate_images, column_names=["content", "style"], num_parallel_workers=4)
    dataset = dataset.batch(batch_size)
    return dataset

# 创建一个简化版的训练过程
def train_style_transfer_model(train_dataset, vgg19, loss_fn, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for batch in train_dataset.create_tuple_iterator():
            content_image, style_image = batch
            generated_image = Tensor(np.random.randn(*content_image.shape), dtype=mindspore.float32)  # Initialize generated image

            # 计算损失
            loss = loss_fn(content_image, style_image, generated_image)

            # 反向传播并更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 输出损失
            print(f"Epoch [{epoch + 1}], Loss: {loss.asnumpy()}")

# 配置训练参数
vgg19 = VGG19()
loss_fn = LossFunctions(vgg19)
optimizer = nn.Adam(vgg19.trainable_params(), learning_rate=0.001)

# 加载数据集（路径替换为你自己的FaceScrub数据集路径）
image_paths = ["/path/to/facescrub/images/image1.jpg", "/path/to/facescrub/images/image2.jpg"]  # Example paths
train_dataset = load_facescrub_data(image_paths)

# 训练模型
train_style_transfer_model(train_dataset, vgg19, loss_fn, optimizer)
