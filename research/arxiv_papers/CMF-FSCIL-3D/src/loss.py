import mindspore
from mindspore import nn, ops
import mindspore.ops.functional as F

def normalize(x, dim, p=2):
    """
    Normalize the input tensor x along the specified dimension.

    Args:
        x: Input tensor.
        dim: The dimension to normalize along.
        p: The order of the norm (default is 2 for L2 norm).

    Returns:
        Normalized tensor.
    """
    
    norm = ops.pow(ops.reduce_sum(ops.pow(x, p), axis=dim), 1/p)
    
    norm = ops.maximum(norm, mindspore.Tensor(1e-12, x.dtype))
    
    norm = ops.expand_dims(norm, axis=dim)
    return x / norm

class LosswithIMG(nn.Cell):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.main_labels = None
        self.all_labels = None
        self.batch_size = None
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def compute_embeddings(self, pc_embed, text_embed, image_embed, logit_scale):
        logits_per_pc_text = logit_scale * ops.matmul(pc_embed, text_embed.T)
        logits_per_text_pc = logit_scale * ops.matmul(text_embed, pc_embed.T)
        logits_per_pc_image = logit_scale * ops.matmul(pc_embed, image_embed.T)
        logits_per_image_pc = logit_scale * ops.matmul(image_embed, pc_embed.T)
        return logits_per_pc_text, logits_per_text_pc, logits_per_pc_image, logits_per_image_pc
        

    def construct(self, main_labels, all_labels, outputs):
        pc_embed = outputs['pc_embed']
        text_embed = outputs['text_embed']
        image_embed = outputs['image_embed']
        logit_scale = outputs['logit_scale']
        logits_per_pc_all = outputs['pc_all']
        logits_per_pc_main = outputs['pc_main']

        local_batch_size = pc_embed.shape[0]
        self.batch_size = local_batch_size
        self.labels = ops.arange(self.batch_size)
        self.main_labels = main_labels
        self.all_labels = all_labels

        
        pc_embed = normalize(pc_embed, dim=-1, p=2)
        text_embed = normalize(text_embed, dim=-1, p=2)
        image_embed = normalize(image_embed, dim=-1, p=2)

        
        logits = self.compute_embeddings(pc_embed, text_embed, image_embed, logit_scale)
        logits_per_pc_text, logits_per_text_pc, logits_per_pc_image, logits_per_image_pc = logits

        
        loss_pc_text = self.cross_entropy(logits_per_pc_text, self.labels)
        loss_text_pc = self.cross_entropy(logits_per_text_pc, self.labels)
        loss_pc_image = self.cross_entropy(logits_per_pc_image, self.labels)
        loss_image_pc = self.cross_entropy(logits_per_image_pc, self.labels)
        loss_fake = self.get_fake_loss(logits_per_pc_text, logits_per_text_pc, logits_per_pc_image, logits_per_image_pc)
        loss_pc_all = self.cross_entropy(logits_per_pc_all, self.all_labels)
        loss_pc_main = self.cross_entropy(logits_per_pc_main, self.main_labels)

        loss = 0.3 * ((loss_pc_text + loss_text_pc) / 2 + (loss_pc_image + loss_image_pc) / 2) \
               + 0.1 * loss_fake \
               + 0.4 * loss_pc_all \
               + 0.2 * loss_pc_main

        
        pred = ops.argmax(logits_per_pc_text, dim=-1)
        correct = ops.equal(pred, self.labels)
        pc_text_acc = ops.ReduceMean()(correct.astype(mindspore.float32)) * 100

        pred = ops.argmax(logits_per_pc_image, dim=-1)
        correct = ops.equal(pred, self.labels)
        pc_image_acc = ops.ReduceMean()(correct.astype(mindspore.float32)) * 100

        pred = ops.argmax(logits_per_pc_all, dim=-1)
        correct = ops.equal(pred, self.all_labels)
        pc_all_acc = ops.ReduceMean()(correct.astype(mindspore.float32)) * 100

        pred = ops.argmax(logits_per_pc_main, dim=-1)
        correct = ops.equal(pred, self.main_labels)
        pc_main_acc = ops.ReduceMean()(correct.astype(mindspore.float32)) * 100

        return {'loss': loss, 'pc_image_acc': pc_image_acc, 'pc_text_acc': pc_text_acc,
                'pc_all_acc': pc_all_acc, 'pc_main_acc': pc_main_acc}

    def get_fake_loss(self, logits_pc_text, logits_text_pc, logits_pc_image, logits_image_pc):
        mask_value = -1e9  
        batch_size = logits_pc_text.shape[0]
        counts = batch_size // 5

        
        mask = ops.eye(batch_size, batch_size, mindspore.bool_)
        for i in range(batch_size):
            if i % 5 != 0:
                mask[i, i] = False

        
        logits_pc_text = ops.select(mask, logits_pc_text, ops.full_like(logits_pc_text, mask_value))
        logits_text_pc = ops.select(mask, logits_text_pc, ops.full_like(logits_text_pc, mask_value))
        logits_pc_image = ops.select(mask, logits_pc_image, ops.full_like(logits_pc_image, mask_value))
        logits_image_pc = ops.select(mask, logits_image_pc, ops.full_like(logits_image_pc, mask_value))

        
        new_labels = ops.arange(batch_size)
        for category in range(counts):
            new_labels[category * 5:category * 5 + 5] = category * 5

        
        loss_pc_text = self.cross_entropy(logits_pc_text, new_labels)
        loss_text_pc = self.cross_entropy(logits_text_pc, new_labels)
        loss_pc_image = self.cross_entropy(logits_pc_image, new_labels)
        loss_image_pc = self.cross_entropy(logits_image_pc, new_labels)

        loss = (loss_pc_text + loss_text_pc) / 2 + (loss_pc_image + loss_image_pc) / 2

        return loss

