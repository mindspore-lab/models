# FixMatch semi-supervised learning algorithm implementation
import os
import logging
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore.dataset.vision import c_transforms as vision
from mindspore.dataset import transforms
from PIL import Image
from tqdm import tqdm
import argparse
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Command line arguments
parser = argparse.ArgumentParser(description='FixMatch semi-supervised learning')
# Model parameters
parser.add_argument('--model', type=str, default='resnet18', help='Model to use (resnet18, vgg16)')
parser.add_argument('--pretrained', type=str, default='', help='Path to pretrained model')

# Dataset parameters
parser.add_argument('--dataset', type=str, default='cub200', help='Dataset to use (cifar10, cifar100, cub200)')
parser.add_argument('--num_classes', type=int, default=200, help='Number of classes (10 for cifar10, 100 for cifar100, 200 for cub200)')
parser.add_argument('--data_dir', type=str, default='CUB_200_2011', help='Dataset path')
parser.add_argument('--labeled_ratio', type=float, default=0.1, help='Ratio of labeled data')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--num_workers', type=int, default=2, help='Number of data loader workers')

# Training parameters
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--confidence_threshold', type=float, default=0.95, help='Pseudo-label confidence threshold')
parser.add_argument('--lambda_u', type=float, default=1.0, help='Unlabeled loss weight')

# Device parameters
parser.add_argument('--device_target', type=str, default='Ascend', help='Device to run (Ascend, GPU, CPU)')
parser.add_argument('--device_id', type=int, default=0, help='Device ID')
parser.add_argument('--amp_level', type=str, default='O2', help='Mixed precision level (O0, O1, O2, O3)')

# Save parameters
parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Model save path')
parser.add_argument('--save_interval', type=int, default=10, help='Epoch interval for saving checkpoints')

# Parse arguments
args = parser.parse_args()

# Setup device
ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE)

# Create save directory
os.makedirs(args.save_dir, exist_ok=True)

# Data augmentation
def create_weak_augmentation():
    """Create weak data augmentation"""
    # Choose appropriate normalization parameters based on dataset
    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    else:  # cub200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
    return [
        vision.Resize(256),
        vision.CenterCrop(224),
        vision.RandomHorizontalFlip(prob=0.5),
        vision.Normalize(mean=mean, std=std),
        vision.HWC2CHW()
    ]

def create_strong_augmentation():
    """Create strong data augmentation"""
    # Choose appropriate normalization parameters based on dataset
    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    else:  # cub200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
    return [
        vision.Resize(256),
        vision.RandomCrop(224),
        vision.RandomHorizontalFlip(prob=0.5),
        # RandAugment replacement
        vision.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        vision.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=5),
        # Remove unsupported RandomErasing
        # vision.RandomErasing(prob=0.2),
        vision.Normalize(mean=mean, std=std),
        vision.HWC2CHW()
    ]

# Create dataset
def create_dataset(name, root, split, download=False):
    """Create CIFAR dataset"""
    if name == 'cifar10':
        dataset = ds.Cifar10Dataset(root, split == 'train', download=download)
    elif name == 'cifar100':
        dataset = ds.Cifar100Dataset(root, split == 'train', download=download)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    
    return dataset

# Load CUB_200_2011 dataset
def load_cub200_dataset(root):
    """Load CUB_200_2011 dataset"""
    # Define CUB_200_2011 dataset class
    class CUB200Dataset:
        def __init__(self, root, is_train=True):
            self.root = root
            self.is_train = is_train
            
            # Read category information
            self.classes = {}
            with open(os.path.join(root, 'classes.txt'), 'r') as f:
                for line in f:
                    class_id, class_name = line.strip().split()
                    self.classes[class_id] = class_name
            
            # Read image paths
            self.images = {}
            with open(os.path.join(root, 'images.txt'), 'r') as f:
                for line in f:
                    image_id, image_path = line.strip().split()
                    self.images[image_id] = image_path
            
            # Read image categories
            self.image_classes = {}
            with open(os.path.join(root, 'image_class_labels.txt'), 'r') as f:
                for line in f:
                    image_id, class_id = line.strip().split()
                    self.image_classes[image_id] = int(class_id) - 1  # Subtract 1 to make classes start from 0
            
            # Read train/test split
            self.train_test_split = {}
            with open(os.path.join(root, 'train_test_split.txt'), 'r') as f:
                for line in f:
                    image_id, is_train = line.strip().split()
                    self.train_test_split[image_id] = int(is_train)
            
            # Filter training or test images
            self.data = []
            for image_id, path in self.images.items():
                if self.train_test_split[image_id] == (1 if is_train else 0):
                    self.data.append((os.path.join(root, 'images', path), 
                                     self.image_classes[image_id]))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            img_path, label = self.data[idx]
            try:
                img = Image.open(img_path).convert('RGB')
                return img, label
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Return random image and label
                random_idx = random.randint(0, len(self.data)-1)
                return self.__getitem__(random_idx)
    
    # Create training set and test set
    train_dataset = CUB200Dataset(root, is_train=True)
    test_dataset = CUB200Dataset(root, is_train=False)
    
    print(f"CUB200 dataset loaded, train set contains {len(train_dataset)} images")
    print(f"CUB200 dataset loaded, test set contains {len(test_dataset)} images")
    
    return train_dataset, test_dataset

# Create model
def create_model(name, num_classes):
    """Create model"""
    if name == 'resnet18':
        import mindcv.models as models
        model = models.resnet18(num_classes=num_classes)
    elif name == 'vgg16':
        import mindcv.models as models
        model = models.vgg16(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")
    
    return model

# Main training function
def train_fixmatch():
    """FixMatch semi-supervised learning main function"""
    logging.info("Starting FixMatch semi-supervised learning training")
    logging.info(f"Configuration parameters: {vars(args)}")
    
    # Load dataset
    print("Loading CUB_200_2011 dataset...")
    train_dataset, test_dataset = load_cub200_dataset(args.data_dir)
    
    # Split labeled and unlabeled data
    labeled_size = int(len(train_dataset) * args.labeled_ratio)
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    labeled_indices = indices[:labeled_size]
    unlabeled_indices = indices[labeled_size:]
    
    labeled_data = [train_dataset[i] for i in labeled_indices]
    unlabeled_data = [train_dataset[i] for i in unlabeled_indices]
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Labeled data size: {len(labeled_data)}, unlabeled data size: {len(unlabeled_data)}")
    
    # Create data augmentation
    weak_transform = create_weak_augmentation()
    strong_transform = create_strong_augmentation()
    
    # Create labeled dataset
    def labeled_generator():
        for img, label in labeled_data:
            img_np = np.array(img)
            # Apply weak augmentation
            transformed_img = img_np
            for op in weak_transform:
                transformed_img = op(transformed_img)
            yield transformed_img, label
    
    labeled_ds = ds.GeneratorDataset(
        source=labeled_generator(),
        column_names=["image", "label"],
        shuffle=True
    )
    labeled_ds = labeled_ds.batch(args.batch_size)
    
    # Create unlabeled dataset
    def unlabeled_generator():
        for img, _ in unlabeled_data:
            img_np = np.array(img)
            # Apply weak augmentation
            weak_img = img_np
            for op in weak_transform:
                weak_img = op(weak_img)
            
            # Apply strong augmentation
            strong_img = img_np
            for op in strong_transform:
                strong_img = op(strong_img)
            
            yield weak_img, strong_img
    
    unlabeled_ds = ds.GeneratorDataset(
        source=unlabeled_generator(),
        column_names=["weak_image", "strong_image"],
        shuffle=True
    )
    unlabeled_ds = unlabeled_ds.batch(args.batch_size)
    
    # Create test dataset
    def test_generator():
        for img, label in test_dataset:
            img_np = np.array(img)
            # Apply test transform
            transformed_img = img_np
            for op in weak_transform:
                transformed_img = op(transformed_img)
            yield transformed_img, label
    
    test_ds = ds.GeneratorDataset(
        source=test_generator(),
        column_names=["image", "label"],
        shuffle=False
    )
    test_ds = test_ds.batch(args.batch_size)
    
    # Create model
    model = create_model(args.model, args.num_classes)
    
    # If pretrained model is specified, load pretrained weights
    if args.pretrained:
        ms.load_param_into_net(model, ms.load_checkpoint(args.pretrained))
        logging.info(f"Pretrained weights loaded: {args.pretrained}")
    
    # Create optimizer
    optimizer = nn.Momentum(
        params=model.trainable_params(),
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Create loss function
    ce_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    
    # Define evaluation function
    def evaluate():
        """Evaluate model on test set"""
        correct = 0
        total = 0
        model.set_train(False)
        
        # Re-create test dataset
        eval_ds = ds.GeneratorDataset(
            source=test_generator(),
            column_names=["image", "label"],
            shuffle=False
        )
        eval_ds = eval_ds.batch(args.batch_size)
        
        for data in eval_ds.create_tuple_iterator():
            images, labels = data
            outputs = model(images)
            _, predicted = ops.max(outputs, axis=1)
            predicted = predicted.astype(ms.int32)
            labels = labels.astype(ms.int32)
            total += labels.shape[0]
            correct += (predicted == labels).astype(ms.float32).sum().asnumpy().item()
        
        if total == 0:
            logging.warning("No samples processed in test set, please check test dataset")
            return 0.0
        
        return correct / total
    
    # Define training step
    class FixMatchTrainStep(nn.Cell):
        def __init__(self, network, optimizer):
            super(FixMatchTrainStep, self).__init__()
            self.network = network
            self.optimizer = optimizer
            self.grad_fn = ops.value_and_grad(self.forward, None, optimizer.parameters)
            self.ce_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')
            self.softmax = nn.Softmax(axis=1)
            self.confidence_threshold = ms.Tensor(args.confidence_threshold, ms.float32)
            self.lambda_u = ms.Tensor(args.lambda_u, ms.float32)
        
        def forward(self, labeled_images, labeled_labels, unlabeled_weak, unlabeled_strong):
            # Supervised loss
            logits_x = self.network(labeled_images)
            loss_x = self.ce_loss(logits_x, labeled_labels).mean()
            
            # Unsupervised loss - Replace ms.stop_gradient
            # Use pseudo-label computation instead of stop_gradient
            logits_w = self.network(unlabeled_weak)
            probs_w = self.softmax(logits_w)
            max_probs, pseudo_labels = ops.max(probs_w, axis=1)
            mask = (max_probs >= self.confidence_threshold).astype(ms.float32)
            
            # Ensure gradients do not propagate through pseudo-labels
            pseudo_labels = ops.stop_gradient(pseudo_labels)
            mask = ops.stop_gradient(mask)
            
            logits_s = self.network(unlabeled_strong)
            loss_u = (self.ce_loss(logits_s, pseudo_labels) * mask).mean()
            
            # Total loss
            loss = loss_x + self.lambda_u * loss_u
            return loss
        
        def construct(self, labeled_images, labeled_labels, unlabeled_weak, unlabeled_strong):
            loss, grads = self.grad_fn(labeled_images, labeled_labels, unlabeled_weak, unlabeled_strong)
            self.optimizer(grads)
            return loss
    
    # Create training network
    train_step = FixMatchTrainStep(model, optimizer)
    
    # Start training
    logging.info("Starting FixMatch semi-supervised training...")
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.set_train(True)
        total_loss = 0.0
        steps = 0
        
        # Re-create data loaders and iterators at each epoch
        # Labeled data loader
        labeled_generator_dataset = ds.GeneratorDataset(
            source=labeled_generator(),
            column_names=["image", "label"],
            shuffle=True
        )
        labeled_ds_epoch = labeled_generator_dataset.batch(args.batch_size)
        labeled_iter = labeled_ds_epoch.create_dict_iterator()
        
        # Unlabeled data loader
        unlabeled_generator_dataset = ds.GeneratorDataset(
            source=unlabeled_generator(),
            column_names=["weak_image", "strong_image"],
            shuffle=True
        )
        unlabeled_ds_epoch = unlabeled_generator_dataset.batch(args.batch_size)
        unlabeled_iter = unlabeled_ds_epoch.create_dict_iterator()
        
        # Calculate total steps per epoch
        total_steps = min(len(labeled_data) // args.batch_size, len(unlabeled_data) // args.batch_size)
        train_iter = tqdm(range(total_steps), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step in train_iter:
            try:
                labeled_batch = next(labeled_iter)
                unlabeled_batch = next(unlabeled_iter)
                
                labeled_images = labeled_batch["image"]
                labeled_labels = labeled_batch["label"]
                unlabeled_weak = unlabeled_batch["weak_image"]
                unlabeled_strong = unlabeled_batch["strong_image"]
                
                loss = train_step(labeled_images, labeled_labels, unlabeled_weak, unlabeled_strong)
                
                total_loss += loss.asnumpy().item()
                steps += 1
                
                # Update progress bar
                train_iter.set_postfix(loss=f"{loss.asnumpy().item():.4f}")
            except StopIteration:
                # If iterator is exhausted, stop this epoch
                logging.warning(f"Epoch {epoch+1} data iterator exhausted, completed {step}/{total_steps} steps")
                break
        
        # Calculate average loss
        avg_loss = total_loss / steps
        
        # Evaluate every 5 epochs or at the last epoch
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            acc = evaluate()
            logging.info(f"Epoch: {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
            
            # Save best model
            if acc > best_acc:
                best_acc = acc
                ms.save_checkpoint(model, os.path.join(args.save_dir, f"{args.model}_fixmatch_best.ckpt"))
                logging.info(f"Best model saved, accuracy: {best_acc:.4f}")
        else:
            logging.info(f"Epoch: {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint every specified interval
        if (epoch + 1) % args.save_interval == 0:
            ms.save_checkpoint(model, os.path.join(args.save_dir, f"{args.model}_fixmatch_epoch{epoch+1}.ckpt"))
            logging.info(f"Checkpoint saved at epoch {epoch+1}")
    
    logging.info(f"FixMatch semi-supervised training completed, best accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train_fixmatch()