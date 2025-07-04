"""
CMT-Mamba Knowledge Distillation Semi-supervised Learning Framework
Dataset: CIFAR10
Teacher Model: CMT-Small with Mamba
Student Model: ResNet18
Training Method: Knowledge Distillation + Semi-supervised Learning
"""

import os
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.common import set_seed
from mindspore.dataset import GeneratorDataset
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore.train.callback import ModelCheckpoint, LossMonitor, TimeMonitor
import sys
import datetime
import logging
from tqdm import tqdm
from PIL import Image
from typing import Optional, Callable, Tuple, Any

# Import MindCV related modules
from mindcv.models import create_model
from mindcv.data import create_dataset, create_transforms, create_loader
from mindcv.loss import create_loss
from mindcv.scheduler import create_scheduler

# Set random seed to ensure experiment reproducibility
set_seed(42)

# Define command line arguments
parser = argparse.ArgumentParser(description='Semi-supervised Knowledge Distillation Learning CMT-Mamba → ResNet')

# Model parameters
parser.add_argument('--teacher_model', type=str, default='cmt_small', help='Teacher model name')
parser.add_argument('--student_model', type=str, default='resnet18', help='Student model name')
parser.add_argument('--use_mamba', action='store_true', default=True, help='Whether teacher model uses Mamba module')
parser.add_argument('--pretrained_teacher', type=str, default='', help='Pretrained teacher model path')

# Dataset parameters
parser.add_argument('--dataset', type=str, default='cub200', choices=['cifar10', 'cifar100', 'cub200'], 
                    help='Dataset name')
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
parser.add_argument('--T', type=float, default=2.0, help='Distillation temperature')
parser.add_argument('--alpha', type=float, default=0.5, help='Distillation loss weight')
parser.add_argument('--beta', type=float, default=0.3, help='Consistency loss weight')
parser.add_argument('--ema_decay', type=float, default=0.999, help='Exponential moving average decay rate')

# Device parameters
parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'], 
                    help='Device to run')
parser.add_argument('--device_id', type=int, default=0, help='Device ID')
parser.add_argument('--amp_level', type=str, default='O2', help='Mixed precision level')

# Save path parameters
parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Model save path')
parser.add_argument('--save_interval', type=int, default=10, help='Model save interval')

args = parser.parse_args()

# Configure runtime environment
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)

# Create data augmentation
def create_weak_augmentation():
    """Create weak data augmentation"""
    # Choose appropriate normalization parameters based on dataset
    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    else:
        mean = [0.485, 0.456, 0.406]  # ImageNet normalization parameters, suitable for most natural images
        std = [0.229, 0.224, 0.225]
    
    return [
        # Use Resize+CenterCrop to ensure fixed size
        vision.Resize(256),  # First resize to larger size
        vision.CenterCrop(224),  # Then crop to target size
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
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    return [
        # Use Resize+RandomCrop to ensure fixed size
        vision.Resize(256),
        vision.RandomCrop(224),
        vision.RandomHorizontalFlip(prob=0.5),
        # RandAugment alternative
        vision.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        vision.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=5),
        vision.RandomErasing(prob=0.2),
        vision.Normalize(mean=mean, std=std),
        vision.HWC2CHW()
    ]

# Create a function to handle test data
def process_test_batch(batch_data):
    """Uniformly process test batch data"""
    if args.dataset == 'cifar10':
        images, labels = batch_data
    else:  # cifar100
        images, fine_labels, _ = batch_data  # Unpack three return values
        labels = fine_labels
    
    # Apply data transformation
    # Choose appropriate normalization parameters based on dataset
    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    else:  # cifar100
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        
    test_transform = [
        vision.Resize(224),  # Ensure test images are also resized to correct size
        vision.Normalize(mean=mean, std=std),
        vision.HWC2CHW()
    ]
    
    # If input is a dictionary, extract images and labels
    if isinstance(batch_data, dict):
        images = batch_data.get("image")
        labels = batch_data.get("label")
    else:
        # Already processed tuple
        pass
    
    # Apply transformation
    for op in test_transform:
        images = op(images)
    
    return images, labels

# Custom EMA model updater
class EMA:
    def __init__(self, model, shadow_model, decay=0.999):
        self.model = model
        self.shadow_model = shadow_model
        self.decay = decay
        self.shadow_params = [p.clone() for p in shadow_model.get_parameters()]
        self.backup_params = []
        
    def update(self):
        """Update Shadow model parameters"""
        model_params = list(self.model.get_parameters())
        for i, param in enumerate(self.shadow_params):
            param.assign_value((self.decay * param + (1 - self.decay) * model_params[i]))
            
    def apply_shadow(self):
        """Apply Shadow parameters to model"""
        model_params = list(self.model.get_parameters())
        self.backup_params = [p.clone() for p in model_params]
        for i, param in enumerate(model_params):
            param.assign_value(self.shadow_params[i])
            
    def restore(self):
        """Restore model original parameters"""
        model_params = list(self.model.get_parameters())
        for i, param in enumerate(model_params):
            param.assign_value(self.backup_params[i])

# Add CUB200Dataset class
class CUB200Dataset:
    """CUB_200_2011 dataset loader"""
    
    def __init__(self, root, split='train', transform=None):
        """
        Initialize CUB_200_2011 dataset
        
        Args:
            root: Dataset root directory
            split: 'train' or 'test'
            transform: Image transformation
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        
        # Read image list and corresponding classes
        self.image_paths = []
        self.targets = []
        
        # Read image ID and image path mapping
        image_path_file = os.path.join(self.root, 'images.txt')
        image_paths = {}
        with open(image_path_file, 'r') as f:
            for line in f:
                image_id, image_path = line.strip().split()
                image_paths[image_id] = image_path
        
        # Read image ID and class mapping
        image_class_file = os.path.join(self.root, 'image_class_labels.txt')
        image_classes = {}
        with open(image_class_file, 'r') as f:
            for line in f:
                image_id, class_id = line.strip().split()
                # Class ID starts from 1, convert to start from 0
                image_classes[image_id] = int(class_id) - 1
        
        # Read train/test split
        split_file = os.path.join(self.root, 'train_test_split.txt')
        train_test_split = {}
        with open(split_file, 'r') as f:
            for line in f:
                image_id, is_train = line.strip().split()
                train_test_split[image_id] = int(is_train)
        
        # Construct dataset based on split
        for image_id in image_paths:
            # 1 indicates training set, 0 indicates test set
            is_train = train_test_split[image_id] == 1
            if (self.split == 'train' and is_train) or (self.split == 'test' and not is_train):
                self.image_paths.append(os.path.join(self.root, 'images', image_paths[image_id]))
                self.targets.append(image_classes[image_id])
        
        # Get class names
        self.classes = []
        with open(os.path.join(self.root, 'classes.txt'), 'r') as f:
            for line in f:
                class_id, class_name = line.strip().split()
                self.classes.append(class_name)
        
        print(f"CUB200 dataset loaded, {split} set contains {len(self.image_paths)} images")
    
    def __getitem__(self, index):
        path = self.image_paths[index]
        target = self.targets[index]
        
        # Load image
        img = Image.open(path).convert('RGB')
        
        # Apply transformation
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target
    
    def __len__(self):
        return len(self.image_paths)

# Modify command line arguments, add cub200 option
def parse_args():
    parser = argparse.ArgumentParser(description='Semi-supervised Knowledge Distillation Learning CMT-Mamba → ResNet')
    
    # Model parameters
    parser.add_argument('--teacher_model', type=str, default='cmt_small', help='Teacher model name')
    parser.add_argument('--student_model', type=str, default='resnet18', help='Student model name')
    parser.add_argument('--use_mamba', action='store_true', default=True, help='Whether teacher model uses Mamba module')
    parser.add_argument('--pretrained_teacher', type=str, default='', help='Pretrained teacher model path')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cub200', choices=['cifar10', 'cifar100', 'cub200'], 
                        help='Dataset name')
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
    parser.add_argument('--T', type=float, default=2.0, help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.5, help='Distillation loss weight')
    parser.add_argument('--beta', type=float, default=0.3, help='Consistency loss weight')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='Exponential moving average decay rate')

    # Device parameters
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'], 
                        help='Device to run')
    parser.add_argument('--device_id', type=int, default=0, help='Device ID')
    parser.add_argument('--amp_level', type=str, default='O2', help='Mixed precision level')

    # Save path parameters
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Model save path')
    parser.add_argument('--save_interval', type=int, default=10, help='Model save interval')
    
    return parser.parse_args()

# Define CMT-Mamba knowledge distillation semi-supervised training process
def train_distill():
    # Set up logging
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"cmt_mamba_distill_{current_time}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Record training configuration
    logging.info(f"Starting CMT-Mamba knowledge distillation semi-supervised training")
    logging.info(f"Configuration parameters: {vars(args)}")
    
    # Load dataset
    if args.dataset == 'cub200':
        # Create CUB_200_2011 dataset loader
        print(f"Loading CUB_200_2011 dataset...")
        
        # Create CUB200 dataset
        train_dataset = CUB200Dataset(
            root=args.data_dir,
            split='train',
            transform=None  # Do not apply transformation for now, will be applied later
        )
        test_dataset = CUB200Dataset(
            root=args.data_dir,
            split='test',
            transform=None
        )
        
        # Convert dataset to in-memory numpy array for later processing
        dataset_np = []
        for idx in range(len(train_dataset)):
            img, label = train_dataset[idx]
            # Convert to numpy array
            img_np = np.array(img)
            dataset_np.append((img_np, label))
    else:
        # Original CIFAR dataset processing logic
        cifar_dataset = create_dataset(
            name=args.dataset,
            root=args.data_dir,
            split='train',
            download=False
        )
        
        # Manually split labeled and unlabeled data
        dataset_np = []
        print(f"Loading and converting {args.dataset.upper()} dataset...")
        for data in cifar_dataset:
            if args.dataset == 'cifar10':
                img, label = data
            else:  # cifar100
                # Unpack three return values: image, fine-grained label, and coarse-grained label
                img, fine_label, _ = data
                label = fine_label  # Only use fine-grained label (100 classes)
            
            dataset_np.append((img.asnumpy(), label.asnumpy().item()))
    
    print(f"Dataset size: {len(dataset_np)}")
    rng = np.random.RandomState(42)
    rng.shuffle(dataset_np)
    
    # Split by class
    class_indices = [[] for _ in range(args.num_classes)]  # Create index list based on dataset class count
    for i, (img, label) in enumerate(dataset_np):
        class_indices[label].append(i)
    
    # Perform stratified sampling for each class
    labeled_indices = []
    unlabeled_indices = []
    for indices in class_indices:
        n_labeled = max(1, int(len(indices) * args.labeled_ratio))
        labeled_indices.extend(indices[:n_labeled])
        unlabeled_indices.extend(indices[n_labeled:])
    
    # Create labeled and unlabeled datasets
    labeled_data = [(dataset_np[i][0], dataset_np[i][1]) for i in labeled_indices]
    unlabeled_data = [(dataset_np[i][0], dataset_np[i][1]) for i in unlabeled_indices]
    
    print(f"Labeled data size: {len(labeled_data)}, Unlabeled data size: {len(unlabeled_data)}")
    
    # Create data transformation
    weak_transform = create_weak_augmentation()
    strong_transform = create_strong_augmentation()
    
    # Create labeled data loader
    def labeled_generator():
        indices = list(range(len(labeled_data)))
        while True:
            rng.shuffle(indices)
            for idx in indices:
                yield labeled_data[idx]
    
    labeled_ds = ds.GeneratorDataset(
        source=labeled_generator(),
        column_names=["image", "label"],
        shuffle=True
    )
    labeled_ds = labeled_ds.map(operations=weak_transform, input_columns=["image"])
    labeled_ds = labeled_ds.batch(args.batch_size)
    
    # Create unlabeled data loader
    def unlabeled_generator():
        indices = list(range(len(unlabeled_data)))
        while True:
            rng.shuffle(indices)
            for idx in indices:
                img, _ = unlabeled_data[idx]
                yield img, img  # Weak augmentation, strong augmentation of original image
    
    unlabeled_ds = ds.GeneratorDataset(
        source=unlabeled_generator(),
        column_names=["weak_image", "strong_image"],
        shuffle=True
    )
    unlabeled_ds = unlabeled_ds.map(operations=weak_transform, input_columns=["weak_image"])
    unlabeled_ds = unlabeled_ds.map(operations=strong_transform, input_columns=["strong_image"])
    unlabeled_ds = unlabeled_ds.batch(args.batch_size)
    
    # Test set preprocessing
    # Choose appropriate normalization parameters based on dataset
    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    else:  # cifar100
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        
    test_transform = [
        vision.Resize(224),  # Ensure test images are also resized to correct size
        vision.Normalize(mean=mean, std=std),
        vision.HWC2CHW()
    ]
    
    # Create test dataset
    if args.dataset == 'cub200':
        # Test set preprocessing
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        test_transform = [
            vision.Resize(256),
            vision.CenterCrop(224),  # Ensure test images are also fixed size
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]
        
        # Test set generator function
        def test_generator():
            for idx in range(len(test_dataset)):
                img, label = test_dataset[idx]
                img_np = np.array(img)
                
                # Apply test transformation
                for op in test_transform:
                    img_np = op(img_np)
                
                yield img_np, label
        
        test_ds = ds.GeneratorDataset(
            source=test_generator(),
            column_names=["image", "label"],
            shuffle=False
        )
        test_ds = test_ds.batch(args.batch_size)
    else:
        # Original CIFAR test set processing logic
        test_ds = create_dataset(
            name=args.dataset,
            root=args.data_dir,
            split='test',
            download=False
        )
        
        # Apply transformation to test set
        test_ds = test_ds.map(operations=test_transform, input_columns=["image"])
        test_ds = test_ds.batch(args.batch_size)
    
    # Create models
    print("Creating teacher model (CMT-Small) and student model (ResNet18)...")

    # Create teacher model - Use CMT-Small with Mamba
    teacher_model = create_model(
        args.teacher_model,
        num_classes=args.num_classes,
        in_channels=3,
        use_mamba=args.use_mamba,
    )

    # Create student model - Use ResNet18
    student_model = create_model(
        args.student_model,
        num_classes=args.num_classes,
        in_channels=3,
        
    )

    # Check if pretrained teacher model exists
    if not args.pretrained_teacher:
        logging.info("No pretrained teacher model found, start training teacher model using labeled data...")
        
        # Create teacher model optimizer
        teacher_optimizer = nn.Momentum(
            params=teacher_model.trainable_params(),
            learning_rate=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        
        # Train teacher model using only labeled data
        train_teacher(teacher_model, teacher_optimizer, labeled_ds, test_ds, epochs=100, 
                     labeled_size=len(labeled_data), test_dataset_obj=test_dataset)
        
        # Save trained teacher model
        ms.save_checkpoint(teacher_model, "teacher_pretrained.ckpt")
        logging.info("Teacher model pretraining completed, checkpoint saved")
    
    # Create optimizer
    lr = nn.cosine_decay_lr(
        min_lr=args.lr * 0.01,
        max_lr=args.lr,
        total_step=args.epochs * (len(labeled_data) // args.batch_size),
        step_per_epoch=len(labeled_data) // args.batch_size,
        decay_epoch=args.epochs
    )
    
    optimizer = nn.Momentum(
        params=student_model.trainable_params(),
        learning_rate=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Create EMA updater
    teacher_ema = EMA(student_model, teacher_model, decay=args.ema_decay)
    
    # Modify DistillationTrainStep class
    class DistillationTrainStep(nn.Cell):
        def __init__(self, student_model, teacher_model, optimizer, args):
            super(DistillationTrainStep, self).__init__()
            self.student_model = student_model
            self.teacher_model = teacher_model
            self.optimizer = optimizer
            self.args = args
            # Explicitly specify network parameters to match optimizer parameters
            self.weights = self.student_model.trainable_params()
            self.grad_fn = ops.value_and_grad(self.forward, None, self.weights, has_aux=False)
        
        def forward(self, labeled_imgs, labels, unlabeled_weak, unlabeled_strong):
            # Teacher model inference
            teacher_labeled_logits = self.teacher_model(labeled_imgs)
            teacher_unlabeled_logits = self.teacher_model(unlabeled_weak)
            
            # Student model inference
            student_labeled_logits = self.student_model(labeled_imgs)
            student_unlabeled_strong_logits = self.student_model(unlabeled_strong)
            
            # Calculate distillation loss
            distill_loss = self._distill_loss_fn(student_labeled_logits, teacher_labeled_logits, labels)
            
            # Calculate semi-supervised consistency loss
            # Generate pseudo labels
            pseudo_labels = ops.softmax(teacher_unlabeled_logits, axis=1)
            max_probs, targets = ops.max(pseudo_labels, axis=1)
            mask = (max_probs > 0.95).astype(ms.float32)  # High-confidence pseudo label mask
            
            # Calculate consistency loss
            consistency_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')(
                student_unlabeled_strong_logits, 
                targets
            )
            consistency_loss = (consistency_loss * mask).mean()
            
            # Calculate total loss
            total_loss = distill_loss + self.args.beta * consistency_loss
            
            return total_loss
        
        def _distill_loss_fn(self, student_logits, teacher_logits, labels):
            # Supervised loss
            ce_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
            hard_loss = ce_loss(student_logits, labels)
            
            # Distillation soft label loss
            T = self.args.T
            soft_student = ops.softmax(student_logits / T, axis=1)
            soft_teacher = ops.softmax(teacher_logits / T, axis=1)
            
            kl_div = ops.kl_div(
                ops.log(soft_student + 1e-10),
                soft_teacher,
                reduction='batchmean'
            ) * (T * T)
            
            # Combine losses
            return (1 - self.args.alpha) * hard_loss + self.args.alpha * kl_div
        
        def construct(self, labeled_imgs, labels, unlabeled_weak, unlabeled_strong):
            loss, grads = self.grad_fn(labeled_imgs, labels, unlabeled_weak, unlabeled_strong)
            self.optimizer(grads)
            return loss
    
    # Create training network
    net = DistillationTrainStep(student_model, teacher_model, optimizer, args)
    
    # Define evaluation function
    def evaluate():
        """Evaluation function used during teacher model training"""
        correct = 0
        total = 0
        model.set_train(False)
        
        # Recreate test dataset before each evaluation
        if args.dataset == 'cub200' and test_dataset is not None:
            # Recreate CUB200 test set
            test_transform_ops = [
                vision.Resize(256),
                vision.CenterCrop(224),
                vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                vision.HWC2CHW()
            ]
            
            # Create generator function - Use passed-in test_dataset_obj instead of global variable
            def test_generator():
                for idx in range(len(test_dataset)):
                    img, label = test_dataset[idx]
                    img_np = np.array(img)
                    
                    # Apply test transformation
                    for op in test_transform_ops:
                        img_np = op(img_np)
                    
                    yield img_np, label
            
            # Create new test dataset
            eval_ds = ds.GeneratorDataset(
                source=test_generator(),
                column_names=["image", "label"],
                shuffle=False
            )
            eval_ds = eval_ds.batch(args.batch_size)
        else:
            # Recreate CIFAR test set
            eval_ds = create_dataset(
                name=args.dataset,
                root=args.data_dir,
                split='test',
                download=False
            )
            
            # Apply transformation to test set
            eval_ds = eval_ds.map(operations=test_transform, input_columns=["image"])
            eval_ds = eval_ds.batch(args.batch_size)
        
        # Evaluate using new dataset
        for data in eval_ds.create_tuple_iterator():
            # Get images and labels based on dataset type
            if args.dataset == 'cifar10' or args.dataset == 'cub200':
                images, labels = data
            else:  # cifar100
                images, fine_labels, _ = data
                labels = fine_labels
            
            # Use tensor data for prediction
            outputs = model(images)
            _, predicted = ops.max(outputs, axis=1)
            predicted = predicted.astype(ms.int32)
            labels = labels.astype(ms.int32)
            total += labels.shape[0]
            correct += (predicted == labels).astype(ms.float32).sum().asnumpy().item()
        
        # Prevent division by zero error
        if total == 0:
            logging.warning("No samples processed in test set, please check test dataset")
            return 0.0
        
        return correct / total
    
    # Create model save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Start training
    print("Starting knowledge distillation semi-supervised training...")
    best_acc = 0.0
    
    labeled_iter = labeled_ds.create_dict_iterator()
    unlabeled_iter = unlabeled_ds.create_dict_iterator()
    
    for epoch in range(args.epochs):
        # Set training mode
        student_model.set_train(True)
        teacher_model.set_train(False)  # Teacher model always in evaluation mode
        
        # Train one epoch
        total_loss = 0.0
        steps = min(len(labeled_data) // args.batch_size, len(unlabeled_data) // args.batch_size)
        
        logging.info(f"Epoch {epoch+1}/{args.epochs} started...")
        epoch_progress = tqdm(range(steps), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step in epoch_progress:
            try:
                labeled_batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = labeled_ds.create_dict_iterator()
                labeled_batch = next(labeled_iter)
            
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = unlabeled_ds.create_dict_iterator()
                unlabeled_batch = next(unlabeled_iter)
            
            labeled_imgs = labeled_batch["image"]
            labels = labeled_batch["label"]
            weak_imgs = unlabeled_batch["weak_image"]
            strong_imgs = unlabeled_batch["strong_image"]
            
            loss = net(labeled_imgs, labels, weak_imgs, strong_imgs)
            total_loss += loss.asnumpy().item()
            
            # Update progress bar
            epoch_progress.set_postfix(loss=f"{loss.asnumpy().item():.4f}")
            
            if step % 50 == 0:
                logging.info(f"Epoch: {epoch+1}/{args.epochs}, Step: {step+1}/{steps}, Loss: {loss.asnumpy().item():.4f}")
        
        # Evaluate model
        avg_loss = total_loss / steps
        logging.info(f"Epoch: {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")
        
        # Evaluate model only every 5 epochs or last epoch
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            top1_acc = evaluate()
            logging.info(f"Epoch: {epoch+1}/{args.epochs}, TOP1 Accuracy: {top1_acc:.4f}")
            
            # Save best model (based on TOP1 accuracy)
            if top1_acc > best_acc:
                best_acc = top1_acc
                ms.save_checkpoint(student_model, os.path.join(args.save_dir, f"{args.student_model}_best.ckpt"))
                logging.info(f"Best model saved, TOP1 accuracy: {best_acc:.4f}")
        
        # Save checkpoint every certain number of epochs
        if (epoch + 1) % args.save_interval == 0:
            ms.save_checkpoint(student_model, os.path.join(args.save_dir, f"{args.student_model}_epoch{epoch+1}.ckpt"))
            logging.info(f"Checkpoint saved at epoch {epoch+1}")
    
    print(f"Training completed! Best TOP1 accuracy: {best_acc:.4f}")


def train_teacher(model, optimizer, train_ds, test_ds, epochs, labeled_size, test_dataset_obj=None):
    """Train teacher model using labeled data"""
    
    # Define evaluation function
    def evaluate():
        """Evaluation function used during teacher model training"""
        correct = 0
        total = 0
        model.set_train(False)
        
        # Recreate test dataset before each evaluation
        if args.dataset == 'cub200' and test_dataset_obj is not None:
            # Recreate CUB200 test set
            test_transform_ops = [
                vision.Resize(256),
                vision.CenterCrop(224),
                vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                vision.HWC2CHW()
            ]
            
            # Create generator function - Use passed-in test_dataset_obj instead of global variable
            def test_generator():
                for idx in range(len(test_dataset_obj)):
                    img, label = test_dataset_obj[idx]
                    img_np = np.array(img)
                    
                    # Apply test transformation
                    for op in test_transform_ops:
                        img_np = op(img_np)
                    
                    yield img_np, label
            
            # Create new test dataset
            eval_ds = ds.GeneratorDataset(
                source=test_generator(),
                column_names=["image", "label"],
                shuffle=False
            )
            eval_ds = eval_ds.batch(args.batch_size)
        else:
            # Recreate CIFAR test set
            eval_ds = create_dataset(
                name=args.dataset,
                root=args.data_dir,
                split='test',
                download=False
            )
            
            # Apply transformation to test set
            eval_ds = eval_ds.map(operations=test_transform, input_columns=["image"])
            eval_ds = eval_ds.batch(args.batch_size)
        
        # Evaluate using new dataset
        for data in eval_ds.create_tuple_iterator():
            # Get images and labels based on dataset type
            if args.dataset == 'cifar10' or args.dataset == 'cub200':
                images, labels = data
            else:  # cifar100
                images, fine_labels, _ = data
                labels = fine_labels
            
            # Use tensor data for prediction
            outputs = model(images)
            _, predicted = ops.max(outputs, axis=1)
            predicted = predicted.astype(ms.int32)
            labels = labels.astype(ms.int32)
            total += labels.shape[0]
            correct += (predicted == labels).astype(ms.float32).sum().asnumpy().item()
        
        # Prevent division by zero error
        if total == 0:
            logging.warning("No samples processed in test set, please check test dataset")
            return 0.0
        
        return correct / total
    
    logging.info("Starting teacher model training...")
    best_acc = 0.0
    
    # Create dynamic learning rate
    total_steps = epochs * (labeled_size // args.batch_size)
    lr = nn.cosine_decay_lr(
        min_lr=0.0,
        max_lr=args.lr,
        total_step=total_steps,
        step_per_epoch=labeled_size // args.batch_size,
        decay_epoch=epochs
    )
    dynamic_lr = ms.Parameter(ms.Tensor(lr, ms.float32))
    
    # Create new optimizer with dynamic learning rate
    optimizer = nn.Momentum(
        params=model.trainable_params(),
        learning_rate=dynamic_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Define training step
    class TeacherTrainStep(nn.Cell):
        def __init__(self, network, optimizer):
            super(TeacherTrainStep, self).__init__()
            self.network = network
            self.optimizer = optimizer
            self.grad_fn = ops.value_and_grad(self.forward, None, self.network.trainable_params())
            self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        
        def forward(self, images, labels):
            logits = self.network(images)
            loss = self.loss_fn(logits, labels)
            return loss
        
        def construct(self, images, labels):
            loss, grads = self.grad_fn(images, labels)
            self.optimizer(grads)
            return loss
    
    # Create training network
    train_step = TeacherTrainStep(model, optimizer)
    
    # Training loop
    for epoch in range(epochs):
        model.set_train(True)
        total_loss = 0.0
        steps = 0
        
        # Calculate total steps per epoch
        total_steps = labeled_size // args.batch_size
        
        # Use tqdm to display progress
        train_iter = tqdm(range(total_steps), desc=f"Teacher Epoch {epoch+1}/{epochs}")
        
        data_iter = train_ds.create_dict_iterator()
        for _ in train_iter:
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = train_ds.create_dict_iterator()
                data = next(data_iter)
            
            images, labels = data["image"], data["label"]
            loss = train_step(images, labels)
            total_loss += loss.asnumpy().item()
            steps += 1
            
            # Update progress bar
            train_iter.set_postfix(loss=f"{loss.asnumpy().item():.4f}")
            
            # Train for fixed number of steps per epoch
            if steps >= total_steps:
                break
        
        # Calculate average loss
        avg_loss = total_loss / steps
        
        # Evaluate model only every 5 epochs or last epoch
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            acc = evaluate()
            
            # Record training information
            logging.info(f"Teacher Epoch: {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
            
            # Save best model
            if acc > best_acc:
                best_acc = acc
                ms.save_checkpoint(model, "teacher_best.ckpt")
                logging.info(f"Best teacher model saved, accuracy: {acc:.4f}")
        else:
            # Only record training loss when not evaluating
            logging.info(f"Teacher Epoch: {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    
    # Load best model after training
    param_dict = ms.load_checkpoint("teacher_best.ckpt")
    ms.load_param_into_net(model, param_dict)
    logging.info(f"Teacher model training completed, best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    train_distill() 