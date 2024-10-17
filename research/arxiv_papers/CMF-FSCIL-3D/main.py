import os
import logging
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import context,Parameter,Tensor
from mindspore.train.serialization import (
    load_checkpoint,
    load_param_into_net,
    save_checkpoint,
)
from mindspore.nn.optim import AdamWeightDecay
from tqdm import tqdm
from src.model import PN_SSG, NCMClassifier
from src.dataset import ShapeNetTest, ShapeNetTrain
from src.tokenizer import SimpleTokenizer
from src.config.config import config
from pprint import pformat
import math
from src.config.label import shapenet_label
from src.loss import LosswithIMG


context.set_context(device_target="GPU")
ms.set_context(pynative_synchronize=False)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
log_dir = "log/"
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = ops.TopK()(output, maxk)
    pred = pred.transpose()
    correct = ops.Equal()(pred, ops.ExpandDims()(target, 0))

    res = []
    for k in topk:
        correct_k = ops.ReduceSum()(correct[:k].view(-1), 0, keep_dims=True)
        res.append(correct_k * 100.0 / batch_size)
    return res


class WithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, pc, texts, image, main_label, all_label):
        outputs = self.backbone(pc, texts, image)
        loss_dict = self.loss_fn(main_label, all_label, outputs)
        return loss_dict['loss']  


def train_single_card(config):
    model = PN_SSG()

    
    initial_lr = config.lr

    
    criterion = LosswithIMG()
    net_with_loss = WithLossCell(model, criterion)

    start_epoch = 0
    best_acc = 0.0

    
    train_dataset = ShapeNetTrain(task_id=0)
    train_loader = ds.GeneratorDataset(
        train_dataset,
        ["pc", "image", "texts", "labels"],
        shuffle=True,  
        num_parallel_workers=config.workers,
    ).batch(config.batch_size, drop_remainder=True)

    val_dataset = ShapeNetTest(task_id=0)
    val_loader = ds.GeneratorDataset(
        val_dataset,
        ["pc", "label", "other_column"],  
        shuffle=False,
        num_parallel_workers=config.workers,
    ).batch(config.batch_size, drop_remainder=True)

    total_steps = train_loader.get_dataset_size()  

    for epoch in tqdm(range(start_epoch, config.epochs), desc="training"):
        logger.info(f"Epoch {epoch + 1}/{config.epochs} started.")

        
        cos_decay = 0.5 * (1 + math.cos(math.pi * epoch / config.epochs))
        decayed_lr = initial_lr * cos_decay

        
        optimizer = AdamWeightDecay(
            model.trainable_params(),
            learning_rate=ms.Tensor(decayed_lr, ms.float32),  
            weight_decay=config.wd,
        )

        train_net = nn.TrainOneStepCell(net_with_loss, optimizer)
        train_net.set_train()

        total_loss = 0.0  

        
        with tqdm(total=total_steps, desc=f"Training Epoch {epoch + 1}") as pbar:
            for data in train_loader.create_dict_iterator():
                pc = data["pc"]
                image = data["image"]
                texts = data["texts"]
                all_label = data["labels"]
                main_label = ops.floor_div(all_label, 5)

                
                loss = train_net(pc, texts, image, main_label, all_label)
                total_loss += loss.asnumpy()  

                
                pbar.update(1)

        avg_loss = total_loss / total_steps
        
        logger.info(f"Epoch {epoch + 1} completed with avg loss: {avg_loss:.4f}, learning rate: {decayed_lr:.6f}")

        
        if epoch % config.eval_freq == 0 or epoch == config.epochs - 1:
            eval_metric = evaluate_model(model, val_loader)
            is_best = eval_metric["top1_label_main"] > best_acc
            if is_best:
                best_acc = eval_metric["top1_label_main"]

            logger.info(f"Validation results at epoch {epoch + 1}: {pformat(eval_metric)}")

            
            save_checkpoint(model, os.path.join(config.output_dir, f"checkpoint_{epoch}.ckpt"))
            logger.info(f"Checkpoint saved at epoch {epoch + 1}.")
            if is_best:
                save_checkpoint(model, os.path.join(config.output_dir, "best_checkpoint.ckpt"))
                logger.info(f"Best checkpoint updated at epoch {epoch} with accuracy {best_acc:.4f}.")


def evaluate_model(model, val_loader):
    model.set_train(False)

    tokenizer = SimpleTokenizer()
    text_features = []
    labels = shapenet_label[:25]
    
    for label in labels:
        text = f"a point cloud model of {label}."
        captions = [tokenizer(text)]
        texts = ms.Tensor(captions)
        class_embeddings = model.encode_text(texts)
        class_embeddings = ops.L2Normalize(axis=-1)(class_embeddings)
        class_embeddings = class_embeddings.mean(axis=0)
        class_embeddings = ops.L2Normalize(axis=-1)(class_embeddings)
        text_features.append(class_embeddings)

    text_features = ops.stack(text_features, axis=0)

    metrics = {
        "top1_label_main": 0,
        "top5_label_main": 0,
        "top1_label_all": 0,
        "top5_label_all": 0,
        "top1_text": 0,
        "top5_text": 0,
    }

    total_steps = val_loader.get_dataset_size()

    with tqdm(total=total_steps, desc="Evaluating") as pbar:
        for data in val_loader.create_dict_iterator():
            pc = data["pc"]
            target = data["labels"]

            logits_pc_text, logits_label_all, logits_label_main = model.encode_pc(pc)
            logits_text = ops.matmul(logits_pc_text, text_features.T)

            acc1_label_main, acc5_label_main = accuracy(
                logits_label_main, target, topk=(1, 5)
            )
            metrics["top1_label_main"] += acc1_label_main
            metrics["top5_label_main"] += acc5_label_main

            logits_label_all_summed = logits_label_all.reshape(
                logits_label_all.shape[0], 25, 5
            ).sum(axis=2)
            acc1_label_all, acc5_label_all = accuracy(
                logits_label_all_summed, target, topk=(1, 5)
            )
            metrics["top1_label_all"] += acc1_label_all
            metrics["top5_label_all"] += acc5_label_all

            acc1_text, acc5_text = accuracy(logits_text, target, topk=(1, 5))
            metrics["top1_text"] += acc1_text
            metrics["top5_text"] += acc5_text

            
            pbar.update(1)

    
    for key in metrics:
        metrics[key] /= total_steps

    logger.info(
        f"Evaluation results - Acc@1_label_main: {metrics['top1_label_main']:.2f}, "
        f"Acc@5_label_main: {metrics['top5_label_main']:.2f}, "
        f"Acc@1_label_all: {metrics['top1_label_all']:.2f}, "
        f"Acc@5_label_all: {metrics['top5_label_all']:.2f}, "
        f"Acc@1_text: {metrics['top1_text']:.2f}, Acc@5_text: {metrics['top5_text']:.2f}"
    )

    return metrics


def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def check_and_fix_model_parameters(model, checkpoint_path, new_checkpoint_path):
    
    checkpoint = load_checkpoint(checkpoint_path)

    
    model_params = model.parameters_dict()
    checkpoint_params = {name: param for name, param in checkpoint.items()}

    logger.info("开始检查模型参数和预训练参数的一致性...")

    
    param_name_mapping = {
        'weight': 'gamma',
        'bias': 'beta'
    }

    updated_checkpoint = {}
    missing_in_checkpoint = []
    missing_in_model = []
    updated_names = []

    for name, param in model_params.items():
        
        if name not in checkpoint_params:
            
            for old_name, new_name in param_name_mapping.items():
                mapped_name = name.replace(new_name, old_name)
                if mapped_name in checkpoint_params:
                    updated_checkpoint[name] = checkpoint_params[mapped_name]
                    updated_names.append((mapped_name, name))  
                    break
            else:
                missing_in_checkpoint.append(name)
        else:
            updated_checkpoint[name] = checkpoint_params[name]

    for name in checkpoint_params.keys():
        if name not in model_params:
            missing_in_model.append(name)

    
    if missing_in_checkpoint:
        logger.warning(f"模型中存在 {len(missing_in_checkpoint)} 个参数未在预训练参数中找到: {missing_in_checkpoint}")

    if missing_in_model:
        logger.warning(f"预训练参数中存在 {len(missing_in_model)} 个参数未在模型中找到: {missing_in_model}")

    if updated_names:
        logger.info(f"预训练参数名进行了以下替换: {updated_names}")

    
    save_checkpoint(updated_checkpoint, new_checkpoint_path)
    logger.info(f"新的预训练模型参数已保存到: {new_checkpoint_path}")

    logger.info("模型参数检查和调整完成。")


def increament_init(checkpoint_path, method, mode, device):
    logger = setup_logger(mode, f"log/test_{mode}_ssg_{method}_0-6.log")

    
    model_ssg = PN_SSG()
    
    
    checkpoint = load_checkpoint(checkpoint_path)
    load_param_into_net(model_ssg, checkpoint)
    
    for name, param in model_ssg.parameters_and_names():
        param.requires_grad = False
        print(f"load {name} and freeze")
    model_ssg.set_train(False)
    model_ssg.point_encoder.set_train(False)
    model = NCMClassifier(model_ssg.point_encoder).to_float(ms.float32)  
    
    return model, logger


def increament_test(task_id, model, batch_size=80, mode="with"):
    test_dataset = ShapeNetTest(task_id=task_id)
    test_loader = ds.GeneratorDataset(test_dataset, ["pc", "labels", "other_column"], shuffle=False).batch(batch_size)

    cos_count, ed_count, dp_count = 0, 0, 0
    l = len(test_dataset)

    total_steps = test_loader.get_dataset_size()

    with tqdm(total=total_steps, desc="testing") as pbar:
        for data in test_loader.create_dict_iterator():
            batch_pcs = data["pc"]
            batch_targets = data["labels"]

            if len(batch_pcs) == 1:
                cos_count += 1
                ed_count += 1
                dp_count += 1
                continue

            preds = model.predict(batch_pcs)
            if mode == "with":
                _, _, logits_main = model.encoder(batch_pcs)
            for k, target in enumerate(batch_targets):
                if mode == "with" and target < 25:
                    if logits_main[k].argmax(-1).asnumpy() == target.asnumpy():
                        cos_count += 1
                        ed_count += 1
                        dp_count += 1
                else:
                    if preds["cos"][k]["cate"] == target.asnumpy():
                        cos_count += 1
                    if preds["ed"][k]["cate"] == target.asnumpy():
                        ed_count += 1
                    if preds["dp"][k]["cate"] == target.asnumpy():
                        dp_count += 1

            pbar.update(1)

    return cos_count / l, ed_count / l, dp_count / l


def increament_train(i, model, train_batch_size=80, method="mean"):
    task_id = i if i != 0 else -2
    train_dataset = ShapeNetTrain(task_id=task_id)
    pc_datas, cates = [], []

    for j in tqdm(range(0, len(train_dataset), train_batch_size), desc="training"):
        samples = [train_dataset[k] for k in range(j, min(j + train_batch_size, len(train_dataset)))]
        pc_datas = [sample[0] for sample in samples]

        if i == 0:
            cates = [ms.ops.floor_div(sample[1], Tensor(5, ms.int32)).asnumpy().item() for sample in samples]
        else:
            cates = [sample[1] for sample in samples]

        model.train(cates, pc_datas, method=method)
    model.train_last(method=method)
    return model


def train_and_test_with(start, end, model_with, batch_size, acc_with, method):
    for i in range(start, end):
        if i != 0:
            model_with = increament_train(i, model_with, train_batch_size=batch_size, method=method)
        
        for lst, val in zip(
            (acc_with["cos"], acc_with["ed"], acc_with["dp"]),
            increament_test(i, model_with, batch_size=batch_size, mode="with"),
        ):
            lst.append(val)


def train_and_test_without(start, end, model_without, batch_size, acc_without, method):
    for i in range(start, end):
        model_without = increament_train(i, model_without, train_batch_size=batch_size, method=method)
        
        for lst, val in zip(
            (acc_without["cos"], acc_without["ed"], acc_without["dp"]),
            increament_test(i, model_without, batch_size=batch_size, mode="without"),
        ):
            lst.append(val)


def increament_process(checkpoint_path, method="mean"):
    
    
    
    
    
    
    model_without, logger_without = increament_init(checkpoint_path, method, "without", "cuda:0")
    acc_without = {"cos": [], "ed": [], "dp": []}
    train_and_test_without(0, 7, model_without, 120, acc_without, method)
    logger_without.info("Accuracies:\n")
    logger_without.info(pformat(acc_without))


def main():
    os.makedirs(config.output_dir, exist_ok=True)
    logger.info("Starting training on a single GPU")
    train_single_card(config)
    logger.info("Training completed")


if __name__ == "__main__":
    ms.set_context(pynative_synchronize=False)
    
    
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    increament_process(
        "outputs/checkpoint_79.ckpt", method="median"
    )