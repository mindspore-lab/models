import time
import tqdm
from sklearn.metrics import roc_auc_score
from dataset.douban import Douban,DoubanMusic
import mindspore as ms

from model.fnn import FactorizationSupportedNeuralNetworkModel

def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    if name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(256,256,256), dropout=0)
    return None

class EarlyStopper():

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            ms.save_checkpoint(model, self.save_path)
            return True
        if self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        return False

def train(model, optimizer, data_loader, criterion, log_interval=100):
    model.set_train(True)
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    def forward_fn(data, label):
        logits = model(data)
        loss = criterion(logits, label)
        return loss, logits
    grad_fn = ms.ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    for i, (fields, target) in enumerate(tk0):
        (loss, _), grads = grad_fn(fields, target)
        optimizer(grads)
        total_loss += sum(loss)
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

def test(model, data_loader):
    model.set_train(False)
    targets, predicts = list(), list()
    for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
        y = model(fields)
        targets.extend(target.asnumpy().tolist())
        predicts.extend(y.asnumpy().tolist())
    return roc_auc_score(targets, predicts)

def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         job):
    ms.set_context(device_target=device)
    train_dataset = Douban('train')
    valid_dataset = Douban('val')
    test_dataset = DoubanMusic('test')
    train_data_loader = ms.dataset.GeneratorDataset(train_dataset, column_names=["item", "target"], shuffle=True)
    train_data_loader = train_data_loader.batch(batch_size, drop_remainder=False)
    valid_data_loader = ms.dataset.GeneratorDataset(valid_dataset, column_names=["item", "target"], shuffle=False)
    valid_data_loader = valid_data_loader.batch(batch_size, drop_remainder=False)
    test_data_loader = ms.dataset.GeneratorDataset(test_dataset, column_names=["item", "target"], shuffle=False)
    test_data_loader = test_data_loader.batch(batch_size, drop_remainder=False)
    model = get_model(model_name, train_dataset)
    criterion = ms.nn.BCELoss()
    optimizer = ms.nn.Adam(params=model.trainable_params(), learning_rate=learning_rate, weight_decay=weight_decay)
    save_path = f'{save_dir}/douban_{model_name}_train_v2_{job}.ckpt'
    early_stopper = EarlyStopper(num_trials=5, save_path=save_path)
    start = time.time()
    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion)
        auc = test(model, valid_data_loader)
        print('epoch:', epoch_i, 'validation: auc:', auc)
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
    end = time.time()
    param_dict = ms.load_checkpoint(save_path)
    ms.load_param_into_net(model, param_dict)
    auc = test(model, test_data_loader)
    print(f'test auc: {auc}')
    print('running time = ', end - start)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='douban')
    parser.add_argument('--dataset_path', default='dataset/')
    parser.add_argument('--model_name', default='fnn')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='GPU', help='CPU, GPU, Ascend, Davinci')
    parser.add_argument('--save_dir', default='/chkpt/')
    parser.add_argument('--job', type=int, default=6)
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir,
         args.job)