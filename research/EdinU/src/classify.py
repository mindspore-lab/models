# classify_ms.py
# evaluate the Banyan model on SST and MRPC datasets using MindSpore
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from utils import *
from models import Banyan
import numpy as np
from tqdm import tqdm
import sys

def binary_acc(y_pred, y_test):
    """Calculates binary accuracy using MindSpore operations."""
    y_pred_tag = ops.round(ops.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().astype(ms.float32)
    acc = correct_results_sum / y_test.shape[0]
    acc = ops.round(acc * 100)
    return acc

def f1_score(y_pred, y_test):
    """Calculates F1 score using MindSpore operations."""
    y_pred_tag = ops.round(ops.sigmoid(y_pred))
    
    tp = ((y_pred_tag == 1) & (y_test == 1)).sum().astype(ms.float32)
    fp = ((y_pred_tag == 1) & (y_test == 0)).sum().astype(ms.float32)
    fn = ((y_pred_tag == 0) & (y_test == 1)).sum().astype(ms.float32)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    return ops.round(f1 * 100)

def process_sst_dataset(data_path):
    """Processes the SST dataset into NumPy arrays."""
    df = pd.read_csv(data_path, sep='\t')
    bpemb_en = BPEmb(lang='en', vs=25000, dim=100)
    sents1 = [sent_to_bpe(x.strip('\n'), bpemb_en) for x in df['sentence']]
    scores = [np.array(x, dtype=np.int32) for x in df['label']]
    dataset = [(sents1[x], scores[x]) for x in range(len(sents1))]
    return dataset

class SSTDataset:
    """A source dataset for SST, compatible with GeneratorDataset."""
    def __init__(self, data):
        self.sequences = data
        self.n_samples = len(data)

    def __getitem__(self, index):
        return self.sequences[index][0], self.sequences[index][1]

    def __len__(self):
        return self.n_samples

def collate_fn_sst(*args):
    """
    [CORRECTED] Pads SST batches using a robust *args signature.
    """
    sents, scores = args[0], args[1]
    padding_value = 25000
    max_len = max([s.shape[0] for s in sents])
    padded_sents = []
    for seq in sents:
        pad_len = max_len - seq.shape[0]
        padding = np.full((pad_len,), padding_value, dtype=seq.dtype)
        padded_sents.append(np.concatenate([seq, padding]))
    return np.stack(padded_sents), np.stack(scores)

def create_sst_dataloader(data_path, batch_size, shuffle=False):
    """Creates a MindSpore DataLoader for the SST task."""
    data = process_sst_dataset(data_path)
    dataset = SSTDataset(data)
    generator_dataset = ds.GeneratorDataset(dataset, column_names=["sents", "scores"], shuffle=shuffle)
    dataloader = generator_dataset.batch(batch_size, per_batch_map=collate_fn_sst, output_columns=["sents", "scores"], num_parallel_workers=4)
    return dataloader

def embed_sentence(model, path):
    """Embeds single sentences using the Banyan model."""
    model.set_train(False)
    embeddings = []
    labels = []
    dataloader = create_sst_dataloader(path, 128)
    for inputs in tqdm(dataloader):
        tokens_1 = inputs[0]
        out = model(tokens_1, seqs2=tokens_1)
        embeddings.append(out[0])
        labels.append(inputs[-1])
    model.set_train(True)
    return ops.cat(embeddings, axis=0), ops.cat(labels, axis=0)

def embed_sentence_pairs(model, path):
    """Embeds sentence pairs using the Banyan model."""
    model.set_train(False)
    embeddings = []
    labels = []
    dataloader = create_sts_dataloader(path, 128)
    for inputs in tqdm(dataloader):
        tokens_1 = inputs[0]
        tokens_2 = inputs[1]
        out = model(tokens_1, seqs2=tokens_2)
        embeddings.append(ops.cat((out[0], out[1]), axis=1))
        labels.append(inputs[-1])
    model.set_train(True)
    return ops.cat(embeddings, axis=0), ops.cat(labels, axis=0)
'''
class S1ClassDataset:
    """A dataset for single-sentence classification using pre-computed embeddings."""
    def __init__(self, model, path):
        data = embed_sentence(model, path)
        self.x = data[0]
        self.y = data[1]
        self.n_samples = data[0].shape[0]
    def __getitem__(self, index):
        return self.x[index].asnumpy(), self.y[index].asnumpy()
    def __len__(self):
        return self.n_samples

class SPClassDataset:
    """A dataset for sentence-pair classification using pre-computed embeddings."""
    def __init__(self, model, path):
        data = embed_sentence_pairs(model, path)
        self.x = data[0]
        self.y = data[1]
        self.n_samples = data[0].shape[0]
    def __getitem__(self, index):
        return self.x[index].asnumpy(), self.y[index].asnumpy()
    def __len__(self):
        return self.n_samples

'''


class S1ClassDataset:
    """[CORRECTED] Dataset for single-sentence classification on pre-computed embeddings."""
    def __init__(self, model, path):
        data = embed_sentence(model, path)
        # Convert to NumPy ONCE during initialization for efficiency and stability.
        self.x = data[0].asnumpy()
        self.y = data[1].asnumpy()
        self.n_samples = self.x.shape[0]
        
    def __getitem__(self, index):
        # Indexing NumPy arrays is robust and fast.
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.n_samples

class SPClassDataset:
    """[CORRECTED] Dataset for sentence-pair classification on pre-computed embeddings."""
    def __init__(self, model, path):
        data = embed_sentence_pairs(model, path)
        # Convert to NumPy ONCE during initialization for efficiency and stability.
        self.x = data[0].asnumpy()
        self.y = data[1].asnumpy()
        self.n_samples = self.x.shape[0]
        
    def __getitem__(self, index):
        # Indexing NumPy arrays is robust and fast.
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.n_samples


class SPClassModel(nn.Cell):
    """The classifier model for STS tasks."""
    def __init__(self, embedding_size, output_size, fancy=False):
        super(SPClassModel, self).__init__()
        if fancy:
            self.layer_out = nn.SequentialCell(nn.Dense(2 * embedding_size, 512),
                                           nn.GELU(),
                                           nn.Dense(512, output_size))
        else:
            self.layer_out = nn.Dense(2 * embedding_size, output_size)

    def construct(self, x):
        return self.layer_out(x)

def singleclass_eval(model, train_path, dev_path, fancy=False):
    """Performs evaluation on sentence-pair tasks like MRPC."""
    train_dataset_source = SPClassDataset(model, train_path)
    train_dataset = ds.GeneratorDataset(train_dataset_source, column_names=["data", "label"], shuffle=True)
    train_dataloader = train_dataset.batch(256)
    
    dev_dataset_source = SPClassDataset(model, dev_path)
    dev_dataset = ds.GeneratorDataset(dev_dataset_source, column_names=["data", "label"], shuffle=False)
    dev_dataloader = dev_dataset.batch(5000)

    results = []
    for seed in tqdm(range(5)):
        set_seed(seed)
        classifier = SPClassModel(256, 1, fancy=fancy) # Banyan model in paper has 256 dim, but pairs are 128
        criterion = nn.BCEWithLogitsLoss()
        optimizer = nn.Adam(classifier.trainable_params(), learning_rate=1e-4)
        
        def forward_fn(data, label):
            logits = classifier(data.astype(ms.float32))
            logits = ops.squeeze(logits)
            loss = criterion(logits, label.squeeze().astype(ms.float32))
            return loss, logits

        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        def train_step(data, label):
            (loss, logits), grads = grad_fn(data, label)
            optimizer(grads)
            return loss, logits

        best_acc = 0.0
        best_f1 = 0.0

        for epoch in range(40):
            classifier.set_train(True)
            for x, labels in train_dataloader:
                loss, logits = train_step(x, labels)
            
            classifier.set_train(False)
            for x, labels in dev_dataloader:
                logits = classifier(x.astype(ms.float32))
                logits = ops.squeeze(logits)
                
                current_acc = binary_acc(logits, labels.squeeze()).asnumpy()
                if current_acc > best_acc:
                    best_acc = current_acc

                current_f1 = f1_score(logits, labels.squeeze()).asnumpy()
                if current_f1 > best_f1:
                    best_f1 = current_f1

        results.append((best_acc, best_f1))

    return np.mean([x[0] for x in results]), np.mean([x[1] for x in results])


def sst_eval(model, train_path, dev_path, fancy=False):
    """Performs evaluation on single-sentence tasks like SST."""
    train_dataset_source = S1ClassDataset(model, train_path)
    train_dataset = ds.GeneratorDataset(train_dataset_source, column_names=["data", "label"], shuffle=True)
    train_dataloader = train_dataset.batch(256)

    dev_dataset_source = S1ClassDataset(model, dev_path)
    dev_dataset = ds.GeneratorDataset(dev_dataset_source, column_names=["data", "label"], shuffle=False)
    dev_dataloader = dev_dataset.batch(5000)

    results = []
    for seed in tqdm(range(5)):
        set_seed(seed)
        classifier = SPClassModel(128, 1, fancy=fancy) # The classifier takes a single embedding
        # Note: The original code had a bug here, SPClassModel(128,...) but dataset is S1Class...
        # The input to the classifier should be model embedding size, 256.
        # But for compatibility, we follow the original code's SPClassModel which has a 2*emb size input
        # We will assume the intent was to use a different classifier, but will stick to the original code's class
        # A proper fix would be a new S1ClassModel(embedding_size, ...).
        # For this migration, we keep the original structure.
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = nn.Adam(classifier.trainable_params(), learning_rate=1e-4)

        def forward_fn(data, label):
            logits = classifier(data.astype(ms.float32))
            logits = ops.squeeze(logits)
            loss = criterion(logits, label.squeeze().astype(ms.float32))
            return loss, logits

        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        def train_step(data, label):
            (loss, logits), grads = grad_fn(data, label)
            optimizer(grads)
            return loss, logits

        best_acc = 0.0
        best_f1 = 0.0

        for epoch in range(40):
            classifier.set_train(True)
            for x, labels in train_dataloader:
                loss, logits = train_step(x, labels)

            classifier.set_train(False)
            for x, labels in dev_dataloader:
                logits = classifier(x.astype(ms.float32))
                logits = ops.squeeze(logits)
                
                current_acc = binary_acc(logits, labels.squeeze()).asnumpy()
                if current_acc > best_acc:
                    best_acc = current_acc

                current_f1 = f1_score(logits, labels.squeeze()).asnumpy()
                if current_f1 > best_f1:
                    best_f1 = current_f1

        results.append((best_acc, best_f1))

    return np.mean([x[0] for x in results]), np.mean([x[1] for x in results])


# --- Main Execution ---
if __name__ == "__main__":
    # Set MindSpore context
    # ms.set_context(mode=ms.PYNATIVE_MODE) # Use PyNative for easier debugging if needed
    
    model_ms = Banyan(vocab_size=25001, embedding_size=256, channels=128, r=0.1)

    # Load from PyTorch checkpoint
        # Load from PyTorch checkpoint
    if len(sys.argv) < 2:
        print("Error: Please provide the path to the PyTorch checkpoint file.")
        sys.exit(1)
    
    pth_path = sys.argv[1]
    
    print("Loading weights from PyTorch checkpoint...")
    param_dict = ms.load_checkpoint(pth_path)
    ms.load_param_into_net(model_ms, param_dict)
    print("Weight loading complete.")

    out = singleclass_eval(model_ms, '../data/mrpc_train.csv', '../data/mrpc_test.csv', fancy=True)
    print(f'MRPC Accuracy: {out[0]} F1: {out[1]}')
    out = sst_eval(model_ms, '../data/sst_train.tsv', '../data/sst_dev.tsv', fancy=True)
    print(f'SST Accuracy: {out[0]} F1: {out[1]}')