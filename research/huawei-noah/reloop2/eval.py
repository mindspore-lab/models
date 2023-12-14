"""train_criteo."""
import os
import sys
import time
import faiss
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import deque
from mindspore import context
from mindspore.nn.metrics import Metric
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.deepfm import ModelBuilder
from src.dataset import create_dataset, DataType

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
device_id = get_device_id() # int(os.getenv('DEVICE_ID', '0'))
context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=device_id)


class GetHiddenState(Metric):
    """
    Get hidden state of Reloop2
    """
    def __init__(self):
        super(GetHiddenState, self).__init__()
        self.pred_probs = []
        self.true_labels = []
        self.hidden_states = deque(maxlen=300)

    def clear(self):
        """Clear the internal values."""
        self.pred_probs = []
        self.true_labels = []
        self.hidden_states.clear()

    def update(self, *inputs):
        batch_predict = inputs[1].asnumpy()
        batch_label = inputs[2].asnumpy()
        self.hidden_batch = inputs[3].asnumpy()
        self.pred_probs.extend(batch_predict.flatten().tolist())
        self.true_labels.extend(batch_label.flatten().tolist())
        self.hidden_states.append(self.hidden_batch)

    def eval(self):
        return self.pred_probs, self.true_labels, self.hidden_states

def add_write(file_path, print_str):
    with open(file_path, 'a+', encoding='utf-8') as file_out:
        file_out.write(print_str + '\n')


def modelarts_process():
    pass

def find_test_path(path):
    """
    finding the test files in the path.
    """
    ret = []
    filelist = os.listdir(path)
    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath):
            if filename.startswith("test") and ".db" not in filename:
                ret.append(filepath)
    return ret 

@moxing_wrapper(pre_process=modelarts_process)
def eval_deepfm():
    """ eval_deepfm """
    test_files = find_test_path(config.dataset_path)
    for test_file_index in range(int(len(test_files))):
        print(f"test_file: {test_files[test_file_index]}")
        ds_eval = create_dataset(config.dataset_path, train_mode=False,
                                epochs=1, batch_size=config.batch_size,
                                data_type=DataType(config.data_format), file_index=str(test_file_index))
        if config.convert_dtype:
            config.convert_dtype = config.device_target != "CPU"
        model_builder = ModelBuilder(config, config)
        train_net, eval_net = model_builder.get_train_eval_net()
        train_net.set_train()
        eval_net.set_train(False)
        hs_metric = GetHiddenState()
        model = Model(train_net, eval_network=eval_net, metrics={"hs": hs_metric})
        param_dict = load_checkpoint(config.checkpoint_path)
        load_param_into_net(eval_net, param_dict)
        res = model.eval(ds_eval)
        pred_probs = np.asarray(list(res.values())[0][0])
        true_labels = np.asarray(list(res.values())[0][1])
        hidden_states = np.asarray(list(res.values())[0][2])
        dim = hidden_states.shape[-1]
        hidden_states = np.reshape(hidden_states, (-1, dim)).astype('float32')
        if test_file_index == 0:
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, 100)
            index.train(hidden_states)  
            index.add(hidden_states)
            continue
        _, I = index.search(hidden_states, 2)
        index.add(hidden_states)
        pred_probs = 0.1 * np.mean(true_labels[I], axis=-1) + 0.9 * pred_probs
        auc = roc_auc_score(true_labels, pred_probs)
        out_str = f"test_file: {test_files[test_file_index]}, AUC : {auc}"
        print(out_str)
        add_write('./auc.log', out_str)

if __name__ == '__main__':
    eval_deepfm()
