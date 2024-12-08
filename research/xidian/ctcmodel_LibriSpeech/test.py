from src.model_utils.config import config
from src.model_utils.device_adapter import get_rank_id, get_device_num, get_device_id
from src.dataset import create_dataset
from ipdb import set_trace

device_num=get_device_num()
batch_size=64
rank=0
train_path = "./min_dataset/train.mindrecord0"
ds_train = create_dataset(train_path, True, batch_size, num_shards=device_num, shard_id=rank)
# set_trace()