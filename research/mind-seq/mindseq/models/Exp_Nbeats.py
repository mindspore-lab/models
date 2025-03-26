from .Exp_basic import Exp_Basic
from ..data.data_loader import Dataset_M3C
from .nbeats import NBeatsNet
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.transforms as transforms
import mindspore as ms

def get_data(args, flag):
        data_dict = {
            'M3C':Dataset_M3C,
        }
        Data = data_dict[args.data]
        source_data = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                freq=args.freq,
                flag=flag,
                seq_len=args.seq_len,
                pred_len=args.pred_len
            )
        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size
        if flag == 'test':
            data_set = GeneratorDataset(source=source_data, column_names=["seq_x", "seq_y"])
        else:
            data_set = GeneratorDataset(source=source_data, column_names=["seq_x", "seq_y"], num_shards=args.device_num, shard_id=args.rank_id)
        data_set = data_set.map(operations=transforms.TypeCast(ms.float32), input_columns="seq_x")
        data_set = data_set.map(operations=transforms.TypeCast(ms.float32), input_columns="seq_y")
        print(flag, data_set.get_dataset_size())
        if shuffle_flag:
            data_set = data_set.shuffle(data_set.get_dataset_size())
        data_set = data_set.batch(batch_size=batch_size, drop_remainder=drop_last) 
        return data_set

class Exp_Nbeats(Exp_Basic):
    def __init__(self, args):
        super(Exp_Nbeats, self).__init__(args)

    def _build_model(self):
        model = NBeatsNet(
            backcast_length=self.args.seq_len, forecast_length=self.args.pred_len,
            stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
            nb_blocks_per_stack=2, thetas_dim=(4, 4), share_weights_in_stack=True,
            hidden_layer_units=64
        )
        model.compile(loss='smape', optimizer='adam')
        return model
    
    def train(self):
        train_data = get_data(self.args, 'train')
        test_data = get_data(self.args, 'test')
        rank_id = 0 if self.args.rank_id is None else self.args.rank_id
        self.model.fit(
            train_data=train_data, test_data=test_data,
            epochs=self.args.train_epochs,
            sv_name=f"Nbeats_M3C_Year_{self.args.seq_len}_{self.args.pred_len}_{rank_id}", reducer_flag=self.args.distribute
        )

    def test(self):
        test_data = get_data(self.args, 'test')
        self.model.test(test_data, ckpt_path=self.args.ckpt_path)