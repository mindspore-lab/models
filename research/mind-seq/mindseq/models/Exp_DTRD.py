from .Exp_basic import Exp_Basic
import numpy as np
import logging
from .DTRD import GPT, GPTConfig
from .DTRD import Trainer, TrainerConfig

class StateActionReturnDataset:
    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = np.array(self.data[idx:done_idx]).reshape(block_size, -1)
        states = states / 255.
        actions = np.expand_dims(self.actions[idx:done_idx], axis=1)
        rtgs = np.expand_dims(self.rtgs[idx:done_idx], axis=1)
        timesteps = np.expand_dims(self.timesteps[idx:idx + 1], axis=1)

        return states, actions, rtgs, timesteps

class Exp_DTRD(Exp_Basic):
    def __init__(self, args):
        args.game = args.data
        super(Exp_DTRD, self).__init__(args)

    def _build_model(self):
        train_data = np.load(self.args.data_dir + self.args.game + '/train_set.npz')
        obss, actions, returns, done_idxs, rtgs, timesteps = \
            train_data['obss'].astype(np.uint8), train_data['actions'].astype(np.int32), \
            train_data['returns'].astype(np.float64), train_data['done_idxs'].astype(np.int64), \
            train_data['rtgs'].astype(np.float32), train_data['timesteps'].astype(np.int64)
        # set up logging
        logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO,
        )
        train_dataset = StateActionReturnDataset(obss, self.args.context_length*3, actions, done_idxs, rtgs, timesteps)
        mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=6, n_head=8, n_embd=128, model_type=self.args.model_type, max_timestep=max(timesteps))
        model = GPT(mconf)
        epochs = self.args.epochs
        tconf = TrainerConfig(max_epochs=epochs, batch_size=self.args.batch_size, learning_rate=6e-4,
                            lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*self.args.context_length*3,
                            num_workers=4, seed=self.args.seed, model_type=self.args.model_type, game=self.args.game, max_timestep=max(timesteps),
                            ckpt_path=self.args.ckpt_path, device_num=self.args.device_num, rank_id=self.args.rank_id)
        self.trainer = Trainer(model, train_dataset, None, tconf)
        return model
    
    def train(self):
        self.trainer.train()

    def test(self):
        self.trainer.test()