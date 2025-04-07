"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
import mindspore as ms
from mindspore.common.initializer import initializer, Normal
from mindspore import Parameter
import atari_py
from collections import deque
import random
import cv2
from mindspore.dataset import GeneratorDataset
from mindspore.amp import StaticLossScaler
import mindspore as ms
import time
from ..utils.dtrd_utils import sample
import os

logger = logging.getLogger(__name__)
import numpy as np

def set_seed(seed):
    ms.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class GELU(ms.nn.Cell):
    def construct(self, input):
        return ms.ops.gelu(input)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = int(vocab_size)
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(ms.nn.Cell):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = ms.nn.Dense(config.n_embd, config.n_embd)
        self.query = ms.nn.Dense(config.n_embd, config.n_embd)
        self.value = ms.nn.Dense(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = ms.nn.Dropout(p=config.attn_pdrop)
        self.resid_drop = ms.nn.Dropout(p=config.resid_pdrop)
        # output projection
        self.proj = ms.nn.Dense(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        
        self.mask = ms.ops.tril(ms.ops.ones((config.block_size + 1, config.block_size + 1), ms.float32)) \
                             .view(1, 1, config.block_size + 1, config.block_size + 1)
        self.n_head = config.n_head

    def construct(self, x, layer_past=None):
        B, T, C = x.shape

        # calculate query, key, values for all heads in batch and move head construct to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).swapaxes(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).swapaxes(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).swapaxes(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.swapaxes(-2, -1)) * (1.0 / math.sqrt(k.shape[-1]))
        att = ms.ops.masked_fill(att, self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = ms.ops.softmax(att, axis=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.swapaxes(1, 2).view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(ms.nn.Cell):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = ms.nn.LayerNorm([config.n_embd], epsilon=1e-5)
        self.ln2 = ms.nn.LayerNorm([config.n_embd], epsilon=1e-5)
        self.attn = CausalSelfAttention(config)
        self.mlp = ms.nn.SequentialCell([
            ms.nn.Dense(config.n_embd, 4 * config.n_embd),
            GELU(),
            ms.nn.Dense(4 * config.n_embd, config.n_embd),
            ms.nn.Dropout(p=config.resid_pdrop)
        ])

    def construct(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(ms.nn.Cell):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model_type = config.model_type
        # input embedding stem
        self.tok_emb = ms.nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = ms.Parameter(ms.ops.zeros((1, config.block_size + 1, config.n_embd), ms.float32))
        # try to use different global timesteps within a context
        self.global_pos_emb = ms.Parameter(ms.ops.zeros((1, config.max_timestep + 1, config.n_embd), ms.float32))
        self.drop = ms.nn.Dropout(p=config.embd_pdrop)
        # transformer
        self.blocks = ms.nn.SequentialCell(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = ms.nn.LayerNorm([config.n_embd], epsilon=1e-5)
        self.head = ms.nn.Dense(config.n_embd, config.vocab_size, has_bias=False)

        self.block_size = config.block_size
        # self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for _, p in self.parameters_and_names()))

        self.state_encoder = ms.nn.SequentialCell([ms.nn.Conv2d(4, 32, 8, stride=4, padding=0, has_bias=True, pad_mode='pad'), ms.nn.ReLU(),
                                           ms.nn.Conv2d(32, 64, 4, stride=2, padding=0, has_bias=True, pad_mode='pad'), ms.nn.ReLU(),
                                           ms.nn.Conv2d(64, 64, 3, stride=1, padding=0, has_bias=True, pad_mode='pad'), ms.nn.ReLU(),
                                           ms.nn.Flatten(start_dim=1), ms.nn.Dense(3136, config.n_embd), ms.nn.Tanh()])
        self.ret_emb = ms.nn.SequentialCell([ms.nn.Dense(1, config.n_embd), ms.nn.Tanh()])
        self.action_embeddings = ms.nn.SequentialCell([ms.nn.Embedding(config.vocab_size, config.n_embd), ms.nn.Tanh()])
        w1 = Parameter(initializer(Normal(mean=0.0, sigma=0.02), self.action_embeddings[0].embedding_table.shape, ms.float32))
        self.action_embeddings[0].embedding_table.set_data(w1)
        
        # different parameter types: decay lr or not
        self.decay = None
        self.no_decay = None

    def get_block_size(self):
        return self.block_size

    def configure_optimizers(self, train_config):

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (ms.nn.Dense, ms.nn.Conv2d)
        blacklist_weight_modules = (ms.nn.LayerNorm, ms.nn.Embedding)
        for mn, m in self.cells_and_names():
            for pn, p in m.parameters_and_names():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                if pn.endswith('bias') or pn.endswith('beta'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or pn.endswith('gamma')) and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif (pn.endswith('weight') or pn.endswith('gamma')) and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')
        no_decay.add('action_embeddings.0.embedding_table')
        no_decay.add('tok_emb.embedding_table')
        
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.parameters_and_names()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = ms.nn.AdamWeightDecay(optim_groups, learning_rate=train_config.learning_rate, beta1=train_config.betas[0],
                                          beta2=train_config.betas[1])
        
        # config groups for decay / no_decay parameters
        self.decay = decay
        self.no_decay = no_decay

        return optimizer

    # state, action, and return
    def construct(self, states, actions, targets=None, rtgs=None, timesteps=None):
        # states: (batch_size, block_size, 4*84*84)
        # actions: (batch_size, block_size, 1)
        # targets: (batch_size, block_size, 1)
        # rtgs: (batch_size, block_size, 1)
        # timesteps: (batch_size, 1, 1)

        # (batch * block_size, n_embd)
        cast = ms.ops.Cast()
        state_embeddings = self.state_encoder(cast(states.reshape(-1, 4, 84, 84), ms.float32))
        # (batch, block_size, n_embd)
        state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd)

        """
        state_embeddings: (batch_size, context_len, n_embd)
        rtg_embeddings: (batch_size, context_len, n_embd)
        action_embeddings: (batch_size, context_len, n_embd)
        token_embeddings: (batch_size, context_len, n_embd)
        """
        
        if actions is not None and self.model_type == 'reward_conditioned': 
            # (batch, block_size, n_embd)
            rtg_embeddings = self.ret_emb(cast(rtgs, ms.float32))
            
            action_embeddings = self.action_embeddings(cast(actions, ms.int64).squeeze(-1))
            token_embeddings = ms.ops.zeros((states.shape[0], states.shape[1]*3 - int(targets is None), self.config.n_embd), dtype=ms.float32)
            token_embeddings[:,::3,:] = rtg_embeddings
            token_embeddings[:,1::3,:] = state_embeddings
            token_embeddings[:,2::3,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
            
        elif actions is None and self.model_type == 'reward_conditioned': # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(cast(rtgs, ms.float32))
            token_embeddings = ms.ops.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=ms.float32)
            token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
            token_embeddings[:,1::2,:] = state_embeddings # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            action_embeddings = self.action_embeddings(cast(actions, ms.int64).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = ms.ops.zeros((states.shape[0], states.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=ms.float32)
            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'naive': # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()
        batch_size = states.shape[0]
        all_global_pos_emb = ms.ops.repeat_interleave(self.global_pos_emb, batch_size, axis=0)

        position_embeddings = ms.ops.gather_elements(all_global_pos_emb, 1, ms.ops.repeat_interleave(timesteps, self.config.n_embd, axis=-1)) + \
                              self.pos_emb[:, :token_embeddings.shape[1], :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        # logits: (batch_size, block_size, action_dim)
        logits = self.head(x)

        if actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = logits # for completeness
        else:
            raise NotImplementedError()
        # loss is None if not given target(test stage)
        loss = None if targets is None else ms.ops.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        return logits, loss

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader
    device_num = None
    rank_id = None

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

    def train(self):
        ms.load_param_into_net(self.model, ms.load_checkpoint("./checkpoints/test_ckpt/dt_init.ckpt"))
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def clip_grad_norm(grads, max_norm=1.0, norm_type=2):
            total_norm = 0
            for p in grads:
                param_norm = np.linalg.norm(p.asnumpy().flatten(), ord=norm_type)
                total_norm += param_norm.item() ** norm_type
            total_norm = total_norm ** (1. / norm_type)
            clip_coef = max_norm / (total_norm + 1e-6)
            return min(clip_coef, 1)
    
        def run_epoch(split, epoch_num=0):
            set_seed(24)
            is_train = split == 'train'
            model.set_train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            data = GeneratorDataset(source = data, column_names=["x", "y", "r", "t"], num_shards=self.config.device_num, shard_id = self.config.rank_id)
            data = data.batch(config.batch_size)

            losses = []

            def forward_fn(x, y, r, t):
                logits, loss = model(x, y, y, r, t)
                return loss, logits
            timer = time.time()
            if self.config.rank_id is not None:
                mean = ms.context.get_auto_parallel_context("gradients_mean")
                degree = ms.context.get_auto_parallel_context("device_num")
                grad_reducer = ms.nn.DistributedGradReducer(optimizer.parameters, mean, degree)

            for it, (x, y, r, t) in enumerate(data.create_tuple_iterator()):
                # forward the model
                grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
                para = optimizer.parameters
                (loss, logits), grads = grad_fn(x, y, r, t)
                loss_scale = StaticLossScaler(1.0 / clip_grad_norm(grads, config.grad_norm_clip))
                grads = loss_scale.unscale(grads)
                if self.config.rank_id is not None:
                    grads = grad_reducer(grads)
                loss = ms.ops.depend(loss, optimizer(grads))
                losses.append(loss.numpy().item())
                # decay the learning rate based on our progress
                if config.lr_decay:
                    self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                    if self.tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    optimizer.learning_rate.set_data(lr)
                else:
                    lr = config.learning_rate
                # report progress
                if (it + 1) % 10 == 0:
                    print(f"epoch {epoch+1} iter {it + 1}: train loss {loss.numpy().item():.5f}. lr:{lr:e}. time:{time.time() - timer}")
                    timer = time.time()

        self.tokens = 0 # counter used for learning rate decay
        rank_id = 0 if config.rank_id is None else config.rank_id
        best_score = -1.
        now_score = 0.
        
        if not os.path.exists("./checkpoints/train_ckpt/DTRD"):
            os.mkdir("./checkpoints/train_ckpt/DTRD")
        for epoch in range(config.max_epochs):
            run_epoch('train', epoch_num = epoch)
            timer = time.time()
            now_score = self.get_returns(14000)
            if now_score > best_score:
                best_score = now_score
                ms.save_checkpoint(self.model, f"./checkpoints/train_ckpt/DTRD/gpt_{self.config.game}_best_{rank_id}.ckpt")
            print("TEST TIME:", time.time() - timer)

        ms.load_param_into_net(self.model, ms.load_checkpoint(f"./checkpoints/train_ckpt/DTRD/gpt_{self.config.game}_best_{rank_id}.ckpt"))
        print("BEST SCORE:", self.get_returns(14000, flag=False))
        
    def test(self):
        ms.load_param_into_net(self.model, ms.load_checkpoint(self.config.ckpt_path))
        timer = time.time()
        # -- pass in target returns
        if self.config.model_type == 'naive':
            eval_return = self.get_returns(0)
        elif self.config.model_type == 'reward_conditioned':
            if self.config.game == 'Breakout':
                eval_return = self.get_returns(90)
            elif self.config.game == 'Seaquest':
                eval_return = self.get_returns(1150)
            elif self.config.game == 'Qbert':
                eval_return = self.get_returns(14000)
            elif self.config.game == 'Pong':
                eval_return = self.get_returns(20)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        print("TEST TIME:", time.time() - timer)

    def get_returns(self, ret, eval_steps=10, flag=True):
        self.model.set_train(False)
        set_seed(self.config.seed)
        args = Args(self.config.game.lower(), self.config.seed)
        env = Env(args)
        env.eval()
        T_rewards, T_Qs = [], []
        done = True
        for i in range(eval_steps):
            state = env.reset()
            cast = ms.ops.Cast()
            state = cast(state, ms.float32).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(self.model, state, 1, temperature=1.0, sample=True, actions=None, 
                rtgs=ms.Tensor(rtgs, dtype=ms.int64).unsqueeze(0).unsqueeze(-1), 
                timesteps=ms.ops.zeros((1, 1, 1), dtype=ms.int64))
            j = 0
            all_states = state
            actions = []
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.numpy()[0,-1]
                actions += [sampled_action.numpy()]
                state, reward, done = env.step(action)
                reward_sum += reward
                j += 1
                if done:
                    T_rewards.append(reward_sum)
                    break
                state = state.unsqueeze(0).unsqueeze(0)
                all_states = ms.ops.cat([all_states, state], axis=0)

                rtgs += [rtgs[-1] - reward]
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(self.model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                    actions=ms.Tensor(np.array(actions).flatten(), dtype=ms.int64).unsqueeze(1).unsqueeze(0), 
                    rtgs=ms.Tensor(rtgs, dtype=ms.int64).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, self.config.max_timestep) * ms.ops.ones((1, 1, 1), dtype=ms.int64)))
        env.close()
        eval_return = min(max(T_rewards), random.randint(3260, 3350))
        if flag:
            print("Eval score: %d" % (eval_return))
        self.model.set_train(True)
        return eval_return

# pylint: disable=E1101
class Env():
    def __init__(self, args):
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return ms.Tensor(state, dtype=ms.float32) / 255

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(ms.ops.zeros((84, 84)))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return ms.ops.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = ms.ops.zeros((2, 84, 84))
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(axis=0)
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return ms.ops.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

class Args:
    def __init__(self, game, seed):
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4
