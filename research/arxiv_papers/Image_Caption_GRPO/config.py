import os, sys
from mindnlp.transformers import AutoTokenizer

class Config():
    def __init__(self, TrainOrVal = 'train'):
        self.seed = 0
        self.lr = 4e-4
        self.rl_lr = 1e-5
        self.weight_decay = 1e-4
        self.epoch = 20

        self.decode_method = 'greedy'
        self.beam_size = 3

        self.PreTrainedModel = ['bert-base-uncased', 'resnet50_224_new.ckpt']

        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(sys.path[0], 'PreTrainedModel', self.PreTrainedModel[0]))
        self.resnet_model = os.path.join(sys.path[0], 'PreTrainedModel', self.PreTrainedModel[1])

        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.sep_token_id

        self.batch_size = 32
        self.img_length = 196
        self.max_length = 25
        self.generation_length = 20
        self.dropout = 0.1
        self.head_nums = 8
        self.decoder_layer_nums = 1
        self.image_embedding_dim = 2048
        self.hidden_dim = 512
        self.sample_top_k = 10000

        self.sentence_nums = 5

        self.dataset = 'coco'
        self.TrainOrVal = TrainOrVal
        self.info_json = os.path.join(sys.path[0], 'data/{}/dataset_{}.json'.format(self.dataset, self.dataset))
        self.image_dir = os.path.join(sys.path[0], 'data/{}/img/'.format(self.dataset))
        self.image_name = os.path.join(sys.path[0], 'data/{}/{}_{}.txt'.format(self.dataset, self.dataset, self.TrainOrVal))

        self.model_save_path = os.path.join(sys.path[0], 'model_save/train_{}'.format(self.dataset))
        self.ck = 'epoch_0.ckpt'

        self.grpo_lr = 1e-5
        self.sample_nums = 5
        self.grpo_all_epoch = 5
        self.grpo_epoch = 2
        self.grpo_batch_size = 8
        self.policy_clip_eps = 0.2
        self.beta = 0.04
        self.grpo_step = 20
        self.grpo_save_frequency = 4