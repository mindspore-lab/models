import os
import argparse
import importlib
import numpy as np
import mindspore
from mindspore import ops, context, nn, load_checkpoint, load_param_into_net
from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.common import set_seed
from mindnlp.models import BertConfig, CLIPVisionConfig, BertModel, CLIPModel, CLIPConfig
from src.models import TransformerLitModel, UnimoForMaskedLM
from src.data import KGC
from src.utils import LearningRate, AnalogyMetric

def load_model(args):
    # load pretrained visual and textual configs, models
    vision_config = CLIPVisionConfig.from_json_file(os.path.join(args.visual_model_path, 'vision_config.json'))
    text_config = BertConfig.from_pretrained(os.path.join(args.model_name_or_path, 'config.json'))
    vision_config.device = 'cpu'
    model = UnimoForMaskedLM(vision_config, text_config)

    bert = BertModel.from_pretrained(args.model_name_or_path)
    text_model_dict = bert.parameters_dict()
    clip_model_dict = load_checkpoint(os.path.join(args.visual_model_path, 'mindspore.ckpt'))

    def load_state_dict():
        """Load bert and vit pretrained weights"""
        vision_names, text_names = [], []
        model_dict = model.parameters_dict()
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '').replace('unimo.', 'vision_model.')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '').replace('unimo.', '')
                if text_name in text_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = text_model_dict[text_name]
        load_param_into_net(model, model_dict)
        print('Load model state dict successful.')
    load_state_dict()
    return model

def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Basic arguments
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--chunk", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--pretrain_path", type=str, default=None)
    parser.add_argument("--entity_img_path", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--visual_model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default='CPU')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--overwrite_cache", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.4, help="the weight of similarity loss")
    parser.add_argument("--only_test", action="store_true", default=False)
    return parser

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    set_seed(args.seed)

    if args.device == 'CPU':
        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    elif args.device == 'Ascend':
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    elif args.device == 'GPU':
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    data = KGC(args)
    tokenizer = data.tokenizer
    dataset = data.setup()

    model = load_model(args)
    model = TransformerLitModel(args=args, model=model, tokenizer=tokenizer, data_config=data.get_config())

    model._init_relation_word()
    if args.checkpoint:
        load_param_into_net(model, mindspore.load_checkpoint(args.checkpoint), strict_load=False)

    epoch_num = args.epochs
    step_per_epoch = dataset['train'].get_dataset_size()

    lr = LearningRate(learning_rate=args.lr,
                      end_learning_rate=0.0,
                      warmup_steps=int(args.warmup_ratio*epoch_num*step_per_epoch),
                      decay_steps=epoch_num*step_per_epoch)

    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    params = model.trainable_params()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{'params': decay_params, 'weight_decay': 1e-2},
                    {'params': other_params, 'weight_decay': 0.0},
                    
                    
                    
                    
                      
                    {'order_params': params}]

    optimizer = nn.AdamWeightDecay(group_params, learning_rate=lr)

    callback_size = 1
    actual_epoch_num = int(epoch_num * step_per_epoch / callback_size)
    callback = [TimeMonitor(callback_size), LossMonitor(callback_size)]

    config_ck = CheckpointConfig(save_checkpoint_steps=step_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix="mkg_analogy", config=config_ck)
    callback.append(ckpoint_cb)

    model = Model(model, eval_network=model, optimizer=optimizer, metrics={'analogy_metric': AnalogyMetric()})
    model.train(actual_epoch_num, dataset['train'], callbacks=callback)
    
    model._eval_network.construct = model._eval_network._eval
    eval_results = model.eval(dataset['test'])
    print(eval_results)


if __name__ == "__main__":
    main()
