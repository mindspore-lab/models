import warnings
import os
import sys
import logging
import numpy as np
from tqdm import tqdm
import hydra
from hydra import utils
import mindspore
from mindspore import ops, load_param_into_net, context, nn
from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.common import set_seed
from mindformers import AutoConfig, T5Tokenizer
from src.seq2seq.trainer_arguments import ModelArguments, DataTrainingArguments, PromptArguments
from src.seq2seq.argument_parser import MsArgumentParser
from src.seq2seq.models import T5Prompt
from src.extraction import constants
from src.extraction.record_schema import RecordSchema
from src.extraction.extraction_metrics import get_extract_metrics
from src.extraction.dataset_processer import PrefixGenerator
from src.seq2seq.constrained_seq2seq import ConstraintSeq2SeqTrainingArguments
from src.utils import LearningRate
from dataset import create_data

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

def postprocess_text(x_str, to_remove_token_list):
    # Clean `bos` `eos` `pad` for cleaned text
    for to_remove_token in to_remove_token_list:
        x_str = x_str.replace(to_remove_token, '')
    return x_str.strip()

def compute_metrics(eval_preds, tokenizer, record_schema, to_remove_token_list, data_args):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False,
                                           clean_up_tokenization_spaces=False)
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False)

    decoded_preds = [postprocess_text(x, to_remove_token_list) for x in decoded_preds]
    decoded_labels = [postprocess_text(x, to_remove_token_list) for x in decoded_labels]

    result = get_extract_metrics(
        pred_lns=decoded_preds,
        tgt_lns=decoded_labels,
        label_constraint=record_schema,
        decoding_format=data_args.decoding_format,
    )

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def prediction_step(model, inputs, constraint_decoder=None):
    def prefix_allowed_tokens_fn(batch_id, sent):
        src_sentence = inputs['input_ids'][batch_id]
        return constraint_decoder.constraint_decoding(src_sentence=src_sentence,
                                                            tgt_generated=sent)

    gen_kwargs = {
        "max_length": model.config.max_length,
        "num_beams": 1,
        "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn if constraint_decoder else None,
    }

    generated_tokens = model.generate(**inputs, **gen_kwargs)
    return generated_tokens

def prediction(model, dataset, constraint_decoder=None,
               tokenizer=None, record_schema=None, to_remove_token_list=None, data_args=None):
    preds, labels = [], []
    i = 0
    for batch in tqdm(dataset, total=len(dataset), desc='Predict'):
        inputs = {'input_ids': batch[0].tolist(), 'attention_mask': batch[1].tolist()}
        generated_tokens = prediction_step(model, inputs, constraint_decoder)
        labels.extend(batch[2].tolist())
        preds.extend(generated_tokens)
        i += 1
        if i > 5:
            break

    labels = np.array(labels)
    result = compute_metrics([preds, labels], tokenizer, record_schema,
                             to_remove_token_list, data_args)
    return result, preds

def get_model(model_args, prompt_args):
    model = T5Prompt(
        model_args.model_name_or_path,
        prompt_args
    )
    if model_args.model_ckpt_path:
        load_param_into_net(model, mindspore.load_checkpoint(os.path.join(
                                model_args.model_ckpt_path,'pytorch_model.ckpt')),
                                strict_load=False)
    elif prompt_args.source_prefix_path and prompt_args.target_prefix_path:
        model_dict = model.parameters_dict()
        source_model_dict = mindspore.load_checkpoint(os.path.join(
                                        prompt_args.source_prefix_path,'pytorch_model.ckpt'))
        target_model_dict = mindspore.load_checkpoint(os.path.join(
                                        prompt_args.target_prefix_path, 'pytorch_model.ckpt'))
        for key in model_dict:
            if 't5' not in key:
                if prompt_args.prefix_fusion_way == 'add':
                    # 1. add source and target prefix
                    model_dict[key] = (source_model_dict[key] + target_model_dict[key]) / 2
                elif 'concat' in prompt_args.prefix_fusion_way:
                    # 2. concatenate source and target prefix -- in model forward
                    if 'source' in key:
                        model_dict[key] = source_model_dict[key.replace('source_', '')]
                    elif 'lstm' not in key:
                        model_dict[key] = target_model_dict[key]
            else:
                if 'adapter' not in key:    # adapter is just in transfer models
                    model_dict[key] = target_model_dict[key]
        load_param_into_net(model, model_dict)
    return model

def get_tokenizer(model_args, training_args):
    # tokenizer
    tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name \
                                                else model_args.model_name_or_path
    tokenizer = T5Tokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    to_remove_token_list = list()
    if tokenizer.bos_token:
        to_remove_token_list += [tokenizer.bos_token]
    if tokenizer.eos_token:
        to_remove_token_list += [tokenizer.eos_token]
    if tokenizer.pad_token:
        to_remove_token_list += [tokenizer.pad_token]

    # if training_args.do_train:
    #     to_add_special_token = list()
    #     for special_token in [constants.type_start, constants.type_end, constants.text_start,
    #                           constants.span_start, constants.spot_prompt, constants.asoc_prompt]:
    #         if special_token not in tokenizer.get_vocab():
    #             to_add_special_token += [special_token]

    #     tokenizer.add_special_tokens(
    #         {"additional_special_tokens":
    #          tokenizer.special_tokens_map_extended['additional_special_tokens']
    #          + to_add_special_token}
    #     )

    logger.info(tokenizer)

    return tokenizer, to_remove_token_list

def get_config(model_args, prompt_args, data_args):
    # config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )

    config.use_adapter = prompt_args.use_adapter
    config.max_length = data_args.max_target_length
    return config

def get_dataset(data_args, training_args, tokenizer):
    if data_args.record_schema and os.path.exists(data_args.record_schema):
        record_schema = RecordSchema.read_from_file(data_args.record_schema)
    else:
        record_schema = None

    if data_args.source_prefix is not None:
        if data_args.source_prefix == 'schema':
            prefix = PrefixGenerator.get_schema_prefix(schema=record_schema)
        elif data_args.source_prefix.startswith('meta'):
            prefix = ""
        else:
            prefix = data_args.source_prefix
    else:
        prefix = ""
    logger.info(f"Prefix: {prefix}")
    logger.info(f"Prefix Length: {len(tokenizer.tokenize(prefix))}")

    padding = "max_length" if data_args.pad_to_max_length else False
    dataset = create_data(data_args, tokenizer, prefix, padding, record_schema=record_schema,
                          constants=constants,
                          batch_size=training_args.per_device_train_batch_size)

    return dataset, record_schema

@hydra.main(config_path="conf", config_name='config')
def main(cfg):
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd

    parser = MsArgumentParser((ModelArguments, DataTrainingArguments,
                                ConstraintSeq2SeqTrainingArguments, PromptArguments))
    model_args, data_args, training_args, prompt_args = parser.parse_dict(cfg,
                                                                          allow_extra_keys=True)

    if model_args.device == 'CPU':
        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    elif model_args.device == 'Ascend':
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    elif model_args.device == 'GPU':
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    logger.info("Options:")
    logger.info(model_args)
    logger.info(data_args)
    logger.info(training_args)
    logger.info(prompt_args)

    tokenizer, to_remove_token_list = get_tokenizer(model_args, training_args)

    # model
    model = get_model(model_args, prompt_args)

    dataset, record_schema = get_dataset(data_args, training_args, tokenizer)

    epoch_num = training_args.num_train_epochs
    step_per_epoch = dataset['train'].get_dataset_size()

    lr = LearningRate(learning_rate=training_args.learning_rate,
                      end_learning_rate=0.0,
                      warmup_steps=int(training_args.warmup_ratio*epoch_num*step_per_epoch),
                      decay_steps=epoch_num*step_per_epoch)

    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    params = model.trainable_params()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{'params': decay_params, 'weight_decay': 1e-2},
                    {'params': other_params, 'weight_decay': 0.0},
                    {'order_params': params}]

    optimizer = nn.AdamWeightDecay(group_params, learning_rate=lr)

    callback_size = 10
    actual_epoch_num = int(epoch_num * step_per_epoch/callback_size)
    callback = [TimeMonitor(callback_size), LossMonitor(callback_size)]

    config_ck = CheckpointConfig(save_checkpoint_steps=step_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix="CP-NER", config=config_ck)
    callback.append(ckpoint_cb)

    if prompt_args.save_prefix:
        past_prompt = model.get_prompt(bsz=1)
        mindspore.save_mindir(past_prompt, os.path.join(training_args.output_dir, 'prefix.mindir'))

    if prompt_args.save_label_word:
        label_word = record_schema.type_list
        label_word_id = [tokenizer.encode(label, add_special_tokens=False) for label in label_word]
        label2embed = {}
        for label, label_id in zip(label_word, label_word_id):
            if len(label_id) > 1:
                label2embed[label] = ops.stack([model.t5.shared.embedding_table.data[id].cpu()
                                                for id in label_id], axis=0).mean(0)
            else:
                label2embed[label] = \
                    model.t5.shared.embedding_table.data[label_id[0]].cpu()
        mindspore.save_mindir(label2embed, os.path.join(
                            training_args.output_dir, 'label_word.mindir'))

    if prompt_args.save_prefix or prompt_args.save_label_word:
        return

    if prompt_args.multi_source_path:
        model.normalize_multi_source(record_schema.type_list, tokenizer, training_args.device)

    if training_args.do_train:
        model = Model(model, optimizer=optimizer)
        model.train(actual_epoch_num, dataset['train'], callbacks=callback)

    if training_args.do_predict:
        logger.info("*** Test ***")
        test_metrics, preds = prediction(model, dataset['validation'], None,
                   tokenizer, record_schema, to_remove_token_list, data_args)

        print(test_metrics)
        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)

        output_test_result_file = os.path.join(training_args.output_dir,
                                               "test_results_seq2seq.txt")
        with open(output_test_result_file, "w") as writer:
            logger.info("***** Test results *****")
            for key, value in sorted(test_metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        if training_args.predict_with_generate:
            test_preds = tokenizer.batch_decode(
                preds, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            test_preds = [postprocess_text(pred, to_remove_token_list) for pred in test_preds]
            output_test_preds_file = os.path.join(training_args.output_dir,
                                                  "test_preds_seq2seq.txt")
            with open(output_test_preds_file, "w") as writer:
                writer.write("\n".join(test_preds))

if __name__ == "__main__":
    set_seed(12315)
    main()
