
from datasets import load_dataset as hf_load
from datasets import Dataset
import mindspore.dataset.transforms as C
import mindspore.common.dtype as mstype
from mindspore.dataset import GeneratorDataset
from src.seq2seq.data_collator import DynamicSSIGenerator
from src.extraction.spot_asoc_noiser import SpotAsocNoiser
from src.extraction.utils import convert_to_record_function
from src.extraction.constants import BaseStructureMarker

class TransferDataset:
    """TransferDataset for Huggingface Dataset."""
    def __init__(self, arrow_ds, column_names):
        self.ds = arrow_ds
        self.column_names = column_names

    def __getitem__(self, index):
        return tuple(self.ds[int(index)][name] for name in self.column_names)

    def __len__(self):
        return self.ds.dataset_size

class TransferIterableDataset:
    """TransferIterableDataset for Huggingface IterableDataset."""
    def __init__(self, arrow_ds, column_names):
        self.ds = arrow_ds
        self.column_names = column_names

    def __iter__(self):
        for data in self.ds:
            yield tuple(data[name] for name in self.column_names)

def create_data(data_args, tokenizer, prefix, padding, record_schema, constants, batch_size=1):
    text_column = data_args.text_column
    record_column = data_args.record_column
    max_target_length = data_args.max_target_length

    def collator_features(examples):
        def sample_ssi(feature):
            # Sample SSI
            _, positive_spot, negative_spot = negative_sampler.sample_spot(
                                                    positive=feature.get('spots', []))
            converted_asoc_prefix, negative_asoc = negative_sampler.sample_asoc(
                                                    positive=feature.get('asocs', []))

            # Dynamic generating spot-asoc during training
            if 'spot_asoc' in feature:

                # Deleted positive example Spot in Target that was not sampled by Prefix
                feature['spot_asoc'] = [spot_asoc for spot_asoc in feature['spot_asoc']
                                            if spot_asoc["label"] in positive_spot]

                # Inject rejection noise
                if spot_asoc_nosier is not None:
                    if isinstance(spot_asoc_nosier, SpotAsocNoiser):
                        feature['spot_asoc'] = spot_asoc_nosier.add_noise(
                            feature['spot_asoc'],
                            spot_label_list=negative_spot,
                            asoc_label_list=negative_asoc,
                        )
                    else:
                        raise NotImplementedError(f'{spot_asoc_nosier} is not implemented.')

                # Generate new record
                record = convert_to_record_function[data_args.decoding_format](
                    feature['spot_asoc'],
                    structure_maker=BaseStructureMarker()
                )
                feature["labels"] = tokenizer.encode(record)
            return feature, converted_asoc_prefix

        def extract_feature(examples):
            for feature in examples:
                sample_prompt = feature['sample_prompt']

                if not sample_prompt:
                    # Evaluation using Ordered SSI
                    _ = negative_sampler.full_spot(shuffle=False)
                    converted_asoc_prefix = negative_sampler.full_asoc(shuffle=False)
                else:
                    feature, converted_asoc_prefix = sample_ssi(feature)
                if 'sample_prompt' in feature:
                    feature.pop('sample_prompt')
                if 'spot_asoc' in feature:
                    feature.pop('spot_asoc')
                if 'spots' in feature:
                    feature.pop('spots')
                if 'asocs' in feature:
                    feature.pop('asocs')
                # prefix = converted_spot_prefix + converted_asoc_prefix
                prefix = converted_asoc_prefix
                # truncate `prefix` to max length
                if data_args.max_prefix_length is not None and data_args.max_prefix_length >= 0:
                    prefix = prefix[:data_args.max_prefix_length]

                feature['input_ids'] = prefix + [negative_sampler.text_start] + \
                                        feature['input_ids']

                # truncate `input_ids` to max length
                if data_args.max_source_length:
                    feature['input_ids'] = feature['input_ids'][:data_args.max_source_length]
                if data_args.max_target_length and 'labels' in feature:
                    feature['labels'] = feature['labels'][:data_args.max_target_length]

                feature['attention_mask'] = [1] * len(feature['input_ids'])

            labels = [feature["labels"] for feature in examples] \
                                if "labels" in examples[0].keys() else None
            # We have to pad the labels before calling `tokenizer.pad` as this method won't
            # pad them and needs them of the same length to return tensors.
            if labels is not None:
                max_label_length = max(len(_label) for _label in labels)
                padding_side = "left"
                for feature in examples:
                    remainder = [label_pad_token_id] * (max_label_length - len(feature["labels"]))
                    feature["labels"] = (feature["labels"] + remainder if padding_side == "right" \
                                         else remainder + feature["labels"])

            # examples = tokenizer.pad(
            #     examples,
            #     padding='max_length',
            #     max_length=data_args.max_source_length,
            #     return_tensors="np"
            # )
            examples = {key: [feature[key] for feature in examples] for key in examples[0]}

            return examples

        negative_sampler = DynamicSSIGenerator(
                tokenizer=tokenizer,
                schema=record_schema,
                positive_rate=data_args.meta_positive_rate,
                negative=data_args.meta_negative,
                ordered_prompt=data_args.ordered_prompt,
        )
        spot_asoc_nosier = SpotAsocNoiser(
            spot_noise_ratio=data_args.spot_noise,
            asoc_noise_ratio=data_args.asoc_noise,
            null_span=constants.null_span,
        )
        label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else \
                                tokenizer.pad_token_id
        examples = [{key: examples[key][i] for key in examples.keys()}
                            for i in range(len(examples['input_ids']))]

        return extract_feature(examples)

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[record_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer([item for item in inputs if len(item) > 0],
                                 max_length=data_args.max_source_length,
                                 padding='max_length',
                                 truncation=True)

        # Setup the tokenizer for targets
        # with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding='max_length',
                            truncation=True)

        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(_label if _label != tokenizer.pad_token_id else -100) \
                        for _label in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        model_inputs['sample_prompt'] = [False] * len(model_inputs['input_ids'])
        if data_args.source_prefix is not None and data_args.source_prefix.startswith('meta'):
            model_inputs['spots'] = examples['spot']
            model_inputs['asocs'] = examples['asoc']
            model_inputs['spot_asoc'] = examples['spot_asoc']
            # sample_prompt=True for Finetune and Pretrain
            model_inputs['sample_prompt'] = [True] * len(model_inputs['input_ids'])
        return collator_features(model_inputs)

    ds_ret = hf_load(data_args.data_file)
    

    datasets_dict = {}
    for key, raw_ds in ds_ret.items():
        raw_ds = raw_ds.map(preprocess_function, batched=True, num_proc=1,
                            remove_columns=raw_ds.column_names)
        column_names = list(raw_ds.features.keys())
        if isinstance(raw_ds, Dataset):
            source = TransferDataset(raw_ds, column_names)
        else:
            source = TransferIterableDataset(raw_ds, column_names)
        ms_ds = GeneratorDataset(
            source=source,
            column_names=column_names,
            shuffle=False,
            num_parallel_workers=1)

        type_cast_op = C.TypeCast(mstype.int32)
        ms_ds = ms_ds.map(input_columns="input_ids", operations=type_cast_op)
        ms_ds = ms_ds.map(input_columns="attention_mask", operations=type_cast_op)
        ms_ds = ms_ds.map(input_columns="labels", operations=type_cast_op)
        ms_ds = ms_ds.batch(batch_size, drop_remainder=True)

        datasets_dict[key] = ms_ds

    if len(datasets_dict) == 1:
        return datasets_dict.popitem()[1]
    return datasets_dict
