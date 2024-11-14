#!/usr/bin/env python
# -*- coding:utf-8 -*-
import mindspore
from dataclasses import dataclass, field
from typing import Union, List, Dict, Tuple, Any, Optional

from mindformers import TrainingArguments, Trainer
from .constraint_decoder import get_constraint_decoder


@dataclass
class ConstraintSeq2SeqTrainingArguments(TrainingArguments):
    """
    Parameters:
        constraint_decoding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use Constraint Decoding
        structure_weight (:obj:`float`, `optional`, defaults to :obj:`None`):
    """
    constraint_decoding: bool = field(default=False,
                metadata={"help": "Whether to Constraint Decoding or not."})
    save_better_checkpoint: bool = field(default=False,
                metadata={"help": "Whether to save better metric checkpoint"})
    start_eval_step: int = field(default=0, metadata={"help": "Start Evaluation after Eval Step"})
    do_train: bool = field(
        default=False, metadata={"help": "Whether to eval current model while Training. "}
    )
    do_predict: bool = field(
        default=False, metadata={"help": "Whether to eval current model while Training. "}
    )
    warmup_ratio: float = field(default=0.0, metadata={"help": "warmup_ratio."})
    predict_with_generate: bool = field(
        default=False,
        metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
