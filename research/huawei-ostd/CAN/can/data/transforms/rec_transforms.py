"""
transform for text recognition tasks.
"""
import logging
import math
from random import sample
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import mindspore as ms
import mindspore.ops as ops

__all__ = [
    "CANLabelEncode",
]
_logger = logging.getLogger(__name__)


class CANLabelEncode(object):
    """Convert between text-label and text-index"""
    def __init__(
        self,
        max_text_length=100,
        character_dict_path=None,
        use_space_char=False,
        lower=False,
        task="infer",
        **kwargs,
    ):
        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"
        self.lower = lower
        self.task = task

        if character_dict_path is None:
            _logger.warning(
                "The character_dict_path is None, model can only recognize number and lower letters"
            )
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
            self.lower = True
        else:
            self.character_str = []
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def encode(self, text_seq):
        """convert text-label into text-index.
        input:
            text_seq: text labels of each image. [batch_size]
        output:
            test_seq_encoded: text-index of each image. [batch_size]
        """
        text_seq_encoded = []
        for text in text_seq:
            if text not in self.character:
                continue
            text_seq_encoded.append(self.dict.get(text))
        if len(text_seq_encoded) == 0:
            return None
        return text_seq_encoded

    def __call__(self, data):
        if self.task == "train":
            label = data["label"]
            if isinstance(label, str):
                label = label.strip().split()
            label.append(self.end_str)
            data["label"] = self.encode(label)
            return data
        elif self.task == "eval":
            data["ones_label"] = ops.ones((36), ms.int64)
            data["image_mask"] = ops.ones(data["image"].shape, ms.float32)
            data["label_len"] = len(data["label"])
            return data
        else:
            return data

