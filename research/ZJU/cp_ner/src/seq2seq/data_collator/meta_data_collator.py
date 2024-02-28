#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
from dataclasses import dataclass
import logging
import random
import math
from typing import Optional, Union
from collections import OrderedDict
from mindspore import ops
from ...extraction.record_schema import RecordSchema
from ...extraction.dataset_processer import spot_prompt, asoc_prompt
from ...extraction.constants import BaseStructureMarker, text_start
from ...extraction.utils import convert_to_record_function
from ...extraction.spot_asoc_noiser import SpotAsocNoiser


logger = logging.getLogger("__main__")


class DynamicSSIGenerator():
    """
    Sample negative spot and asoc to construct SSI
    """
    def __init__(self, tokenizer,schema,positive_rate=1,
                 negative=5, ordered_prompt=False) -> None:
        self.spot_dict = self.get_ordered_dict(schema.type_list, tokenizer)
        self.asoc_dict = self.get_ordered_dict(schema.role_list, tokenizer)
        self.spot_list = list(self.spot_dict.keys())
        self.asoc_list = list(self.asoc_dict.keys())
        self.spot_prompt = tokenizer.s.piece_to_id(spot_prompt)
        self.asoc_prompt = tokenizer.s.piece_to_id(asoc_prompt)
        self.text_start = tokenizer.s.piece_to_id(text_start)
        self.positive_rate = positive_rate if positive_rate > 0 and positive_rate < 1 else 1
        self.negative = negative
        self.ordered_prompt = ordered_prompt

    @staticmethod
    def get_ordered_dict(schema_name_list, tokenizer):
        schema_ordered_dict = OrderedDict()
        for name in schema_name_list:
            schema_ordered_dict[name] = tokenizer.encode(name, add_special_tokens=False)
        return schema_ordered_dict

    @staticmethod
    def sample_negative(postive, candidates, k=5):
        if k < 0:
            k = len(candidates)
        negative_set = set()
        if len(candidates) > 0:
            for index in ops.randperm(len(candidates))[:k].tolist():
                negative = candidates[index]
                if negative not in postive:
                    negative_set.add(negative)
        return list(negative_set)

    def sample_spot(self, positive):
        """ Sample spot
        """
        negative_spot = self.sample_negative(postive=positive,
                    candidates=self.spot_list, k=self.negative)
        positive_spot = random.sample(positive, math.floor(len(positive) * self.positive_rate))

        prefix_spot_candidates = positive_spot + negative_spot
        converted_spot_prefix = self.convert_prefix(
            candidates=prefix_spot_candidates,
            prompt=self.spot_prompt,
            mapper=self.spot_dict,
            ordered_prompt=self.ordered_prompt,
        )

        return converted_spot_prefix, positive_spot, negative_spot

    def sample_asoc(self, positive):
        """ Sample Asoc
        """
        negative_asoc = self.sample_negative(postive=positive,
                        candidates=self.asoc_list, k=self.negative)
        prefix_asoc_candidates = positive + negative_asoc
        converted_asoc_prefix = self.convert_prefix(
            candidates=prefix_asoc_candidates,
            prompt=self.asoc_prompt,
            mapper=self.asoc_dict,
            ordered_prompt=self.ordered_prompt,
        )
        return converted_asoc_prefix, negative_asoc

    def full_spot(self, shuffle=False):
        # Random Prompt + Shuffle
        if not self.ordered_prompt and shuffle:
            ordered_prompt = False
        else:
            ordered_prompt = True
        return self.convert_prefix(
            candidates=self.spot_list,
            prompt=self.spot_prompt,
            mapper=self.spot_dict,
            ordered_prompt=ordered_prompt,
        )

    def full_asoc(self, shuffle=False):
        # Random Prompt + Shuffle
        if not self.ordered_prompt and shuffle:
            ordered_prompt = False
        else:
            ordered_prompt = True
        return self.convert_prefix(
            candidates=self.asoc_list,
            prompt=self.asoc_prompt,
            mapper=self.asoc_dict,
            ordered_prompt=ordered_prompt,
        )

    @staticmethod
    def convert_prefix(candidates, prompt, mapper, ordered_prompt=True):
        prefix = list()

        if ordered_prompt:
            candidate_sorted = sorted([(candidate, index)
                            for index, candidate in enumerate(candidates)])
            index_list = [index for _, index in candidate_sorted]
        else:
            index_list = ops.randperm(len(candidates)).tolist() if len(candidates) >  0 else []

        for index in index_list:
            prefix += [prompt]
            prefix += mapper[candidates[index]]
        return prefix
