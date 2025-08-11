# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""LLaMA tokenizer APIs."""

import os
from shutil import copyfile
from typing import Any, Dict, List, Optional

import sentencepiece as spm

from mindformers.tools.logger import logger
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.tokenization_utils import AddedToken, PreTrainedTokenizer
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from ..convert_slow_tokenizer import import_protobuf

__all__ = ['LlamaTokenizer']

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

SPIECE_UNDERLINE = "▁"

DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class LlamaTokenizer(PreTrainedTokenizer):
    r"""
    Construct a Llama tokenizer based on byte-level Byte-Pair-Encoding.

    The default padding token is unset as there isno padding token in the original model.

    Args:
        vocab_file (str):
            Path to the vocabulary file.
        unk_token (Union[str, AddedToken], optional):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. Default: ``"<unk>"`` .
        bos_token (Union[str, AddedToken], optional):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
            Default: ``"<s>"`` .
        eos_token (Union[str, AddedToken], optional):
            The end of sequence token. Default: ``"</s>"`` .
        pad_token (Union[str, AddedToken], optional):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation. Default: ``"<unk>"`` .
        sp_model_kwargs (Dict[str, Any], optional):
            Will be passed to the `SentencePieceProcessor.__init__()` method.
            The `Python wrapper for SentencePiece <https://github.com/google/sentencepiece/tree/master/python>`_
            can be used, among other things, to set keys below. Default: ``None`` , an empty dict will be passed.
        add_bos_token (bool, optional):
            Whether to add an `bos_token` at the start of sequences. Default: ``True`` .
        add_eos_token (bool, optional):
            Whether to add an `eos_token` at the end of sequences. Default: ``False`` .
        clean_up_tokenization_spaces (bool, optional):
            Whether to clean up spaces after decoding. Cleanup includes removing potential artifacts like
            extra spaces. Default: ``False`` .
        use_default_system_prompt (bool, optional):
            Whether the default system prompt for Llama should be used. Default: ``False`` .
        spaces_between_special_tokens (bool, optional):
            Whether to add spaces between special tokens. Default: ``False`` .
        legacy (bool, optional):
            Whether the `legacy` behavior of the tokenizer should be used. Default: ``True`` .

    Returns:
        A LlamaTokenizer instance.

    Examples:
        >>> from mindformers import LlamaTokenizer
        >>> tokenizer = LlamaTokenizer.from_pretrained("llama2_7b")
        >>> res = tokenizer("hello world")
        >>> print(res)
        {'input_ids': [1, 27701, 924], 'attention_mask': [1, 1, 1]}
        >>> res = tokenizer("hello world", padding='max_length', max_length=10)
        >>> print(res)
        {'input_ids': [1, 27701, 924, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]}
        >>> res = tokenizer("hello world", return_tensors='ms')
        >>> print(res)
        {'input_ids': Tensor(shape=[3], dtype=Int32, value= [    1, 27701,  924]), 'attention_mask': Tensor(shape=[3],
        dtype=Int32, value= [1, 1, 1])}
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    FILE_LIST = ['tokenizer_config.json']
    _support_list = MindFormerBook.get_tokenizer_support_list()['llama']

    def __init__(
            self,
            vocab_file,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<unk>",
            sp_model_kwargs: Optional[Dict[str, Any]] = None,
            add_bos_token=True,
            add_eos_token=False,
            clean_up_tokenization_spaces=False,
            legacy=True,
            **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        self.legacy = legacy
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", False))

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs
        )

    @property
    def unk_token_length(self):
        return len(self.sp_model.encode(str(self.unk_token)))

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_spm_processor
    def get_spm_processor(self, from_slow=False):
        """get_spm_processor"""
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        if self.legacy or from_slow:  # no dependency on protobuf
            tokenizer.Load(self.vocab_file)
            return tokenizer

        with open(self.vocab_file, "rb") as f:
            sp_model = f.read()
            model_pb2 = import_protobuf(f"The new behaviour of {self.__class__.__name__} (with `self.legacy = False`)")
            model = model_pb2.ModelProto.FromString(sp_model)
            normalizer_spec = model_pb2.NormalizerSpec()
            normalizer_spec.add_dummy_prefix = False
            model.normalizer_spec.MergeFrom(normalizer_spec)
            sp_model = model.SerializeToString()
            tokenizer.LoadFromSerializedProto(sp_model)
        return tokenizer

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.tokenize
    def tokenize(
            self, text: "TextInput", pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs
    ) -> List[str]:
        """
        Converts a string to a list of tokens. If `self.legacy` is set to `False`, a prefix token is added unless the
        first token is special.
        """
        if self.legacy or not text:
            return super().tokenize(text, **kwargs)

        tokens = super().tokenize(SPIECE_UNDERLINE + text.replace(SPIECE_UNDERLINE, " "), **kwargs)

        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]
        return tokens

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer._tokenize
    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
        `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus we always encode `f"{unk_token}text"` and strip the
        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        tokens = self.sp_model.encode(text, out_type=str)
        if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):
            return tokens

        # 1. Encode string + prefix ex: "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. Remove self.unk_token from ['<','unk','>', '▁Hey']
        return tokens[self.unk_token_length:] if len(tokens) >= self.unk_token_length else tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # since we manually add the prefix space, we have to remove it when decoding
        if tokens[0].startswith(SPIECE_UNDERLINE):
            tokens[0] = tokens[0][1:]

        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for i, token in enumerate(tokens):
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special and i != 0 and self.legacy:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`, paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path (%s) should be a directory", save_directory)
            return None
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            with os.fdopen(os.open(out_vocab_file, flags_, 0o750), 'wb') as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None):
        r"""
        Insert the special tokens to the input_ids. Current this method will add `bos_token` and 'eos_token'
        to the head and end of sequence respectively.

        Args:
            token_ids_0 (List[int]):
                List of token IDs.
            token_ids_1 (List[int], optional):
                Second list of token IDs for sequence pairs. Default: ``None`` , only use one sequence.

        Returns:
            list of the tokens after inserting special tokens.
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (List[int]):
                List of token IDs.
            token_ids_1 (List[int], optional):
                Second list of token IDs for sequence pairs. Default: ``None`` , only use one sequence.
            already_has_special_tokens (bool, optional):
                Whether the token list is already formatted with special tokens for the model. Default: ``False`` .

        Returns:
            A list consists of integer 0 and 1, where 1 for a special token and 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        An ALBERT sequence pair mask has the following format:

        .. code-block::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1
            |   first sequence   | second sequence  |

        if `token_ids_1` is None, then only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (List[int]):
                List of token IDs.
            token_ids_1 (List[int], optional):
                Second list of token IDs for sequence pairs. Default: ``None`` , only use one sequence.

        Returns:
            A List consists of integer 0 and 1 according to the given sequence(s),
            where 0 for tokens in `token_ids_0` and 1 for tokens in `token_ids_1`.
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output
