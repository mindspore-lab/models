# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""
 Tokenization classes for python tokenizers. For fast tokenizers (provided by HuggingFace's tokenizers library) see
 tokenization_utils_fast.py
"""

import re
import unicodedata
import bisect
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import OrderedDict

from .tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    AddedToken,
    BatchEncoding,
    EncodedInput,
    EncodedInputPair,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PreTrainedTokenizerBase,
    TextInput,
    TextInputPair,
    TruncationStrategy,
    add_end_docstrings,
    PaddingStrategy,
    TensorType
)
from ..tools.logger import logger

__all__ = ["PreTrainedTokenizer", "PreTrainedTokenizerBase"]


class Trie:
    """
    Trie in Python. Creates a Trie out of a list of words. The trie is used to split on `added_tokens` in one pass
    Loose reference https://en.wikipedia.org/wiki/Trie
    """

    def __init__(self):
        self.data = {}
        self._tokens = set()

    def add(self, word: str):
        """
        Passes over every char (utf-8 char) on word and recursively adds it to the internal `data` trie representation.
        The special key `""` is used to represent termination.

        This function is idempotent, adding twice the same word will leave the trie unchanged

        Examples:
            >>> trie = Trie()
            >>> trie.add("Hello 友達")
            >>> trie.data
            {"H": {"e": {"l": {"l": {"o": {" ": {"友": {"達": {"": 1}}}}}}}}}
            >>> trie.add("Hello")
            >>> trie.data
            {"H": {"e": {"l": {"l": {"o": {"": 1, " ": {"友": {"達": {"": 1}}}}}}}}}
        """
        if not word:
            # Prevent empty string
            return

        self._tokens.add(word)
        ref = self.data
        for char in word:
            ref[char] = ref[char] if char in ref else {}
            ref = ref[char]
        ref[""] = 1

    # pylint: disable=R1702
    def split(self, text: str) -> List[str]:
        """
        Will look for the words added to the trie within `text`. Output is the original string split along the
        boundaries of the words found.

        This trie will match the longest possible word first !

        Examples:
            >>> trie = Trie()
            >>> trie.split("[CLS] This is a extra_id_100")
            ["[CLS] This is a extra_id_100"]
            >>> trie.add("[CLS]")
            >>> trie.add("extra_id_1")
            >>> trie.add("extra_id_100")
            >>> trie.split("[CLS] This is a extra_id_100")
            ["[CLS]", " This is a ", "extra_id_100"]
        """
        # indexes are counted left of the chars index.
        # "hello", index 0, is left of h, index 1 is between h and e.
        # index 5 is right of the "o".

        # States are going to capture every possible start (indexes as above)
        # as keys, and have as values, a pointer to the position in the trie
        # where we're at. This is a partial match for now.
        # This enables to keep track of multiple matches while we're iterating
        # the string
        # If the trie contains, "blowing", and "lower" and we encounter the
        # string "blower", we need to split into ["b", "lower"].
        # This is where we need to keep track of multiple possible starts.
        states = OrderedDict()

        # This will contain every indices where we need
        # to cut.
        # We force to cut at offset 0 and len(text) (added later)
        offsets = [0]

        # This is used by the lookahead which needs to skip over
        # some text where the full match exceeded the place in the initial
        # for loop
        skip = 0
        # Main loop, Giving this algorithm O(n) complexity
        for current, current_char in enumerate(text):
            if skip and current < skip:
                # Prevents the lookahead for matching twice
                # like extra_id_100 and id_100
                continue

            # This will track every state
            # that stop matching, we need to stop tracking them.
            # If we look at "lowball", we're going to match "l" (add it to states), "o", "w", then
            # fail on "b", we need to remove 0 from the valid states.
            to_remove = set()
            # Whenever we found a match, we need to drop everything
            # this is a greedy algorithm, it will match on the first found token
            reset = False

            # In this case, we already have partial matches (But unfinished)
            for start, trie_pointer in states.items():
                if "" in trie_pointer:
                    # This is a final match, we need to reset and
                    # store the results in `offsets`.

                    # Lookahead to match longest first
                    # Important in case of extra_id_1 vs extra_id_100
                    # Here we are also actively looking for other earlier partial
                    # matches
                    # "[CLS]", "L", we need to match CLS even if L is special
                    for lookstart, looktrie_pointer in states.items():
                        if lookstart > start:
                            # This partial match is later, we can stop looking
                            break
                        elif lookstart < start:
                            # This partial match is earlier, the trie pointer
                            # was already updated, so index is + 1
                            lookahead_index = current + 1
                            end = current + 1
                        else:
                            # Here lookstart == start and
                            #      looktrie_pointer == trie_pointer
                            # It wasn't updated yet so indices are current ones
                            lookahead_index = current
                            end = current
                        next_char = text[lookahead_index] if lookahead_index < len(text) else None
                        if "" in looktrie_pointer:
                            start = lookstart
                            end = lookahead_index
                            skip = lookahead_index

                        while next_char in looktrie_pointer:
                            looktrie_pointer = looktrie_pointer[next_char]
                            lookahead_index += 1
                            if "" in looktrie_pointer:
                                start = lookstart
                                end = lookahead_index
                                skip = lookahead_index

                            if lookahead_index == len(text):
                                # End of string
                                break
                            next_char = text[lookahead_index]
                        # End lookahead

                    # Storing and resetting
                    offsets.append(start)
                    offsets.append(end)
                    reset = True
                    break
                elif current_char in trie_pointer:
                    # The current character being looked at has a match within the trie
                    # update the pointer (it will be stored back into states later).
                    trie_pointer = trie_pointer[current_char]

                    # Storing back the new pointer into the states.
                    # Partial matches got longer by one.
                    states[start] = trie_pointer
                else:
                    # The new character has not match in the trie, we need
                    # to stop keeping track of this partial match.
                    # We can't do it directly within the loop because of how
                    # python iteration works
                    to_remove.add(start)

            # Either clearing the full start (we found a real match)
            # Or clearing only the partial matches that didn't work.
            if reset:
                states = {}
            else:
                for start in to_remove:
                    del states[start]

            # If this character is a starting character within the trie
            # start keeping track of this partial match.
            if current >= skip and current_char in self.data:
                states[current] = self.data[current_char]

        # We have a cut at the end with states.
        for start, trie_pointer in states.items():
            if "" in trie_pointer:
                # This is a final match, we need to reset and
                # store the results in `offsets`.
                end = len(text)
                offsets.append(start)
                offsets.append(end)
                # Longest cut is always the one with lower start so the first
                # item so we need to break.
                break

        return self.cut_text(text, offsets)

    def split_atom_1(self, states=None, text=None, offsets=None):
        """atomic act of split"""
        for start, trie_pointer in states.items():
            if "" in trie_pointer:
                # This is a final match, we need to reset and
                # store the results in `offsets`.
                end = len(text)
                offsets.append(start)
                offsets.append(end)
                # Longest cut is always the one with lower start so the first
                # item so we need to break.
                break
        return offsets

    def split_atom_2(self, reset=None, to_remove=None, states=None):
        """atomic act of split"""
        if reset:
            states = {}
        else:
            for start in to_remove:
                del states[start]
        return states

    def split_atom_3(self, states=None, current=None, text=None, start=None, skip=None):
        """atomic act of split"""
        for lookstart, looktrie_pointer in states.items():
            if lookstart > start:
                # This partial match is later, we can stop looking
                break
            elif lookstart < start:
                # This partial match is earlier, the trie pointer
                # was already updated, so index is + 1
                lookahead_index = current + 1
                end = current + 1
            else:
                # Here lookstart == start and
                #      looktrie_pointer == trie_pointer
                # It wasn't updated yet so indices are current ones
                lookahead_index = current
                end = current
            next_char = text[lookahead_index] if lookahead_index < len(text) else None
            if "" in looktrie_pointer:
                start = lookstart
                end = lookahead_index
                skip = lookahead_index

            while next_char in looktrie_pointer:
                looktrie_pointer = looktrie_pointer[next_char]
                lookahead_index += 1
                if "" in looktrie_pointer:
                    start = lookstart
                    end = lookahead_index
                    skip = lookahead_index

                if lookahead_index == len(text):
                    # End of string
                    break
                next_char = text[lookahead_index]
        return states, start, end, skip

    def cut_text(self, text, offsets):
        """cut_text"""
        # We have all the offsets now, we just need to do the actual splitting.
        # We need to eventually add the first part of the string and the eventual
        # last part.
        offsets.append(len(text))
        tokens = []
        start = 0
        for end in offsets:
            if start > end:
                logger.error(
                    "There was a bug in Trie algorithm in tokenization. Attempting to recover. Please report it"
                    " anyway."
                )
                continue
            elif start == end:
                # This might happen if there's a match at index 0
                # we're also preventing zero-width cuts in case of two
                # consecutive matches
                continue
            tokens.append(text[start:end])
            start = end

        return tokens


def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char in (' ', '\t', '\n', '\r'):
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char in ('\t', '\n', '\r'):
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_end_of_word(text):
    """Checks whether the last character in text is one of a punctuation, control or whitespace character."""
    last_char = text[-1]
    return bool(_is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(last_char))


def _is_start_of_word(text):
    """Checks whether the first character in text is one of a punctuation, control or whitespace character."""
    first_char = text[0]
    return bool(_is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(first_char))


def _insert_one_token_to_ordered_list(token_list: List[str], new_token: str):
    """
    Inserts one token to an ordered list if it does not already exist. Note: token_list must be sorted.
    """
    insertion_idx = bisect.bisect_left(token_list, new_token)
    # Checks if new_token is already in the ordered token_list
    if not (insertion_idx < len(token_list) and token_list[insertion_idx] == new_token):
        # new_token is in token_list, don't add
        token_list.insert(insertion_idx, new_token)


class PreTrainedTokenizer(PreTrainedTokenizerBase):
    r"""
    Base class for all slow tokenizers.

    Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading
    pretrained tokenizers as well as adding tokens to the vocabulary.

    This class also contain the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).

    Note:
        Initialize the basic configuration of the tokenizer.

        Steps:

        1. Initialize the parent class.
        2. If the subclass has not been initialized with `_added_tokens_decoder`, it will be initialized.
        3. Use the passed `added_tokens_decoder` to update the `_added_tokens_decoder`.
        4. Add special tokens that are not in the vocabulary to the vocabulary in the same order as
           the `SPECIAL_TOKENS_ATTRIBUTES` in `tokenizers`.

        Characteristic:

        1. Ensure that all special tokens are added to the vocabulary, even if they were not originally
           in the vocabulary.
        2. Use Trie structure to store tokens.

    Args:
        model_max_length (int, optional):
            The maximum length (in number of tokens) for the inputs to the transformer model.
            Set when the tokenizer is loaded with ``from_pretrained()`` based on the model's
            ``max_model_input_sizes`` attribute. Default: ``1e-30`` .
        padding_side (str, optional):
            Specifies the side on which the model should have padding applied. Options are
            ['right', 'left']. The default value is picked from the class attribute of the
            same name.
        truncation_side (str, optional):
            Specifies the side on which the model should have truncation applied.
            Options are ['right', 'left']. The default value is picked from the class
            attribute of the same name.
        chat_template (str, optional):
            A Jinja template string used to format lists of chat messages.
            Default: ``"{% for message in messages %}{{'<|im_start|>' + message['role'] +
            '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{
            '<|im_start|>assistant\n' }}{% endif %}"`` .
        model_input_names (List[str], optional):
            Lists the names of inputs accepted by the forward pass of the model,
            such as "token_type_ids" or "attention_mask". Defaults to values picked
            from the class attribute of the same name. Default: ``None`` .
        bos_token (Union[str, tokenizers.AddedToken], optional):
            Represents the beginning of a sentence and is associated with
            ``self.bos_token`` and ``self.bos_token_id``. Default: ``None`` .
        eos_token (Union[str, tokenizers.AddedToken], optional):
            Represents the end of a sentence and is associated with ``self.eos_token``
            and ``self.eos_token_id``. Default: ``None`` .
        unk_token (Union[str, tokenizers.AddedToken], optional):
            Represents an out-of-vocabulary token and is associated with
            ``self.unk_token`` and ``self.unk_token_id``. Default: ``None`` .
        sep_token (Union[str, tokenizers.AddedToken], optional):
            A special token separating two different sentences in the same input
            (used by BERT, for example) and is associated with ``self.sep_token``
            and ``self.sep_token_id``. Default: ``None`` .
        pad_token (Union[str, tokenizers.AddedToken], optional):
            Used to make arrays of tokens the same size for batching purposes and
            will be ignored by attention mechanisms or loss computation. It is
            associated with ``self.pad_token`` and ``self.pad_token_id``. Default: ``None`` .
        cls_token (Union[str, tokenizers.AddedToken], optional):
            Represents the class of the input (used by BERT, for example) and is
            associated with ``self.cls_token`` and ``self.cls_token_id``. Default: ``None`` .
        mask_token (Union[str, tokenizers.AddedToken], optional):
            Represents a masked token (used by masked-language modeling pretraining
            objectives like BERT) and is associated with ``self.mask_token`` and
            ``self.mask_token_id``. Default: ``None`` .
        additional_special_tokens (Union[tuple, list, tokenizers.AddedToken], optional):
            Lists additional special tokens that are ensured to be skipped when
            decoding with ``skip_special_tokens`` set to True. They will be added
            at the end of the vocabulary if not already part of it. Default: ``None`` .
        clean_up_tokenization_spaces (bool, optional):
            Determines whether to clean-up spaces that were added when splitting the
            input text during the tokenization process. Default: ``True`` .
        split_special_tokens (bool, optional):
            Specifies whether special tokens should be split during the tokenization
            process. This affects the internal state of the tokenizer. By default, special
            tokens are not split. For example, if '<s>' is the bos_token, then
            ``tokenizer.tokenize("<s>")`` results in ['<s>']. If ``split_special_tokens``
            is True, then ``tokenizer.tokenize("<s>")`` would result in ['<','s', '>']. Default: ``False`` .

    Returns:
        PreTrainedTokenizer instance.

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
    _support_list = []

    def __init__(self, **kwargs):
        # 1. Init the parent class

        self.tokens_trie = Trie()

        # 2. init `_added_tokens_decoder` if child class did not
        if not hasattr(self, "_added_tokens_decoder"):
            self._added_tokens_decoder: Dict[int, AddedToken] = {}

        # 3. if a `added_tokens_decoder` is passed, we are loading from a saved tokenizer, we overwrite
        self._added_tokens_decoder.update(kwargs.pop("added_tokens_decoder", {}))
        self._added_tokens_encoder: Dict[str, int] = {k.content: v for v, k in self._added_tokens_decoder.items()}

        # 4 init the parent class
        super().__init__(**kwargs)

        # 4. If some of the special tokens are not part of the vocab, we add them, at the end.
        # the order of addition is the same as self.SPECIAL_TOKENS_ATTRIBUTES following `tokenizers`
        self._add_tokens(
            [token for token in self.all_special_tokens_extended if token not in self._added_tokens_encoder],
            special_tokens=True,
        )

        self._decode_use_source_tokenizer = False

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        raise NotImplementedError

    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        """
        Returns the sorted mapping from string to index. The added tokens encoder is cached for performance
        optimisation in `self._added_tokens_encoder` for the slow tokenizers.

        Returns:
            A dict, the added tokens.
        """
        return {k.content: v for v, k in sorted(self._added_tokens_decoder.items(), key=lambda item: item[0])}

    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        """
        Returns the added tokens in the vocabulary as a dictionary of index to AddedToken.

        Returns:
            A dict, the added tokens.
        """
        return dict(sorted(self._added_tokens_decoder.items(), key=lambda item: item[0]))

    @added_tokens_decoder.setter
    def added_tokens_decoder(self, value: Dict[int, Union[AddedToken, str]]) -> Dict[int, AddedToken]:
        # Always raise an error if string because users should define the behavior
        for index, token in value.items():
            if not isinstance(token, (str, AddedToken)) or not isinstance(index, int):
                raise ValueError(
                    f"The provided `added_tokens_decoder` has an element of type {index.__class__, token.__class__}, "
                    f"should be a dict of {int, Union[AddedToken, str]}"
                )

            self._added_tokens_decoder[index] = AddedToken(token) if isinstance(token, str) else token
            self._added_tokens_encoder[str(token)] = index

    def get_added_vocab(self) -> Dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index.

        Returns:
            A dict, the added tokens.
        """
        return self.added_tokens_encoder

    def __len__(self):
        """
        Size of the full vocabulary with the added tokens.
        """
        return self.vocab_size + len(self.added_tokens_encoder)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary. Special tokens are sometimes already in the
        vocab which is why they have to be handled specifically.

        Args:
            new_tokens (`List[str]`or `List[tokenizers.AddedToken]`):
                Token(s) to add in vocabulary. A token is counted as added if it's not already in the vocabulary
                (tested by checking if the tokenizer assign the index of the `unk_token` to them). If a token is part
                of the vocabulary then we simply mark this token as an `AddedToken` which allows to control the
                stripping and normalization of this token. This is NOT possible in `tokenizers`.
            special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the tokens should be added as special tokens.

        Returns:
            The number of tokens actually added to the vocabulary, type is `int`.

        Examples:

        ```python
        # Let's see how to increase the vocabulary of Bert model and tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")

        num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
        print("We have added", num_added_toks, "tokens")
        # Note: resize_token_embeddings expects to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
        ```"""
        added_tokens = 0
        if new_tokens is None:
            return added_tokens
        current_vocab = self.get_vocab().copy()
        new_idx = len(current_vocab)  # only call this once, len gives the last index + 1
        for token in new_tokens:
            if not isinstance(token, (str, AddedToken)):
                raise TypeError(f"Token {token} is not a string but a {type(token)}.")
            if str(token) == "":
                continue
            if isinstance(token, str):
                if token in self._added_tokens_encoder:
                    continue
                else:
                    # very important for fast and slow equivalence!
                    is_special = token in self.all_special_tokens or special_tokens
                    token = AddedToken(
                        token, rstrip=False, lstrip=False, normalized=not is_special, special=is_special
                    )
            elif special_tokens:
                # doing token.special=True changes the normalization! will fix in rust
                # this is important and the only reason why the AddedTokens in each class are normalized by default
                token.__setstate__({"special": True, "normalized": token.normalized})
            if token in self._added_tokens_decoder:
                continue
            if not token.special and token.normalized and getattr(self, "do_lower_case", False):
                # Normalize if requested
                token.content = token.content.lower()
            if token.content not in current_vocab:
                token_index = new_idx + added_tokens
                current_vocab[token.content] = token_index
                added_tokens += 1
            else:
                token_index = current_vocab[token.content]

            if token.special and str(token) not in self.all_special_tokens:
                self.reset_special_tokens_cache()
                self._additional_special_tokens.append(token)
            # the setter automatically updates the reverse map
            self._added_tokens_decoder[token_index] = token
            self._added_tokens_encoder[token.content] = token_index
            if self.verbose:
                logger.info(f"Adding {token} to the vocabulary")

        self._update_trie()
        return added_tokens

    # pylint: disable=W0212
    def _update_trie(self, unique_no_split_tokens: Optional[str] = None):
        if unique_no_split_tokens is None:
            unique_no_split_tokens = []
        for token in self._added_tokens_decoder.values():
            if token not in self.tokens_trie._tokens:
                self.tokens_trie.add(token.content)
        for token in unique_no_split_tokens:
            if token not in self.tokens_trie._tokens:
                self.tokens_trie.add(token)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        Note:
            This encodes a dummy input and checks the number of added tokens, and is therefore not efficient.
            Do not put this inside your training loop.


        Args:
            pair (bool, optional): Whether the number of added tokens should be computed in the case of
                a sequence pair or a single sequence. Default: ``False`` .

        Returns:
            Number of special tokens added to sequences.
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def tokenize(
            self, text: TextInput, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs
    ) -> List[str]:
        """
        Converts a string in a sequence of tokens, using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.

        Args:
            text (TextInput): The sequence to be encoded.
            pair (str, optional): A second sequence to be encoded with the first. Default: ``None`` .
            add_special_tokens (bool, optional): Whether to add the special tokens associated with
                the corresponding model. Default: ``False`` .
            kwargs (Any, optional): Will be passed to the underlying model specific
                encode method. See details in `~PreTrainedTokenizerBase.__call__`.

        Returns:
            `tokenized_text`, the list of tokens, type is `List[str]`.
        """
        split_special_tokens = kwargs.pop("split_special_tokens", self.split_special_tokens)

        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        if kwargs:
            logger.warning(f"Keyword arguments {kwargs} not recognized.")

        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase. Might be super slow as well?
            escaped_special_toks = [re.escape(s_tok) for s_tok in self.all_special_tokens]
            escaped_special_toks += [
                re.escape(s_tok.content)
                for s_tok in (self._added_tokens_decoder.values())
                if not s_tok.special and s_tok.normalized
            ]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        if split_special_tokens:
            no_split_token = []
            tokens = [text]
        else:
            no_split_token = self._added_tokens_encoder.keys()  # don't split on any of the added tokens
            # "This is something<special_token_1>  else"
            tokens = self.tokens_trie.split(text)

        # ["This is something", "<special_token_1>", "  else"]
        tokens = self.tokenize_atom(tokens, no_split_token)
        # ["This is something", "<special_token_1>", "else"]
        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text

    def tokenize_atom(self, tokens, no_split_token):
        """tokenize_atom"""
        for i, token in enumerate(tokens):
            if token in no_split_token:
                tok_extended = self._added_tokens_decoder.get(self._added_tokens_encoder[token], None)
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.rstrip and right:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        tokens[i + 1] = right.lstrip()
                    # Strip white spaces on the left
                    if tok_extended.lstrip and left:
                        tokens[i - 1] = left.rstrip()  # Opposite here
                    if tok_extended.single_word and left and left[-1] != " ":
                        tokens[i - 1] += token
                        tokens[i] = ""
                    elif tok_extended.single_word and right and right[0] != " ":
                        tokens[i + 1] = token + tokens[i + 1]
                        tokens[i] = ""
                else:
                    raise ValueError(
                        f"{tok_extended} cannot be tokenized because it was not properly added"
                        f" to the tokenizer. This means that it is not an `AddedToken` but a {type(tok_extended)}"
                    )
        return tokens

    def _tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (Union[str, List[str]]): One or several token(s) to convert to token id(s).

        Returns:
            `ids` , the token id or list of token ids, type is `int` or `List[int]`.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        raise NotImplementedError

    def _encode_plus(
            self,
            text: Union[TextInput, PreTokenizedInput, EncodedInput],
            text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs,
    ) -> BatchEncoding:
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            if isinstance(text, (list, tuple)) and text and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(
                        itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
                    )
                    return self.convert_tokens_to_ids(tokens)
                return self.convert_tokens_to_ids(text)
            if isinstance(text, (list, tuple)) and text and isinstance(text[0], int):
                return text
            if is_split_into_words:
                raise ValueError(
                    f"Input {text} is not valid. Should be a string or a list/tuple of strings when"
                    " `is_split_into_words=True`."
                )
            raise ValueError(
                f"Input {text} is not valid. Should be a string, a list/tuple of strings or a list/tuple of"
                " integers."
            )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "mindformers.PreTrainedTokenizerFast. "
            )

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None

        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )

    def _batch_encode_plus(
            self,
            batch_text_or_text_pairs: Union[
                List[TextInput],
                List[TextInputPair],
                List[PreTokenizedInput],
                List[PreTokenizedInputPair],
                List[EncodedInput],
                List[EncodedInputPair],
            ],
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs,
    ) -> BatchEncoding:
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            if isinstance(text, (list, tuple)) and text and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(
                        itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
                    )
                    return self.convert_tokens_to_ids(tokens)
                return self.convert_tokens_to_ids(text)
            if isinstance(text, (list, tuple)) and text and isinstance(text[0], int):
                return text
            raise ValueError(
                "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
            )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "mindformers.PreTrainedTokenizerFast."
            )

        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids

            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        return BatchEncoding(batch_outputs)

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def _batch_prepare_for_model(
            self,
            batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[str] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_length: bool = False,
            verbose: bool = True,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """

        batch_outputs = {}
        for first_ids, second_ids in batch_ids_pairs:
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    def prepare_for_tokenization(
            self, text: str, **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Performs any necessary transformations before tokenization.

        This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the
        `kwargs` at the end of the encoding process to be sure all the arguments have been used.

        Args:
            text (str): The text to prepare.
            kwargs (Any, optional): Keyword arguments to use for the tokenization.

        Returns:
            Tuple[str, Dict[str, Any]], means the prepared text and the unused kwargs.
        """
        return (text, kwargs)

    def get_special_tokens_mask(
            self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1], 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))

    def convert_ids_to_tokens(
            self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (Union[int, list[int]]):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (bool, optional):
                Whether to remove special tokens in the decoding. Default: ``False`` .

        Returns:
            Str or List[str], the decoded token(s).
        """
        if isinstance(ids, int):
            if ids in self._added_tokens_decoder:
                return self._added_tokens_decoder[ids].content
            if ids >= self.vocab_size:
                raise IndexError(f"The token id {ids} is out of the size of vocabulary, please check your tokenizer "
                                 f"and corresponding vocabulary files.")
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self._added_tokens_decoder:
                tokens.append(self._added_tokens_decoder[index].content)
            else:
                if index >= self.vocab_size:
                    raise IndexError(
                        f"The token id {index} is out of the size of vocabulary, please check your tokenizer "
                        f"and corresponding vocabulary files.")
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, index: int) -> str:
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def _decode(
            self,
            token_ids: List[int],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = None,
            **kwargs,
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        legacy_added_tokens = set(self._added_tokens_encoder.keys()) - set(self.all_special_tokens) | {
            token for token in self.additional_special_tokens if self.convert_tokens_to_ids(token) >= self.vocab_size
        }
        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in legacy_added_tokens:
                if current_sub_text:
                    string = self.convert_tokens_to_string(current_sub_text)
                    if string:
                        sub_texts.append(string)
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        spaces_between_special_tokens = kwargs.pop("spaces_between_special_tokens", True)
        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        return text
