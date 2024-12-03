import logging
import numpy as np
from typing import Dict, List, Optional

__all__ = [
    "RecCTCLabelEncode",
]
_logger = logging.getLogger(__name__)

def str2idx(
    text: str,
    label_dict: Dict[str, int],
    max_text_len: int = 23,
    lower: bool = False,
    unknown_idx: Optional[int] = None,
    ignore_warning: bool = False,
) -> List[int]:
    """
    Encode text (string) to a squence of char indices
    Args:
        text (str): text string
    Returns:
        char_indices (List[int]): char index seq
    """
    if len(text) == 0 or len(text) > max_text_len:
        return None

    if lower:
        text = text.lower()

    char_indices = []
    for char in text:
        if char not in label_dict:
            if unknown_idx is not None:
                char_indices.append(unknown_idx)
        else:
            char_indices.append(label_dict[char])

    if len(char_indices) == 0 and not ignore_warning:
        _logger.warning("`{}` does not contain any valid character in the dictionary.".format(text))
        return None

    return char_indices

class RecCTCLabelEncode(object):
    def __init__(
        self,
        max_text_len=23,
        character_dict_path=None,
        use_space_char=False,
        blank_at_last=True,
        lower=False,
        **kwargs,
    ):
        self.max_text_len = max_text_len
        self.space_idx = None
        self.lower = lower

        # read dict
        if character_dict_path is None:
            char_list = [c for c in "0123456789abcdefghijklmnopqrstuvwxyz"]

            self.lower = True
        else:
            char_list = []
            with open(character_dict_path, "r") as f:
                for line in f:
                    c = line.rstrip("\n\r")
                    char_list.append(c)
        # add space char if set
        if use_space_char:
            if " " not in char_list:
                char_list.append(" ")
            self.space_idx = len(char_list) - 1
        else:
            if " " in char_list:
                _logger.warning(
                    "The dict still contains space char in dict although use_space_char is set to be False, "
                    f"because the space char is coded in the dictionary file {character_dict_path}"
                )

        self.num_valid_chars = len(char_list)

        # add blank token for padding
        if blank_at_last:
            # the index of a char in dict is [0, num_chars-1], blank index is set to num_chars
            char_list.append("<PAD>")
            self.blank_idx = self.num_valid_chars
        else:
            char_list = ["<PAD>"] + char_list
            self.blank_idx = 0

        self.dict = {c: idx for idx, c in enumerate(char_list)}

        self.num_classes = len(self.dict)

    def __call__(self, data: dict):
        char_indices = str2idx(data["label"], self.dict, max_text_len=self.max_text_len, lower=self.lower)

        if char_indices is None:
            char_indices = []
        data["length"] = np.array(len(char_indices), dtype=np.int32)
        # padding with blank index
        char_indices = char_indices + [self.blank_idx] * (self.max_text_len - len(char_indices))
        data["text_seq"] = np.array(char_indices, dtype=np.int32)
        data["text_length"] = len(data["label"])
        data["text_padded"] = data["label"] + " " * (self.max_text_len - len(data["label"]))

        return data
