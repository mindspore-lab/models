import json
import h5py
import random
import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Tuple, Any

from datasets import Dataset

def load_csv_as_df(file_path):
    dataset = pd.read_csv(file_path, dtype={'Zipcode': 'str'})
    dataset['labels'] = dataset['Label']
    dataset['Film genre'] = dataset['First genre']
    fields = ["User ID", "Gender", "Age", "Job", "Zipcode", "Movie ID", "Title", "Film genre", "labels"]
    dataset = dataset[fields]
    return dataset


class PLM4CTRDataset(Dataset):
    """ PLM4CTR Dataset
    The PLM4CTRDataset overwrites the _getitem function of the Dataset Class to include the templating step.
    """
    def _post_setups(
        self, 
        tokenizer, 
        shuffle_fields: bool,
        meta_data_dir: str, 
        h5_data_dir: str,
        mode: str,
        model_fusion: str,
        do_mlm_only: str,
    ):
        """ Set up the parameters
        Args:
            tokenizer: Tokenizer from HuggingFace
            shuffle_fields: Whether to shuffle the fields for lossless augmentation
            meta_data_dir: The data path for meta CTR data
            h5_data_dir: The data path for CTR data
            mode: `train`/`test`
            model_fusion: Method to fuse CTR & NLP model for prediction
            do_mlm_only: Whether to do MLM pretraining only
        """
        self.tokenizer = tokenizer
        self.shuffle_fields = shuffle_fields
        self.meta_data_dir = meta_data_dir
        self.h5_data_dir = h5_data_dir
        self.mode = mode
        self.model_fusion = model_fusion
        self.do_mlm_only = do_mlm_only
        
        self.get_meta_data()
        self.get_h5_data(mode)
    
    def get_meta_data(self):
        meta_data = json.load(open(self.meta_data_dir, 'r'))
        self.field_names = meta_data['field_names']
        self.feature_count = meta_data['feature_count']
        self.feature_dict = meta_data['feature_dict']
        self.feature_offset = meta_data['feature_offset']
        self.num_fields = len(self.field_names)
        self.input_size = sum(self.feature_count)
       
    def get_h5_data(self, mode):
        assert mode in ["train", "valid", "test"]
        with h5py.File(self.h5_data_dir, 'r') as f:
            mode_name = mode if mode != "valid" else "train"
            self.ctr_X = f[f"{mode_name} data"][:]
            self.ctr_Y = f[f"{mode_name} label"][:]
        if mode == "train" and not self.do_mlm_only: # The validation set is also used for mlm pretraining.
            self.ctr_X = self.ctr_X[:len(self.ctr_X) // 9 * 8]
            self.ctr_Y = self.ctr_Y[:len(self.ctr_Y) // 9 * 8]
        if mode == "valid":
            self.ctr_X = self.ctr_X[len(self.ctr_X) // 9 * 8:]
            self.ctr_Y = self.ctr_Y[len(self.ctr_Y) // 9 * 8:]
        offset = np.array(self.feature_offset).reshape(1, self.num_fields)
        assert self.__len__() == len(self.ctr_X)
        assert self.__len__() == len(self.ctr_Y)
        self.ctr_X += offset

    def _getitem(self, key: Union[int, slice, str], decoded: bool = True, **kwargs) -> Union[Dict, List]:
        """ Get Item from Tabular Data
        Get one instance of the tabular data, permuted, converted to text and tokenized.
        """
        if self.model_fusion == "ctr_only":
            return self.ctr_X[key], self.ctr_Y[key]
        
        row = self._data.fast_slice(key, 1)

        shuffle_fields = list(row.column_names)
        shuffle_fields.remove("labels")
        if self.shuffle_fields:
            random.shuffle(shuffle_fields)

        shuffled_text = " ".join(
            ["%s is %s." % (field, str(row[field].to_pylist()[0]).strip()) for field in shuffle_fields]
        )
        
        tokenized_output = self.tokenizer(
            shuffled_text, 
            padding="max_length", 
            max_length=100,
            return_tensors="ms"
        )

        return self.ctr_X[key], self.ctr_Y[key], tokenized_output["input_ids"], tokenized_output["attention_mask"]

