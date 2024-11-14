from typing import cast, List, Union, Tuple
from tqdm import tqdm
import numpy as np

import mindspore
from mindnlp.transformers import MSBertModel, AutoTokenizer

class FlagModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            use_fp16: bool = True
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = MSBertModel.from_pretrained(model_name_or_path)
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method

        self.l2_normalize = mindspore.ops.L2Normalize(axis=-1, epsilon=1e-12)
        
        if use_fp16: self.model.half()

    def encode_queries(self, queries: Union[List[str], str],
                       batch_size: int = 256,
                       max_length: int = 512,
                       convert_to_numpy: bool = True) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if self.query_instruction_for_retrieval is not None:
            if isinstance(queries, str):
                input_texts = self.query_instruction_for_retrieval + queries
            else:
                input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
        else:
            input_texts = queries
        return self.encode(input_texts, batch_size=batch_size, max_length=max_length, convert_to_numpy=convert_to_numpy)

    def encode_corpus(self,
                      corpus: Union[List[str], str],
                      batch_size: int = 256,
                      max_length: int = 512,
                      convert_to_numpy: bool = True) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        return self.encode(corpus, batch_size=batch_size, max_length=max_length, convert_to_numpy=convert_to_numpy)
    def encode(self,
               sentences: Union[List[str], str],
               batch_size: int = 256,
               max_length: int = 512,
               convert_to_numpy: bool = True) -> np.ndarray:

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []

        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs = self.tokenizer(sentences_batch, padding="max_length", truncation=True, return_tensors="ms", max_length=max_length)
            outputs = self.model(**inputs)
            last_hidden_state = outputs[0]
            embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
            if self.normalize_embeddings:
                embeddings = self.l2_normalize(embeddings)
                
            embeddings = cast(mindspore.Tensor, embeddings)

            if convert_to_numpy:
                embeddings = embeddings.asnumpy()
            all_embeddings.append(embeddings)

        if convert_to_numpy:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = mindspore.ops.stack(all_embeddings)

        if input_was_string:
            return all_embeddings[0]
        return all_embeddings

    def pooling(self,
                last_hidden_state: mindspore.Tensor,
                attention_mask: mindspore.Tensor = None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = mindspore.ops.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(axis=1, keepdim=True).float()
            return s / d
