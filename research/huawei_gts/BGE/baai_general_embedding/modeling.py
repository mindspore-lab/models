from dataclasses import dataclass
import mindspore
from mindspore import nn
from mindspore.ops import L2Normalize
from mindnlp.transformers import AutoModel
from mindnlp.utils import ModelOutput
import mindspore.ops as ops
from mindspore.ops import arange, zeros
from mindnlp.transformers import MSBertModel


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: mindspore.Tensor = None
    p_reps: mindspore.Tensor = None
    loss: mindspore.Tensor = None
    scores: mindspore.Tensor = None


class BiEncoderModel(nn.Cell):

    def __init__(self,
                 model_name: str = None,
                 normlized: bool = True,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 0.02,
                 use_inbatch_neg: bool = True,
                 graph_mode: bool = True
                 ):
        super().__init__()
        self.graph_mode = graph_mode
        if graph_mode:
            self.model = MSBertModel.from_pretrained(model_name)
        else:
            self.model = AutoModel.from_pretrained(model_name)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model.config
        self.l2_normalize = L2Normalize(axis=-1, epsilon=1e-12)
        if not normlized:
            self.temperature = 1.0

        self.negatives_cross_device = negatives_cross_device
        self.concat = ops.Concat(axis=0)

    def sentence_embedding(self, hidden_state):
        return hidden_state[:, 0]

    def encode_graph(self, input_ids, token_type_ids, attention_mask):
        psg_out = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        res = self.sentence_embedding(psg_out[0])

        if self.normlized:
            res = self.l2_normalize(res)
        return res

    def encode_dynamic(self, input_ids, token_type_ids, attention_mask):
        psg_out = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=True)
        res = self.sentence_embedding(psg_out.last_hidden_state)

        if self.normlized:
            res = self.l2_normalize(res)
        return res

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.shape) == 2:
            return ops.matmul(q_reps, p_reps.transpose(1, 0))
        return ops.matmul(q_reps, p_reps.transpose(-2, -1))


    def construct(self, query_input_ids=None, query_token_type_ids=None, query_attention_mask=None,
                  passages_input_ids=None, passages_token_type_ids=None, passages_attention_mask=None,
                  teacher_score=None):
        batch_size, query_num, seq_len = query_input_ids.shape
        batch_size, passage_num, pseq_len = passages_input_ids.shape

        query_input_ids = query_input_ids.reshape(batch_size * query_num, seq_len)
        query_token_type_ids = query_token_type_ids.reshape(batch_size * query_num, seq_len)
        query_attention_mask = query_attention_mask.reshape(batch_size * query_num, seq_len)

        passages_input_ids = passages_input_ids.reshape(batch_size * passage_num, pseq_len)
        passages_token_type_ids = passages_token_type_ids.reshape(batch_size * passage_num, pseq_len)
        passages_attention_mask = passages_attention_mask.reshape(batch_size * passage_num, pseq_len)

        input_ids = self.concat((query_input_ids, passages_input_ids))
        token_type_ids = self.concat((query_token_type_ids, passages_token_type_ids))
        attention_mask = self.concat((query_attention_mask, passages_attention_mask))
        if self.graph_mode:
            res = self.encode_graph(input_ids, token_type_ids, attention_mask)
        else:
            res = self.encode_dynamic(input_ids, token_type_ids, attention_mask)

        q_reps = res[:batch_size, :]
        p_reps = res[batch_size:, :]

        if self.training:

            group_size = p_reps.shape[0] // q_reps.shape[0]
            if self.use_inbatch_neg:
                scores = self.compute_similarity(q_reps, p_reps) / self.temperature  # B B*G
                scores = scores.view(q_reps.shape[0], -1)

                target = arange(scores.shape[0], dtype=mindspore.int32)

                target = target * group_size
                loss = self.compute_loss(scores, target)
            else:
                scores = self.compute_similarity(q_reps[:, None, :, ],
                                                 p_reps.view(q_reps.shape[0], group_size, -1)).squeeze(
                    1) / self.temperature  # B G

                scores = scores.view(q_reps.shape[0], -1)
                target = zeros(scores.shape[0], dtype=mindspore.int32)
                loss = self.compute_loss(scores, target)
            return loss

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

            return EncoderOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps,
            )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)
