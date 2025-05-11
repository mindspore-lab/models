import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import XavierUniform
# from mindspore.common.parameter import Parameter.setdata
class PromptEncoder(nn.Cell):
    def __init__(self, template, hidden_size, tokenizer, args):
        super().__init__()
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        
        # Convert mask to MindSpore Tensor
        self.cloze_mask = ms.Tensor(self.cloze_mask, dtype=ms.bool_)
        
        # Sequence indices
        self.seq_indices = ms.Tensor(list(range(len(self.cloze_mask[0]))), dtype=ms.int32)
        
        # Embedding layer
        self.embedding = nn.Embedding(len(self.cloze_mask[0]), 
                                      self.hidden_size,
                                    #   False,
                                    #   XavierUniform()
                                      )
        
        # RNN layer - bidirectional
        self.rnn = nn.RNN(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # MLP head
        self.mlp_head = nn.SequentialCell(
            nn.Dense(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dense(self.hidden_size, self.hidden_size)
        )
        
        print("init prompt encoder...")
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize embedding weights
        self.embedding.embedding_table.set_data(
            ms.common.initializer.initializer(
                XavierUniform(),
                self.embedding.embedding_table.shape,
                self.embedding.embedding_table.dtype
            )
        )
        
        # Initialize MLP weights
        for _, cell in self.mlp_head.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    ms.common.initializer.initializer(
                        XavierUniform(),
                        cell.weight.shape,
                        cell.weight.dtype
                    )
                )
                if cell.bias is not None:
                    cell.bias.set_data(
                        ms.common.initializer.initializer(
                            'zeros',
                            cell.bias.shape,
                            cell.bias.dtype
                        )
                    )

    def construct(self):
        # Get input embeddings
        input_embeds = self.embedding(self.seq_indices)
        input_embeds = ops.expand_dims(input_embeds, 0)  # Add batch dimension
        
        # Process through RNN
        rnn_output, _ = self.rnn(input_embeds)
        
        # Process through MLP
        output_embeds = self.mlp_head(rnn_output)
        output_embeds = ops.squeeze(output_embeds)  # Remove batch dimension
        
        return self.seq_indices
