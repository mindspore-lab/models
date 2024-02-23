from mindspore import nn
import mindspore.ops as ops
from .layers import EmbeddingLayer, MultiLayerPerceptron


class PLEModel(nn.Cell):
    """
    A pytorch implementation of PLE Model.

    Reference:
        Tang, Hongyan, et al. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations. RecSys 2020.
    """

    def __init__(self, categorical_field_dims, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, shared_expert_num, specific_expert_num, dropout):
        super(PLEModel, self).__init__(auto_prefix=True)
        self.emb = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.embed_output_dim = len(categorical_field_dims) * embed_dim
        self.task_num = task_num
        self.shared_expert_num = shared_expert_num
        self.specific_expert_num = specific_expert_num
        self.layers_num = len(bottom_mlp_dims)

        self.task_experts=[[0] * self.task_num for _ in range(self.layers_num)]
        self.task_gates=[[0] * self.task_num for _ in range(self.layers_num)]
        self.share_experts=[0] * self.layers_num
        self.share_gates=[0] * self.layers_num
        for i in range(self.layers_num):
            input_dim = self.embed_output_dim if 0 == i else bottom_mlp_dims[i - 1]
            self.share_experts[i] = nn.CellList([MultiLayerPerceptron(input_dim, [bottom_mlp_dims[i]], dropout, output_layer=False) for k in range(self.specific_expert_num)])
            self.share_gates[i]=nn.SequentialCell(nn.Dense(input_dim, shared_expert_num + task_num * specific_expert_num), nn.Softmax(axis=1))
            for j in range(task_num):
                self.task_experts[i][j]=nn.CellList([MultiLayerPerceptron(input_dim, [bottom_mlp_dims[i]], dropout, output_layer=False) for k in range(self.specific_expert_num)])
                self.task_gates[i][j]=nn.SequentialCell(nn.Dense(input_dim, shared_expert_num + specific_expert_num), nn.Softmax(axis=1))
            self.task_experts[i]=nn.CellList(self.task_experts[i])
        self.task_experts= nn.CellList(self.task_experts)
        self.share_experts= nn.CellList(self.share_experts)
        self.tower = nn.CellList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])
        self.sigmoid = ops.Sigmoid()
        self.bmm = ops.BatchMatMul()

    def construct(self, categorical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        categorical_emb = self.emb(categorical_x)
        categorical_emb = self.duplicate(categorical_emb)
        emb = categorical_emb
        task_fea = task_fea = [emb[i].view(-1,self.embed_output_dim) for i in range(self.task_num + 1)]
        for i in range(self.layers_num):
            share_output=[ops.expand_dims(expert(task_fea[-1]),1) for expert in self.share_experts[i]]
            task_output_list=[]
            for j in range(self.task_num):
                task_output=[ops.expand_dims(expert(task_fea[j]),1) for expert in self.task_experts[i][j]]
                task_output_list.extend(task_output)
                mix_ouput=ops.concat(task_output+share_output,axis=1)
                gate_value = ops.expand_dims(self.task_gates[i][j](task_fea[j]),1)
                task_fea[j] = ops.squeeze(self.bmm(gate_value, mix_ouput),1)
            if i != self.layers_num-1:#最后一层不需要计算share expert 的输出
                gate_value = ops.expand_dims(self.share_gates[i](task_fea[-1]),1)
                mix_ouput = ops.concat(task_output_list + share_output, axis=1)
                task_fea[-1] = ops.squeeze(self.bmm(gate_value, mix_ouput),1) 
        
        results = [self.sigmoid(ops.squeeze(self.tower[i](task_fea[i]),1)) for i in range(self.task_num)]
        return results
    
    def duplicate(self,x):
        if len(x) != (self.task_num+1): # Don't set batch_size as two  
            return [x] * (self.task_num+1)
        return x