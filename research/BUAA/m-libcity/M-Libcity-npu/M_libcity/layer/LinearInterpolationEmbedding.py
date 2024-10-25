import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as np

class LinearInterpolationEmbedding(nn.Cell):
    def __init__(self,loc_size=10,time_size=11,up_loc=10,hidden_size=1):
        super().__init__()
        self.hidden_size=hidden_size
        self.loc_size = loc_size
        self.lw_time = 0.0
        self.up_time = time_size
        self.lw_loc = 0.0
        self.up_loc = up_loc
        self.h0 = ms.Parameter(np.randn([self.hidden_size, 1]))  # h0
        self.weight_ih = ms.Parameter(np.randn([self.hidden_size, self.hidden_size]))  # C
        self.weight_th_upper = ms.Parameter(np.randn([self.hidden_size, self.hidden_size]))  # T Tu
        self.weight_th_lower = ms.Parameter(np.randn([self.hidden_size, self.hidden_size]))  # T Tl
        self.weight_sh_upper = ms.Parameter(np.randn([self.hidden_size, self.hidden_size]))  # S
        self.weight_sh_lower = ms.Parameter(np.randn([self.hidden_size, self.hidden_size]))  # S
        self.location_weight = nn.Embedding(self.loc_size, self.hidden_size)  # 还是按编号来的，但是需要经纬度额外信息
    
    def construct(self, td_upper, td_lower, ld_upper, ld_lower, current_loc,loc_len):
        
        batch_size = current_loc.shape[0]
        output = []
        for i in range(batch_size):
            ttd = [((self.weight_th_upper * td_upper[i][j] +
                     self.weight_th_lower * td_lower[i][j])
                    / (td_upper[i][j] + td_lower[i][j]))
                   for j in range(loc_len[i])]
            sld = [((self.weight_sh_upper * ld_upper[i][j] +
                     self.weight_sh_lower * ld_lower[i][j])
                    / (ld_upper[i][j] + ld_lower[i][j]))
                   for j in range(loc_len[i])]
            loc = current_loc[i][:loc_len[i]]  # sequence_len
            loc = self.location_weight(loc).expand_dims(2)
            loc_vec = np.concatenate([np.matmul(sld[j],np.matmul(ttd[j],loc[j])).expand_dims(0)
                                         for j in range(loc_len[i])],0)
            loc_vec = loc_vec.sum(0)
            output.append(loc_vec)
        output=np.concatenate(output)
        return output
    
    
if __name__ == '__main__':
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE,device_target='CPU')
    dim=1
    loc_len=[4]
    td_lower=ms.Tensor([[[0],[0],[0],[0]]])
    td_upper=ms.Tensor([[[10],[10],[10],[10]]])
    ld_lower=ms.Tensor([[[0],[0],[0],[0]]])
    ld_upper=ms.Tensor([[[10],[10],[10],[10]]])
    loc=ms.Tensor([[5,5,5,5]])
    model=LinearInterpolationEmbedding()
    output=model(td_upper,td_lower,ld_upper,ld_lower,loc,loc_len)
    print(output.shape)