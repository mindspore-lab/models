import torch
from torch import nn, tensor
from torch.nn.parameter import Parameter


class GroupEnhance(nn.Module):
    def __init__(self, groups = 5):
        super(GroupEnhance, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.weight   = Parameter(torch.zeros(1, groups, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x): # (b, f, l)=(2,20,2)=80 group = 4
        b, f, l = x.size() # b=l=2,f=30
        x = x.view(b*self.groups, -1, l) # (2*4,5,2)
        dot = x * self.avg_pool(x) 
        dot = dot.sum(dim=1, keepdim=True) 
        norm = dot.view(b * self.groups, -1) # (8,2)
        norm = norm - norm.mean(dim=1, keepdim=True) 
        std = norm.std(dim=1, keepdim=True) + 1e-5
        norm = norm / std 
        norm = norm.view(b, self.groups, l) #(2,4,2)
        norm = norm * self.weight + self.bias
        norm = norm.view(b * self.groups, 1, l)
        x = x * self.sig(norm) #(b*group,f/group,l)(2*4,5,2)
        x = x.view(b, f, l)
        return x
