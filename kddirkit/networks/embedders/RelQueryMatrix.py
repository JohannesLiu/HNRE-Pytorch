import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class RelQueryMatrix(nn.module):
    def __init__(self, hier, hidden_size, relation_level_layer, DEVICE = "cpu", weight_matrix =None):
        nn.Module.__init__(self)

        self.layer = relation_level_layer
        self.hier = hier
        self.hidden_size = hidden_size
        self.weight_matrix = weight_matrix
        self.relation_matrixs = []

        self.DEVICE= DEVICE
        self.reset_parameter()

    def reset_parameter(self):
        if self.weight_matrix != None :
            for i in range(self.hier):
                self.relation_matrixs.append(nn.Embedding(self.layer[i], self.hidden_size, _weight=nn.init.xavier_uniform(
                    torch.Tensor(self.layer[i], self.hidden_size))).to(self.DEVICE))
        else: #这里要放每一层关系的权重
            NotImplemented

    @property
    def relation_matrixs(self):
        return self.relation_matrixs