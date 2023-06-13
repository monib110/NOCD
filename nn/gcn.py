import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from nocd.utils import to_sparse_tensor


def sparse_or_dense_dropout(x, p=0.5, training=True):
    if isinstance(x, (torch.sparse.FloatTensor, torch.sparse.FloatTensor)):
        new_values = F.dropout(x.values(), p=p, training=training)
        return torch.sparse.FloatTensor(x.indices(), new_values, x.size())
    else:
        return F.dropout(x, p=p, training=training)

#########################  gcn layer in paper #############
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        return adj @ (x @ self.weight) + self.bias


#########################  new gcn layer in  an other paper #############
class GraphConvolution1(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

#########################  improve gcn layer  #############

class ImpGraphConvolution(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_own = nn.Parameter(torch.empty(in_features, out_features))
        self.weight_nbr = nn.Parameter(torch.empty(in_features, out_features))

        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_own, gain=2.0)
       
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight_own)
        support1 = torch.mm(x, self.weight_nbr)
        support2 = torch.mm(x, self.weight_temp)

        output = torch.spmm(adj, support)
        output1 = torch.spmm(adj, support1)
        output2 = torch.spmm(adj, support2)

        return output +output1+ output2 + self.bias



#########################  linear batchnorm  block  #############
class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel, bias=False)
        self.bn   = nn.BatchNorm1d(out_channel,eps=1e-05, momentum=0.5) #increase momentum
        self.act  = act

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


########################graph attention##############

class GraphAttention(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=False):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = input @ self.W
        N = h.size()[0]
        
        f_1 = torch.matmul(h, self.a1)
        f_2 = torch.matmul(h, self.a2)
        
        e = self.leakyrelu(f_1 + f_2.transpose(0,1))
        
        zero_vec = -9e15*torch.ones_like(e)
        
        #attention = torch.where(adj > 0, e, zero_vec)
        attention = F.relu(e)
        attention = F.dropout(attention, self.dropout, training=self.training)
     
        h_prime = torch.matmul(attention , h )

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


#########################  gcn model ############# in this model using two gcn layer
class GCN(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, batch_norm=False):
        super().__init__()
        self.dropout = dropout
        layer_dims = np.concatenate([hidden_dims, [output_dim]]).astype(np.int32)
        #layers_dim=[128,18]
        self.layers = nn.ModuleList([GraphConvolution(input_dim, layer_dims[0]),
                                    GraphConvolution(layer_dims[0], layer_dims[1])]) 
        self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_dims
            ]

        ### encoder block =====> for use instead of relu and batchnorm
        self.encoder = nn.Sequential(
            LinearBn(128, 128),
            nn.ReLU(inplace=True),
            LinearBn(128, 128),
            nn.ReLU(inplace=True),
             LinearBn(128, 128),
             nn.ReLU(inplace=True),
            LinearBn(128, layer_dims[0])
        )
    def forward(self, x, adj):
        y = sparse_or_dense_dropout(x, p=self.dropout, training=self.training)
        
        x = self.layers[0](y, adj)
        x = x = F.relu(x)
        x = self.batch_norm[0](x)

        # x = self.layers[1](x, adj)
        # x = F.relu(x)
        # x = self.batch_norm[0](x)
        # x = self.layers[1](x, adj)
        # x = F.relu(x)
        # x = self.batch_norm[0](x)
        x = self.layers[1](x, adj)
        return x

    def get_weights(self):
        return [w for n, w in self.named_parameters() if 'bias' not in n]
    def get_biases(self):
        return [w for n, w in self.named_parameters() if 'bias' in n]
