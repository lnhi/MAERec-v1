import torch.nn.functional as F
from statistics import mean
from params import args
from torch import nn
import numpy as np
from handler import *
from model import *
from utils import *
import torch 


init = nn.init.xavier_uniform_

class Item_Graph(nn.Module):
    def __init__(self, config):
        super(Item_Graph, self).__init__(config)
        self.knn_k = config['knn_k']
        self.k = 40
        self.t_feat = torch.from_numpy(np.load("text_feat-v1.npy", allow_pickle=True)).type(torch.FloatTensor).to(self.device)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

        self.gcn_layers = nn.Sequential(*[GCNLayer() for i in range(args.num_gcn_layers)])
        
        text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
        self.mm_adj = text_adj

    def forward(self, mm_adj):
        h = mm_adj
        for i in self.gcn_layers:
            h = torch.sparse.mm(self.item_emb, h)
        item_rep = mm_adj + h
        return item_rep

    def get_knn_adj_mat(self, mm_embedding):
        context_norm = mm_embedding.div(torch.norm(mm_embedding, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        #k = 5
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0])
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)
    
    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)
    
class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)

item = Item_Graph()
print(item)