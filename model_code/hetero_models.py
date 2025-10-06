# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import torch
import torch.nn.functional as F

from torch.nn import Embedding
from model_code.layers import EdgeConvolution, SAGEConvWithEdgeAttr

from torch_geometric.data import Data, HeteroData
from torch.nn import Dropout, LazyBatchNorm1d
from torch_geometric.nn import Linear, SAGEConv, BatchNorm
from torch_geometric.nn import to_hetero

import inspect

# Trivial Autoencoders
class CompletelyTrivialAutoencoder(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        out_dim = 0
        for node_tensor in data.x_dict.values():
            out_dim = max(node_tensor.size(1), out_dim)
        
        self.lin1 = Linear(-1, out_dim)
    
    def forward(self, x, edge_index, edge_attr):
        x = self.lin1(x)
        return x

class TrivialAutoencoder(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        out_dim = 0
        for node_tensor in data.x_dict.values():
            out_dim = max(node_tensor.size(1), out_dim)

        self.lin1 = Linear(-1, 100)
        self.lin2 = Linear(-1, 100)
        self.lin3 = Linear(-1, out_dim)
    
    def forward(self, x, edge_index, edge_attr):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return x

# Encoder/Decoder

class HeteroAutoencoder(torch.nn.Module):
    
    def __init__(self, data, hetero_encoder, hetero_decoder):
        super().__init__()
        self.encoder = to_hetero(hetero_encoder, data.metadata())
        self.decoder = to_hetero(hetero_decoder, data.metadata())
        self.neighbor_depth = hetero_encoder.neighbor_depth + hetero_decoder.neighbor_depth
    
    def forward(self, x, edge_index, edge_attr):
        # Use the encoder
        encoding_dict = self.encoder(x, edge_index, edge_attr)
        # Use the decoder
        recon_node_values = self.decoder(encoding_dict, edge_index, edge_attr)
        return recon_node_values

# Encoders

class LinearEncoder(torch.nn.Module):
    def __init__(self, data, inner_dim=20, encoding_dim=16, use_activation=True, use_dropout=True): 
        super().__init__()
        self.neighbor_depth = 0

        self.lin1 = Linear(-1, inner_dim)
        self.batchnorm1 = BatchNorm(inner_dim)
        self.dropout1 = Dropout(p=0.2)
        self.lin2 = Linear(-1, inner_dim)
        self.batchnorm2 = BatchNorm(inner_dim)
        self.dropout2 = Dropout(p=0.2)
        self.lin3 = Linear(-1, encoding_dim)

        self.use_activation = use_activation
        self.use_dropout = use_dropout

    def forward(self, x, edge_index, edge_attr):
        x = self.lin1(x)
        x = self.batchnorm1(x)
        if self.use_activation:
            x = F.relu(x)
        if self.use_dropout:
            x = self.dropout1(x)
        x = self.lin2(x)
        x = self.batchnorm2(x)
        if self.use_activation:
            x = F.relu(x)
        if self.use_dropout:
            x = self.dropout2(x)
        x = self.lin3(x)
        return x

class SAGEEncoder(torch.nn.Module):
    def __init__(self, data, inner_dim=20, encoding_dim=16, use_dropout=True):
        super().__init__()
        self.neighbor_depth = 5
        self.use_dropout = use_dropout
        self.sage1 = SAGEConv(-1, inner_dim)
        self.linskip1 = Linear(-1, inner_dim)
        self.lin1 = Linear(-1, inner_dim)
        self.batchnorm1 = BatchNorm(inner_dim)
        self.sage2 = SAGEConv(-1, inner_dim)
        self.linskip2 = Linear(-1, inner_dim)
        self.lin2 = Linear(-1, inner_dim)        
        self.batchnorm2 = BatchNorm(inner_dim)
        self.dropout2 = Dropout(p=0.2)
        self.sage3 = SAGEConv(-1, inner_dim)
        self.linskip3 = Linear(-1, inner_dim)
        self.lin3 = Linear(-1, inner_dim)
        self.batchnorm3 = BatchNorm(inner_dim)
        self.dropout3 = Dropout(p=0.2)
        self.sage4 = SAGEConv(-1, inner_dim)
        self.linskip4 = Linear(-1, inner_dim)
        self.lin4 = Linear(-1, inner_dim)
        self.batchnorm4 = BatchNorm(inner_dim)
        self.sage5 = SAGEConv(-1, inner_dim)
        self.linskip5 = Linear(-1, inner_dim)
        self.lin5 = Linear(-1, encoding_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.sage1(x, edge_index) + self.linskip1(x)
        x = self.lin1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.sage2(x, edge_index) + self.linskip2(x)
        x = self.lin2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        if self.use_dropout:
            x = self.dropout2(x)
        x = self.sage3(x, edge_index) + self.linskip3(x)
        x = self.lin3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        if self.use_dropout:
            x = self.dropout3(x)
        x = self.sage4(x, edge_index) + self.linskip4(x)
        x = self.lin4(x)
        x = self.batchnorm4(x)
        x = F.relu(x)
        x = self.sage5(x, edge_index) + self.linskip5(x)
        x = self.lin5(x)
        return x 
    

class ConvolutionEncoder(torch.nn.Module):
    def __init__(self, data, inner_dim=20, encoding_dim=16):
        super().__init__()
        self.c = EdgeConvolution

# Decoders

class LinearDecoder(torch.nn.Module):
    def __init__(self, data, inner_dim=20, out_dim=None, use_activation=True, use_dropout=True):
        super().__init__()
        self.neighbor_depth = 0
        if out_dim is None:
            out_dim = 0
            for node_tensor in data.x_dict.values():
                out_dim = max(node_tensor.size(1), out_dim)
        self.lin1 = Linear(-1, inner_dim)
        self.batchnorm1 = BatchNorm(inner_dim)
        self.dropout1 = Dropout(p=0.2)
        self.lin2 = Linear(-1, out_dim)
        self.batchnorm2 = BatchNorm(out_dim)
        self.dropout2 = Dropout(p=0.2)
        self.lin3 = Linear(-1, out_dim)
        self.batchnorm3 = BatchNorm(out_dim)

        self.use_activation = use_activation
        self.use_dropout = use_dropout

    def forward(self, x, edge_index=None, edge_attr=None):
        x = self.lin1(x)
        x = self.batchnorm1(x)
        if self.use_activation:
            x = F.relu(x)
        if self.use_dropout:
            x = self.dropout1(x)
        x = self.lin2(x)
        x = self.batchnorm2(x)
        if self.use_activation:
            x = F.relu(x)
        if self.use_dropout:
            x = self.dropout2(x)
        x = self.lin3(x)
        x = self.batchnorm3(x)
        return x

class SAGEDecoder(torch.nn.Module):
    def __init__(self, data, inner_dim=20, use_dropout=True):
        super().__init__()
        self.neighbor_depth = 5
        out_dim = 0
        self.use_dropout = use_dropout
        for node_tensor in data.x_dict.values():
            out_dim = max(node_tensor.size(1), out_dim)
        self.sage1 = SAGEConv(-1, inner_dim)
        self.linskip1 = Linear(-1, inner_dim)
        self.lin1 = Linear(-1, inner_dim)
        self.batchnorm1 = BatchNorm(inner_dim)
        self.sage2 = SAGEConv(-1, inner_dim)
        self.linskip2 = Linear(-1, inner_dim)
        self.lin2 = Linear(-1, inner_dim)        
        self.batchnorm2 = BatchNorm(inner_dim)
        self.dropout2 = Dropout(p=0.2)
        self.sage3 = SAGEConv(-1, inner_dim)
        self.linskip3 = Linear(-1, inner_dim)
        self.lin3 = Linear(-1, inner_dim)
        self.batchnorm3 = BatchNorm(inner_dim)
        self.dropout3 = Dropout(p=0.2)
        self.sage4 = SAGEConv(-1, out_dim)
        self.linskip4 = Linear(-1, out_dim)
        self.lin4 = Linear(-1, out_dim)
        self.batchnorm4 = BatchNorm(out_dim)
        self.sage5 = SAGEConv(-1, out_dim)
        self.linskip5 = Linear(-1, out_dim)
        self.lin5 = Linear(-1, out_dim)
    
    def forward(self, x, edge_index, edge_attr=None):
        x = self.sage1(x, edge_index) + self.linskip1(x)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.sage2(x, edge_index) + self.linskip2(x)
        x = self.lin2(x)
        x = F.relu(x)
        if self.use_dropout:
            x = self.dropout2(x)
        x = self.sage3(x, edge_index) + self.linskip3(x)
        x = self.lin3(x)
        x = F.relu(x)
        if self.use_dropout:
            x = self.dropout3(x)
        x = self.sage4(x, edge_index) + self.linskip4(x)
        x = self.lin4(x)
        x = F.relu(x)
        x = self.sage5(x, edge_index) + self.linskip5(x)
        x = self.lin5(x)
        return x 

# Old
class GCN(torch.nn.Module):
    def __init__(self, data, hidden_dim=16):
        # torch.set_default_dtype(torch.float32)
        super().__init__()
        self.conv1 = EdgeConvolution(hidden_dim)
        self.skip1 = Linear(-1, 16)
        self.conv2 = EdgeConvolution(hidden_dim)
        self.skip2 = Linear(-1, hidden_dim)
        out_dim = 0
        for node_tensor in data.x_dict.values():
            out_dim = max(node_tensor.size(1), out_dim)
        self.lin3 = Linear(-1, out_dim)

    def forward(self, x, edge_index, edge_attr):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # First Graph Convolution layer
        x = self.conv1(x, edge_index, edge_attr) + self.skip1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Second Graph Convolution layer
        x = self.conv2(x, edge_index, edge_attr) + self.skip2(x)
        
        # return F.log_softmax(x, dim=1)
        return self.lin3(x)

class NodeReconstructionDecoder():
    pass