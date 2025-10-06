# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import torch
from torch_geometric.nn import MessagePassing, Linear, SAGEConv
from torch_geometric.utils import add_self_loops

class EdgeConvolution(MessagePassing):
    def __init__(self, out_channels):
        super(EdgeConvolution, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = Linear(-1, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Add self-loops to the adjacency matrix.
        # edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=0)

        # Start propagating messages.
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # Concatenate node features with edge features.
        tmp = torch.cat([x_j, edge_attr], dim=1)
        return self.lin(tmp)
    
class SAGEConvWithEdgeAttr(SAGEConv):
    def message(self, x_j, edge_attr):
        # Concatenate node features with edge features.
        tmp = torch.cat([x_j, edge_attr], dim=1)
        return self.lin(tmp)