# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from task import GraphTask
import torch
from torch.utils.data import DataLoader

class Graph_Embeddings(GraphTask):
    def __init__(self, model, train_data, test_data=None):
        super().__init__(model=model, train_data=train_data, test_data=test_data)

    def check_data(self):
        pass
    def check_model(self):
        pass

