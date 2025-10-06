# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from task import Task

class Cluster(Task):
    def __init__(self, data, model):
        super().__init__(data, model)

    def check_data(self):
        pass
    def check_model(self):
        pass
    def train_model(self):
        pass
    def get_clusters(self, k=10, data="test")
        self.

