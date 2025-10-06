# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from loguru import logger

from abc import ABC
import copy
import math
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch import tensor, float32
from IPython.display import display, clear_output
from datetime import datetime
from itertools import chain
from torch.utils.data import DataLoader

class Task(ABC):
    """
    A Task represents a machine learning problem that involves 
    converting one Torch dataset into another. 
    
    Examples of tasks include: Calculating graph embeddings, automated clustering, and next token prediction.
    """
    def __init__(self, model, train_data, val_data=None, test_data=None, logger=logger):
        """
        Parameters
        ----------
        model : Torch model
            Torch model used for task solving.
        train_data : Torch Dataset 
            Torch Dataset for model training.
        val_data : Torch Dataset
            Torch Dataset for model validation.
        test_data : Torch Dataset, optional 
            Torch Dataset for model testing.
        """
        self.logger=logger
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_test = self.test_data is not None
        self.model = model
        self.name = type(self).__name__
        self._check_data()
        self._check_model()
        
    def _check_data(self):
        """ 
        May be overwritten or extended to check that data has the right properties. 
        Called during initialization.
        """
        pass

    def _check_model(self):
        """ 
        May be overwritten or extended to check that model has the right properties. 
        Called during initialization.
        """
        pass

    def get_data(self, data):
        """
        Return a dataset for this task.

        Parameters
        ----------
        data : ['train', 'val', 'test']
            the name of the dataset

        Returns
        -------
        Torch Dataset
            the dataset
        """
        if data == "train":
            return self.train_data
        if data == "val":
            return self.val_data
        if data == "test":
            return self.test_data
        raise ValueError(f"data must be in {['train', 'val', 'test']}")

    def model_loss(self, batch, model_out):
        """
        Calculate loss function from input data and model output. 
        
        Parameters
        ----------
        batch : torch.Tensor
            model input
        model_out : torch.Tensor
            model output (from Task.model_call)
        
        Returns
        -------
        torch.Tensor
            the loss function from the model call
        """
        x = batch
        return torch.nn.functional.mse_loss(x, model_out)

    def model_call(self, batch):
        """
        Run model on batch data and return results.
        
        Parameters
        ----------
        batch : torch.Tensor
            the input data for the model
        
        Returns
        -------
        torch.Tensor
            model output
        """
        x = batch
        return self.model(x)

    def _plot_init(self):
        """ Initialize plot variables. """
        plt.ion()
        self._fig, self._ax = plt.subplots(1, 1)
        self._lines = {} 
        self._ax.set_xlabel('Epochs')
        self._ax.set_ylabel('Loss')

        self._color_map = ['r', 'b', 'g', 'k', 'y', 'p']

    def _plot_update(self, epoch, losses, percent=0.9):
        """
        Update plot with new data.

        Parameters
        ----------
        epoch : int
            current training epoch and number of losses of each type
        losses : dict(str, list(float))
            losses to plot in ascending order of epoch
        percent : float, default=0.9
            the most recent portion of the graph to show at one time
        """
        for idx, (loss_name, loss_values) in enumerate(losses.items()):
            if loss_name not in self._lines:
                self._lines[loss_name], = self._ax.plot([], [], color=self._color_map[idx], label=loss_name)
            
            self._lines[loss_name].set_xdata(list(range(math.ceil((epoch+1)*(1-percent)), epoch + 1)))
            self._lines[loss_name].set_ydata(loss_values[math.ceil((epoch+1)*(1-percent)):epoch + 1])
        
        self._ax.relim()
        self._ax.autoscale_view(True, True, True)
        self._ax.legend()
        clear_output(wait=True)
        display(self._fig)
        plt.close()

    def _make_data_iterator(self, data, use_dataloader, batch_size=None, shuffle=None, drop_last=False):
        """
        Creates an iterator for the dataset. 

        Parameters
        ----------
        data : ['train', 'val', 'test']
            the name of the dataset to use
        use_dataloader : boolean
            whether to use a pytorch dataloader for batching
        batch_size : int, optional
            the batch size of the dataloader
            must be set if use_dataloader is True
        shuffle : boolean, optional
            whether the dataloader should shuffle inputs
            must be set if use_dataloader is True
        drop_last : boolean, default=False
            whether the dataloader should drop the last, possibly smaller, batch
            must be set if use_dataloader is True
        """
        dataset = self.get_data(data)
        if use_dataloader:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        else:
            dataloader = [dataset]
        return dataloader
    
    def _make_optimizer(self):
        """ Make a new optimizer for training the model. """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        return optimizer

    def train_model(self, epochs=100, use_DataLoader=True, batch_size=64,
                    shuffle=True, validation=False, notify=10, plot=True):
        """
        Runs a training loop to train the underlying model on the training dataset.

        Parameters
        ----------
        epochs : int, default=100
            the number of training loops/epochs
        use_DataLoader : boolean, default=True
            whether a dataloader should be used.
        batch_size : int, default=64
            the number of samples to batch for dataloader
        shuffle : boolean, default=True
            whether the DataLoader should shuffle the samples before batching
        validation : [False, "view", "save_best", "save_load_best"], default=False
            whether to create a validation set and what to do with it
        notify : int, default=10
            how often the training loss should be logged
        """
        raise NotImplementedError("train_model of Task is currently out of date. "
                                  "Subclasses may provide an implementation.")

    def preds(self, data):
        """
        Runs the model for the specified data and returns predicted logits

        Parameters
        ----------
        data : str ["train", "val", "test"]
            which task dataset to use as input

        Returns
        -------
        torch.Tensor
            raw result of model_call
        """
        dataset = self.get_data(data)
        self.model.eval()
        result = self.model_call(dataset)
        return result

    def save_model(self, name=None, timestamp=None, to_torch=True):
        """
        Saves the model to torch pt file.

        If no name is provided, the model will be named after the task and the timestamp.
        
        Parameters
        ----------
        name : str, optional
            name for model file
        timestamp : datetime, optional
            timestamp to use for naming. Not used if name is specified. Current time is used if not specified.
        to_torch : boolean, default=True
            whether to save the model using the pytorch save method or pickle
        """
        now = timestamp
        if now is None:
            now = datetime.now()
        formatted_time = now.strftime('%Y-%m-%d_%H-%M-%S')
        if name is None:
            name = f"{self.name}_{formatted_time}"
        ext = "pt" if to_torch else "pk1"
        if not "." in name:
            name = f"{name}.{ext}"
        foldername = f"models/{self.name}"
        os.makedirs(foldername, exist_ok=True)
        if to_torch:
            torch.save(self.model, os.path.join(foldername, name))
        else:
            with open(os.path.join(foldername, name), 'wb') as file:
                pickle.dump(self.model, file)

    def save_data(self, train_name=None, val_name=None, test_name=None, timestamp=None, to_torch=True):
        """
        Saves the datasets to pickle file.

        If no name is provided, the data will be named after the task and the timestamp.

        Parameters
        ----------
        train_name : str, optional
            name for training data file
        val_name : str, optinoal
            name for validation data file
        test_name : str, optional
            name for testing data file
        timestamp : datetime, optional
            timestamp to use for naming. Not used if name is specified. Current time is used if not specified.
        to_torch : boolean, default=False
            whether to save the data using the pytorch save method or pickle
        """
        now = timestamp
        if now is None:
            now = datetime.now()
        formatted_time = now.strftime('%Y-%m-%d_%H-%M-%S')
        if train_name is None:
            train_name = f"train_{self.name}_{formatted_time}"
        if val_name is None:
            val_name = f"val_{self.name}_{formatted_time}"
        if test_name is None:
            test_name = f"test_{self.name}_{formatted_time}"
        ext = "pt" if to_torch else "pk1"
        if not "." in train_name:
            train_name = f"{train_name}.{ext}"
        if not "." in val_name:
            val_name = f"{val_name}.{ext}"
        if not "." in test_name:
            test_name = f"{test_name}.{ext}"
        
        train_foldername = f"train_data/{self.name}"
        os.makedirs(train_foldername, exist_ok=True)
        if to_torch:
            torch.save(self.train_data, os.path.join(train_foldername, train_name))
        else:
            with open(os.path.join(train_foldername, train_name), 'wb') as train_file:
                pickle.dump(self.train_data, train_file)
        
        if self.val_data is not None:
            val_foldername = f"val_data/{self.name}"
            os.makedirs(val_foldername, exist_ok=True)
            if to_torch:
                torch.save(self.val_data, os.path.join(val_foldername, val_name))
            else:
                with open(os.path.join(val_foldername, val_name), 'wb') as val_file:
                    pickle.dump(self.test_data, test_file)

        if self.train_test:
            test_foldername = f"test_data/{self.name}"
            os.makedirs(test_foldername, exist_ok=True)
            if to_torch:
                torch.save(self.test_data, os.path.join(test_foldername, test_name))
            else:
                with open(os.path.join(test_foldername, test_name), 'wb') as test_file:
                    pickle.dump(self.test_data, test_file)
     
    def save_preds(self, train_name=None, val_name=None, test_name=None, timestamp=None, to_torch=True):
        """
        Saves the predictions to pickle file.

        If no name is provided, the data will be named after the task and the timestamp.

        Parameters
        ----------
        train_name : str, optional
            name for training predictions data file
        val_name : str, optional
            name for validation predictions data file
        test_name : str, optional
            name for testing predictions data file
        timestamp : datetime, optional
            timestamp to use for naming. Not used if name is specified. Current time is used if not specified.
        to_torch : boolean, default=True
            whether to save the predictions using the pytorch save method or pickle
        """
        now = timestamp
        if now is None:
            now = datetime.now()
        formatted_time = now.strftime('%Y-%m-%d_%H-%M-%S')
        if train_name is None:
            train_name = f"train_preds_{self.name}_{formatted_time}"
        if val_name is None:
            val_name = f"val_preds_{self.name}_{formatted_time}"
        if test_name is None:
            test_name = f"test_preds_{self.name}_{formatted_time}"
        ext = "pt" if to_torch else "pk1"
        if not "." in train_name:
            train_name = f"{train_name}.{ext}"
        if not "." in val_name:
            val_name = f"{val_name}.{ext}"
        if not "." in test_name:
            test_name = f"{test_name}.{ext}"

        train_foldername = f"train_preds/{self.name}"
        train_preds = self.preds(data="train")
        os.makedirs(train_foldername, exist_ok=True)
        if to_torch:
            torch.save(train_preds, os.path.join(train_foldername, train_name))
        else:
            with open(os.path.join(train_foldername, train_name), 'wb') as train_file:
                pickle.dump(train_preds, train_file)
        
        if self.val_data is not None:
            val_foldername = f"val_preds/{self.name}"
            val_preds = self.preds(data="val")
            os.makedirs(val_foldername, exist_ok=True)
            if to_torch:
                torch.save(val_preds, os.path.join(val_foldername, val_name))
            else:
                with open(os.path.join(val_foldername, val_name), 'wb') as val_file:
                    pickle.dump(val_preds, test_file)
        
        if self.test_data is not None:
            test_foldername = f"test_preds/{self.name}"
            test_preds = self.preds(data="test")
            os.makedirs(test_foldername, exist_ok=True)
            if to_torch:
                torch.save(test_preds, os.path.join(test_foldername, test_name))
            else:
                with open(os.path.join(test_foldername, test_name), 'wb') as test_file:
                    pickle.dump(test_preds, test_file)

    def save_all(self, timestamp=None):
        """
        Saves the datasets, model, and predictions.

        Parameters
        ----------
        timestamp : datetime, optional
            timestamp to use for naming. All files will use the same timestamp.
        """
        now = timestamp
        if now is None:
            now = datetime.now()
        self.save_data(timestamp=now)
        self.save_model(timestamp=now)
        self.save_preds(timestamp=now)

    @classmethod
    def load_model(cls, model_path):
        """
        Loads model from a file.

        Parameters
        ----------
        model_path : str
            path to model file.
            Must be a torch file (.pt) or pickle file (.pkl)
        
        Returns
        -------
        pytorch model
            the saved model
        """
        if model_path.partition('.')[2] == "pt":
            model = torch.load(model_path)
        elif model_path.partition('.')[2] == "pkl":
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
        else:
            raise Exception("Model file extension must be torch (pt) or pickle (pkl)")
        return model

    @classmethod
    def load_data(cls, data_path):
        """
        Loads data from a file.

        Parameters
        ----------
        data_path : str
            path to data file.
            Must be a torch file (.pt) or pickle file (.pkl)
        
        Returns
        -------
        pytorch tensor
            the saved tensor
        """
        if data_path.partition('.')[2] == "pt":
            data = torch.load(data_path)
        elif data_path.partition('.')[2] == "pkl":
            with open(data_path, 'rb') as data_file:
                data = pickle.load(data_file)
        else:
            raise Exception("Data file extension must be torch (pt) or pickle (pkl)")
        return data

    @classmethod
    def load_task(cls, model_path, train_data_path, val_data_path=None, test_data_path=None):
        """
        Loads a task from specified files.

        Parameters
        ----------
        model_path : str
            filepath to model
        train_data_path : str
            filepath to training data
        val_data_path : str, optional
            filepath to validation data
        test_data_path : str, optional
            filepath to testing data

        Returns
        -------
        Task
            a Task using the given model and data
        """
        model = cls.load_model(model_path)
        train_data = cls.load_data(train_data_path)
        if val_data_path:
            val_data = cls.load_data(val_data_path)
        else:
            val_data = None
        if test_data_path:
            test_data = cls.load_data(test_data_path)
        else:
            test_data = None
        return cls(model=model, train_data=train_data, val_data=None, test_data=test_data)

    @classmethod
    def load_recent(cls):
        """
        Loads the most recent saved task back into a Task object.

        Returns
        -------
        Task
            the most recently saved Task
        """
        model_folder = f"models/{cls.__name__}/"
        train_data_folder = f"train_data/{cls.__name__}/"
        val_data_folder = f"val_data/{cls.__name__}"
        test_data_folder = f"test_data/{cls.__name__}/"
        most_recent_model = max([os.path.join(model_folder, fname) for fname in os.listdir(model_folder)], key=os.path.getctime)
        most_recent_train_data = max([os.path.join(train_data_folder, fname) for fname in os.listdir(train_data_folder)], key=os.path.getctime)
        most_recent_val_data = None
        most_recent_test_data = None
        try: 
            train_datas = os.listdir(train_data_folder)
            train_datas_len = len(train_datas)
        except FileNotFoundError as fnfe:
            logger.warning("No training data found (may not be needed).")
            train_datas = []
            train_datas_len = 0
        try:
            val_datas = os.listdir(val_data_folder)
            val_datas_len = len(val_datas)
        except FileNotFoundError as fnfe:
            logger.warning("No validation data found (may not be needed).")
            val_datas = []
            val_datas_len = 0        
        try:
            test_datas = os.listdir(test_data_folder)
            test_datas_len = len(test_datas)
        except FileNotFoundError as fnfe:
            logger.warning("No test data found (may not be needed).")
            test_datas = []
            test_datas_len = 0
        if train_datas_len > 0:
            most_recent_train_data = max([os.path.join(train_data_folder, fname) for fname in train_datas], key=os.path.getctime)
        if val_datas_len > 0:
            most_recent_val_data_check = max([os.path.join(val_data_folder, fname) for fname in val_datas], key=os.path.getctime)
            base_train = os.path.basename(most_recent_train_data)
            base_val = os.path.basename(most_recent_val_data_check)
            if base_train.partition('_')[-1] == base_val.partition('_')[-1]:
                most_recent_val_data = most_recent_val_data_check
        if test_datas_len > 0:
            most_recent_test_data_check = max([os.path.join(test_data_folder, fname) for fname in test_datas], key=os.path.getctime)
            base_train = os.path.basename(most_recent_train_data)
            base_test = os.path.basename(most_recent_test_data_check)
            if base_train.partition('_')[-1] == base_test.partition('_')[-1]:
                most_recent_test_data = most_recent_test_data_check
        return cls.load(model_path=most_recent_model, 
                        train_data_path=most_recent_train_data, 
                        val_data_path=most_recent_val_data, 
                        test_data_path=most_recent_test_data)


class GraphTask(Task):
    """ Represents a task that requires graph data rather than tabular data. """

    def __init__(self, model, train_data, val_data=None, test_data=None, logger=logger):
        super().__init__(model=model, train_data=train_data, val_data=val_data, test_data=test_data, logger=logger)
    
    def model_call(self, batch):
        x = batch.x_dict
        edge_index = batch.edge_index_dict
        edge_attr = batch.edge_attr_dict

        return self.model(x, edge_index, edge_attr)
    
    def train_model(self, epochs=100, use_DataLoader=True, batch_size=64,
                    shuffle=True, validation=False, optimizer = None, notify=10):
        """
        Runs a training loop to train the underlying model on the training graph(s).

        Parameters
        ----------
        epochs : int
            the number of training loops/epochs
        use_DataLoader : boolean
            whether a dataloader should be used.
        batch_size : int
            the number of samples to batch for dataloader
        shuffle : boolean
            whether the DataLoader should shuffle the samples before batching
        validation : [False, "view", "save_best", "save_load_best"]
            whether to create a validation set and what to do with it
        optimizer : torch_optimizer, optional
            the optimization method the model should use. Defaults to Adam(lr=0.01)
        notify : int
            how often the training loss should be logged
        """
        raise NotImplementedError("train_model of GraphTask is currently out of date.")


class HeteroGraphTask(GraphTask):
    """ Represents a task that requires HeteroData graph data rather than tabular data. """

    class _HeteroDataFixer:
        """
        This class fixes an issue where HeteroData objects lack the "get" method
        despite the to_hetero class expecting and calling "get" on the object.

        It wraps the HeteroData object into a _HeteroDataFixer object, acting like
        a HeteroData in almost every way. When the underlying HeteroData object is
        needed, ._hetero_data may be accessed. 

        Known ways this object does not act like HeteroData:
        1. _HeteroDataFixer.get(key, default) works as expected but 
            HeteroData.get(key, default) throws an error
        2. type(_HeteroDataFixer) != type(HeteroData)
        3. _HeteroDataFixer._hetero_data attribute holds the original HeteroData object
        """
        def __init__(self, hetero_data):
            self._hetero_data = hetero_data

        def get(self, key, default=None):
            try:
                result = self._hetero_data[key]
            except KeyError:
                result = default
            return result

        def __getattr__(self, name):
            # This method is called for any attribute not found in this class
            try:
                attr = getattr(self._hetero_data, name)
            except AttributeError as ae:
                self.logger.error(f"Attempted access of {name}, but attribute is not available to this HeteroData.")
                raise ae
            return attr

        def __setattr__(self, name, value):
            if name == '_hetero_data':
                # Initialize the _hetero_data attribute
                super().__setattr__(name, value)
            else:
                setattr(self._hetero_data, name, value)

        def __getitem__(self, key):
            result = self._hetero_data[key]
            return result
        
        def __setitem__(self, key, value):
            self._hetero_data[key] = value

    def __init__(self, model, train_data, val_data=None, test_data=None, target_nodes=None, logger=logger):
        """
        Parameters
        ----------
        model : Torch model
            Torch model used for task solving.
        train_data : Torch Dataset 
            Torch Dataset for model training.
        val_data : Torch Dataset
            Torch Dataset for model validation.
        test_data : Torch Dataset, optional 
            Torch Dataset for model testing.
        target_nodes : list[str], optional
            node types that should be reproduced by the autoencoder
        """
        # Wrap HeteroData objects with _HeteroDataFixer class
        train_data = self._HeteroDataFixer(train_data)
        if val_data:
            val_data = self._HeteroDataFixer(val_data)
        if test_data:
            test_data = self._HeteroDataFixer(test_data)
        super().__init__(model=model, train_data=train_data, val_data=val_data, test_data=test_data, logger=logger)
        if target_nodes is None:
            target_nodes = [node for node in train_data.node_types]
        self.target_nodes = target_nodes

    def model_call(self, batch, hide_mask=None):
        x = batch.x_dict
        edge_index = batch.edge_index_dict
        edge_attr = batch.edge_attr_dict

        if hide_mask:
            for node_type, node_mask in hide_mask.items():
                hidden_x = x[node_type].clone()
                hidden_x[node_mask] = 0.0
                x[node_type] = hidden_x

        return self.model(x, edge_index, edge_attr)
    
    def model_loss_nodes(self, batch, model_out, mask):
        """ Calculate loss function for each node of each node type. """
        raise NotImplementedError("If model_loss_node is needed, it must be implemented "
                                  "by a subclass of HeteroGraphTask.")
    
    def model_loss(self, batch, model_out, aggr="mean"):
        """ Calculate loss function for each node type and add them. """
        raise NotImplementedError("No implementation for Generic HeteroGraphTask loss function.")

    def get_mask(self, batch, name, value=1.0, include_non_targets=False):
        meta_names = batch.x_meta_name_dict
        meta_vals = batch.x_meta_dict
        mask_dict = {}
        for node_type in meta_names.keys():
            device = batch[node_type].x.get_device()
            node_names = meta_names[node_type]
            node_vals = meta_vals[node_type]
            try:
                idx = node_names.index(name)
            except ValueError as ve:
                self.logger.error(f"\"{name}\" not a valid metadata column name for node \"{node_type}\".")
                raise ve
            mask = node_vals[:, idx] == value
            mask_dict[node_type] = mask
        if not include_non_targets:
            mask_dict = {key: value for key, value in mask_dict.items() if key in self.target_nodes}
        return mask_dict

    def get_batch_mask(self, batch):
        batch_mask = {}
        for node_type in batch.node_types:
            if hasattr(batch[node_type], "batch_size"):
                batch_node = batch[node_type]
                batch_size = batch_node.batch_size
                batch_mask[node_type] = torch.zeros_like(batch_node.x[:,0], dtype=torch.bool)
                batch_mask[node_type][:batch_size] = True
        return batch_mask
                
    def combine_masks(self, *masks, operation="and"):
        if len(masks) == 0:
            return {}
        result_node_types = set(masks[0].keys())
        for mask in masks:
            if operation == "and":
                result_node_types.intersection_update(mask.keys())
            elif operation == "or":
                result_node_types = result_node_types.union(mask.keys())
        result_mask = {node_type: node_mask for node_type, node_mask in masks[0].items() 
                       if node_type in result_node_types}
        for mask in masks:
            for node_type in result_node_types:
                if node_type not in result_mask:
                    result_mask[node_type] = mask[node_type]
                if operation == "and":
                    result_mask[node_type] = result_mask[node_type] & mask[node_type]
                if operation == "or":
                    result_mask[node_type] = result_mask[node_type] | mask[node_type]
        return result_mask

    def _make_data_iterator(self, data, use_dataloader, batch_size=None, shuffle=None, mask=None, drop_last=False, max_degree=-1):
        dataset = self.get_data(data)
        if use_dataloader:
            if mask is None:
                mask = self.get_mask(dataset, "all", 1.0)
            assert(batch_size is not None)
            assert(shuffle is not None)
            assert(mask is not None)
            assert(max_degree is not None)
            try:
                neighbor_depth = self.model.neighbor_depth
            except AttributeError as ae:
                self.logger.critical("Model must provide its neighbor_depth as an attribute.")
                raise ae
            dataloaders = []
            for node_type, mask_array in mask.items():
                edge_sample = {edge: [max_degree]*neighbor_depth for edge in dataset.edge_types}
                this_mask = node_type, mask_array
                dataloader = torch_geometric.loader.NeighborLoader(
                    dataset._hetero_data, num_neighbors=edge_sample, 
                    input_nodes=this_mask, batch_size=batch_size, shuffle=shuffle,
                    drop_last=drop_last)
                dataloaders.append(dataloader)
        else:
            dataloader = [dataset]
            dataloaders = [dataloader]
        class MultiIterChain:
            def __init__(self, iterables):
                self.iterables = iterables
            def __iter__(self):
                # Create a new iterator every time __iter__ is called
                return chain.from_iterable(self.iterables)
        return MultiIterChain(dataloaders)

    def train_model(self, epochs=100, use_DataLoader=True, batch_size=8192,
                    shuffle=True, notify=10, validation=False, patience=10, plot=True):
        """
        Runs a training loop to train the underlying model on the training graph(s).

        Parameters
        ----------
        epochs : int, default=100
            the number of training loops/epochs
        use_DataLoader : boolean, default=True
            whether a dataloader should be used.
        batch_size : int, default=8192
            the number of samples to batch for dataloader
        shuffle : boolean, default=True
            whether the DataLoader should shuffle the samples before batching
        notify : int, default=10
            how often to log epoch losses (period)
        validation : [False, "view", "save_best", "save_load_best"], default=False
            whether to create a validation set and what to do with it
        patience : int, default=10
            how many epochs to wait for validation loss improvement before termination
        plot : boolean, default=True
            whether to plot the loss across epochs
        """
        assert(validation in [False, "view", "save_best", "save_load_best"])
        # Prepare model
        model = self.model
        model.train()
        optimizer = self._make_optimizer()
        # Initialize training data
        train_data = self.train_data
        train_mask = self.get_mask(train_data, "train", 1.0, include_non_targets=False)
        dataloader = self._make_data_iterator("train", use_DataLoader, 
                    batch_size=batch_size, shuffle=shuffle, mask=train_mask, max_degree=-1)
        # Initialize validation data
        if validation:
            val_data = self.val_data
            val_mask = self.get_mask(val_data, "val", 1.0, include_non_targets=False)
            valdataloader = self._make_data_iterator("val", use_DataLoader, 
                    batch_size=batch_size, shuffle=shuffle, mask=val_mask,
                    drop_last=False, max_degree=-1)
            best_val = float('inf')
            best_model = copy.deepcopy(self.model)
            patience_counter = 0
        # Other trackers
        trackdatadicts = {
            # "train_all": {
            #     "data": "test",
            #     "mask": self.get_mask(self.test_data, "train", 1.0, include_non_targets=False),
            # },
            # "test_benign": {
            #     "data": "test",
            #     "mask": self.get_mask(self.test_data, "test_benign", 1.0, include_non_targets=False),
            # },
            # "test_malicious": {
            #     "data": "test",
            #     "mask": self.get_mask(self.test_data, "malicious", 1.0, include_non_targets=False)
            # }
        }
        # Initialize output
        train_losses = []
        all_losses = {"train": train_losses}
        if validation:
            val_losses = []
            all_losses["val"] = val_losses
        if plot:
            self._plot_init()
        # Begin main training loop
        for epoch in range(epochs):
            model.train()
            # Initialize loop variables
            train_epoch_losses = []
            batch_count = 0
            for model_batch in dataloader:
                # Main training operations
                optimizer.zero_grad()
                model_out = self.model_call(model_batch)
                loss_mask = self.get_mask(model_batch, "train", 1.0)
                if use_DataLoader:
                    batch_mask = self.get_batch_mask(model_batch)
                    loss_mask = self.combine_masks(loss_mask, batch_mask)
                loss = self.model_loss(model_batch, model_out, loss_mask)
                loss.backward()
                optimizer.step()
                # Store outputs
                train_epoch_losses.append(loss)
                batch_count += 1
            # Store outputs
            train_epoch_average_loss = torch.stack(train_epoch_losses).mean(dim=0).item()
            train_losses.append(train_epoch_average_loss)
            # Send outputs
            if (epoch == 0):
                self.logger.info(f'Training batch_count is {batch_count}.')
            if (epoch == 0) or (epoch % notify == notify - 1):
                self.logger.info(f'Epoch {epoch+1}, Loss: {train_epoch_average_loss}')
            else:
                self.logger.trace(f'Epoch {epoch+1}, Loss: {train_epoch_average_loss}')
            # Begin validation 
            if validation:
                model.eval()
                # Initialize validation variables
                val_losses_epoch = []
                val_batch_count = 0
                # Begin validation loop
                for val_batch in valdataloader:
                    # Main validation operations
                    with torch.no_grad():
                        model_out = self.model_call(val_batch)
                    loss_mask = self.get_mask(val_batch, "all", 1.0)
                    if use_DataLoader:
                        batch_mask = self.get_batch_mask(val_batch)
                        loss_mask = self.combine_masks(loss_mask, batch_mask)
                    with torch.no_grad():
                        loss = self.model_loss(val_batch, model_out, mask=loss_mask, aggr="mean")
                    # Store outputs
                    val_losses_epoch.append(loss)
                    val_batch_count += 1
                # Store epoch outputs
                avg_val_loss = torch.stack(val_losses_epoch).mean(dim=0).item()
                val_losses.append(avg_val_loss)
                # Send outputs
                if epoch == 0:
                    self.logger.info(f'Validation batch count is {val_batch_count}')
                if (epoch == 0) or (epoch % notify == notify - 1):
                    self.logger.info(f'Epoch {epoch+1}, Val_loss: {avg_val_loss}')
                else:
                    self.logger.trace(f'Epoch {epoch+1}, Val_loss: {avg_val_loss}')
                # Validation model saving
                decider_val_loss = avg_val_loss
                patience_counter += 1
                if (validation == "save_best" or validation == "save_load_best") and decider_val_loss < best_val:
                    # Save the new model since it's better
                    self.logger.debug(f"Model with loss {decider_val_loss} improved from {best_val}. saving new best model.")
                    best_val = decider_val_loss
                    best_model = copy.deepcopy(self.model)
                    patience_counter = 0
                elif validation == "save_load_best" and decider_val_loss >= best_val and patience_counter != patience:
                    # Don't save or load model since it's not better
                    self.logger.debug(f"Model with loss {decider_val_loss} did not improve from {best_val}. patience = {patience_counter}.")
                elif validation == "save_load_best" and decider_val_loss >= best_val and patience_counter == patience:
                    # Terminate learning because model has stopped improving. Load best model
                    self.logger.info(f"Model with loss {decider_val_loss} did not improve from {best_val}. patience = {patience_counter}. "
                                     f"STOPPING NOW. Loading best model with loss {decider_val_loss}.")
                    decider_val_loss = avg_val_loss
                    self.model = copy.deepcopy(best_model)
                    break
            # Begin tracking loss calculations
            model.eval()
            # Initialize tracking variables
            track_epoch_avgs = {}
            track_batch_counts = {}
            # Begin looping through tracking types
            for track_type_name, track_dict in trackdatadicts.items():
                track_data = track_dict["data"]
                track_dataset = self.get_data(track_data)
                track_mask = track_dict["mask"]
                trackdataloader = self._make_data_iterator(
                    data=track_data, use_dataloader=use_DataLoader,
                    batch_size=batch_size, shuffle=shuffle, mask=track_mask,
                    drop_last=False, max_degree=-1
                )
                # Initialize preloop variables
                track_type_losses = []
                track_batch_count = 0
                # Begin track loop
                for track_batch in trackdataloader:
                    # Main track operations
                    with torch.no_grad():
                        model_out = self.model_call(track_batch)
                    loss_mask = self.get_mask(track_batch, "all", 1.0)
                    if use_DataLoader:
                        batch_mask = self.get_batch_mask(track_batch)
                        loss_mask = self.combine_masks(loss_mask, batch_mask)
                    with torch.no_grad():
                        loss = self.model_loss(track_batch, model_out, mask=loss_mask, aggr="mean")
                    # Store outputs
                    track_type_losses.append(loss)
                    track_batch_count += 1
                # Store epoch outputs
                track_batch_counts[track_type_name] = track_batch_count
                avg_track_loss = torch.stack(track_type_losses).mean(dim=0).item()
                track_epoch_avgs[track_type_name] = avg_track_loss
                # Store overall outputs
                node_track_losses = all_losses.get(track_type_name, [])
                node_track_losses.append(avg_track_loss)
                all_losses[track_type_name] = node_track_losses
            # Send outputs
            if epoch == 0:
                self.logger.info(f'Tracking batch counts are {track_batch_counts}')
            if (epoch == 0) or (epoch % notify == notify - 1):
                self.logger.info(f'Epoch {epoch+1}, Track_loss: {track_epoch_avgs}')
            else:
                self.logger.trace(f'Epoch {epoch+1}, Track_loss: {track_epoch_avgs}')
            if plot:
                self._plot_update(epoch, all_losses)

    def model_call_loss(self, data, use_DataLoader=True, batch_size=8192,
                        shuffle=True, mask=None, notify=10, filter=True):
        model = self.model
        model.eval()
        dataset = self.get_data(data)
        if mask is None:
            mask = self.get_mask(dataset, data, 1.0)
        dataloader = self._make_data_iterator(data, use_DataLoader, batch_size=batch_size,
                                              shuffle=shuffle, mask=mask, drop_last=False, 
                                              max_degree=-1)
        train_losses = {node_type: torch.zeros_like(x[:,0]) for node_type, x in dataset.x_dict.items() if node_type in mask.keys()}
        for i, batch in enumerate(dataloader):
            if i % notify == 0:
                self.logger.info(f"Calculating loss for batch {i+1}")
            with torch.no_grad():
                model_out = self.model_call(batch)
            loss_mask = self.get_mask(batch, data, 1.0)
            if use_DataLoader:
                batch_mask = self.get_batch_mask(batch)
                loss_mask = self.combine_masks(loss_mask, batch_mask)
            with torch.no_grad():
                loss = self.model_loss_nodes(batch, model_out, mask=loss_mask)
            for node_type, loss_vals in loss.items():
                if use_DataLoader:
                    batch_idx = batch[node_type].n_id
                    for i, loss_value in enumerate(loss_vals):
                        original_index = batch_idx[i].item()
                        train_losses[node_type][original_index] = loss_value
                else:
                    train_losses[node_type] = loss_vals
        if filter:
            for node_type, loss_vals in train_losses.items():
                train_losses[node_type] = loss_vals[mask[node_type]]
        return train_losses 

    def save_visual_loss(self, name=None, train=True, node_start = None, jumps = 10):
        """
        Save a visualization of a subset of the nodes in the network of a model.
        
        Includes all nodes with a path to the start node of or below a specified length.
        Individual node loss is provided and represented by saturation.

        Parameters
        ----------
        name : str
            the file location name for the html graph to be saved
        train : boolean
            whether to use the train data or the test
        node_start : tuple[str, int]
            the starting node that defines the region to include
        jumps : int
            the maximum path distance between nodes and the graph center           
        """
        import networkx as nx
        from pyvis.network import Network
        from torch_geometric.utils import to_networkx
        from torch_geometric.data import HeteroData
        from collections import deque
        from numpy.random import default_rng

        def nodes_within_n_jumps(hetero_data, start_node_type, start_node_id, n):
            # Keep track of visited nodes to avoid revisiting them
            visited = set()
            # Queue for BFS: stores tuples of (node_type, node_id, level)
            queue = deque([(start_node_type, start_node_id, 0)])
            # Add the start node to the visited set
            visited.add((start_node_type, start_node_id))

            # Store nodes within n jumps
            nodes_within_n = set()

            # Perform BFS up to n levels
            while queue:
                current_node_type, current_node_id, current_level = queue.popleft()
                
                # Add the current node to the result set if within n jumps
                if current_level <= n:
                    nodes_within_n.add((current_node_type, current_node_id))

                # Stop the search after n levels
                if current_level == n:
                    continue

                # Get all connected edges and neighbors for the current node
                for edge_type in hetero_data.edge_types:
                    if edge_type[0] == current_node_type:
                        # Get the edge indices for the current node type
                        edge_index = hetero_data[edge_type].edge_index
                        # Find indices of edges that involve the current node
                        edge_mask = edge_index[0] == current_node_id
                        # Get neighboring node IDs
                        neighbors = edge_index[1][edge_mask]
                        
                        # Add neighbors to the queue if they haven't been visited
                        for neighbor_id in neighbors.tolist():
                            neighbor_node = (edge_type[2], neighbor_id)
                            if neighbor_node not in visited:
                                visited.add(neighbor_node)
                                queue.append((edge_type[2], neighbor_id, current_level + 1))
                    if edge_type[2] == current_node_type:
                        # Get the edge indices for the current node type
                        edge_index = hetero_data[edge_type].edge_index
                        # Find indices of edges that involve the current node
                        edge_mask = edge_index[1] == current_node_id
                        # Get neighboring node IDs
                        neighbors = edge_index[0][edge_mask]
                        
                        # Add neighbors to the queue if they haven't been visited
                        for neighbor_id in neighbors.tolist():
                            neighbor_node = (edge_type[0], neighbor_id)
                            if neighbor_node not in visited:
                                visited.add(neighbor_node)
                                queue.append((edge_type[0], neighbor_id, current_level + 1))

            return nodes_within_n

        def get_edge_pairs_from_node_set(hetero_data, nodes):
            edge_pairs = []

            # Iterate over all edge types in the hetero_data object
            for edge_type in hetero_data.edge_types:
                # Unpack source and destination types from the edge_type
                src_type, _, dst_type = edge_type
                # Get the edge indices for the current edge type
                edge_index = hetero_data[edge_type].edge_index

                # Find all edges where both source and destination nodes are in the nodes set
                for src_node_id, dst_node_id in edge_index.t().tolist():
                    # Check if both nodes are in the nodes_within_n sets
                    if src_node_id in nodes.get(src_type, []) and dst_node_id in nodes.get(dst_type, []):
                        # If both nodes are in the set, add the edge pair to the list
                        edge_pairs.append(((src_type, src_node_id), (dst_type, dst_node_id)))

            return edge_pairs

        rng = default_rng()
        data: HeteroData = self.train_data if train else self.test_data
        node_types = data.node_types
        edge_types = data.edge_types
        num_node_types = len(node_types)
        num_edge_types = len(edge_types)
        node_type_hues = {nt: int(360 / num_node_types * i) for i, nt in enumerate(node_types)}
        edge_type_hues = {et: 0 for i, et in enumerate(edge_types)}

        if node_start is None:
            nt = rng.choice(node_types)
            ni = rng.integers(0, data[nt].x.size(0))
            node_start = (nt, ni)
        if name is None:
            name = 'loss_graph.html'

        out = self.model_call(data)
        loss = self.model_loss_nodes(data, out)

        nodes = nodes_within_n_jumps(data, node_start[0], node_start[1], jumps)
        
        nodes_dict = {}
        for node_type, node_id in nodes:
            nodes_dict[node_type] = nodes_dict.get(node_type, [])
            nodes_dict[node_type].append(node_id)
        
        edges = get_edge_pairs_from_node_set(data, nodes_dict)

        net = Network(notebook=True, directed=True)
        global_node_id_mapping = {}
        global_id_counter = 0

        max_loss = float('-inf')
        min_loss = float('inf')
        for node_type, node_id in nodes:
            node_loss = loss[node_type][node_id]
            max_loss = max(node_loss, max_loss)
            min_loss = min(node_loss, min_loss)

        for node_type, node_id in nodes:
            node_loss = loss[node_type][node_id] 
            hue = node_type_hues.get(node_type, 0)
            saturation = int((node_loss - min_loss) / (max_loss - min_loss) * -90 + 100)
            brightness = 50 if node_type in node_type_hues else 0
            color = f"hsl({hue}, {saturation}%, {brightness}%)"

            # Create a unique global ID for each node
            global_node_id = f'{node_type}_{node_id}'
            global_node_label = f"{node_type},{node_id},{node_loss:.3f}"
            global_node_id_mapping[global_node_id] = global_id_counter

            # Add nodes to PyVis network with color
            net.add_node(global_id_counter, label=global_node_label, color=color)

            # Increment the global ID counter
            global_id_counter += 1

        for (src_type, src_id), (dst_type, dst_id) in edges:
            source_global_id = global_node_id_mapping[f"{src_type}_{src_id}"]
            dest_global_id = global_node_id_mapping[f"{dst_type}_{dst_id}"]
            net.add_edge(source_global_id, dest_global_id, color='black')

        net.show_buttons(filter_=['physics'])
        net.toggle_physics(True)

        net.show(name)

    def data_dataframe(self, node_type, data="train"):
        dataset = self.get_data(data)
        result_values = dataset[node_type].x.cpu().detach().numpy()
        names = dataset[node_type].x_name
        return pd.DataFrame(result_values, columns=names)


class HeteroNodeEncodingTask(HeteroGraphTask):
    """ Represents a task that produces an encoding for its data input. """
    def __init__(self, model, train_data, val_data=None, test_data=None, target_nodes=None, logger=logger):
        super().__init__(model=model, train_data=train_data, val_data=val_data, 
                         test_data=test_data, target_nodes=target_nodes, logger=logger)

    def model_encode(self, batch):
        x = batch.x_dict
        edge_index = batch.edge_index_dict
        edge_attr = batch.edge_attr_dict

        encoding = self.model.encode(x, edge_index, edge_attr)
        return encoding

    def model_decode(self, encoding):
        decoding = self.model.decode(encoding)
        return decoding


class HeteroNodeAutoEncodingTask(HeteroNodeEncodingTask):
    """ Represents a task that autoencodes node features. """
    def __init__(self, model, train_data, autoencoder_loss_functions, 
                 autoencoder_reconstruction_functions,  
                 val_data=None, test_data=None, target_nodes=None, 
                 logger=logger):
        super().__init__(model=model, train_data=train_data, val_data=val_data, 
                         test_data=test_data, target_nodes=target_nodes,
                         logger=logger)
        self.autoencoder_loss_functions = autoencoder_loss_functions
        self.autoencoder_reconstruction_functions = autoencoder_reconstruction_functions

    def model_loss(self, batch, model_out, mask, aggr="mean"):
        """ Calculate loss function for nodes and aggregate them. """
        losses = []
        for node_type, node_mask in mask.items():
            all_nodes_data_out = batch[node_type].x
            node_features = all_nodes_data_out.size(dim=1)
            all_nodes_model_out = model_out[node_type][:,:node_features]
            data_out = all_nodes_data_out[node_mask]
            model_out = all_nodes_model_out[node_mask]
            loss_function = self.autoencoder_loss_functions[node_type]
            if aggr == "mean" or aggr == "sum":
                loss = loss_function(model_out, data_out, aggr=aggr)
            else:
                loss = loss_function(model_out, data_out, aggr="none")
            losses.append(loss)
        result_loss = torch.stack(losses)
        if aggr == "mean":
            result_loss = result_loss.mean(dim=0)
        if aggr == "median":
            result_loss = torch.stack(losses).median()
        if aggr == "max":
            result_loss = torch.stack(losses).max()
        if aggr == "low99":
            result_loss = torch.stack(losses).flatten()
            result_loss = result_loss.topk(k=int(0.99 * result_loss.size(dim=0)), largest=False).values.mean(dim=0)
        if aggr == "top01":
            result_loss = torch.stack(losses).flatten()
            result_loss = result_loss.topk(k=int(0.01 * result_loss.size(dim=0))).values.mean(dim=0)
        return result_loss

    def model_loss_nodes(self, batch, model_out, mask):
        """ Calculates loss function for each node of each node type. """
        losses = {}
        for node_type, node_mask in mask.items():
            all_nodes_data_out = batch[node_type].x
            node_features = all_nodes_data_out.size(dim=1)
            all_nodes_model_out = model_out[node_type][:,:node_features]
            data_out = all_nodes_data_out[node_mask]
            model_out = all_nodes_model_out[node_mask]
            loss_function = self.autoencoder_loss_functions[node_type]
            loss = loss_function(model_out, data_out, aggr="none")
            losses[node_type] = loss
        result_loss = losses
        return result_loss

    def transformed_preds(self, data="train"):
        pred = self.preds(data)
        node_preds = {}
        for node_type, transform_function in self.autoencoder_reconstruction_functions.items():
            node_out = pred[node_type]
            result = transform_function(node_out)
            node_preds[node_type] = result
        return node_preds
    
    def pred_dataframe(self, node_type, data="train"):
        import pandas as pd
        result_values = self.transformed_preds(data)[node_type].cpu().detach().numpy()
        names = self.train_data[node_type].x_name
        return pd.DataFrame(result_values, columns=names)
    
    def save_reconstruction_hists(self, timestamp=None, log=False):
        now = timestamp
        if now is None:
            now = datetime.now()
        formatted_time = now.strftime('%Y-%m-%d_%H-%M-%S')
        
        types = {"train": self.train_data, "val": self.val_data, "test": self.test_data}
        for type_name, type_data in types.items():
            nodes = list(type_data.node_types)
            for node_type in nodes:
                node_data = self.data_dataframe(node_type, type_name == "train")
                node_pred = self.pred_dataframe(node_type, type_name == "train") # TODO, fix to work with all 3 types
                col_names = node_data.columns                
                for col_name in col_names:
                    fname = f"{type_name}_{self.name}_{node_type}_{col_name}_{formatted_time}"
                    ext = "png"
                    fname = f"{fname}.{ext}"
                    foldername = f"figs/{type_name}/{self.name}/{node_type}"
                    os.makedirs(foldername, exist_ok=True)

                    series_data = node_data[col_name]
                    series_pred = node_pred[col_name]
                    series_diff = series_pred - series_data
                    # all_df = pd.DataFrame({"data": series_data, "pred": series_pred, "diff": series_diff})
                    bins = np.histogram_bin_edges(pd.concat([series_data, series_pred, series_diff]), bins=20)

                    plt.hist(series_data, alpha=1.0, log=log, bins=bins, histtype='step', label='data', stacked=True, color='green')
                    plt.hist(series_pred, alpha=0.5, log=log, bins=bins, histtype='step', label='pred', stacked=True, color='blue')
                    plt.hist(series_diff, alpha=0.5, log=log, bins=bins, histtype='step', label='diff', stacked=True, color='red')
                    plt.legend()
                    plt.savefig(fname)
                    plt.clf()
                    plt.close()


class HeteroGraphNoSelfTask(HeteroNodeAutoEncodingTask):
    """ 
    Represents a task that requires HeteroData graph data and trains/tests without
    it's own features.
    """
    def __init__(self, model, train_data, autoencoder_loss_functions, 
                 autoencoder_reconstruction_functions,  
                 val_data=None, test_data=None, target_nodes=None,
                 logger=logger):
        super().__init__(model=model, train_data=train_data, 
                         autoencoder_loss_functions=autoencoder_loss_functions, 
                 autoencoder_reconstruction_functions=autoencoder_reconstruction_functions,  
                 val_data=val_data, test_data=test_data, target_nodes=target_nodes,logger=logger)
        
    def model_call(self, batch, hide_mask=None):
        batch_mask = self.get_batch_mask(batch)
        mask = batch_mask
        if hide_mask:
            mask = self.combine_masks(mask, batch_mask, operation="or")
        x = batch.x_dict
        edge_index = batch.edge_index_dict
        edge_attr = batch.edge_attr_dict

        for node_type, node_mask in mask.items():
            hidden_x = x[node_type].clone()
            hidden_x[node_mask] = 0.0
            x[node_type] = hidden_x

        return self.model(x, edge_index, edge_attr)


class HeteroEdgePredictionTask(HeteroNodeEncodingTask):
    """ Represents a task that encodes edge connections. """
    def __init__(self, model, train_data, reconstructed_edge_types, 
                 negative_edge_ratio=5, val_data=None, test_data=None,
                 logger=logger):
        super().__init__(model=model, train_data=train_data, val_data=val_data, test_data=test_data, logger=logger)
        self.reconstructed_edge_types = reconstructed_edge_types
        self.negative_edge_ratio = negative_edge_ratio
    
    def model_edge_loss(self, batch, model_encoding):
        """ Calculate loss function for each edge reconstruction and add them. """
        
        def generate_random_edges(num_nodes_1, num_nodes_2, positive_edges, negative_edge_count):
            device = positive_edges.get_device()
            random_edges = []
            # Generate a random edge
            half = int(negative_edge_count / 2)
            node1_index = torch.randint(1, num_nodes_1, size=(half,)).to(device)
            node2_offset = torch.randint(1, num_nodes_2, size=(half,)).to(device)
            node1 = positive_edges[node1_index, 0]
            node2 = torch.remainder(node2_offset + positive_edges[node1_index, 1], num_nodes_2)
            this_random_edges = torch.stack([node1, node2], dim=1)
            random_edges.append(this_random_edges)
            # Generate a random edge from the other node
            other_half = negative_edge_count - half
            node1_offset = torch.randint(1, num_nodes_1, size=(other_half,)).to(device)
            node2_index = torch.randint(1, num_nodes_2, size=(other_half,)).to(device)

            node2 = positive_edges[node2_index, 1]
            node1 = torch.remainder(node1_offset + positive_edges[node2_index,0], num_nodes_1)
            this_random_edges = torch.stack([node1, node2], dim=1)
            random_edges.append(this_random_edges)
            
            return torch.cat(random_edges, dim=0)
        
        losses = []
        
        edge_types = self.reconstructed_edge_types
        negative_edge_ratio = self.negative_edge_ratio

        for send_node_type, edge_type, recieve_node_type in batch.edge_types:
            if edge_type not in edge_types:
                continue
            positive_edges = batch[send_node_type, edge_type, recieve_node_type].edge_index.t()
            positive_edges_count = batch[send_node_type, edge_type, recieve_node_type].num_edges
            negative_edge_count = int(positive_edges_count * negative_edge_ratio)
            send_embeddings = model_encoding[send_node_type]
            receive_embeddings = model_encoding[recieve_node_type]
            positive_scores = (send_embeddings[positive_edges[:, 0]] * receive_embeddings[positive_edges[:, 1]]).sum(dim=1)
            device = positive_edges.get_device()
            # Sample negative edges
            send_node_count = batch[send_node_type].num_nodes
            receive_node_count = batch[recieve_node_type].num_nodes
            negative_edges = generate_random_edges(send_node_count, receive_node_count, positive_edges, negative_edge_count)

            negative_scores = (send_embeddings[negative_edges[:, 0]] * receive_embeddings[negative_edges[:, 1]]).sum(dim=1)

            scores = torch.cat([positive_scores, negative_scores], dim=0)
            labels = torch.cat([
                torch.ones(positive_scores.size(0)),
                torch.zeros(negative_scores.size(0))
            ], dim=0).to(device)

            weight = tensor([negative_edge_ratio]).to(device)
            # Define the binary cross-entropy loss
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

            # Calculate the loss
            loss = criterion(scores, labels)

            # Backpropagate the loss
            losses.append(loss)
        average_loss = torch.stack(losses).mean(dim=0)
        return average_loss

 
class HeteroEdgeOnlyPredictionTask(HeteroEdgePredictionTask):
    """ Represents a task that encodes edge connections. """
    def __init__(self, model, train_data, reconstructed_edge_types, negative_edge_ratio=5, val_data=None, test_data=None, logger=logger):
        super().__init__(model=model, train_data=train_data, reconstructed_edge_types=reconstructed_edge_types, 
                         negative_edge_ratio=negative_edge_ratio, val_data=val_data, test_data=test_data, logger=logger)
    
    def model_loss(self, batch, model_out):
        model_encoding = model_out
        return self.model_edge_loss(batch, model_encoding)

