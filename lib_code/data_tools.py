# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from loguru import logger

import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from data.cleaning import ZeekCleaner
from torch import tensor
from torch import float32, int64
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from data.datasets import Zeek

class ZeekIds:
    """
    Represents a mapping from values across Zeek columns to unique numeric ID's

    Useful for turning a column of ex(IP addresses) into valid integer node identifiers
    for a Zeek_Graph. Can also create a new column of unique ID's.
    """
    def __init__(self, logger=logger):
        self.unique_id_map = {}
        self.inverse_map = {}
        self.logger = logger

    def fit(self, zeeks, log, column):
        """
        Create a mapping for all unique values in columns across given Zeeks.

        Parameters
        ----------
        zeeks : Zeek | list(Zeek)
            the Zeek objects to fit from
        log : str
            the name of the Zeek log to fit from
        column : str
            the name of the Zeek log's column to fit from
        """
        if isinstance(zeeks, Zeek):
            zeeks = [zeeks]
        for zeek in zeeks:
            series = zeek.get(log)[column]

            # Check that the input is a pandas Series
            if not isinstance(series, pd.Series):
                raise ValueError("Input must be a single column.")
            
            unique_values = series.unique()
            for value in unique_values:
                if value not in self.unique_id_map:
                    # Assign a new ID based on the current length of the dictionary
                    self.unique_id_map[value] = len(self.unique_id_map)
                    self.inverse_map[self.unique_id_map[value]] = value

    def transform(self, zeeks, log, column):
        """
        Creates mapped numeric columns from the values in the given columns 
        based on their numeric mapping. 
        
        Parameters
        ----------
        zeeks : Zeek | list(Zeek)
            the Zeek objects to transform
        log : str
            the name of the Zeek log to tranform
        column : str
            the name of the Zeek log's column to tranform
        
        Returns
        -------
        list(Series)
            the transformed colums
        """
        if isinstance(zeeks, Zeek):
            zeeks = [zeeks]
        results = []
        for zeek in zeeks:
            series = zeek.get(log)[column]
            # Replace values in the series with their assigned unique IDs
            results.append(series.map(self.unique_id_map))
        return results

    def reverse_transform(self, zeeks, log, column):
        """
        Creates columns from given numeric previously mapped columns 
        based on their original values.

        Parameters
        ----------
        zeeks : Zeek | list(Zeek)
            the Zeek objects to tranform
        log : str
            the name of the Zeek log to transform
        column : str
            the name of the Zeek log's column to transform

        Returns
        -------
        list(Series)
            the transformed columns
        """
        if isinstance(zeeks, Zeek):
            zeeks = [zeeks]
        results = []
        for zeek in zeeks:
            id_series = zeek.get(log)[column]
            # Replace unique IDs in the series with their original values
            results.append(id_series.map(self.inverse_map))
        return results

    def fit_transform(self, zeeks, log, column):
        """
        Create a mapping for all unique values in columns across given Zeeks. 
        Then creates mapped numeric columns from the values in the given columns 
        based on their numeric mapping. 

        Parameters
        ----------
        zeeks : Zeek | list(Zeek)
            the Zeek objects to fit and transform
        log : str
            the name of the Zeek log to fit and transform
        column : str
            the name of the Zeek log's column to fit and transform

        Returns
        -------
        list(Series)
            the transformed columns
        """
        self.fit(zeeks, log, column)
        return self.transform(zeeks, log, column)
    
    def replace(self, zeeks, log, column):
        """
        Replaces columns with their numeric mapping.

        Parameters
        ----------
        zeeks : Zeek | list(Zeek)
            the Zeek objects to transform from and replace
        log : str
            the name of the Zeek log to transform from and replace
        column : str
            the name of the Zeek log's column to transform from and replace
        """
        if isinstance(zeeks, Zeek):
            zeeks = [zeeks]
        new_vals = self.transform(zeeks, log, column)
        for new_val, zeek in zip(new_vals, zeeks):
            zeek.get(log)[column] = new_val

    def reverse_replace(self, zeeks, log, column):
        """
        Replaces transformed columns with their original unmapped values.

        Parameters
        ----------
        zeeks : Zeek | list(Zeek)
            the Zeek objects to transform from and replace
        log : str
            the name of the Zeek log to transform from and replace
        column : str
            the name of the Zeek log's column to transform from and replace
        """
        if isinstance(zeeks, Zeek):
            zeeks = [zeeks]
        old_vals = self.reverse_transform(zeeks, log, column)
        for old_val, zeek in zip(old_vals, zeeks):
            zeek.get(log)[column] = old_val

    def fit_replace(self, zeeks, log, column):
        """
        Create a mapping for all unique values in columns across given Zeeks.
        Then replaces columns with their numeric mapping.

        Parameters
        ----------
        zeeks : Zeek | list(Zeek)
            the Zeek objects to fit from and replace
        log : str
            the name of the Zeek log to fit from and replace
        column : str
            the name of the Zeek log's column to fit from and replace
        """
        if isinstance(zeeks, Zeek):
            zeeks = [zeeks]
        new_vals = self.fit_transform(zeeks, log, column)
        for new_val, zeek in zip(new_vals, zeeks):
            zeek.get(log)[column] = new_val

    def new_ids(self, zeeks, log, new_column):
        """
        Creates a column with unique values across given Zeeks.
        Then replaces new column with unique numeric mappings.
        Result is a new column with unique values.
        Note: This operation produces unique values for each Zeek given,
        but does not necessarily produce unique values if called in a 
        

        Parameters
        ----------
        zeeks : Zeek | list(Zeek)
            the Zeek objects to alter
        log : str
            the name of the Zeek log to alter
        new_column : str
            the name of the Zeek log's new numeric ID column
        """
        if isinstance(zeeks, Zeek):
            zeeks = [zeeks]
        start = len(self.unique_id_map)
        for zeek in zeeks:
            df = zeek.get(log)
            num_series = pd.Series(range(start, start+len(df))).astype("string")
            num_series.index = df.index
            start += len(num_series)
            df[new_column] = log + "_" + new_column + "_" + num_series
        self.fit_replace(zeeks, log, new_column)


class ZeekTransforms():
    """
    Collection of functions to help with handling Zeek objects.
    """
    def __init__(self, logger=logger):
        self.logger = logger

    @classmethod
    def filter_to(cls, zeeks, log, columns):
        """
        Remove unnamed columns from Zeek logs.

        Parameters
        ----------
        zeeks : Zeek | list(Zeek)
            the Zeek objects to alter
        log : str
            the name of the Zeek log to alter
        column : str
            the name of the Zeek log's column to keep
        """
        if isinstance(zeeks, Zeek):
            zeeks = [zeeks]
        for zeek in zeeks:
            df = zeek.get(log)
            keep_columns = columns
            zeek.set(log, df[keep_columns].copy())

    @classmethod
    def filter_out(cls, zeeks, log, columns):
        """
        Remove named columns from Zeek logs.

        Parameters
        ----------
        zeeks : Zeek | list(Zeek)
            the Zeek objects to alter
        log : str
            the name of the Zeek log to alter
        column : str
            the name of the Zeek log's column to remove
        """
        if isinstance(zeeks, Zeek):
            zeeks = [zeeks]
        for zeek in zeeks:
            df = zeek.get(log)
            keep_columns = [column for column in df.columns if column not in columns]
            zeek.set(log, df[keep_columns].copy())

    @classmethod
    def move_log(cls, zeeks, log, new_log):
        """
        Move log from one name to another.
        Removes initial name and adds new one.

        Parameters
        ----------
        zeeks : Zeek | list(Zeek)
            the Zeek objects to alter
        log : str
            the name of the Zeek log to move
        new_log : str
            the new name for the Zeek log
        """
        if isinstance(zeeks, Zeek):
            zeeks = [zeeks]
        for zeek in zeeks:
            df = zeek.get(log)
            zeek.set(new_log, df)
            zeek.delete_log(log)

    @classmethod
    def copy_log(cls, zeeks, log, new_log):
        """
        Copy log from one name to another.

        Parameters
        ----------
        zeeks : Zeek | list(Zeek)
            the Zeek objects to alter
        log : str
            the name of the Zeek log to copy from
        new_log : str
            the new name for the Zeek log
        """
        if isinstance(zeeks, Zeek):
            zeeks = [zeeks]
        for zeek in zeeks:
            df = zeek.get(log).copy()
            zeek.set(new_log, df)

    @classmethod
    def split_type(cls, log, type_id, delete=False):
        """
        Split one log into multiple based on the values in given column.

        Parameters
        ----------
        zeeks : Zeek | list(Zeek)
            the Zeek objects to alter
        log : str
            the name of the Zeek log source
        type_id : str
            the column name to group by
        delete : boolean, default=False
            whether to delete the original Zeek log
        """
        if isinstance(zeeks, Zeek):
            zeeks = [zeeks]
        for zeek in zeeks:
            df = zeek.get(log).copy()
            other_cols = [column for column in df.columns if column not in set([type_id])]
            gb = df.groupby(type_id)
            result = dict(tuple(gb))
            if delete:
                zeek.delete_log(log)
            for key_ in result.keys():
                new_name = f"{log}_{key_}"
                zeek.set(new_name, result[key_].copy())
    
    @classmethod
    def combine(cls, zeeks):
        """
        Combine given Zeeks into one Zeek object.

        Parameters
        ----------
        zeeks : list(Zeek)
            the Zeek objects to combine

        Returns
        -------
        Zeek
            single combined Zeek object
        """
        new_data_dict = {}
        for zeek in zeeks:
            for log, df in zeek.data.items():
                if log not in new_data_dict.keys():
                    new_data_dict[log] = df.copy()
                else:
                    old = new_data_dict[log].copy().reset_index(drop=True)
                    new = df.reset_index(drop=True)
                    if not df.columns.is_unique:
                        raise Exception(f"Columns in {log} are not unique!")
                    new_data_dict[log] = pd.concat([old, new], ignore_index=True)
        return Zeek(new_data_dict)

    @classmethod
    def remove_unsuccessful_connections(cls, zeeks):
        """
        Removes connections where the "conn_state" != "SF".
        Gathers "uid"s of "conn" logs where "conn_state" != "SF".
        Then removes all rows where "uid" is in that list.

        Parameters
        ----------
        zeeks : Zeek | list(Zeek)
            the Zeek objects to alter
        """
        if isinstance(zeeks, Zeek):
            zeeks = [zeeks]
        for zeek in zeeks:
            conn = zeek.get("conn")
            remove_rows = conn["conn_state"] != "SF"
            remove_uids = conn["uid"][remove_rows].copy()

            for key, df in zeek.data.items():
                mask = df["uid"].isin(remove_uids)
                zeek.data[key] = df[mask]

    @classmethod
    def add_other_features(cls, zc: ZeekCleaner, zeek_clean: Zeek, zeek_old: Zeek):
        """
        Adds features removed by ZeekCleaner transform back to Zeek object

        Parameters
        ----------
        zc : ZeekCleaner
            the ZeekCleaner object that was originally used to transform the data
        zeek : Zeek
            the Zeek object that was transformed to add the other features to
        """
        for log in zeek_clean.data.keys():
            transformed_df = zeek_clean.get(log)
            old_df = zeek_old.get(log)
            other_df = zc.transforms[log].other_features(old_df)
            new_df = pd.concat([transformed_df, other_df], axis=1)
            zeek_clean.set(log, new_df)

    # The process of passing the autoencoder function and reconstruction function
    # from the ZeekCleaner object should be reworked into something better if possible.
    @classmethod
    def create_autoencoder_loss_functions(cls, zc: ZeekCleaner, hd, log_conversion=None):
        """
        Creates a loss function that takes a weighted average of multiple loss functions
        depending on the transformation used during data processing.
        Tracked by name, so renaming features is not allowed.

        Parameters
        ----------
        zc : ZeekCleaner
            the ZeekCleaner object that was originally used to transform the data.
            (It is ok to use a fresh ZeekCleaner object instead, but this may not always be the case.)
        hd : HeteroData
            the heterogeneous graph to which the loss function should be applied
        log_conversion : dict[str, str], optional
            map of new node names to old log file names. 
            The model should only calculate a loss function for the given key nodes.
            The node features should be determined based on the value log.
            By default will associate all nodes with the log file that shares a name.
        
        Returns
        -------
        dict[str, func(input, target, aggr) -> tensor]
            a loss function for each node_type.
            The loss function takes the model output, the node features, and an aggregation method.
            It outputs a single value or a tensor of values depending on the aggregation input.
        """
        if log_conversion is None:
            log_conversion = {log:log for log in hd.node_types}
        transforms = zc.transforms
        result_fns = {}
        for new_log, old_log in log_conversion.items():
            node_store = hd[new_log]
            loss_fn = transforms[old_log].create_autoencoder_loss_function(node_store.x_name)
            result_fns[new_log] = loss_fn
        return result_fns

    @classmethod
    def create_reconstruction_functions(cls, zc: ZeekCleaner, hd, log_conversion=None):
        """
        Creates a node reconstruction function that takes logit predictions and produces 
        reconstructed feature predictions depending on the transformation used during data 
        processing. Tracked by name, so renaming features is not allowed.

        Parameters
        ----------
        zc : ZeekCleaner
            the ZeekCleaner object that was originally used to transform the data.
            (It is ok to use a fresh ZeekCleaner object instead, but this may not always be the case.)
        hd : HeteroData
            the heterogeneous graph input to which the reconstruction function should represent
        log_conversion : dict[str, str], optional
            map of new node names to old log file names. 
            The model should only calculate a reconstruction function for the given key nodes.
            The node features should be determined based on the value log.
            By default will associate all nodes with the log file that shares a name.
        
        Returns
        -------
        dict[str, func(input, aggr) -> tensor]
            a reconstruction function for each node_type.
            The reconstruction function takes the model output.
            It outputs a tensor of values approximating the initial input fields.
        """
        if log_conversion is None:
            log_conversion = {log:log for log in hd.node_types}
        transforms = zc.transforms
        result_fns = {}
        for new_log, old_log in log_conversion.items():
            node_store = hd[new_log]
            transforms_fn = transforms[old_log].create_reconstruction_function(node_store.x_name)
            result_fns[new_log] = transforms_fn
        return result_fns


class ZeekGraph():
    """
    Compiles node and edge features for a Zeek object. 
    Creates a into a Torch Geometric "Data" or "HeteroData" object.
    """
    def __init__(self, zeek: Zeek, logger=logger):
        """
        Initizalizes the object.

        Parameters
        ----------
        zeek : Zeek
            the Zeek object from which to create the graph
        """
        self.zeek = zeek
        self.nodes = {}
        self.edge_idxs = {}
        self.edge_attrs = {}
        self.logger = logger

    def add_nodes_from_log(self, log, node_id=None, node_data=None, node_type_name = None, meta=None):
        """
        Interpret a Zeek log as graph nodes. 

        Parameters
        ----------
        log : str
            the Zeek log source
        node_id : str, optional
            the column holding node ID's
            best if this column was set using Zeek_ID.replace
            uses index if blank
        node_data : list(str), optional
            columns with feature values
            default includes all unnasigned columns
        node_type_name : str, optional
            name of this node type
            default name is None object
        meta : list(str), optional
            columns with nonfeature metadata
            default includes no columns
        """
        df = self.zeek.get(log).copy()

        if node_id:
            if not df[node_id].is_unique:
                raise Exception("df node_ids must be unique!")
            df = df.set_index(node_id, drop=False)
        if meta is None:
            meta = []
        if node_data is None:
            node_data = [column for column in df.columns if column not in set([node_id]).union(set(meta))]

        result = df[node_data]
        result_meta = df[meta]
        names = pd.Series(data=result.index)

        if node_type_name in self.nodes.keys():
            old_names, old_results, old_result_meta = self.nodes[node_type_name]
            names = pd.concat([old_names, names], ignore_index=True)
            result = pd.concat([old_results, result], ignore_index=True)
            result_meta = pd.concat([old_result_meta, result_meta], ignore_index=True)
            if not result.index.is_unique:
                raise Exception("Concattenated nodes must have unique node_ids")
            assert(len(names.values.duplicated()) == 0)
        self.nodes[node_type_name] = (names, result, result_meta)
   
    def add_edges_from_log(self, log:str, start_id: str, end_id: str, edge_data: list[str] = None, edge_type_name: str=None, node_type_start=None, node_type_end=None, meta=None):
        """
        Interpret a Zeek log as graph edges. 

        Parameters
        ----------
        log : str
            the Zeek log source
        start_id : str
            the column holding edge start node ID's
            must match node_id column value for nodes
        end_id : str
            the column holding edge end node ID's
            must match node_id column value for nodes
        edge_data : list(str), optional
            columns with feature values
            default includes all unnasigned columns
        edge_type_name : str, optional
            name of this edge type
            default name is None object
        node_type_start : str, optional
            name of start node type
            default name is None object
        node_type_end : str, optional
            name of end node type
            default name is None object
        meta : list(str), optional
            columns with nonfeature metadata
            default includes no columns
        """
        df = self.zeek.get(log).copy()
        
        names = df[[start_id, end_id]]
        names = names.rename(columns={start_id: "start", end_id: "end"})
        result_ids = df[[start_id] + [end_id]]
        result_ids = result_ids.rename(columns={start_id: "start", end_id: "end"})
        
        if (node_type_start, edge_type_name, node_type_end) in self.edge_idxs.keys():
            old_names, old_result_ids = self.edge_idxs[node_type_start, edge_type_name, node_type_end]
            names = pd.concat([old_names, names], ignore_index=True)
            result_ids = pd.concat([old_result_ids, result_ids], ignore_index=True)
        self.edge_idxs[node_type_start, edge_type_name, node_type_end] = (names, result_ids)
        
        names = df[[start_id, end_id]]

        if meta is None:
            meta = []
        if edge_data is None:
            edge_data = [column for column in df.columns if column not in set([start_id]).union(set([end_id])).union(set(meta))]
        if edge_type_name is None:
            edge_type_name = log
        result_attrs = df[edge_data]
        result_attrs_meta = df[meta]
        if (node_type_start, edge_type_name, node_type_end) in self.edge_attrs.keys():
            old_names, old_result_attrs, old_result_attrs_meta = self.edge_attrs[node_type_start, edge_type_name, node_type_end]
            names = pd.concat([old_names, names], ignore_index=True)
            result_attrs = pd.concat([old_result_attrs, result_attrs], ignore_index=True)
            result_attrs_meta = pd.concat([old_result_attrs_meta, result_attrs_meta], ignore_index=True)
        self.edge_attrs[node_type_start, edge_type_name, node_type_end] = (names, result_attrs, result_attrs_meta)

    class _CombinedDFRowsIter():
        """
        Creates an iterator that returns rows across multiple zeek logs.
        Order specified by column values. 
        """
        def __init__(self, zeek, log_columns_dict_order, presorted=False):
            """
            Creates the iterator.
            Expects the dataframes to be previously sorted.
            
            Paramters
            ---------
            zeek : Zeek
                the Zeek object source
            log_columns_dict_order : dict(str, str)
                mapping from log name to column name to use for determining order 
            """
            log_dict = zeek.data
            if not presorted:
                log_dict = {log: zeek.get(log).copy() for log in log_columns_dict_order.keys()}
                for log, df in log_dict.items():
                    order_col = log_columns_dict_order[log]
                    df.sort_values(by=order_col, ascending=True, inplace=True)
            self.iterframes = [log_dict[log].iterrows() for log in log_columns_dict_order.keys()]
            self.logs = [log for log in log_columns_dict_order.keys()]
            self.order_cols = [order_col for order_col in log_columns_dict_order.values()]
            self.all = [(iterframe, log, order_col) for (iterframe, log, order_col) in zip(self.iterframes, self.logs, self.order_cols)]
            self.next_items = []
            # Initialize the next item for each dataframe
            for df_iter, log, col in self.all:
                try:
                    self.next_items.append((next(df_iter), log, col))
                except StopIteration:
                    # Append None if the dataframe is empty
                    self.next_items.append(None)

        def __iter__(self):
            return self

        def __next__(self):
            # Filter out dataframes that are already fully iterated
            valid_items = [(i, item) for i, item in enumerate(self.next_items) if item is not None]

            if not valid_items:
                raise StopIteration

            # Find the dataframe with the minimum "time"
            min_time_index, min_time_tuple = min(valid_items, key=lambda x: x[1][0][1][x[1][2]])

            # Get the next item from the dataframe that provided the min_time_row
            try:
                self.next_items[min_time_index] = (next(self.all[min_time_index][0]), self.all[min_time_index][1], self.all[min_time_index][2])
            except StopIteration:
                # If the dataframe is exhausted, set its next item to None
                self.next_items[min_time_index] = None

            # Return the row with the minimum "time" value
            return min_time_tuple

    def add_ordered_edges_between_logs(self, log_columns_dict_order, log_columns_dict_node, log_columns_dict_groups, log_dict_node_names=None, edge_name="time", include_duration=True):
        """
        Adds edges between node logs between matching rows ordered by a column value.

        Used to product temporal edges in dataset.

        Parameters
        ----------
        log_columns_dict_order : dict(str, str)
            mapping from log name to column name to use for ordering values
            ex: {"conn_send": "ts", "conn_receive": "ts}
        log_columns_dict_node : dict(str, str)
            mapping from log name to column name to use for node ID values
            ex: {"conn_send": "sender_node", "conn_receive": "receiver_node"}
        log_columns_dict_ids : dict(str, str)
            mapping from log name to column name to use for node groupings
            ex: {"conn_send": "id.orig_h", "conn_receive": "id.resp_h"}
        log_dict_node_names : dict(str, str)
            mapping from log name to node type name
            ex: {"conn_send": "senders", "conn_receive": "receivers"}
            default maps node name to the log name itself
        edge_name : str, default="time"
            name of this edge_type
        include_duration : boolean, default=True
            whether the difference used for determining order should be included as an edge feature
        """
        i = 0
        edges = {}
        last = {}
        for ((index, row), log, order_col) in self._CombinedDFRowsIter(self.zeek, log_columns_dict_order):
            i += 1
            logger.trace(f"Adding {i}th ordered edge.") if i % 10000 == 0 else None
            if log in log_columns_dict_groups.keys():
                end_node_name = log_dict_node_names[log] if log_dict_node_names else log
                id_cols = log_columns_dict_groups[log]
                node_col = log_columns_dict_node[log]
                node = row[node_col]
                for id_col in id_cols:
                    id = row[id_col]
                    time_since = -1.0
                    this_time = row[order_col]
                    if id in last.keys():
                        last_node, last_time, last_log = last[id]
                        start_node_name = log_dict_node_names[last_log] if log_dict_node_names else last_log
                        time_since = (this_time - last_time)
                        if isinstance(time_since, pd.Timedelta):
                            time_since = time_since.total_seconds()
                        else:
                            time_since = float(time_since)
                        time_edge_id = {"start": last_node, "end": node}
                        time_edge_attr = {"duration": time_since} if include_duration else {}
                        edge_id_list, edge_attb_list = edges.get((start_node_name, edge_name, end_node_name), ([],[]))
                        edge_id_list.append(time_edge_id)
                        edge_attb_list.append(time_edge_attr)
                        edges[start_node_name, edge_name, end_node_name] = edge_id_list, edge_attb_list
                    last[id] = (node, this_time, log)
        for (node_type_start, edge_type, node_type_end), (time_edge_ids, time_edge_attrs) in edges.items():
            time_edge_ids = pd.DataFrame(time_edge_ids)
            time_edge_attrs = pd.DataFrame(time_edge_attrs)
            names = time_edge_ids
            result_ids = time_edge_ids
            if (node_type_start, edge_type, node_type_end) in self.edge_idxs.keys():
                old_names, old_result_attrs = self.edge_idxs[node_type_start, edge_type, node_type_end]
                names = pd.concat([old_names, names], ignore_index=True) 
                result_ids = pd.concat([old_result_attrs, result_ids], ignore_index=True)
            self.edge_idxs[node_type_start, edge_type, node_type_end] = names, result_ids
            names = time_edge_ids
            result_attrs = time_edge_attrs
            result_attrs_meta = time_edge_attrs[[]].copy()
            if (node_type_start, edge_type, node_type_end) in self.edge_attrs.keys():
                old_names, old_result_attrs, old_result_attrs_meta = self.edge_attrs[node_type_start, edge_type, node_type_end]
                names = pd.concat([old_names, names], ignore_index=True)
                result_attrs = pd.concat([old_result_attrs, result_attrs], ignore_index=True)
                result_attrs_meta = pd.concat([old_result_attrs_meta, result_attrs_meta], ignore_index=True)
            self.edge_attrs[node_type_start, edge_type, node_type_end] = names, result_attrs, result_attrs_meta

    def to_torch_graph(self, undirected=True):
        """
        Create a torch_geometric.data.Data graph object

        Parameters
        ----------
        undirected : boolean, default=True
            whether to create an undirected graph from the data
        
        Returns
        -------
        torch_geometric.data.Data
            graph of this object
        """
        if len(self.nodes) == 0:
            raise Exception("Run \"add_nodes_from_log\" to add nodes to the graph.")
        if len(self.edge_idxs) == 0:
            raise Exception("Run \"add_edges_from_logs\" to add nodes to the graph.")
        node_name_lookups = {}
        if (len(self.nodes) == 1) and (len(self.edge_idxs) == 1):
            node_type, (x_name, x, x_meta) = list(self.nodes.items())[0]
            x = tensor(x.values).to(float32)
            node_name_lookups[node_type] = {value: index for index, value in x_name.items()}
            (node_type_start, edge_name, node_type_end), (edge_index_name, edge_index) = list(self.edge_idxs.items())[0]
            edge_index["start"] = edge_index["start"].map(node_name_lookups[node_type_start])
            edge_index["end"] = edge_index["end"].map(node_name_lookups[node_type_end])
            edge_index = edge_index[["start", "end"]]
            edge_index = tensor(edge_index.values).to(int64)
            edge_name, (edge_attr_name, edge_attr, edge_attr_meta) = list(self.edge_attrs.items())[0]
            edge_attr = tensor(edge_attr.values).to(float32)
            result = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        else:
            raise ValueError("Graph has more than one type of edge or more than one type of node. "
                             "Use .to_torch_hetero_graph instead.")
        if undirected:
            result.edge_index = to_undirected(result.edge_index, result.edge_attr)
        return result
    
    def to_torch_hetero_graph(self, reverse_edges=True):
        """
        Create a torch_geometric.data.Data graph object

        Parameters
        ----------
        reverse_edges : boolean, default=True
            whether to create distinct reverse edges for each edge in the data
        
        Returns
        -------
        torch_geometric.data.HeterData
            graph of this object
        """
        if len(self.nodes) == 0:
            raise Exception("Run \"add_nodes_from_log\" to add nodes to the graph.")
        if len(self.edge_idxs) == 0:
            raise Exception("Run \"add_edges_from_logs\" to add edges to the graph.")
        node_name_lookups = {}
        if (len(self.nodes) == 1) and (len(self.edge_idxs) == 1):
            logger.warning("Graph has only one node and edge type. Consider creating a "
                           "normal graph using .to_torch_graph instead.")
        result = HeteroData()
        for node_type, (node_names, node_x, node_x_meta) in self.nodes.items():
            node_c = list(node_x.columns)
            node_x = tensor(node_x.values).to(float32)
            node_c_meta = list(node_x_meta.columns)
            node_x_meta = tensor(node_x_meta.values).to(float32)
            # node_x_meta = list(node_x_meta.values)
            result[node_type].x_name = node_c
            result[node_type].x = node_x                
            result[node_type].x_meta_name = node_c_meta
            result[node_type].x_meta = node_x_meta
            node_name_lookups[node_type] = {value: index for index, value in node_names.items()}
        for (node_type_start, edge_type, node_type_end), (edge_names, edge_index) in self.edge_idxs.items():
            edge_index = edge_index.copy()
            edge_start = edge_index["start"]
            node_name_start_lookup = node_name_lookups[node_type_start]
            edge_start_remapped = edge_start.map(lambda x: node_name_start_lookup[x])
            edge_index["start"] = edge_start_remapped
            # edge_index["start"] = edge_index["start"].map(node_name_lookups[node_type_start])
            edge_index["end"] = edge_index["end"].map(node_name_lookups[node_type_end])
            edge_index = edge_index[["start", "end"]]
            edge_index = tensor(edge_index.values).T.to(int64)
            result[node_type_start, edge_type, node_type_end].edge_index = edge_index
            if reverse_edges:
                reverse_edge_index = edge_index.flip(dims=[0])
                new_edge_type = edge_type + "_reversed"
                result[node_type_end, new_edge_type, node_type_start].edge_index = reverse_edge_index
        for (node_type_start, edge_type, node_type_end), (edge_names, edge_attr, edge_attr_meta) in self.edge_attrs.items():
            edge_c = list(edge_attr.columns)
            edge_attr = tensor(edge_attr.values).to(float32)
            edge_c_meta = list(edge_attr_meta.columns)
            edge_attr_meta = tensor(edge_attr_meta.values).to(float32)
            # edge_attr_meta = list(edge_attr_meta.values)
            result[node_type_start, edge_type, node_type_end].edge_attr_name = edge_c
            result[node_type_start, edge_type, node_type_end].edge_attr = edge_attr
            result[node_type_start, edge_type, node_type_end].edge_attr_meta_name = edge_c_meta
            result[node_type_start, edge_type, node_type_end].edge_attr_meta = edge_attr_meta
            if reverse_edges:
                new_edge_type = edge_type + "_reversed"
                result[node_type_end, new_edge_type, node_type_start].edge_attr_name = edge_c
                result[node_type_end, new_edge_type, node_type_start].edge_attr = edge_attr
                result[node_type_end, new_edge_type, node_type_start].edge_attr_name = edge_c_meta
                result[node_type_end, new_edge_type, node_type_start].edge_attr_meat = edge_attr_meta
        return result