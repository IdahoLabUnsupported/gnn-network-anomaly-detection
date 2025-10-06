# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from loguru import logger

import numpy as np
import torch
from pathlib import Path
import os
import sys
import pickle
from pyvis.network import Network

from data.datasets import Zeek
from data.cleaning import ZeekCleaner
from lib_code.data_tools import ZeekGraph, ZeekIds, ZeekTransforms

class CreateDataset():
    """
    A high level class that creates HeteroGraph objects from Zeek data 
    from a folderpath

    .zeek_cleaner : ZeekCleaner
        used to transform the data
    .logs : list[str]
        names of logs used to make connection nodes
    .zclean_dict : dict[str, Zeek]
        mapping from name of Zeek dataset to Zeek dataset
    .heterograph_dict : dict[str, HeteroData]
        mapping from name of Zeek dataset to created HeteroData graph object
    """
    def __init__(self, data_folderpath, model_folderpath, 
                 val_split, test_split, seed,
                 do_train, attacker_ips=[], logger=logger):
        """
        Parameters
        ----------
        data_folderpath : str
            folderpath to Zeek data
        model_folderpath : str
            folderpath to model files
        val_split : float
            percent of data to randomly assign as validation data
        test_split : float
            percent of data to randomly assign as test data
        seed : int
            random seed for numpy operations
        do_train : boolean
            whether to train the ZeekCleaner or load from file
        attacker_ips : list[str], default=[]
            ips to include in the test set as labeled malicious
        """
        np.random.seed(seed)
        data_folderpath = Path(data_folderpath)
        
        # Create the Zeek object that holds the log data
        logger.info(f"Loading data from folder {data_folderpath}")
        zeek = Zeek(data_path=data_folderpath)

        # Filter noisy connections from the dataset object
        logger.info("Filtering empty and unsuccessful connections")
        zeek.remove_empty_connections()
        ZeekTransforms.remove_unsuccessful_connections(zeek)

        # Split data into disjoint subsets
        logger.info(f"Splitting data into subsets of {1.0-val_split-test_split:<.2f} training, {val_split:i<.2f} validating, and {test_split:i<.2f} testing.")
        train_val_ratio = (val_split/(1.0-test_split)) if (test_split != 1.0) else (0.0)
        zeek_dict = {}
        # Zeek normal includes all non-attacker_ip connections
        # Zeek malicious includes only attacker_ip connections
        zeek_dict["normal"], zeek_dict["malicious"] = zeek.train_test_split(test_ips=attacker_ips, ratio=0.0, shuffle=False)
        # Zeek non_test includes train and validation connections
        # Zeek test_normal includes non-attacker_ip connections in the test set
        zeek_dict["non_test"], zeek_dict["test_normal"] = zeek_dict["normal"].train_test_split(ratio=test_split, shuffle=True)
        # Zeek train includes only non-malicious training data
        # Zeek val includes only non-malicious validation data
        zeek_dict["train"], zeek_dict["val"] = zeek_dict["non_test"].train_test_split(ratio=train_val_ratio, shuffle=True)

        # Remove empty datasets
        zeek_dict = {name: zeek for name, zeek in zeek_dict.items() 
                     if zeek.n_logs() > 0}

        # Sort subsets by timestamp
        logger.info("Sorting zeeks by timestamp")
        for name, named_zeek in zeek_dict.items():
            for log in named_zeek.data.keys():
                named_zeek: Zeek
                df = named_zeek.get(log).copy()
                named_zeek.set(log, df)
                named_zeek.sort(log, "ts")

        if do_train:
            # Train new ZeekCleaner to scale and transform data
            logger.info("Making ZeekCleaner")
            zeek_cleaner = ZeekCleaner()
            zeek_cleaner.fit(zeek_dict["train"])
        else:
            # Load ZeekCleaner from filepath
            logger.info("Loading ZeekCleaner")
            cleaner_filepath = os.path.join(model_folderpath, "cleaner.pkl")
            with open(cleaner_filepath, 'rb') as file:
                zeek_cleaner = pickle.load(file)
        # Run ZeekCleaner to scale and transform data
        logger.info("Transforming Zeeks with ZeekCleaner")
        zclean_dict = {zeek_name: zeek_cleaner.transform(zeek) 
                       for zeek_name, zeek in zeek_dict.items()}
        self.zeek_cleaner = zeek_cleaner
        # Determine which columns are data columns
        basis_name = "train" if do_train else "test_normal"
        data_columns = {log: list(df.columns) for log, df in zclean_dict[basis_name].data.items()}
        # Re-add the non-data columns (metadata)
        logger.info("Adding identifying back to Zeek objects")
        for zeek_name, zeek in zclean_dict.items():
            old_zeek = zeek_dict[zeek_name]
            ZeekTransforms.add_other_features(zeek_cleaner, zeek_clean=zeek, zeek_old=old_zeek)

        # Add metadata columns to each subset
        logger.info("Adding metadata to Zeek objects")
        for name, zclean in zclean_dict.items():
            for log in zclean.data.keys():
                zclean.get(log)["all"] = 1
                zclean.get(log)["train"] = int(name == "train")
                zclean.get(log)["val"] = int(name == "val")
                zclean.get(log)["test"] = int(name in ["test_normal", "malicious"])
                zclean.get(log)["malicious"] = int(name == "malicious")
                zclean.get(log)["test_benign"] = int(name == "test_normal")

        zclean_list = list(zclean_dict.values())

        # Assign unique node ID' across all subsets
        logger.info("Adding unique ID's to Zeek object")
        logs = ["conn"]
        self.logs = logs
        node_names = ZeekIds()
        for log in logs:
            node_names.new_ids(zclean_list, log, "sender_node")
            node_names.new_ids(zclean_list, log, "connection_node")
            node_names.new_ids(zclean_list, log, "receiver_node")

        # Create Zeek objects for important combinations of split subsets
        logger.info("Combining Zeek objects to form larger graphs")
        zclean_new_dict = {}
        if "train" in zclean_dict.keys() and "val" in zclean_dict.keys():
            zclean_new_dict["train_val"] = ZeekTransforms.combine([zclean_dict["train"], zclean_dict["val"]])
        elif "train" in zclean_dict.keys():
            zclean_new_dict["train_val"] = ZeekTransforms.combine([zclean_dict["train"]])
        elif "val" in zclean_dict.keys():
            zclean_new_dict["train_val"] = ZeekTransforms.combine([zclean_dict["val"]])
        if "test_normal" in zclean_dict.keys() and "malicious" in zclean_dict.keys():
            zclean_new_dict["test"] = ZeekTransforms.combine([zclean_dict["test_normal"], zclean_dict["malicious"]])
        elif "test_normal" in zclean_dict.keys():
            zclean_new_dict["test"] = ZeekTransforms.combine([zclean_dict["test_normal"]])
        elif "malicious" in zclean_dict.keys():
            zclean_new_dict["test"] = ZeekTransforms.combine([zclean_dict["malicious"]])
        if "train_val" in zclean_new_dict.keys() and "test" in zclean_new_dict.keys():
            zclean_new_dict["full"] = ZeekTransforms.combine([zclean_new_dict["train_val"], zclean_new_dict["test"]])
        elif "train_val" in zclean_new_dict.keys():
            zclean_new_dict["full"] = ZeekTransforms.combine([zclean_new_dict["train_val"]])
        elif "test" in zclean_new_dict.keys():
            zclean_new_dict["full"] = ZeekTransforms.combine([zclean_new_dict["test"]])
 
        for name, zclean in zclean_new_dict.items():
            for log in zclean.data.keys():
                try:
                    zclean.sort(log, "ts")
                except KeyError as ke:
                    logger.critical(f"Could not sort {name}'s {log} by ts.")
                    sys.exit(1)
        zclean_dict = zclean_dict | zclean_new_dict
        zclean_list = list(zclean_dict.values())

        self.zclean_dict = zclean_dict

        meta_features = ["all", "train", "val", "test", "malicious", "test_benign"]
        # Copy log files as different node types
        logger.info("Copying log files with new names.")
        for log in logs:
            # Sender nodes should only have the their node_id and meta_features
            ZeekTransforms.copy_log(zclean_list, log, f"{log}_sender")
            ZeekTransforms.filter_to(zclean_list, f"{log}_sender", ["sender_node"] + meta_features)

            # Receiver nodes should also only have their node_id and meta_features
            ZeekTransforms.copy_log(zclean_list, log, f"{log}_receiver")
            ZeekTransforms.filter_to(zclean_list, f"{log}_receiver", ["receiver_node"] + meta_features)

            # Connection nodes should have all features except sender's node_id and receiver's node_id
            # Can add features to filter if desired
            ZeekTransforms.copy_log(zclean_list, log, f"{log}_node")
            ZeekTransforms.filter_to(zclean_list, f"{log}_node", ["connection_node"] + data_columns[log] + meta_features)
            
            # Edges should just have their incident node_ids
            ZeekTransforms.copy_log(zclean_list, log, f"{log}_send_edge")
            ZeekTransforms.filter_to(zclean_list, f"{log}_send_edge", ["sender_node", "connection_node"])

            ZeekTransforms.copy_log(zclean_list, log, f"{log}_receive_edge")
            ZeekTransforms.filter_to(zclean_list, f"{log}_receive_edge", ["connection_node", "receiver_node"])

            # Temporal edges need a timestamp and ip addresses
            ZeekTransforms.copy_log(zclean_list, log, f"{log}_time_send")
            ZeekTransforms.filter_to(zclean_list, f"{log}_time_send", ["ts", "id.orig_h", "id.resp_h", "sender_node", "receiver_node"])
            
            ZeekTransforms.copy_log(zclean_list, log, f"{log}_time_receive")
            ZeekTransforms.filter_to(zclean_list, f"{log}_time_receive", ["ts", "id.orig_h", "id.resp_h", "sender_node", "receiver_node"])

        # Make ZeekGraphs
        logger.info("Making ZeekGraphs")
        zgraph_dict = {name: ZeekGraph(zclean) for name, zclean in zclean_dict.items()}

        heterograph_dict = {}
        for zg_name, zg in zgraph_dict.items():
            for log in logs:
                # Make sender nodes
                zg.add_nodes_from_log(f"{log}_sender", "sender_node", node_data=[], node_type_name="senders", meta=meta_features + ["sender_node"])
                # Make receiver nodes
                zg.add_nodes_from_log(f"{log}_receiver", "receiver_node", node_data=[], node_type_name="receivers", meta=meta_features + ["receiver_node"])
                # Make log_type nodes (ex: conn_node)
                zg.add_nodes_from_log(f"{log}_node", "connection_node", node_data=data_columns[log], node_type_name=f"{log}_node", meta=meta_features + ["connection_node"])
                # Make sender-to-log_type edges
                zg.add_edges_from_log(f"{log}_send_edge", "sender_node", "connection_node", edge_data=[], edge_type_name="sent", node_type_start="senders", node_type_end=f"{log}_node")
                # Make log_type-to-receiver edges
                zg.add_edges_from_log(f"{log}_receive_edge", "connection_node", "receiver_node", edge_data = [], edge_type_name="received", node_type_start=f"{log}_node", node_type_end="receivers")
            # Make temporal edges between nodes with shared Ip addresses
            dict_order = {f"{log}_time_send": "ts" for log in logs} | {f"{log}_time_receive": "ts" for log in logs}
            dict_node = {f"{log}_time_send": "sender_node" for log in logs} | {f"{log}_time_receive": "receiver_node" for log in logs}
            dict_ids = {f"{log}_time_send": ["id.orig_h"] for log in logs} | {f"{log}_time_receive": ["id.resp_h"] for log in logs}
            dict_node_name = {f"{log}_time_send": "senders" for log in logs} | {f"{log}_time_receive": "receivers" for log in logs}

            zg.add_ordered_edges_between_logs(log_columns_dict_order=dict_order, 
                                            log_columns_dict_node=dict_node,
                                            log_columns_dict_groups=dict_ids,
                                            log_dict_node_names=dict_node_name,
                                            edge_name="time", include_duration=False)

            # Construct pytorch HeteroData graph
            heterograph = zg.to_torch_hetero_graph()
            heterograph_dict[zg_name] = heterograph
        
        # Ensure each node and edge holds at least 1 feature. (Required for pytorch algorithms)
        logger.info("Ensure each node and edge holds at least 1 feature.")
        min_node_features = 1
        min_edge_features = 1
        for graph_name, graph in heterograph_dict.items():
            for node_type in graph.node_types:
                node_features = graph[node_type].x
                if node_features is None or node_features.size(dim=1) == 0:
                    # Assign default features (zeros) to nodes without features
                    num_nodes = graph[node_type].num_nodes
                    zeros = torch.zeros(num_nodes, min_node_features, dtype=torch.float32)
                    graph[node_type].x = zeros
                if node_features.size(dim=1) < min_node_features:
                    num_nodes = graph[node_type].num_nodes
                    num_features = node_features.size(dim=1)
                    zeros = torch.zeros(num_nodes, min_node_features - num_features, dtype=torch.float32)
                    graph[node_type].x = torch.cat((node_features, zeros), dim=1)
            for edge_type in graph.edge_types:
                edge_features = graph[edge_type].edge_attr
                if edge_features is None or edge_features.size(dim=1) == 0:
                    num_edges = graph[edge_type].num_edges
                    zeros = torch.zeros(num_edges, min_edge_features, dtype=torch.float32)
                    graph[edge_type].edge_attr = zeros
                if edge_features.size(dim=1) < min_edge_features:
                    num_edges = graph[edge_type].num_edges
                    num_features = edge_features.size(dim=1)
                    zeros = torch.zeros(num_edges, min_edge_features - num_features, dtype=torch.float32)
                    graph[edge_type].edge_attr = torch.cat((edge_features, zeros), dim=1)

        # Hold graphs
        self.heterograph_dict = heterograph_dict

    def save_graphs(self, graph_path):
        logger.info(f"Saving graphs in {graph_path}")
        graph_path = Path(graph_path)
        for graph_name, graph in self.heterograph_dict.items():
            with open(os.path.join(graph_path, f"{graph_name}.pickle"), "wb") as handle:
                pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    CreateDataset()