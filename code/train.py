# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from loguru import logger

import os
import sys
import torch
import argparse
import pickle

from create_dataset import CreateDataset
from model_code import hetero_models
from lib_code.data_tools import ZeekTransforms
from lib_code.task import HeteroGraphNoSelfTask

def main():
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    config = get_cl_args(logger)

    # Assign config variables
    data_folderpath = config["data_folderpath"]
    model_folderpath = config["model_folderpath"]
    graph_folderpath = config["graph_folderpath"]
    epochs = config["epochs"]
    patience = config["patience"]
    batch_size = config["batch_size"]
    seed = config["seed"]
    logger_level = config["logger"]
    logger_level = logger_level.upper()

    # For valid logger levels, https://loguru.readthedocs.io/en/stable/api/logger.html (add method)
    logger.remove()
    logger.add(sys.stderr, level=logger_level)

    # Throw this warning asap since it impacts just about everything
    device = get_device(logger, True, True)

    # Creating datasets
    logger.info("Creating datasets")
    datasets = CreateDataset(data_folderpath=data_folderpath, model_folderpath=model_folderpath,
                   val_split=0.1, test_split=0.0, seed=seed, do_train=True, logger=logger)
    datasets.save_graphs(graph_path=graph_folderpath)

    # Train model
    # Get graphs from "datasets"
    # Training graph contains only training data nodes
    train_graph = datasets.heterograph_dict["train"].to(device)
    # Validation graph contains training and validation nodes (only evaluates on validation nodes)
    val_graph = datasets.heterograph_dict["train_val"].to(device)
    # No test graph for training
    test_graph = None

    # Initialize model
    logger.info("Initializing model")
    encoder = hetero_models.SAGEEncoder(val_graph, use_dropout=False)
    decoder = hetero_models.SAGEDecoder(val_graph, use_dropout=False)
    model = hetero_models.HeteroAutoencoder(val_graph, encoder, decoder).to(device)

    # Construct loss function from standardization operations
    logger.info("Constructing loss function from standardization operations.")
    zeek_cleaner = datasets.zeek_cleaner

    zeektransforms = ZeekTransforms(logger)
    loss_fn = zeektransforms.create_autoencoder_loss_functions(zeek_cleaner, val_graph, {"conn_node": "conn"})
    reconstruction_fn = zeektransforms.create_reconstruction_functions(zeek_cleaner, val_graph, {"conn_node": "conn"})

    # Create and start task
    task = HeteroGraphNoSelfTask(model, train_graph, autoencoder_loss_functions=loss_fn, 
                                    autoencoder_reconstruction_functions=reconstruction_fn,
                                    val_data=val_graph, test_data=test_graph, target_nodes=["conn_node"],
                                    logger=logger)

    logger.info("Begin training")
    task.train_model(use_DataLoader=True, epochs=epochs, notify=1, validation="save_load_best", 
                     patience=patience, batch_size=batch_size, plot=False)

    # Save model to provided filepath
    cleaner_filepath = os.path.join(model_folderpath, "cleaner.pkl")
    logger.info(f"Saving cleaner to file at {cleaner_filepath}")
    with open(cleaner_filepath, "wb") as file:
        pickle.dump(zeek_cleaner, file)
    model_filepath = os.path.join(model_folderpath, "model.pkl")
    logger.info(f"Saving model to file at {model_filepath}")
    with open(model_filepath, "wb") as file:
        pickle.dump(task.model, file)

def get_device(logger=logger, prefer_cuda=True, require_cuda=False):
    """
    Get the device to run inference on.
    
    Parameters
    ----------
    prefer_cuda : boolean, default=True
        whether a cuda gpu should be preferred to cpu
    
    Returns
    -------
    pytorch device
        the device where pytorch operations should be run
    """
    if require_cuda and not prefer_cuda:
        logger.critical("If cuda is required, it must be preferred.")
        sys.exit(1)
    if not prefer_cuda:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if require_cuda:
        logger.critical("Cuda GPU unavailable.")
        sys.exit(1)
    logger.warning("No CUDA enabled device detected. Models will run on the CPU")
    return torch.device('cpu')

def get_cl_args(logger=logger):
    """
    Read arguments from the command line.

    Returns
    -------
    dict[str, object]
        all of the arguments in name: value format
    """
    parser = argparse.ArgumentParser(description="Graphical Anomaly Detection Training Arguments")
    parser.add_argument('--data_folderpath', type=str, required=True, help=('Path to the data folder containing all of the '
                        'protocol files.'))
    parser.add_argument('--graph_folderpath', type=str, required=True, help=('Path to the data folder to hold the produced graphs.'))
    parser.add_argument('--model_folderpath', type=str, required=True, help=('Path to the base models folder containing all '
                                                                       'of the models. This is where everything that '
                                                                       'is learned is stored.'))
    parser.add_argument('--epochs', type=int, default=200, required=False, help=('Maximum number of training epochs.'))
    parser.add_argument('--patience', type=int, default=10, required=False, help=('Early stopping parameter. How long'
                                                                                  'validation can fail to improve before'
                                                                                  'training will terminate.'))
    parser.add_argument('--batch_size', type=int, default=1024, required=False, help=('Mini batch size. High values'
                                                                                      'are problematic for the algorithm.'))
    parser.add_argument('--seed', type=int, default=1, required=False, help=('Random seed for any random processes.'))
    parser.add_argument('--logger', type=str, default='INFO', required=False, help=('Level for the Loguru logger. '
                                                                                     'Must be one of the predefined '
                                                                                     'levels specified by loguru.'))
    args = parser.parse_args()
    config = vars(args)

    if config["epochs"] < 0:
        logger.critical("Epochs must be greater than 0.")
        sys.exit(1)
    if config["patience"] == -1:
        config["patience"] = config["epochs"]
    if config["patience"] != -1 and config["patience"] <= 0:
        logger.critical("Patience must be greater than 0 (-1 to turn off).")
        sys.exit(1)
    if config["batch_size"] <= 0:
        logger.critical("Batch size must be greater than 0. (batch size cannot be turned off)")
        sys.exit(1)
    valid_log_levels = ["CRITICAL", "ERROR", "INFO", "DEBUG", "TRACE"]
    if config["logger"].upper() not in valid_log_levels:
        logger.critical(f"Logger level must be in {valid_log_levels}.")
        sys.exit(1)
    return config


if __name__ == "__main__":
    main()