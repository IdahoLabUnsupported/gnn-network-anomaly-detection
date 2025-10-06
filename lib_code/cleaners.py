# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

# Warning! Objects in here may need to be pickled, so the logger cannot be part of the object
from loguru import logger

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import *
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
import pandas as pd
import numpy as np

from data.log_cleaners import LogCleaner

class DictCleaner(LogCleaner):
    """
    This is a general cleaner class that uses a dictionary configuration 
    of sklearn transforms to determine how to clean the data.
    """
    def __init__(self, logger=logger):
        super().__init__()
        self.type = "Dict"
        self.transformer = None
    
    def get_new_transforms(self):
        """
        Provides the sklearn transforms for this DictCleaner.
        Must be overrided by child class.

        Returns
        -------
        - {'str', list[sklearn.transformer]/None}
            dict with lists of sklearn transformers to use or None for columns that should be removed 
        """
        if self.__name__ == "DictCleaner":
            raise NotImplementedError("DictCleaner is not a valid cleaner. "
                "Make a cleaner class that derives from DictCleaner and implements \"get_new_transforms\".")
        else:
            raise NotImplementedError("Cleaner must implement \"get_new_tranforms\" "
                "by having it return a dictionary from column name to list of sklearn transforms.")

    def _coerce_object_to_category(self, df):
        # Convert objects to categories
        for column in df.select_dtypes(include=['object']).columns:
            # This definitelyyyyy needs changing
            try:
                # Attempt to convert the column to a 'category' type
                df[column] = df[column].astype('category')
            except TypeError:
                # If a TypeError is raised, drop the column
                logger.warning(f"Column {column} could not be coerced to category type, so it will be dropped.")
                df.drop(column, axis=1, inplace=True)

    def _coerce_duration_to_seconds(self, df):
        # Convert timedelta64[ns] to float (seconds) and standard scale
        timedelta_columns = df.select_dtypes(include=['timedelta64[ns]']).columns
        for col in timedelta_columns:
            df[col] = df[col].dt.total_seconds()

    def _coerce_one_nan(self, df):
        for col in df.columns:
            ser = df[col]
            # Need to rename because true nan's cause duplicated columns later on.
            if ser.dtype.name == 'category':
                ser = ser.cat.rename_categories({'nan': 'nan_str'})
            else:
                ser = ser.replace('nan', 'nan_str')
            df[col] = ser

    def fit(self, data: pd.DataFrame):
        df = data.copy()
        transform_dict = self.get_new_transforms()
        self.transform_dict = transform_dict
        
        self._coerce_duration_to_seconds(df)
        self._coerce_object_to_category(df)
        self._coerce_one_nan(df)
    
        # Define a column transformer
        transformers = []

        zeek_cols = list(df.columns)
        zeek_leftover_check = list(df.columns)
        transform_keys = list(transform_dict.keys())

        i = 0
        for col in transform_keys:
            logger.trace(f"Cleaner loop: Adding {col} column to ColumnTransformer.")
            transform_func = transform_dict[col]
            if col in zeek_leftover_check:
                zeek_leftover_check.remove(col)
            else:
                logger.warning(f"Using {col} column for multiple transformation functions. This may work but is not tested.")
            if col not in zeek_cols and transform_func is not None:
                raise Exception(f"Column \"{col}\" in {type(self).__name__} not in zeek log.") from ve 
            if transform_func is None:
                continue
            transform_func.sparse_output = False
            # The number will be removed during transform
            transformers.append((f"{i:>010}", transform_func, tuple([col])))
            i += 1
        if len(zeek_leftover_check) > 0:
            logger.warning(f"Not all columns accounted for. Skipping: {zeek_leftover_check}")

        # Fit and save the transform pipeline
        column_transformer = ColumnTransformer(transformers, remainder='drop')
        column_transformer.set_output(transform="pandas")
        column_transformer.fit(df)
        self.transformer = column_transformer
        self._fitted = True

    def transform(self, data: pd.DataFrame):
        df = data.copy()
        
        if not self.transformer:
            raise RuntimeError("Transform not yet learned. Did you fit all cleaner objects?")

        self._coerce_duration_to_seconds(df)
        self._coerce_object_to_category(df)
        self._coerce_one_nan(df)

        # Apply Transforms
        pipeline = self.transformer
        transformed_df = pipeline.transform(df)

        transformed_df.columns = [col[12:] for col in transformed_df.columns]

        numeric_df = transformed_df.select_dtypes(include=['number'])

        transformed_df[numeric_df.columns] = numeric_df.fillna(0)

        return transformed_df

    def other_features(self, data:pd.DataFrame):
        """
        Gets the other features not transformed by this ZeekClean object.

        Parameters
        ----------
        data : DataFrame
            the data to filter from

        Returns
        -------
        DataFrame
            the filtered data
        """
        df = data.copy()

        if not self.transformer:
            raise RuntimeError("Transform not yet learned. Did you fit all cleaner objects?")
        
        other_columns = [col for col in df.columns 
                         if (col not in self.transform_dict) or (self.transform_dict[col] is None)]
        
        result_df = df[other_columns]
        return result_df

    def create_autoencoder_loss_function(self, columns: list):
        """
        Creates a  loss function for a model based on these
        transforms and the provided columns.

        The reason for using this is to automatically create a loss function
        based on the transformations performed on the data. Scaled data is best
        measured using MSELoss, while one-hot encoded features are better measured
        using CrossEntropyLoss.

        Parameters
        ----------
        columns : list[str]
            the names of the columns after they were converted by this cleaner

        Returns
        -------
        func(input, target, aggr) --> torch.tensor
            a function that calculates autoencoder loss from the model output
            and target values, based on up to one aggregation function
        """
        def find_matching_key(dictionary, query):
            for key in dictionary.keys():
                if query.startswith(key):
                    return key
            return None
        
        transforms = self.get_new_transforms()
        cols = columns
        indexer = {}
        for col in cols:
            match = find_matching_key(transforms, col)
            if match is not None:
                col_id = cols.index(col)
                indexer[match] = indexer.get(match, [])
                indexer[match].append(col_id)
        losses = []
        mse_cols = []

        for match, indices in indexer.items():
            transform = transforms[match]
            if isinstance(transform, OneHotEncoder):
                losses.append({"indices": indices, "func": CrossEntropyLoss, "weight": 1})
            else:
                mse_cols.extend(indices)
        if mse_cols:
            losses.append({"indices": mse_cols, "func": MSELoss, "weight": len(mse_cols)})

        def loss_func(input, target, aggr = "mean"):
            loss = None
            total_weight = 0
            for loss_dict in losses:
                idx = loss_dict["indices"]
                this_loss_func = loss_dict["func"]
                this_input = input[:,idx]
                this_target = target[:,idx]
                this_weight = loss_dict["weight"]
                if isinstance(this_loss_func, CrossEntropyLoss):
                    this_target = torch.argmax(this_target, dim=1)
                this_loss = this_loss_func(reduction=aggr)(this_input, this_target)
                this_loss = this_weight * this_loss
                if len(this_loss.shape) > 1:
                    this_loss = this_loss.mean(dim=1)
                if loss is None:
                    loss = this_loss
                else:
                    loss = loss + this_loss
                total_weight += this_weight
            return loss / total_weight
        
        return loss_func

    def create_reconstruction_function(self, columns: list):
        """
        Creates a prediction reconstruction function for a model based on these
        transforms and the provided columns. This returns predicted output in the
        same units as the model input.

        The reason for using this is to automatically create the function
        based on the transformations performed on the data. MSELoss data can be
        interpreted directly, while one-hot encoded features need to be translated
        to value categories.

        Parameters
        ----------
        columns : list[str]
            the names of the columns after they were converted by this cleaner

        Returns
        -------
        func(input) --> torch.tensor
            a function that calculates autoencoder reconstruction from the model output
        """
        def find_matching_key(dictionary, query):
            for key in dictionary.keys():
                if query.startswith(key):
                    return key
            return None


        z_transforms = self.get_new_transforms()
        cols = columns
        indexer = {}
        no_match = []
        for col in cols:
            match = find_matching_key(z_transforms, col)
            col_id = cols.index(col)
            if match is not None:
                indexer[match] = indexer.get(match, [])
                indexer[match].append(col_id)
            else:
                no_match.append(col_id)
        if no_match:
            logger.warning(f"Columns had no matching loss function: {no_match}")

        transforms_ = []
        identity_cols = []
        for match, indices in indexer.items():
            z_transform = z_transforms[match]
            if isinstance(z_transform, OneHotEncoder):
                transforms_.append({"indices": indices, "func": F.softmax})
            else:
                identity_cols.extend(indices)
        if identity_cols:
            transforms_.append({"indices": identity_cols, "func": lambda x: x})

        def transform_func(input):
            ar_transform = torch.zeros_like(input)
            for transform_dict in transforms_:
                idx = transform_dict["indices"]
                this_transform_func = transform_dict["func"]
                this_input = input[:,idx]
                this_transform = this_transform_func(this_input)
                ar_transform[:,idx] = this_transform
            return ar_transform
        
        return transform_func

def _log_positive_function(x):
    return np.log(x+1.0).astype(np.float32)

class ConnCleaner(DictCleaner):
    def get_new_transforms(self):
        return {"ts":                None,
                "uid":               None,
                "id.orig_h":         None,
                "id.orig_p":         OneHotEncoder(min_frequency=0.01, handle_unknown='infrequent_if_exist'),
                "id.resp_h":         None,
                "id.resp_p":         OneHotEncoder(min_frequency=0.01, handle_unknown='infrequent_if_exist'),
                "proto":             OneHotEncoder(min_frequency=0.001, handle_unknown='infrequent_if_exist'),
                "service":           OneHotEncoder(min_frequency=0.001, handle_unknown='infrequent_if_exist'),
                "duration":          FunctionTransformer(_log_positive_function),
                "orig_bytes":        FunctionTransformer(_log_positive_function), 
                "resp_bytes":        FunctionTransformer(_log_positive_function), 
                "conn_state":        OneHotEncoder(min_frequency=0.001, handle_unknown='ignore'),
                "local_orig":        None,  
                "local_resp":        None,  
                "missed_bytes":      FunctionTransformer(_log_positive_function), 
                "history":           None,
                "orig_pkts":         FunctionTransformer(_log_positive_function),
                "resp_pkts":         FunctionTransformer(_log_positive_function),
                "tunnel_parents":    None,
                "orig_l2_addr":      None,
                "resp_l2_addr":      None,
                "vlan":              None,
                "inner_vlan":        None,
                "orig_ip_bytes":     None,
                "resp_ip_bytes":     None
                }