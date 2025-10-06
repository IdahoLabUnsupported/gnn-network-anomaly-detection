# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import os
import time
import pandas as pd
import numpy as np
import warnings
from utils.load_data import load_log_data

"""
Load Zeek files from the specified folder/folders.

Usefule fields:
.frames = dictionary(str,DataFrame) log type name to combined pandas frames including all logs of that type
.log_types = list(str) log type names included in dataset
.count = int row count across log types
"""
class log_loader():
    DEFAULT_LOG_PARENT_FOLDER = \
        "/projects/data/zeeklogs/"

    """
    log_parent_folderpath = str folderpath that contains several folders of log files
    log_folderpaths = list(str) list of folderpaths that contain log _files
    limit = int will only include the first n folders
    """
    def __init__(self, log_parent_folderpath=None, log_folderpaths=None, limit=None):
        if log_folderpaths is None:
            if log_parent_folderpath is None:
                # log_folderpaths = [self.DEFAULT_LOG_DIRECTORY]
                log_parent_folderpath = self.DEFAULT_LOG_PARENT_FOLDER
            log_folderpaths = [os.path.join(log_parent_folderpath, folder) for folder in os.listdir(log_parent_folderpath)]
        elif not isinstance(log_folderpaths, list):
            log_folderpaths = [log_folderpaths]
        if limit is not None and len(log_folderpaths) > limit:
            log_folderpaths = log_folderpaths[0:limit]
        self.log_folderpaths = log_folderpaths
        self._load_dataframes()

    def _load_dataframes(self):
        frames_dicts = []
        log_types = set()
        for folderpath in self.log_folderpaths:
            print("Loading", folderpath)
            frames_dict = load_log_data(folderpath)
            log_types = log_types.union(set(frames_dict.keys()))
            frames_dicts.append(frames_dict)
        if len(log_types) < 1:
            raise Exception(f"No log files found in folderpaths: {self.log_folderpaths[0]}.")
        self.log_types = list(log_types)
        self.count = 0
        log_types = self.log_types
        combined_frames_dict = {}
        for log in log_types:
            combined_frame = None
            for frames_dict in frames_dicts:
                if log in frames_dict:
                    if combined_frame is None:
                        combined_frame = frames_dict[log]
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter(action='ignore', category=FutureWarning)
                            combined_frame = pd.concat([combined_frame, frames_dict[log]])
            combined_frame = combined_frame.sort_values('ts').reset_index(drop=True)
            combined_frames_dict[log] = combined_frame
            self.count += len(combined_frame)
        self.frames = combined_frames_dict

"""
Filter and reshape data according to specified excel config

Useful fields:
.frames = dictionary(str,DataFrame) the parsed data as log_type to DataFrame

.data_dframes_dict = dictionary(str,DataFrame) log type name to DataFrame for those logs
.data_names_dict = dictionary(str, list(str)) log type names to DataFrame field names for those logs
.log_types = log type names for which this digestor has a data dframe
.count = int total data row count across log types

.entities = dict(str, dict(str, DataFrame)) log type name to entity field name to unique entity name
.entity_maps = dict(str, dict(str, int)) entity field name to unique entity name to unique int
"""
class digestor():
    DEFAULT_FIELD_PARSE_FILEPATH = \
        "/home/brueja1/internship2024/brueja1/ZeekUsefulFields.xlsx"

    """
    loader = log_loader the log loader which loaded the raw data to use
    field_parse_filepath = str the filepath for the excel configuration file
    """
    def __init__(self, loader=None, field_parse_filepath=None):
        if loader is None:
            print("Running default loader")
            loader = log_loader()
        if isinstance(loader, str):
            print(f"Running loader with parent folder {loader}")
            loader = log_loader(log_parent_folderpath=loader)
        self.loader = loader
        if field_parse_filepath is None:
            field_parse_filepath = self.DEFAULT_FIELD_PARSE_FILEPATH 
        self.field_parse_filepath = field_parse_filepath
        self._parse_data()
        self._set_train_test()
        
    def _get_new_names(self, log_type, old_names):
        new_names = []
        for old_name in old_names:
            print(old_name)
            if old_name in self.data_old_names_dict[log_type]:        
                print("OLD NAME FOUND:", old_name)
                print(self.data_old_names_dict[log_type][old_name])
                new_names.extend(self.data_old_names_dict[log_type][old_name])
            else:
                new_names.append(old_name)
        return new_names
    
    def _parse_data(self):
        field_parse_dict = {}
        with open(self.field_parse_filepath, 'rb') as field_parse_file:
            for log_type in self.loader.log_types:
                field_parse_dict[log_type] = pd.read_excel(field_parse_file, sheet_name=log_type, index_col=0)
        self.data_old_names_dict = {}
        for log_type in self.loader.log_types:
            self.data_old_names_dict[log_type] = {}
        self.frames = {}
        self.count = 0
        self.entities = {}
        self.entity_maps = {"all": {}}
        self.field_agg_methods = {}
        self.log_types = self.loader.log_types.copy()
        for log_type in self.loader.log_types:
            self.frames[log_type] = self.loader.frames[log_type].copy()
            self.count += len(self.frames[log_type])
            self.field_agg_methods[log_type] = {}
        for log_type in self.loader.log_types:
            field_parse = field_parse_dict[log_type]
            for method in field_parse.index.values:
                if method[0:6] == "DropNA":                
                    self._dropna(field_parse_dict, log_type, field_parse.loc[method])
                elif method[0:6] == "AsData":
                    self._parse_dframe(field_parse_dict, log_type, field_parse.loc[method])
                elif method[0:11] == "Aggregation":
                    self._run_aggregation(field_parse_dict, log_type, field_parse.loc[method])
                elif method[0:6] == "Entity":
                    self._run_entity(field_parse_dict, log_type, field_parse.loc[method])
                elif method[0:7] == "GroupBy":
                    self._run_groupby(field_parse_dict, log_type, field_parse.loc[method])

    def _dropna(self, field_parse_dict, log_type, field_vals):
        to_drop = []
        field_parse = field_parse_dict[log_type]
        log_frame = self.loader.frames[log_type].copy()
        for field_name in field_parse.columns:
            fields_check = field_vals[(field_vals==True)].to_frame().index.to_list()
            if len(fields_check) > 0:
                log_frame = log_frame.dropna(subset=fields_check)
        self.frames[log_type] = log_frame

    def _parse_dframe(self, field_parse_dict, log_type, field_vals):
        frames = self.frames
        data_dframes_dict = {}
        data_names_dict = {}
        count_change = -len(self.frames[log_type])
        log_types = self.log_types.copy()

        field_parse = field_parse_dict[log_type]
        log_frame = self.frames[log_type]
        log_data_dframes = []
        log_data_names = []           
        log_data_old_to_new_names = {}

        for field_name in field_vals.index.values.tolist():
            parse_method = field_vals[field_name]
            if pd.notna(parse_method):
                new_dframe, new_names = self._parse_field(log_frame[field_name], parse_method, field_name)
                log_data_dframes.append(new_dframe)
                log_data_names.extend(new_names)
                log_data_old_to_new_names[field_name] = new_names
                print(new_names)
        print(log_data_old_to_new_names)
        if len(log_data_dframes) > 0:
            log_data_dframe = pd.concat(log_data_dframes, axis=1)
            data_dframes_dict[log_type] = log_data_dframe
            data_names_dict[log_type] = log_data_names
            for field_name in log_data_old_to_new_names:
                new_names = self.data_old_names_dict[log_type].get(field_name, [])
                new_names.extend(log_data_old_to_new_names[field_name])
                self.data_old_names_dict[log_type][field_name] = new_names
            count_change += len(log_data_dframe)
        else:
            log_types.remove(log_type)
        self.count += count_change
        self.log_types = log_types
        if log_type in self.log_types:
            self.frames[log_type] = data_dframes_dict[log_type]
    
    def _parse_field(self, series, parse_method, field_name):
        new_field_names = [field_name]
        if parse_method == "enum":
            print("enum")
            new_frame = pd.get_dummies(series, prefix=field_name)
            new_field_names = new_frame.columns.tolist()
        elif parse_method == "duration":
            new_frame = series.dt.total_seconds().astype("float64")
        elif parse_method == "int": # TODO FIX
            new_frame = series.astype("float64")
        elif parse_method == "bool":
            new_frame = series.map({"T": True, "F": False}).astype("float64")
        elif parse_method == "default":
            new_frame = series.copy()
        else:
            print(f"Warning: {parse_method} is not a defined parse method so {field_name} is not parsable. Using 0s instead.") 
            # new_series = np.zeros(len(series))[...,np.newaxis]
            new_series = series.copy().astype("float64")
            new_series[field_name] = 0.0
        new_frame[new_frame.isna()] = 0.0
        return new_frame, new_field_names
    def _run_aggregation(self, field_parse_dict, log_type, field_vals):
        frames = self.frames
        full_frame = frames[log_type]
        field_parse = field_parse_dict[log_type]
        for field in field_vals.index.values.tolist():
            agg_method = field_vals[field]
            if not pd.isna(agg_method):
                agg_list = self.field_agg_methods[log_type].get(field, [])
                agg_list.append(agg_method)
                self.field_agg_methods[log_type][field] = agg_list
    
    def _run_entity(self, field_parse_dict, log_type, field_vals):
        frames = self.frames

        print("Not Running Entities at this time")
        # FROM RUN_OPERATION(self, log_type, operation, frame, parse_method, field_names)
        # print(log_type, parse_method)
        # name_frame = frame.apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        # self.entities[log_type][entity_name]=name_frame
        # mapped = set(self.entity_maps["all"])
        # if entity_name not in self.entity_maps:
        #     self.entity_maps[entity_name] = {}
        # for entity in set(name_frame) - mapped:
        #     map_num = len(self.entity_maps["all"])
        #     self.entity_maps["all"][entity] = map_num
        #     self.entity_maps[entity_name][entity] = map_num
    
    def _run_groupby(self, field_parse_dict, log_type, field_vals):
        frames = self.frames
        full_frame = frames[log_type]
        field_parse = field_parse_dict[log_type]
        field_names = field_vals[field_vals==True].to_frame().index.to_list()
        field_names = self._get_new_names(log_type, field_names)
        if len(field_names) == 0:
            print("GroupBy Empty for", log_type)
            return
        print(field_names, full_frame.columns)
        method_frame = full_frame[field_names]
        
        #self._run_operation(log_type, operation, method_frame, method, field_names) 
        # From RUN_OPERATION(self, log_type, operation, frame, parse_method, field_names)
        
        print("GroupBy", log_type)
        groupby_frame = self.frames[log_type].groupby(field_names, as_index=False)
        base_frame = groupby_frame[field_names].agg(lambda x: x.iloc[0])
        agg_frames = [base_frame]
        agg_methods = {}
        for field, methods in self.field_agg_methods[log_type].items():
            for method in methods:
                fields = agg_methods.get(method, [])
                fields.append(field)
                agg_methods[method] = fields
        new_names_to_add = {}
        for method in agg_methods:
            fields = agg_methods[method]
            fields = self._get_new_names(log_type, fields)
            suffix = "_" + method
            gb1 = groupby_frame[fields]
            if method == "mean":
                agg_frame = gb1.mean()
            elif method == "max":
                agg_frame = gb1.max()
            elif method == "min":
                agg_frame = gb1.min()
            elif method == "count":
                agg_frame = gb1.count()
            elif method == "sum":
                agg_frame = gb1.sum()
            else:
                print(f"Aggregation method {method} not implemented")
                agg_frame = None
            agg_frame = agg_frame[fields].copy().add_suffix(suffix)
            agg_frames.append(agg_frame)
        for method in agg_methods:
            fields = agg_methods[method]
            for field in fields:
                new_names = self.data_old_names_dict[log_type].get(field, [])
                new_names.append(field + "_" + method)
                self.data_old_names_dict[log_type][field] = new_names
        combined_frame = pd.concat(agg_frames, axis=1)
        self.frames[log_type] = combined_frame

    def _set_train_test(self, p=0.8, random=False): 
        self.train_data_arrays_dict = None
        self.test_data_arrays_dict = None

def main():
    digestor()

if __name__ == "__main__":
    main()
