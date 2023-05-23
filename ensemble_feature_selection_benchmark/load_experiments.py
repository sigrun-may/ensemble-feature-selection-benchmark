# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Load load data from specific experiment."""
from pprint import pprint

import logging
import os
import pickle
from os.path import exists
from typing import Dict, List, Optional

import pandas as pd
import toml
from pymongo import MongoClient

from config import settings
from ensemble_feature_selection_benchmark import check_feature_selection_result

_logger = logging.getLogger(__name__)


def _parse_raw_selection_results(
    raw_selection_results_dict: dict, stored_settings
) -> dict:
    print(raw_selection_results_dict.keys())
    for key, selection_result in raw_selection_results_dict.items():
        if isinstance(selection_result[0], List):
            new_list = []
            for cv_iteration_result in selection_result:
                new_list.append(
                    pd.DataFrame(
                        cv_iteration_result,
                        index=raw_selection_results_dict["feature_names"][1:],
                    )
                )
            raw_selection_results_dict[key] = new_list
    check_feature_selection_result.check_raw_feature_selection_result_dict(
        raw_selection_results_dict, stored_settings
    )
    return raw_selection_results_dict


def _load_meta_data_from_toml(experiment_id, cwd_path):
    filename_list = os.listdir(f"{cwd_path}/selection_results")
    toml_file_path = ""
    for filename in filename_list:
        if filename.endswith(".toml"):
            experiment_id_str = filename[
                filename.find(start := "experiment")
                + len(start) : filename.find("_meta_data")
            ]
            if int(experiment_id_str) == experiment_id:
                print(experiment_id)
                toml_file_path = filename
    loaded_meta_data = toml.load(f"{cwd_path}/selection_results/{toml_file_path}")
    return loaded_meta_data


def _load_raw_selection_results_from_files(experiment_id, cwd_path):
    # load meta data
    meta_data = _load_meta_data_from_toml(experiment_id, cwd_path)
    assert isinstance(meta_data, dict)

    # load pickled data
    result_per_method = {}
    filename_list = os.listdir(f"{cwd_path}/selection_results")
    for filename in filename_list:
        # load feature selection results
        if f"experiment{experiment_id}_raw_selection" in filename:
            path = f"{cwd_path}/selection_results/{filename}"
            if exists(path):
                with open(path, "rb") as handle:
                    feature_selection_method = filename[
                        filename.find(start := "selection_")
                        + len(start) : filename.find(".pkl")
                    ]
                    result_per_method[feature_selection_method] = pickle.load(handle)
    if meta_data is None:
        raise ValueError(
            f"Meta data for experiment {experiment_id} was not found in {cwd_path}/selection_results"
        )
    # get feature names
    data_name = meta_data["data"]["name"]
    input_data_path = (
        f"{cwd_path}/{meta_data['data']['folder']}/{data_name}/{data_name}.csv"
    )
    feature_names = list(pd.read_csv(input_data_path).columns)[
        : meta_data["data"]["number_of_features"]
    ]
    assert feature_names[0] == "label"
    result_per_method["feature_names"] = feature_names
    check_feature_selection_result.check_raw_feature_selection_result_dict(
        result_per_method, meta_data
    )
    return result_per_method, meta_data


def _load_meta_data_from_mongodb(experiment_id, mongo_db_url):
    client = MongoClient(mongo_db_url)
    db = client.admin
    experiments_meta_data = db.experiments_meta_data
    loaded_experiment_data = experiments_meta_data.find_one(
        {"experiment_id": experiment_id}
    )

    if loaded_experiment_data is None:
        _logger.error(
            f"Experiment meta data from experiment {experiment_id} does not exist!"
        )
    else:
        # Close the connection to MongoDB when you're done.
        client.close()
        return loaded_experiment_data


def _load_raw_selection_results_from_mongodb(experiment_id, mongo_db_url):
    client = MongoClient(mongo_db_url)
    db = client.admin
    experiments_data = db.experiments_data
    experiments_meta_data = db.experiments_meta_data

    # get meta data
    meta_data = experiments_meta_data.find_one({"experiment_id": experiment_id})
    if meta_data is None:
        raise ValueError(
            f"Meta data for experiment {experiment_id} could not be loaded from mongodb "
            f"{mongo_db_url}!"
        )
    assert len(meta_data) > 0
    assert isinstance(meta_data, dict)
    pprint(meta_data)

    # load stored data
    result_per_method = {}
    for result in experiments_data.find({"experiment_id": experiment_id}):
        print("Feature selection methods:")
        for key, value in result.items():
            if "id" not in key:
                print(key)
                result_per_method[key] = value
    print(f"loaded experiment has experiment_id {experiment_id}")

    # close the connection to MongoDB
    client.close()

    # get feature names
    print(meta_data.keys())
    data_name = meta_data["data"]["name"]
    input_data_path = f"{meta_data['cwd_path']}/{meta_data['data']['folder']}/{data_name}/{data_name}.csv"
    feature_names = list(pd.read_csv(input_data_path).columns)[
        : meta_data["data"]["number_of_features"]
    ]
    assert feature_names[0] == "label"
    assert len(feature_names) == meta_data["data"]["number_of_features"]
    result_per_method["feature_names"] = feature_names

    result_per_method = _parse_raw_selection_results(result_per_method, meta_data)
    check_feature_selection_result.check_raw_feature_selection_result_dict(
        result_per_method, meta_data
    )
    return result_per_method, meta_data


def _load_raw_selection_results(experiment_id, cwd_path=None, mongo_db_url=None):
    if mongo_db_url is not None:
        return _load_raw_selection_results_from_mongodb(experiment_id, mongo_db_url)
    elif cwd_path is not None:
        return _load_raw_selection_results_from_files(experiment_id, cwd_path)


def _load_preprocessed_data():
    path = settings.data_storage.path_preprocessing_results
    if exists(path):
        print("Path to preprocessing results is ", path)
        with open(path, "rb") as handle:
            return pickle.load(handle)
