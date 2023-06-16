# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import os
from datetime import datetime

import git
import toml
from pymongo import MongoClient, WriteConcern


def _calculate_local_experiment_id(cwd_path):
    filename_list = os.listdir(f"{cwd_path}/selection_results")
    highest_experiment_id = 0
    for filename in filename_list:
        if filename.endswith(".pkl") and ("raw_selection" in filename):
            current_experiment_id_str = filename[
                filename.find(start := "experiment")
                + len(start) : filename.find("_raw_selection")
            ]
            current_experiment_id = int(current_experiment_id_str)
            if current_experiment_id > highest_experiment_id:
                highest_experiment_id = current_experiment_id
    print(f"Experiment id is {highest_experiment_id + 1}")
    return highest_experiment_id + 1


def _save_meta_data_in_mongodb(settings, commit_sha):
    MONGODB_URL = settings.mongo_db_url
    client = MongoClient(MONGODB_URL)
    db = client.admin
    write_concern = WriteConcern(w=1, wtimeout=10000, fsync=True)
    meta_data_collection = db.experiments_meta_data.with_options(
        write_concern=write_concern
    )

    max_id = 0
    for document in meta_data_collection.find():
        if document["experiment_id"] > max_id:
            max_id = document["experiment_id"]
    print("Last experiment was", max_id)
    experiment_id = max_id + 1

    # create experiment
    experiment_meta_data = toml.load(
        [f"{settings['cwd_path']}/settings.toml"], _dict=dict
    )
    experiment_meta_data.update(
        {
            "experiment_id": experiment_id,
            "date": datetime.utcnow(),
            "git_commit": commit_sha,
            "cwd_path": settings["cwd_path"],
        }
    )
    mongodb_id = meta_data_collection.insert_one(
        experiment_meta_data, write_concern
    ).inserted_id

    # check inserted meta data
    new_loaded_data = meta_data_collection.find_one({"_id": mongodb_id})
    for key in new_loaded_data:
        if "date" in key:
            assert new_loaded_data[key].date() == experiment_meta_data[key].date()
        elif "_id" not in key:
            assert (
                new_loaded_data[key] == experiment_meta_data[key]
            ), f"{new_loaded_data[key]} == {experiment_meta_data[key]}"
        elif key == "experiment_id":
            print(f"Loaded experiment id is {new_loaded_data[key]}")
    return mongodb_id, experiment_id


def post(settings):
    assert (settings.env == "local") or (settings.env == "cluster")
    data = {"dynaconf_merge": True}
    git_repository = git.Repo(search_parent_directories=True)
    commit_sha = git_repository.head.object.hexsha
    if settings.env == "cluster":
        print("settings.env", settings.env)
        cwd_path = settings["cwd_path"]
        if settings.store_result:
            mongodb_id, experiment_id = _save_meta_data_in_mongodb(settings, commit_sha)
            data["experiment_id"] = experiment_id
            data["mongodb_id"] = mongodb_id
    elif settings.env == "local":
        cwd_path = os.path.dirname(os.path.realpath(__file__))
        data["cwd_path"] = cwd_path
        if settings.store_result:
            experiment_id = _calculate_local_experiment_id(cwd_path)
            data["experiment_id"] = experiment_id
            data["git_commit"] = commit_sha
            # pickle meta data
            data_path = (
                f"{cwd_path}{settings['data_storage']['path_selection_results']}"
                f"/{settings['data']['name']}_experiment{experiment_id}_meta_data.toml"
            )
            meta_data = settings.to_dict()
            meta_data = dict((k.lower(), v) for k, v in meta_data.items())
            with open(data_path, "w") as handle:
                toml.dump(meta_data, handle)
    else:
        raise ValueError("Environment must be env='cluster' or env='local'")

    data[
        "path_yeo_johnson_c_module"
    ] = f"{cwd_path}{settings['preprocessing']['path_yeo_johnson_c_module']}".replace(
        "ensemble-feature-selection-benchmark", ""
    )
    data[
        "path_yeo_johnson_c_library"
    ] = f"{cwd_path}{settings['preprocessing']['path_yeo_johnson_c_library']}".replace(
        "ensemble-feature-selection-benchmark", ""
    )
    data[
        "path_yeo_johnson_fpga_module"
    ] = f"{cwd_path}{settings['preprocessing']['path_yeo_johnson_fpga_module']}".replace(
        "ensemble-feature-selection-benchmark", ""
    )
    print("Return from hook")
    return data
