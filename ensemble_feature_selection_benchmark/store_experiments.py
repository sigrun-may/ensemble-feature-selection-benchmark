# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Store experiment results."""

import datetime
import logging
import pickle
from typing import List, Optional

import pandas as pd
from pymongo import MongoClient, WriteConcern

from ensemble_feature_selection_benchmark import check_feature_selection_result


_logger = logging.getLogger(__name__)


def _convert_raw_selection_result_from_dataframe_to_list(selection_result: List):
    new_list = []
    for cv_iteration_result in selection_result:
        if isinstance(cv_iteration_result, pd.DataFrame):
            new_list.append(cv_iteration_result.to_dict(orient="records"))
    if len(new_list) > 0:
        assert len(new_list) == len(selection_result)
        return new_list
    else:
        return selection_result


def _parse_raw_selection_results(raw_selection_results_dict: dict) -> dict:
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
    return raw_selection_results_dict


def pickle_data(data, path):
    """TODO

    Args:
        data: TODO
        path: TODO

    Returns:
        TODO

    """
    assert data is not None
    assert path is not None
    with open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # check pickled data
    with open(path, "rb") as handle:
        pickled_data = pickle.load(handle)
    return pickled_data


def save_experiment_in_mongodb(mongo_client, document_dict):
    """TODO

    Args:
        mongo_client: TODO
        document_dict: TODO

    Returns:
        TODO

    """
    # connect to mongo db
    db = mongo_client.admin
    experiments_data = db.experiments_data.with_options(
        write_concern=WriteConcern(w=1, wtimeout=10000, fsync=True)
    )
    mongodb_id = experiments_data.insert_one(document_dict).inserted_id

    # test loading stored data
    return experiments_data.find_one({"_id": mongodb_id})


def _get_results_from_ray(selection_result_list):
    assert isinstance(selection_result_list, list)
    if not (
        isinstance(selection_result_list[0], pd.DataFrame)
        or isinstance(selection_result_list[0], dict)
    ):
        import ray

        assert isinstance(selection_result_list[0], ray._raylet.ObjectRef)
        selection_result_list_with_data = []
        for selection_result_id in selection_result_list:
            selection_result_list_with_data.append(ray.get(selection_result_id))
            del selection_result_id
        del selection_result_list
        return selection_result_list_with_data
    else:
        return selection_result_list


def save_raw_selection_result_per_method(
    selection_result_list, feature_selection_method_name, settings
):
    """Save feature selection result for the given feature selection method.

    Args:
        selection_result_list: List including results for each outer cross-validation iteration.
        feature_selection_method_name: Embedded feature selection method.
        settings: Settings.

    """
    selection_result_list = _get_results_from_ray(selection_result_list)
    check_feature_selection_result.check_feature_selection_result_outer_cv(
        selection_result_list, feature_selection_method_name, settings
    )

    if settings.store_result:
        # # store data in mongodb
        # if settings["env"] == "cluster":
        #     converted_raw_selection_result = (
        #         _convert_raw_selection_result_from_dataframe_to_list(
        #             selection_result_list
        #         )
        #     )
        #     assert isinstance(converted_raw_selection_result, List)
        #
        #     # create experiment
        #     experiment = {
        #         "experiment_id": settings["experiment_id"],
        #         "mongodb_settings_id": settings["mongodb_id"],
        #         feature_selection_method_name: converted_raw_selection_result,
        #     }
        #     saved_data = save_experiment_in_mongodb(
        #         mongo_client=MongoClient(settings["mongo_db_url"]),
        #         document_dict=experiment,
        #     )
        #
        #     # test loaded data
        #     for key in saved_data:
        #         if key == "experiment_id":
        #             assert saved_data[key] == settings["experiment_id"]
        #         if key == "mongodb_settings_id":
        #             assert saved_data[key] == settings["mongodb_id"]
        #         if key == feature_selection_method_name:
        #             assert converted_raw_selection_result == saved_data[key]
        #             if "Reverse" in feature_selection_method_name:
        #                 assert isinstance(saved_data[key][0], List)
        #                 # convert back to pandas
        #                 new_list = []
        #                 for cv_iteration_result in saved_data[key]:
        #                     new_list.append(pd.DataFrame(cv_iteration_result))
        #                 check_feature_selection_result.check_feature_selection_result_outer_cv(
        #                     new_list, feature_selection_method_name, settings
        #                 )
        #     print(f"Result {feature_selection_method_name} stored in DB")
        #     del converted_raw_selection_result
        #     del selection_result_list
        #     del saved_data
        #     # close the connection to MongoDB
        #     # client.close()

        # pickle data
        # else:
        assert isinstance(settings["data_storage"]["path_selection_results"], str)
        assert len(settings["data_storage"]["path_selection_results"]) > 0
        path = (
            f"{settings['cwd_path']}{settings['data_storage']['path_selection_results']}"
            f"/{settings['data']['name']}_experiment{settings['experiment_id']}_raw_selection_{feature_selection_method_name}.pkl"
        )
        pickled_data = pickle_data(selection_result_list, path)
        check_feature_selection_result.check_feature_selection_result_outer_cv(
            pickled_data, feature_selection_method_name, settings
        )
        print(f"Pickled {feature_selection_method_name} at {datetime.datetime.now()}")
        del selection_result_list
        del pickled_data


def save_duration_and_power_consumption(
    settings,
    benchmark_dict: Optional[dict],
    element: str,
):
    """TODO

    Args:
        settings: TODO
        benchmark_dict: TODO
        element: TODO

    Returns:
        TODO

    """
    if settings["env"] == "cluster":
        _logger.debug(benchmark_dict)
        assert (
            element == "preprocessing"
            or element == "feature_selection"
            or element == "baseline"
        )
        if element != "baseline":
            assert type(benchmark_dict["time"]) == datetime.timedelta

            # convert to seconds for mongodb
            duration = benchmark_dict["time"]
            duration_in_seconds = benchmark_dict["time"].total_seconds()
            benchmark_dict["time"] = duration_in_seconds
            print(type(duration_in_seconds), type(duration_in_seconds))

        client = MongoClient(settings["mongo_db_url"])
        db = client.admin
        experiments_meta_data = db.experiments_meta_data
        loaded_experiment_data = experiments_meta_data.find_one(
            {"_id": settings["mongodb_id"]}
        )
        updated_experiment_data = {"$set": {f"benchmark_{element}": benchmark_dict}}
        experiments_meta_data.update_one(
            loaded_experiment_data, updated_experiment_data
        )

        if element != "baseline":
            # test loading stored duration data
            new_loaded_data = experiments_meta_data.find_one(
                {"_id": settings["mongodb_id"]}
            )
            for key in new_loaded_data:
                if key == f"benchmark_{element}":
                    assert new_loaded_data[key] == benchmark_dict
                    timedelta_str = new_loaded_data[key]["time"]
                    assert datetime.timedelta(seconds=timedelta_str) == duration
                    _logger.debug(f" loaded_duration {element} = {timedelta_str}")
                    _logger.debug(
                        f" loaded_duration {element} timedelta = {datetime.timedelta(seconds=timedelta_str)}"
                    )

        # close the connection to MongoDB
        client.close()
