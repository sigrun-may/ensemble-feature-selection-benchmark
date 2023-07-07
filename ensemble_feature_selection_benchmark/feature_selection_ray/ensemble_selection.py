# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Ensemble feature selection."""
import time

import datetime

import ray

from config import settings
from ensemble_feature_selection_benchmark import store_experiments
from ensemble_feature_selection_benchmark.feature_selection_ray.feature_selection_classes import (
    str_to_class,
)


@ray.remote
def _remote_select_feature_subsets(
    settings_id, preprocessed_data_id, feature_selection_class, outer_cv_iteration
):
    return feature_selection_class.select_feature_subsets(
        data=preprocessed_data_id,
        outer_cv_iteration=outer_cv_iteration,
        settings_id=settings_id,
    )


# @ray.remote(num_gpus=0.3)
@ray.remote(scheduling_strategy="SPREAD")
def _remote_serial_outer_cv(
    settings_id,
    preprocessed_data_id,
    feature_selection_class,
    feature_selection_method_name,
):
    raw_selection_result_list = []
    for outer_cv_iteration in range(settings_id.cv.n_outer_folds):
        raw_selection_result = feature_selection_class.select_feature_subsets(
            data=preprocessed_data_id,
            outer_cv_iteration=outer_cv_iteration,
            settings_id=settings_id,
        )
        raw_selection_result_list.append(raw_selection_result)

    store_experiments.save_raw_selection_result_per_method(
        selection_result_list=raw_selection_result_list,
        feature_selection_method_name=feature_selection_method_name,
        settings=settings_id,
    )
    del raw_selection_result_list


def _parallel_outer_cv(
    preprocessed_data_id, feature_selection_class, n_outer_folds: int, settings_id=None
):
    raw_selection_result_object_list = []
    for outer_cv_iteration in range(n_outer_folds):
        raw_selection_result_object = _remote_select_feature_subsets.remote(
            settings_id,
            preprocessed_data_id,
            feature_selection_class,
            outer_cv_iteration,
        )
        raw_selection_result_object_list.append(raw_selection_result_object)
        del raw_selection_result_object
    return raw_selection_result_object_list


@ray.remote(scheduling_strategy="SPREAD", num_cpus=6)  # (num_cpus=3)
def _remote_parallel_outer_cv(
    settings_id,
    preprocessed_data_id,
    feature_selection_class,
    feature_selection_method_name,
):
    raw_selection_result_object_list = []
    for outer_cv_iteration in range(6):
        raw_selection_result_object = _remote_select_feature_subsets.remote(
            settings_id,
            preprocessed_data_id,
            feature_selection_class,
            outer_cv_iteration,
        )
        raw_selection_result_object_list.append(raw_selection_result_object)
        del raw_selection_result_object
    #     _parallel_outer_cv(
    #     settings_id=settings_id,
    #     preprocessed_data_id=preprocessed_data_id,
    #     feature_selection_class=feature_selection_class,
    #     n_outer_folds=settings_id.cv.n_outer_folds,
    # )
    print("_remote_parallel_outer_cv")
    store_experiments.save_raw_selection_result_per_method(
        selection_result_list=raw_selection_result_object_list,
        feature_selection_method_name=feature_selection_method_name,
        settings=settings_id,
    )
    del raw_selection_result_object_list


def ensemble_feature_selection(preprocessed_data_id):
    """Ensemble feature selection.

    Args:
        preprocessed_data_id: Ids of preprocessed input data in the ray object store.

    Returns:
        Current UTC time.

    """
    settings_id = ray.put(settings)
    parallel_methods_selection_object_references_list = []
    for feature_selection_method in settings.selection_method.methods:
        # skip reverse feature selection
        if "everse" not in feature_selection_method:
            print(f"Start {feature_selection_method}")
            feature_selection_class = str_to_class(feature_selection_method)

            # parallel outer cross-validation and parallel feature selection methods
            if (settings.parallel_processes.feature_selection_methods > 1) and (
                settings.parallel_processes.outer_cv > 1
            ):
                none_object_reference = _remote_parallel_outer_cv.remote(
                    settings_id,
                    preprocessed_data_id,
                    feature_selection_class,
                    feature_selection_method,
                )
                parallel_methods_selection_object_references_list.append(
                    none_object_reference
                )
                # _remote_parallel_outer_cv.remote(
                #     settings_id,
                #     preprocessed_data_id,
                #     feature_selection_class,
                #     feature_selection_method,
                # )

            # parallel outer cross-validation and serial feature selection methods
            elif (settings.parallel_processes.feature_selection_methods < 2) and (
                settings.parallel_processes.outer_cv > 1
            ):
                raw_selection_result = _parallel_outer_cv(
                    settings_id=settings_id,
                    preprocessed_data_id=preprocessed_data_id,
                    feature_selection_class=feature_selection_class,
                    n_outer_folds=settings.cv.n_outer_folds,
                )
                store_experiments.save_raw_selection_result_per_method(
                    raw_selection_result, feature_selection_method, settings
                )
                del raw_selection_result

            # serial outer cross-validation and parallel feature selection methods
            elif (settings.parallel_processes.feature_selection_methods > 1) and (
                settings.parallel_processes.outer_cv < 2
            ):
                none_object_reference = _remote_serial_outer_cv.remote(
                    settings_id,
                    preprocessed_data_id,
                    feature_selection_class,
                    feature_selection_method,
                )
                parallel_methods_selection_object_references_list.append(
                    none_object_reference
                )

            # serial outer cross-validation and serial feature selection methods
            else:
                raw_selection_result_list = []
                for outer_cv_iteration in range(settings.cv.n_outer_folds):
                    raw_selection_result = (
                        feature_selection_class.select_feature_subsets(
                            data=preprocessed_data_id,
                            outer_cv_iteration=outer_cv_iteration,
                            settings_id=settings,
                        )
                    )
                    print(raw_selection_result)
                    raw_selection_result_list.append(raw_selection_result)
                store_experiments.save_raw_selection_result_per_method(
                    raw_selection_result_list, feature_selection_method, settings
                )
                del raw_selection_result

    ray.get(parallel_methods_selection_object_references_list)

    # reverse feature selection with parallel features
    for feature_selection_method in settings.selection_method.methods:
        if "everse" in feature_selection_method:
            print(f"Start {feature_selection_method}")
            feature_selection_class = str_to_class(feature_selection_method)
            raw_selection_result_list = []
            for outer_cv_iteration in range(settings.cv.n_outer_folds):
                raw_selection_result = feature_selection_class.select_feature_subsets(
                    data=preprocessed_data_id,
                    outer_cv_iteration=outer_cv_iteration,
                    settings_id=settings,
                )
                print(raw_selection_result)
                raw_selection_result_list.append(raw_selection_result)
            store_experiments.save_raw_selection_result_per_method(
                raw_selection_result_list, feature_selection_method, settings
            )
            del raw_selection_result

    return datetime.datetime.utcnow(), settings_id
