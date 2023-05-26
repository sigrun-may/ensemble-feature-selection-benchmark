# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import logging
from os import makedirs
from os.path import exists

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import settings
from ensemble_feature_selection_benchmark.aggregation import aggregation_methods

"""Aggregate the results of feautre selection methods."""

_logger = logging.getLogger(__name__)


def _aggregate_outer_cv(aggregation_method_cv: str, result_per_method_dict: dict):
    cv_aggregation_df = pd.DataFrame()
    aggregator = aggregation_methods.str_to_class(aggregation_method_cv)
    # iterate over the different feature selection methods
    for feature_selection_method, data in result_per_method_dict.items():
        if feature_selection_method in settings.ensemble.methods:
            assert isinstance(data, pd.DataFrame), f"{aggregation_method_cv}, {data}"
            aggregated_data = aggregator.aggregate(data)
            # scale aggregated results from outer cv
            scaler = MinMaxScaler()
            scaled_data_np = scaler.fit_transform(
                np.asarray(aggregated_data).reshape(-1, 1)
            ).flatten()
            cv_aggregation_df[feature_selection_method] = pd.Series(
                scaled_data_np, index=data.index
            )
    return cv_aggregation_df


def aggregate_selections(result_per_method_dict, experiment_id=None):
    """Aggregates the results of the feature selection methods.

    Args:
        result_per_method_dict: dictionary containing the results per selection method.
        experiment_id: the id of the experiment the results of the selection methods should be aggregated.

    Returns:
        dictionary containing the aggregation results per aggregation method that is selected in the settings.

    """
    if experiment_id is None:
        experiment_id = settings.experiment_id
    # path to store the csv result files
    path = f"{settings.cwd_path}/aggregation_results/experiment{experiment_id}"
    if not exists(path):
        makedirs(path)

    ensemble_aggregation_dict = {}
    # iterate over aggregation methods for the outer cv
    for aggregation_method_cv in settings.aggregation_method_cv.methods:
        aggregated_cv_data_df = _aggregate_outer_cv(
            aggregation_method_cv, result_per_method_dict
        )
        aggregated_cv_data_df.to_csv(
            f"{path}/single_feature_selection_aggregated_cv_with_{aggregation_method_cv}.csv"
        )

        # iterate over the aggregation methods for ensemble feature selection
        for (
            ensemble_aggregation_method
        ) in settings.aggregation_ensemble_feature_selection.methods:
            # aggregate ensemble methods
            ensemble_aggregator = aggregation_methods.str_to_class(
                ensemble_aggregation_method
            )
            aggregated_ensemble_data = ensemble_aggregator.aggregate(
                aggregated_cv_data_df
            )
            ensemble_aggregation_dict[
                ensemble_aggregation_method
            ] = aggregated_ensemble_data

        result_df = pd.DataFrame(ensemble_aggregation_dict)
        result_df.to_csv(
            f"{path}/ensemble_aggregated_cv_with_{aggregation_method_cv}.csv"
        )
    return ensemble_aggregation_dict
