# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Ensemble feature selection."""

import datetime

from config import settings
from ensemble_feature_selection_benchmark import (
    check_feature_selection_result,
    store_experiments,
)
from ensemble_feature_selection_benchmark.data_types import PreprocessedData
from ensemble_feature_selection_benchmark.feature_selection_standard.feature_selection_classes import (
    str_to_class,
)


def cross_validate_feature_subsets(
    preprocessed_data: PreprocessedData, feature_selection_method: str
):
    """Outer cross-validation for given feature selection method.

    Args:
        preprocessed_data:
        feature_selection_method:

    Returns:
        None. Results are stored.

    """
    raw_selection_result_list = []
    feature_selection_class = str_to_class(feature_selection_method)
    for outer_cv_iteration in range(settings["cv"]["n_outer_folds"]):
        raw_selection_result = feature_selection_class.select_feature_subsets(
            data=preprocessed_data, outer_cv_iteration=outer_cv_iteration
        )
        raw_selection_result_list.append(raw_selection_result)
    store_experiments.save_raw_selection_result_per_method(
        raw_selection_result_list, feature_selection_method, settings
    )


def ensemble_feature_selection(preprocessed_data: PreprocessedData):
    """Ensemble feature selection.

    Args:
        preprocessed_data: Preprocessed data to analyse as output from 'preprocessing.py'.

    Returns:
        Current UTC time.

    """
    for feature_selection_method in settings["selection_method"]["methods"]:
        print(f"Start {feature_selection_method}")
        cross_validate_feature_subsets(preprocessed_data, feature_selection_method)
    return datetime.datetime.utcnow(), None
