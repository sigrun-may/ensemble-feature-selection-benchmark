# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Verify that data has the correct structure and content."""

import math
from typing import List

import pandas as pd


def _check_result_standard_feature_selection(
    standard_feature_selection_result_dict: dict, settings
):
    # check if standard_feature_selection_result_dict has the expected 5 elements:
    # shap values, validation metric, best hyperparameters, macro feature importance, micro feature importance
    assert len(standard_feature_selection_result_dict) == 5

    # check shap values
    assert "shap_values" in standard_feature_selection_result_dict.keys()
    assert isinstance(standard_feature_selection_result_dict["shap_values"], List)
    for element in standard_feature_selection_result_dict["shap_values"]:
        assert isinstance(element, List)
        if len(element) > 0:
            for entry in element:
                assert isinstance(entry, float)
                assert not math.isnan(entry)
    assert (
        len(standard_feature_selection_result_dict["shap_values"])
        == settings["cv"]["n_inner_folds"]
    )

    # check validation metric
    assert "validation_metric_value" in standard_feature_selection_result_dict.keys()
    assert isinstance(
        standard_feature_selection_result_dict["validation_metric_value"], float
    )

    # check hyper parameters
    assert "best_hyper_parameters" in standard_feature_selection_result_dict.keys()
    assert isinstance(
        standard_feature_selection_result_dict["best_hyper_parameters"], dict
    )

    # check macro feature importance
    assert "macro_feature_importances" in standard_feature_selection_result_dict.keys()
    assert (
        len(standard_feature_selection_result_dict["macro_feature_importances"])
        == settings["cv"]["n_inner_folds"]
    )
    assert isinstance(
        standard_feature_selection_result_dict["macro_feature_importances"], List
    )
    for element in standard_feature_selection_result_dict["macro_feature_importances"]:
        assert isinstance(element, List)
        if len(element) > 0:
            for entry in element:
                assert isinstance(entry, float)
                assert not math.isnan(entry)

    # check micro feature importance
    assert "micro_feature_importance" in standard_feature_selection_result_dict.keys()
    assert isinstance(
        standard_feature_selection_result_dict["micro_feature_importance"], List
    )
    if len(standard_feature_selection_result_dict["micro_feature_importance"]) > 0:
        for element in standard_feature_selection_result_dict[
            "micro_feature_importance"
        ]:
            assert isinstance(element, float)
            assert not math.isnan(element)


def _check_result_reverse_feature_selection(  # NOTE: kann private?
    reverse_feature_selection_result_df: pd.DataFrame,
):
    assert len(reverse_feature_selection_result_df.columns) == 2
    assert list(reverse_feature_selection_result_df.columns) == ["unlabeled", "labeled"]
    for column_name in reverse_feature_selection_result_df:
        assert pd.api.types.is_float_dtype(
            reverse_feature_selection_result_df[column_name]
        )


def check_feature_selection_result_outer_cv(
    feature_selection_result_list: List, feature_selection_method: str, settings
):
    """Check structure and data types of feature selection result.

    Args:
        feature_selection_result_list: List with all selection results for each iteration of the outer cross-validation.
        feature_selection_method: Feature selection method name.
        settings: Project settings (dict or dynaconf object).
    """
    assert (
        len(feature_selection_result_list) == settings["cv"]["n_outer_folds"]
    ), f"{feature_selection_result_list}, {feature_selection_method}"
    for single_outer_fold_result in feature_selection_result_list:
        assert isinstance(single_outer_fold_result, pd.DataFrame) or isinstance(
            single_outer_fold_result, dict
        ), f"{type(single_outer_fold_result)}"
        if isinstance(single_outer_fold_result, dict):
            _check_result_standard_feature_selection(single_outer_fold_result, settings)
        elif isinstance(single_outer_fold_result, pd.DataFrame):
            _check_result_reverse_feature_selection(single_outer_fold_result)
            # exclude the label from feature names
            assert (
                len(single_outer_fold_result)
                == settings["data"]["number_of_features"] - 1
            )


def check_raw_feature_selection_result_dict(
    raw_feature_selection_result_dict: dict, settings
):
    """Check structure and data types of all feature selection results.

    Args:
        raw_feature_selection_result_dict: Dict with all selection results of all feature selection methods.
        Each selection results includes a list of results for each iteration of each outer cross-validation.
        settings: Project settings (dict or dynaconf object).
    """
    assert isinstance(raw_feature_selection_result_dict, dict)

    # all selection methods + feature names
    assert (
        len(raw_feature_selection_result_dict)
        == len(settings["selection_method"]["methods"]) + 1
    )
    for key, value in raw_feature_selection_result_dict.items():
        assert isinstance(value, List), f"{value}{type(value)}"
        if key == "feature_names":
            assert len(value) == settings["data"]["number_of_features"]
            for feature_name in value:
                assert isinstance(feature_name, str)
        else:
            check_feature_selection_result_outer_cv(value, key, settings)
