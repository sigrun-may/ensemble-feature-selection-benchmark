# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Working on results."""

import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ensemble_feature_selection_benchmark import check_feature_selection_result

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def _get_selected_feature_subsets(feature_weights_pd: pd.DataFrame) -> Dict:
    selected_feature_subsets_dict = {}
    for method in feature_weights_pd:
        assert (
            not feature_weights_pd[method].values.min() < 0
        ), f"{feature_weights_pd[method].values.min()}"
        selected_features_mask = feature_weights_pd[method].gt(0)
        selected_feature_subsets_dict[method] = set(
            feature_weights_pd[method][selected_features_mask].index
        )
    return selected_feature_subsets_dict


def _evaluate_selection_result(
    selection_result, selection_method_name: str, settings
) -> pd.DataFrame:
    # analyze reverse feature selection
    if "Reverse" in selection_method_name:
        feature_weights_df = pd.DataFrame(index=selection_result.index)
        difference = selection_result["labeled"].subtract(selection_result["unlabeled"])
        difference[difference <= 0] = 0
        _logger.debug(difference.sort_values(ascending=False))
        feature_weights_df[f"{selection_method_name}"] = difference

    #  analyze standard feature selection
    else:
        feature_weights_df = pd.DataFrame()
        #  aggregate coefficients (sum up absolute values)
        cumulated_coefficients = np.sum(
            np.abs(np.asarray(selection_result["macro_feature_importances"])), axis=0
        )
        feature_weights_df[
            f"{selection_method_name} cumulated_macro_importance"
        ] = cumulated_coefficients
        feature_weights_df[
            f"{selection_method_name} cumulated_micro_importance"
        ] = selection_result["micro_feature_importance"]

        #  aggregate shap values (sum up absolute values)
        if len(selection_result["shap_values"][0]) > 0:
            feature_weights_df[
                f"{selection_method_name} cumulated_shap_values"
            ] = np.sum(
                np.abs(selection_result["shap_values"]),
                axis=0,
            )
    assert (
        feature_weights_df.shape[0]
        == settings["data"]["number_of_features"] - 1  # exclude label
    ), feature_weights_df.info(verbose=True)
    return feature_weights_df


def find_robust_features(feature_subsets_list: List[dict]):
    # TODO: add DocString

    # initialize robust subset
    robust_feature_subsets = feature_subsets_list[0]

    # find intersection between the outer cross-validation folds
    for feature_subsets_dict in feature_subsets_list:
        for key in feature_subsets_dict.keys():
            robust_feature_subsets[key] = robust_feature_subsets[key].intersection(
                feature_subsets_dict[key]
            )

    # print result
    print(robust_feature_subsets, robust_feature_subsets)
    if len(robust_feature_subsets) > 1:
        for key, robust_feature_subset in robust_feature_subsets.items():
            robust_biomarkers = {
                feature for feature in robust_feature_subsets[key] if "bm" in feature
            }
            print("robust_biomarkers", robust_biomarkers)
            print("robust_feature_subset", robust_feature_subset)
    return robust_feature_subsets


def get_feature_selection_results_per_method(raw_selection_results_dict, settings):
    # TODO: add DocString
    check_feature_selection_result.check_raw_feature_selection_result_dict(
        raw_selection_results_dict, settings
    )
    feature_weights_dict = defaultdict(list)
    # iterate over different machine learning methods
    for (
        feature_selection_method,
        raw_selection_results_list,
    ) in raw_selection_results_dict.items():
        print(feature_selection_method)
        assert isinstance(raw_selection_results_list, List)
        if feature_selection_method == "feature_names":
            continue
        # evaluate selection result of each outer cv iteration
        for raw_selection_result in raw_selection_results_list:
            # calculate and scale feature weights
            all_feature_weights_df = _evaluate_selection_result(
                raw_selection_result, feature_selection_method, settings
            )
            assert isinstance(all_feature_weights_df, pd.DataFrame)
            # index = pd.Index(raw_selection_results_dict["feature_names"][1:])  # exclude label
            # feature_weights_df.set_index(index, inplace=True, verify_integrity=True)
            scaler = MinMaxScaler()
            scaled_data_np = scaler.fit_transform(all_feature_weights_df.values)
            scaled_feature_weights_df = pd.DataFrame(
                data=scaled_data_np,
                index=pd.Index(
                    raw_selection_results_dict["feature_names"][1:]
                ),  # exclude label
                columns=all_feature_weights_df.columns,
            )
            # iterate over feature selection methods (micro / macro importance, shap values...)
            for single_feature_selection_method in all_feature_weights_df.columns:
                feature_weights_dict[single_feature_selection_method].append(
                    scaled_feature_weights_df[single_feature_selection_method]
                )

    # concatenate result for each outer cv
    for single_feature_selection_method, list_of_series in feature_weights_dict.items():
        feature_weights_dict[single_feature_selection_method] = pd.concat(
            list_of_series, axis=1
        )
    return feature_weights_dict
