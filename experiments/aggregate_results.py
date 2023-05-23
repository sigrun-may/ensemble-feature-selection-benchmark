# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Main program to aggregate ensemble feature selection results."""

from ensemble_feature_selection_benchmark.aggregation import result
from ensemble_feature_selection_benchmark.aggregation.aggregation import (
    aggregate_selections,
)
from ensemble_feature_selection_benchmark import load_experiments
from config import settings

# SETTINGS must have TESTING = TRUE!!!!

experiment_id = 1
if settings.env == "local":
    (
        raw_feature_selection_result,
        stored_settings,
    ) = load_experiments._load_raw_selection_results(
        experiment_id, cwd_path=settings["cwd_path"]
    )
elif settings.env == "cluster":
    (
        raw_feature_selection_result,
        stored_settings,
    ) = load_experiments._load_raw_selection_results(
        experiment_id, mongo_db_url=settings["mongo_db_url"]
    )
else:
    raise ValueError(
        "No valid environment found in settings. Set environment to 'local' or 'cluster'."
    )
feature_selection_result_per_method = result.get_feature_selection_results_per_method(
    raw_feature_selection_result, settings
)
aggregated_data = aggregate_selections(
    feature_selection_result_per_method, experiment_id
)
print(aggregated_data)
