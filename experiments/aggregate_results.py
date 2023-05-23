# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Main program to aggregate ensemble feature selection results."""
import tomlkit

from ensemble_feature_selection_benchmark.aggregation import result
from ensemble_feature_selection_benchmark.aggregation.aggregation import (
    aggregate_selections,
)
from ensemble_feature_selection_benchmark import load_experiments
from config import settings


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

# set testing to avoid starting a new experiment
file_name = "../ensemble-feature-selection-benchmark/settings.toml"
with open(file_name, mode="rt", encoding="utf-8") as fp:
    config_toml_file = tomlkit.load(fp)
config_toml_file["testing"] = True
config_toml_file["parallel_processes"]["init_ray"] = False
with open(file_name, mode="wt", encoding="utf-8") as fp:
    tomlkit.dump(config_toml_file, fp)

feature_selection_result_per_method = result.get_feature_selection_results_per_method(
    raw_feature_selection_result, stored_settings
)
aggregated_data = aggregate_selections(
    feature_selection_result_per_method, experiment_id
)
print(aggregated_data)
