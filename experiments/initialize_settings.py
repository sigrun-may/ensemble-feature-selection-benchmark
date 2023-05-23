# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import tomlkit

file_name = "../ensemble-feature-selection-benchmark/settings.toml"


def serial_init():
    # serial
    with open(file_name, mode="rt", encoding="utf-8") as fp:
        config_toml_file = tomlkit.load(fp)
    config_toml_file["store_result"] = True
    config_toml_file["parallel_processes"]["init_ray"] = False
    config_toml_file["parallel_processes"]["hpo_reverse"] = 1
    config_toml_file["parallel_processes"]["hpo_standard"] = 1
    config_toml_file["parallel_processes"]["max_concurrent_trials_hpo_ray"] = 1
    config_toml_file["parallel_processes"]["inner_cv"] = 1
    config_toml_file["parallel_processes"]["outer_cv"] = 1
    config_toml_file["parallel_processes"]["feature_selection_methods"] = 1
    config_toml_file["parallel_processes"]["reverse_feature_selection"] = 1
    config_toml_file["parallel_processes"]["n_jobs_training"] = 1
    config_toml_file["parallel_processes"]["num_threads_lightgbm"] = 1
    # set device_type_lightgbm to "cpu" for CPU training
    config_toml_file["parallel_processes"]["device_type_lightgbm"] = "cpu"
    # more details in https://lightgbm.readthedocs.io/en/latest/Parallel-Learning-Guide.html
    config_toml_file["parallel_processes"]["tree_learner"] = "serial"
    with open(file_name, mode="wt", encoding="utf-8") as fp:
        tomlkit.dump(config_toml_file, fp)
