# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import tomlkit
import initialize_settings

n_jobs = 4


def parallel_training_gpu_single_node():
    # parallel hpo single node, threading
    with open(initialize_settings.file_name, mode="rt", encoding="utf-8") as fp:
        config_toml_file = tomlkit.load(fp)
    config_toml_file["parallel_processes"]["n_jobs_training"] = n_jobs
    config_toml_file["parallel_processes"]["num_threads_lightgbm"] = 0
    # set device_type_lightgbm to "cpu" for CPU instead of GPU training
    config_toml_file["parallel_processes"]["device_type_lightgbm"] = "cuda"
    # more details in https://lightgbm.readthedocs.io/en/latest/Parallel-Learning-Guide.html
    config_toml_file["parallel_processes"]["tree_learner"] = "feature"
    with open(initialize_settings.file_name, mode="wt", encoding="utf-8") as fp:
        tomlkit.dump(config_toml_file, fp)


def main():
    initialize_settings.serial_init()
    parallel_training_gpu_single_node()


if __name__ == "__main__":
    print("parallel training gpu single node")
    main()
