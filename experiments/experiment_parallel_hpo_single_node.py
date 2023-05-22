# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import tomlkit
import initialize_settings


def parallel_hpo_single_node_threading():
    # parallel hpo single node, threading
    with open(initialize_settings.file_name, mode="rt", encoding="utf-8") as fp:
        config_toml_file = tomlkit.load(fp)
    # If this argument is set to -1, the number is set to CPU count.
    # Parallelization using threading. May suffer from Python’s GIL.
    # It is recommended to use process-based parallelization if func is CPU bound.
    # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
    config_toml_file["parallel_processes"]["hpo_reverse"] = -1
    config_toml_file["parallel_processes"]["hpo_standard"] = -1
    with open(initialize_settings.file_name, mode="wt", encoding="utf-8") as fp:
        tomlkit.dump(config_toml_file, fp)


def main():
    initialize_settings.serial_init()
    parallel_hpo_single_node_threading()


if __name__ == "__main__":
    print("parallel hpo single node")
    main()
