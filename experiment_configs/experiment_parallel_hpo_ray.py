# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import tomlkit
import initialize_settings


def parallel_hpo_ray_tune():
    # parallel hpo ray tune
    with open(initialize_settings.file_name, mode="rt", encoding="utf-8") as fp:
        config_toml_file = tomlkit.load(fp)
    config_toml_file["parallel_processes"]["init_ray"] = True
    # Maximum number of trials to run concurrently. Must be non-negative.
    # If None or 0, no limit will be applied. https://docs.ray.io/en/latest/_modules/ray/tune/tune.html
    config_toml_file["parallel_processes"]["max_concurrent_trials_hpo_ray"] = 16
    with open(initialize_settings.file_name, mode="wt", encoding="utf-8") as fp:
        tomlkit.dump(config_toml_file, fp)


def main():
    initialize_settings.serial_init()
    parallel_hpo_ray_tune()


if __name__ == "__main__":
    print("parallel hpo ray")
    main()
