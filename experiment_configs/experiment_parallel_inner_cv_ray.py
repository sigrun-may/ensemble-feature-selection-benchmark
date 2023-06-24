# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import tomlkit
import initialize_settings


def parallel_inner_cv_ray():
    # parallel hpo ray tune
    with open(initialize_settings.file_name, mode="rt", encoding="utf-8") as fp:
        config_toml_file = tomlkit.load(fp)
    config_toml_file["parallel_processes"]["init_ray"] = True
    config_toml_file["parallel_processes"]["inner_cv"] = 2
    with open(initialize_settings.file_name, mode="wt", encoding="utf-8") as fp:
        tomlkit.dump(config_toml_file, fp)


def main():
    initialize_settings.serial_init()
    parallel_inner_cv_ray()


if __name__ == "__main__":
    print("parallel inner cv ray")
    main()
