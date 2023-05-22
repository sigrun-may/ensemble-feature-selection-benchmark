# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import os

from dynaconf import Dynaconf


settings = Dynaconf(
    root_path=os.path.dirname(os.path.realpath(__file__)),
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml", "experiment_settings.toml", ".secrets.toml"],
)
