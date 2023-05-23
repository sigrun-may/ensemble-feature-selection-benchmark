# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Access mongodb."""
from pprint import pprint

from config import settings
from ensemble_feature_selection_benchmark import load_experiments


loaded_experiment_data = {}
counter = 1
while loaded_experiment_data is not None:
    loaded_experiment_data = load_experiments._load_meta_data_from_mongodb(
        experiment_id=counter, mongo_db_url=settings["mongodb_url"]
    )
    pprint(loaded_experiment_data)
    counter += 1
