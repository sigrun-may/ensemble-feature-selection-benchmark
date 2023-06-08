# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Main program for benchmarking."""

import logging
import threading
from datetime import datetime
from threading import Event
from config import settings

from ensemble_feature_selection_benchmark import (
    load_experiments,
    power_measurement,
    preprocessing,
    store_experiments,
)

if settings.parallel_processes.init_ray:
    import ray

    from ensemble_feature_selection_benchmark.feature_selection_ray import (
        ensemble_selection,
    )

    print("Parallelize with ray")
    print("cwd_path ", settings.cwd_path)
    exclude_data_path = f"{settings.cwd_path}/experiments/data/*"
    exclude_serial_feature_selection_path = f"{settings.cwd_path}/ensemble_feature_selection_benchmark/feature_selection_standard/*"
    exclude_git_directory_path = ".git"
    runtime_env = {
        "working_dir": settings.cwd_path,
        "excludes": [
            exclude_data_path,
            exclude_serial_feature_selection_path,
            exclude_git_directory_path,
        ],
    }
    status = ray.init(
        address="auto",
        runtime_env=runtime_env,
        ignore_reinit_error=True,
        include_dashboard=True,
        log_to_driver=False,
    )
    print(status)
else:
    print("Serial calculation")
    from ensemble_feature_selection_benchmark.feature_selection_standard import (
        ensemble_selection,
    )

_logger = logging.getLogger(__name__)


def run_experiment():
    start_time_workflow = datetime.utcnow()

    # start preprocessing
    preprocessed_data = preprocessing.preprocess_data()

    # feature selection
    if settings.parallel_processes.init_ray:
        assert (
            len(preprocessed_data.inner_preprocessed_data_splits_list)
            == settings.cv.n_outer_folds
        )
        inner_preprocessed_data_id_list = []
        for (
            preprocessed_inner_cv
        ) in preprocessed_data.inner_preprocessed_data_splits_list:
            assert len(preprocessed_inner_cv) == settings.cv.n_inner_folds
            preprocessed_inner_cv_ids = []
            for preprocessed_inner_cv_iteration in preprocessed_inner_cv:
                preprocessed_inner_cv_iteration_id = ray.put(
                    preprocessed_inner_cv_iteration
                )
                preprocessed_inner_cv_ids.append(preprocessed_inner_cv_iteration_id)
            inner_preprocessed_data_id_list.append(preprocessed_inner_cv_ids)
            # inner_preprocessed_data_id_list.append(ray.put(preprocessed_inner_cv))
            del preprocessed_inner_cv_ids
            del preprocessed_inner_cv
        assert len(inner_preprocessed_data_id_list) == settings.cv.n_outer_folds
        preprocessed_data.inner_preprocessed_data_splits_list = (
            inner_preprocessed_data_id_list
        )
        del inner_preprocessed_data_id_list

        outer_preprocessed_data_id_list = []
        for (
            preprocessed_outer_cv_iteration
        ) in preprocessed_data.outer_preprocessed_data_splits:
            preprocessed_outer_cv_iteration_id = ray.put(
                preprocessed_outer_cv_iteration
            )
            outer_preprocessed_data_id_list.append(preprocessed_outer_cv_iteration_id)
            del preprocessed_outer_cv_iteration
            del preprocessed_outer_cv_iteration_id
        preprocessed_data.outer_preprocessed_data_splits = (
            outer_preprocessed_data_id_list
        )
        del outer_preprocessed_data_id_list

        ensemble_selection.ensemble_feature_selection(ray.put(preprocessed_data))
    else:
        ensemble_selection.ensemble_feature_selection(preprocessed_data)
    print(f"finished workflow in {datetime.utcnow() - start_time_workflow}")


run_experiment()
