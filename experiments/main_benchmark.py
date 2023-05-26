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
    # try to load preprocessed data
    preprocessed_data = load_experiments._load_preprocessed_data()
    if preprocessed_data is None:
        # start preprocessing
        preprocessed_data = benchmark("preprocessing", start_time_workflow)

    # feature selection
    if settings.parallel_processes.init_ray:
        inner_preprocessed_data_id_list = []
        for (
            preprocessed_inner_cv
        ) in preprocessed_data.inner_preprocessed_data_splits_list:
            # preprocessed_inner_cv_ids = []
            # for preprocessed_inner_cv_iteration in preprocessed_inner_cv:
            #     preprocessed_inner_cv_iteration_id = ray.put(
            #         preprocessed_inner_cv_iteration
            #     )
            #     preprocessed_inner_cv_ids.append(preprocessed_inner_cv_iteration_id)
            # inner_preprocessed_data_id_list.append(preprocessed_inner_cv_ids)
            inner_preprocessed_data_id_list.append(ray.put(preprocessed_inner_cv))
        preprocessed_data.inner_preprocessed_data_splits_list = (
            inner_preprocessed_data_id_list
        )

        outer_preprocessed_data_id_list = []
        for (
            preprocessed_outer_cv_iteration
        ) in preprocessed_data.outer_preprocessed_data_splits:
            preprocessed_outer_cv_iteration_id = ray.put(
                preprocessed_outer_cv_iteration
            )
            outer_preprocessed_data_id_list.append(preprocessed_outer_cv_iteration_id)
        preprocessed_data.outer_preprocessed_data_splits = (
            outer_preprocessed_data_id_list
        )

        benchmark("feature_selection", start_time_workflow, ray.put(preprocessed_data))
    else:
        benchmark("feature_selection", start_time_workflow, preprocessed_data)
    print(f"finished workflow in {datetime.utcnow() - start_time_workflow}")


def benchmark(workflow_element, start_time_workflow, data=None):
    print(f"Start {workflow_element}")
    # start time and power measurement
    benchmark_dict = power_measurement.initialize_benchmark_dict()
    event = Event()
    preprocessing_power_measurement_thread = threading.Thread(
        target=power_measurement._measure_power_usage, args=(benchmark_dict, event)
    )
    preprocessing_power_measurement_thread.start()

    # ...calculate
    if workflow_element == "preprocessing":
        result = preprocessing.preprocess_data()
    elif workflow_element == "feature_selection":
        result = ensemble_selection.ensemble_feature_selection(data)
    else:
        raise ValueError(
            "Workflow element can be 'preprocessing' or 'feature_selection'"
        )

    # stop time and power measurement
    benchmark_dict["time"] = datetime.now() - benchmark_dict["time"]
    event.set()
    preprocessing_power_measurement_thread.join()
    assert not preprocessing_power_measurement_thread.is_alive()
    workflow_duration = datetime.utcnow() - start_time_workflow
    benchmark_dict["workflow_duration"] = workflow_duration.total_seconds()
    print(benchmark_dict)
    store_experiments.save_duration_and_power_consumption(
        settings=settings, benchmark_dict=benchmark_dict, element=workflow_element
    )
    return result


run_experiment()
