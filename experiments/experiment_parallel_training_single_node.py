import tomlkit
import initialize_settings

n_jobs = 4


def parallel_training_cpu_single_node():
    # parallel hpo single node, threading
    with open(initialize_settings.file_name, mode="rt", encoding="utf-8") as fp:
        config_toml_file = tomlkit.load(fp)
    config_toml_file["parallel_processes"]["n_jobs_training"] = n_jobs
    config_toml_file["parallel_processes"]["num_threads_lightgbm"] = n_jobs
    # set device_type_lightgbm to "cpu" for CPU instead of GPU training
    config_toml_file["parallel_processes"]["device_type_lightgbm"] = "cpu"
    # more details in https://lightgbm.readthedocs.io/en/latest/Parallel-Learning-Guide.html
    config_toml_file["parallel_processes"]["tree_learner"] = "feature"
    with open(initialize_settings.file_name, mode="wt", encoding="utf-8") as fp:
        tomlkit.dump(config_toml_file, fp)


def main():
    initialize_settings.serial_init()
    parallel_training_cpu_single_node()


if __name__ == "__main__":
    # # benchmark energy consumption baseline
    # baseline_benchmark_dict = power_measurement.get_power_consumption_baseline(
    #     node_power_measurement_dict=initialize_benchmark(), seconds=30
    # )
    # store_experiments.save_duration_and_power_consumption(
    #     settings=settings, benchmark_dict=baseline_benchmark_dict, element="baseline"
    # )
    print("parallel training cpu single node")
    main()
