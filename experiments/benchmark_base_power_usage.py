# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from config import settings
from ensemble_feature_selection_benchmark import power_measurement, store_experiments


def main():
    # benchmark energy consumption baseline
    baseline_benchmark_dict = power_measurement._get_power_consumption_baseline(
        node_power_measurement_dict=power_measurement.initialize_benchmark_dict(),
        seconds=30,
    )
    store_experiments.save_duration_and_power_consumption(
        settings=settings, benchmark_dict=baseline_benchmark_dict, element="baseline"
    )


if __name__ == "__main__":
    print("Benchmark energy consumption baseline")
    main()
