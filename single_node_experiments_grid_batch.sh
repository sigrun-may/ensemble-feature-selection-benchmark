# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

python experiments/experiment_serial_single_node.py
OMP_NUM_THREADS=1 python experiments/main_benchmark.py 2>&1 | sed '/.*LightGBM].*/d'
#
#python experiments/experiment_parallel_hpo_single_node.py
#OMP_NUM_THREADS=1 python experiments/main_benchmark.py 2>&1 | sed '/.*LightGBM].*/d'
#
#python experiments/experiment_parallel_training_single_node.py
#python experiments/main_benchmark.py 2>&1 | sed '/.*LightGBM].*/d'

#python experiments/experiment_parallel_training_gpu.py
#python experiments/main_benchmark.py 2>&1 | sed '/.*LightGBM].*/d'
