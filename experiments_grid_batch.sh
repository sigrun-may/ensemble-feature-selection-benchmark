python experiments/experiment_serial_single_node.py
OMP_NUM_THREADS=1 python experiments/main_benchmark.py 2>&1 | sed '/.*LightGBM].*/d'

python experiments/experiment_parallel_hpo_single_node.py
OMP_NUM_THREADS=1 python experiments/main_benchmark.py 2>&1 | sed '/[LightGBM]/d'
