# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

# start ray cluster in conda environment
# ray start --head

#python experiments/experiment_parallel_hpo_ray.py
#OMP_NUM_THREADS=1 python experiments/main_benchmark.py 2>&1 | sed '/.*LightGBM].*/d'
#
#rm -r /home/sigrun/PycharmProjects/ensemble-feature-selection-benchmark/_objective*

python experiments/experiment_parallel_feature_selection_methods_ray.py
OMP_NUM_THREADS=1 python experiments/main_benchmark.py 2>&1 | sed '/.*LightGBM].*/d'

python experiments/experiment_parallel_inner_cv_ray.py
OMP_NUM_THREADS=1 python experiments/main_benchmark.py 2>&1 | sed '/.*LightGBM].*/d'

python experiments/experiment_parallel_outer_cv_ray.py
OMP_NUM_THREADS=1 python experiments/main_benchmark.py 2>&1 | sed '/.*LightGBM].*/d'

#python experiments/experiment_parallel_features_reverse_ray.py
#OMP_NUM_THREADS=1 python experiments/main_benchmark.py 2>&1 | sed '/.*LightGBM].*/d'


